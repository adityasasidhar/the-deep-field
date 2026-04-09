"""
Creative GRPO — Modified Training Objective
============================================

Replaces:
  - Standard normalized advantage  →  creativity-aware advantage Â* = s_i · η_i · (1 + r_i)
  - KL divergence penalty          →  token-level entropy regularizer R_H

All tensors follow HuggingFace/TRL GRPO conventions:
  - logits: (B, T, V)
  - input_ids / attention_mask: (B, T)
  - rewards: (B,)  ∈ [-1, 1]
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

@dataclass
class CreativeGRPOConfig:
    # PPO clip
    clip_eps: float = 0.2

    # Entropy regularizer coefficient (replaces β·KL)
    lambda_entropy: float = 0.01

    # Confidence threshold: sequence log-prob above this → "confident"
    confidence_threshold: float = -1.5           # tune per model

    # Novelty score placeholder (inject computed η externally or use dummy)
    novelty_score_default: float = 1.0

    # Group size G for GRPO (number of completions per prompt)
    group_size: int = 8

    # Whether to detach old_log_probs from graph (True = standard PPO)
    detach_old: bool = True

    # Optional: weight for ECE-calibrated confidence correction
    use_calibration_weight: bool = False


# ─────────────────────────────────────────────
# Confidence: sequence log-prob over answer span
# ─────────────────────────────────────────────

def compute_sequence_logprob(
    logits: torch.Tensor,        # (B, T, V)
    input_ids: torch.Tensor,     # (B, T)
    answer_mask: torch.Tensor,   # (B, T) — 1 for answer span tokens, 0 for reasoning/prompt
) -> torch.Tensor:
    """
    Returns mean token log-prob over the answer span only (not the reasoning span).
    This is the confidence signal used for reward shaping, following the rationale
    that CoT span probabilities are noisy/unreliable for calibration.

    Returns: (B,) mean log-prob per sequence over answer tokens
    """
    log_probs = F.log_softmax(logits, dim=-1)                          # (B, T, V)
    token_log_probs = log_probs.gather(
        2, input_ids.unsqueeze(-1)                                     # (B, T, 1)
    ).squeeze(-1)                                                       # (B, T)

    # Mask to answer span only
    masked = token_log_probs * answer_mask                             # (B, T)
    denom = answer_mask.sum(dim=-1).clamp(min=1.0)                    # (B,)
    return masked.sum(dim=-1) / denom                                  # (B,) mean log-prob


# ─────────────────────────────────────────────
# Sign function s_i
# ─────────────────────────────────────────────

def compute_sign(
    seq_logprob: torch.Tensor,   # (B,) confidence proxy
    rewards: torch.Tensor,       # (B,) ∈ [-1, 1]
    threshold: float,
) -> torch.Tensor:
    """
    s_i = +1  if uncertain (logprob < threshold) AND correct   (reward > 0)
    s_i = -1  if confident  (logprob ≥ threshold) AND correct  (reward > 0)
    s_i = +1  if incorrect  (reward ≤ 0)   — neither penalized for parrot nor rewarded
                                              but sign positive so reward magnitude flows

    Behavior:
      - Penalizes parrot-correct outputs (high confidence, right answer)
      - Rewards creative-correct outputs (low confidence, right answer)
      - Keeps standard gradient signal for wrong outputs
    """
    confident = (seq_logprob >= threshold)          # (B,) bool
    correct    = (rewards > 0)                      # (B,) bool
    parrot     = confident & correct                # confident-AND-correct → penalize
    creative   = (~confident) & correct            # uncertain-AND-correct → reward

    s = torch.ones_like(rewards)                    # default +1
    s[parrot]  = -1.0
    s[creative] = +1.0
    return s                                        # (B,)


# ─────────────────────────────────────────────
# Modified advantage Â* = s_i · η_i · (1 + r_i)
# ─────────────────────────────────────────────

def compute_creative_advantage(
    rewards: torch.Tensor,       # (B,)  ∈ [-1, 1]
    sign: torch.Tensor,          # (B,)  ∈ {-1, +1}
    novelty: torch.Tensor,       # (B,)  ∈ [0, 1]
) -> torch.Tensor:
    """
    Â*_i = s_i · η_i · (1 + r_i)

    Note: (1 + r_i) shifts the reward to [0, 2] so that correct outputs
    amplify signal and incorrect outputs near r=-1 contribute ≈0 regardless of sign.
    This is intentional: we only want to reward/penalize when there's a meaningful signal.
    """
    return sign * novelty * (1.0 + rewards)         # (B,)


# ─────────────────────────────────────────────
# Token-level entropy regularizer R_H
# ─────────────────────────────────────────────

def compute_entropy_regularizer(
    logits: torch.Tensor,        # (B, T, V)
    attention_mask: torch.Tensor,# (B, T)  — 1 for generated tokens
) -> torch.Tensor:
    """
    R_H = mean over (B, T) of H(π_θ(·|x_{<t}))
        = -Σ_v π(v) log π(v)  averaged over generated positions

    Returns scalar: mean token-level entropy over the batch.
    Higher = more spread distribution = more "creative" token choices.
    """
    probs     = F.softmax(logits, dim=-1)            # (B, T, V)
    log_probs = F.log_softmax(logits, dim=-1)        # (B, T, V)
    entropy   = -(probs * log_probs).sum(dim=-1)     # (B, T)  per-token entropy

    # Average only over generated (non-padding) tokens
    masked_entropy = entropy * attention_mask         # (B, T)
    denom          = attention_mask.sum().clamp(min=1.0)
    return masked_entropy.sum() / denom              # scalar


# ─────────────────────────────────────────────
# PPO-clip ratio term
# ─────────────────────────────────────────────

def compute_ppo_clip_loss(
    new_log_probs: torch.Tensor,   # (B, T) log π_θ per token
    old_log_probs: torch.Tensor,   # (B, T) log π_θ_old per token
    advantages: torch.Tensor,      # (B,)   Â*
    attention_mask: torch.Tensor,  # (B, T)
    clip_eps: float,
) -> torch.Tensor:
    """
    Standard PPO-clip but with our modified per-sequence advantage.
    The advantage is broadcast from (B,) to (B, T) for token-level updates.
    """
    # Per-token importance ratio ρ_i = π_θ(a|s) / π_θ_old(a|s)
    log_ratio = new_log_probs - old_log_probs        # (B, T)
    ratio     = torch.exp(log_ratio)                 # (B, T)

    # Broadcast advantage to token level
    adv = advantages.unsqueeze(1).expand_as(ratio)   # (B, T)

    # PPO-clip
    unclipped = ratio * adv
    clipped   = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    loss_tok  = -torch.min(unclipped, clipped)       # (B, T)  negative = ascent→descent

    # Mask and average
    masked = loss_tok * attention_mask
    return masked.sum() / attention_mask.sum().clamp(min=1.0)


# ─────────────────────────────────────────────
# Full creative GRPO loss
# ─────────────────────────────────────────────

def creative_grpo_loss(
    new_logits: torch.Tensor,      # (B, T, V)  current policy
    old_log_probs: torch.Tensor,   # (B, T)     old policy (detached)
    input_ids: torch.Tensor,       # (B, T)
    attention_mask: torch.Tensor,  # (B, T)     generated token mask
    answer_mask: torch.Tensor,     # (B, T)     answer-span mask (subset of attention_mask)
    rewards: torch.Tensor,         # (B,)  ∈ [-1, 1]
    novelty: torch.Tensor,         # (B,)  ∈ [0, 1]  — externally computed η
    config: CreativeGRPOConfig,
) -> dict:
    """
    Full modified GRPO objective:

        ℒ* = E[ min(ρ·Â*, clip(ρ,1-ε,1+ε)·Â*) ] + λ · R_H(π_θ)

    Returns dict with scalar 'loss' and diagnostic keys.
    """

    # 1. Log-probs under new policy
    new_log_probs = F.log_softmax(new_logits, dim=-1)       # (B, T, V)
    token_new_lp  = new_log_probs.gather(
        2, input_ids.unsqueeze(-1)).squeeze(-1)             # (B, T)

    # 2. Confidence: mean log-prob over answer span
    seq_lp = compute_sequence_logprob(new_logits, input_ids, answer_mask)  # (B,)

    # 3. Sign function
    s = compute_sign(seq_lp, rewards, config.confidence_threshold)         # (B,)

    # 4. Modified advantage Â* = s_i · η_i · (1 + r_i)
    adv_star = compute_creative_advantage(rewards, s, novelty)             # (B,)

    # 5. PPO-clip loss with Â*
    clip_loss = compute_ppo_clip_loss(
        token_new_lp, old_log_probs, adv_star, attention_mask, config.clip_eps
    )

    # 6. Entropy regularizer (replaces KL penalty)
    entropy_reg = compute_entropy_regularizer(new_logits, attention_mask)

    # 7. Final objective (maximize → negate clip_loss already done inside)
    loss = clip_loss - config.lambda_entropy * entropy_reg   # - because we maximize entropy

    return {
        "loss":         loss,
        "clip_loss":    clip_loss.detach(),
        "entropy_reg":  entropy_reg.detach(),
        "adv_mean":     adv_star.mean().detach(),
        "sign_mean":    s.mean().detach(),       # negative = batch skewing toward parrot penalty
        "novelty_mean": novelty.mean().detach(),
        "seq_logprob":  seq_lp.mean().detach(),  # calibration diagnostic
    }


# ─────────────────────────────────────────────
# Novelty score placeholder — η_i  (stub)
# ─────────────────────────────────────────────

def placeholder_novelty_score(
    input_ids: torch.Tensor,   # (B, T)
    rewards: torch.Tensor,     # (B,)
) -> torch.Tensor:
    """
    Stub novelty score returning η = 1.0 for all sequences.

    Replace with one of the candidate operationalizations:
      - Embedding distance from nearest training/seen reasoning chain
      - Compression ratio (Kolmogorov complexity proxy via gzip length)
      - Causal intervention consistency score
      - Cross-domain transfer generalization metric

    All implementations should return (B,) tensor ∈ [0, 1].
    """
    return torch.ones(input_ids.shape[0], device=input_ids.device)


# ─────────────────────────────────────────────
# Minimal smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, V = 4, 32, 8192

    config = CreativeGRPOConfig(
        clip_eps=0.2,
        lambda_entropy=0.01,
        confidence_threshold=-1.5,
    )

    new_logits    = torch.randn(B, T, V)
    old_log_probs = F.log_softmax(torch.randn(B, T, V), dim=-1).gather(
        2, torch.randint(0, V, (B, T, 1))).squeeze(-1).detach()
    input_ids     = torch.randint(0, V, (B, T))
    attn_mask     = torch.ones(B, T)
    answer_mask   = torch.zeros(B, T)
    answer_mask[:, -8:] = 1.0   # last 8 tokens = answer span

    rewards = torch.tensor([1.0, -1.0, 1.0, -1.0])   # alternating correct/wrong
    novelty = placeholder_novelty_score(input_ids, rewards)

    out = creative_grpo_loss(
        new_logits, old_log_probs, input_ids,
        attn_mask, answer_mask, rewards, novelty, config
    )

    print("Creative GRPO output:")
    for k, v in out.items():
        print(f"  {k:16s}: {v.item():.6f}")

    print("\nSign breakdown:")
    seq_lp = compute_sequence_logprob(new_logits, input_ids, answer_mask)
    s      = compute_sign(seq_lp, rewards, config.confidence_threshold)
    print(f"  seq_logprob : {seq_lp.tolist()}")
    print(f"  rewards     : {rewards.tolist()}")
    print(f"  signs       : {s.tolist()}")
    print(f"  adv*        : {compute_creative_advantage(rewards, s, novelty).tolist()}")