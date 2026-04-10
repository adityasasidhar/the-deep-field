import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from main import CreativeGRPOConfig, creative_grpo_loss


NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")


@dataclass
class DatasetStage:
    name: str
    start_step: int
    examples: list[dict[str, str]]


@dataclass
class TrainConfig:
    model_name: str
    output_dir: Path
    batch_size: int
    max_prompt_length: int
    max_new_tokens: int
    learning_rate: float
    num_steps: int
    gradient_accumulation_steps: int
    weight_decay: float
    temperature: float
    top_p: float
    seed: int
    device: str
    save_every: int
    print_every: int
    gsm8k_start_step: int
    math_start_step: int
    math_config_name: str
    train_split: str
    trust_remote_code: bool


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train a Llama 3B-class model with Creative GRPO on GSM8K and MATH."
    )
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--output-dir", default=Path("Learning_with_confidence/checkpoints"), type=Path)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--max-prompt-length", default=512, type=int)
    parser.add_argument("--max-new-tokens", default=160, type=int)
    parser.add_argument("--learning-rate", default=1e-6, type=float)
    parser.add_argument("--num-steps", default=200, type=int)
    parser.add_argument("--gradient-accumulation-steps", default=4, type=int)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--top-p", default=0.9, type=float)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-every", default=50, type=int)
    parser.add_argument("--print-every", default=5, type=int)
    parser.add_argument("--gsm8k-start-step", default=1, type=int)
    parser.add_argument("--math-start-step", default=101, type=int)
    parser.add_argument("--math-config-name", default="algebra")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    return TrainConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        device=args.device,
        save_every=args.save_every,
        print_every=args.print_every,
        gsm8k_start_step=args.gsm8k_start_step,
        math_start_step=args.math_start_step,
        math_config_name=args.math_config_name,
        train_split=args.train_split,
        trust_remote_code=args.trust_remote_code,
    )


# ─────────────────────────────────────────────
# Answer extraction / reward
# ─────────────────────────────────────────────

def normalize_numeric_string(value: str) -> str:
    return value.replace(",", "").strip().rstrip(".")


def extract_final_answer(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    boxed_matches = BOXED_PATTERN.findall(stripped)
    if boxed_matches:
        return boxed_matches[-1].strip()
    if "####" in stripped:
        return stripped.split("####")[-1].strip()
    final_line = stripped.splitlines()[-1].strip()
    answer_markers = ["answer is", "final answer is", "therefore", "thus", "so the answer is"]
    lowered = final_line.lower()
    for marker in answer_markers:
        if marker in lowered:
            start = lowered.rfind(marker)
            return final_line[start + len(marker):].strip(" :.-")
    return final_line


def extract_numeric_answer(text: str) -> str | None:
    final_answer = extract_final_answer(text)
    matches = NUMBER_PATTERN.findall(final_answer)
    if not matches:
        matches = NUMBER_PATTERN.findall(text)
    if not matches:
        return None
    return normalize_numeric_string(matches[-1])


def answers_match(target_text: str, prediction_text: str) -> bool:
    target_numeric = extract_numeric_answer(target_text)
    prediction_numeric = extract_numeric_answer(prediction_text)
    if target_numeric is not None and prediction_numeric is not None:
        try:
            return abs(float(target_numeric) - float(prediction_numeric)) < 1e-6
        except ValueError:
            pass
    target_final = extract_final_answer(target_text)
    prediction_final = extract_final_answer(prediction_text)
    return normalize_numeric_string(target_final.lower()) == normalize_numeric_string(prediction_final.lower())


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def format_prompt(question: str) -> str:
    return (
        "Solve the following math problem. Show your reasoning briefly, then end with a line "
        "formatted exactly as 'Final answer: <answer>'.\n\n"
        f"Problem: {question}\n"
    )


def load_gsm8k_examples(split: str) -> list[dict[str, str]]:
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    return [
        {"prompt": format_prompt(ex["question"]), "answer": extract_final_answer(ex["answer"]), "dataset": "gsm8k"}
        for ex in dataset
    ]


def load_math_examples(split: str, config_name: str) -> list[dict[str, str]]:
    dataset = load_dataset("EleutherAI/hendrycks_math", config_name, split=split)
    return [
        {"prompt": format_prompt(ex["problem"]), "answer": extract_final_answer(ex["solution"]), "dataset": f"math/{config_name}"}
        for ex in dataset
    ]


def build_curriculum(config: TrainConfig) -> list[DatasetStage]:
    stages = [
        DatasetStage("gsm8k", config.gsm8k_start_step, load_gsm8k_examples(config.train_split)),
        DatasetStage(f"math/{config.math_config_name}", config.math_start_step,
                     load_math_examples(config.train_split, config.math_config_name)),
    ]
    return sorted(stages, key=lambda s: s.start_step)


def stage_for_step(stages: list[DatasetStage], step: int) -> DatasetStage:
    active = stages[0]
    for stage in stages:
        if step >= stage.start_step:
            active = stage
        else:
            break
    return active


def cycle_batch(examples: list[dict], batch_size: int, offset: int) -> list[dict]:
    start = (offset * batch_size) % len(examples)
    return [examples[(start + i) % len(examples)] for i in range(batch_size)]


# ─────────────────────────────────────────────
# Rollout — FIX 1: return full attention mask, not answer mask
# ─────────────────────────────────────────────

@torch.no_grad()
def rollout_from_policy(
    policy: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        sequences        (B, T)   full prompt + completion token ids
        old_log_probs    (B, T)   per-token log-probs under rollout policy (prompt region = 0)
        full_attn_mask   (B, T)   full attention mask — 1 everywhere except padding
        answer_mask      (B, T)   1 only on completion tokens (used for confidence / entropy)

    FIX: the forward pass MUST receive full_attn_mask, not answer_mask.
    answer_mask is passed separately to creative_grpo_loss as the signal mask.
    Using answer_mask as the model's attention_mask zeroes out the prompt context
    and causes degenerate logits → NaN loss.
    """
    was_training = policy.training
    policy.eval()

    generated = policy.generate(
        input_ids=prompt_inputs["input_ids"],
        attention_mask=prompt_inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )

    if was_training:
        policy.train()

    sequences = generated.sequences                              # (B, prompt_len + completion_len)
    prompt_len = prompt_inputs["input_ids"].shape[1]
    completion_ids = sequences[:, prompt_len:]                  # (B, completion_len)

    # Old log-probs from generation scores (completion tokens only)
    old_lp_completion = []
    for score_t, tok_t in zip(generated.scores, completion_ids.T):
        lp = torch.log_softmax(score_t, dim=-1)
        old_lp_completion.append(lp.gather(1, tok_t.unsqueeze(1)).squeeze(1))

    if old_lp_completion:
        old_lp_completion = torch.stack(old_lp_completion, dim=1)   # (B, completion_len)
    else:
        old_lp_completion = torch.zeros(sequences.shape[0], 0, device=sequences.device)

    # Pad old_log_probs to full sequence length (prompt region = 0, never used in ratio)
    old_log_probs = torch.cat([
        torch.zeros(sequences.shape[0], prompt_len, device=sequences.device),
        old_lp_completion,
    ], dim=1)                                                    # (B, T)

    # Completion mask: 1 up to and including first EOS, 0 after.
    # If EOS never appears, keep the entire sampled completion.
    cumulative_eos = (completion_ids == tokenizer.eos_token_id).long().cumsum(dim=1)
    completion_mask = (cumulative_eos <= 1).float()

    # answer_mask: prompt region = 0, completion region = completion_mask
    answer_mask = torch.cat([
        torch.zeros_like(prompt_inputs["attention_mask"], dtype=torch.float32),
        completion_mask,
    ], dim=1)                                                    # (B, T)

    # FIX: full attention mask for forward pass — prompt real tokens + completion tokens
    full_attn_mask = torch.cat([
        prompt_inputs["attention_mask"],
        completion_mask.long(),
    ], dim=1)                                                    # (B, T)

    return sequences, old_log_probs, full_attn_mask, answer_mask


# ─────────────────────────────────────────────
# Decoding / rewards / novelty
# ─────────────────────────────────────────────

def decode_completions(tokenizer, sequences, prompt_input_ids) -> list[str]:
    prompt_len = prompt_input_ids.shape[1]
    return tokenizer.batch_decode(sequences[:, prompt_len:], skip_special_tokens=True)


def compute_rewards(batch: list[dict], completions: list[str]) -> torch.Tensor:
    return torch.tensor(
        [1.0 if answers_match(ex["answer"], c) else -1.0 for ex, c in zip(batch, completions)],
        dtype=torch.float32,
    )


def compute_novelty(completions: list[str]) -> torch.Tensor:
    scores = []
    for c in completions:
        tokens = c.split()
        if not tokens:
            scores.append(0.25)
        else:
            scores.append(0.25 + 0.75 * min(1.0, len(set(tokens)) / len(tokens)))
    return torch.tensor(scores, dtype=torch.float32)


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def setup_logging(output_dir: Path) -> Path:
    log_file = output_dir / "train.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return log_file


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────

def main() -> None:
    train_config = parse_args()
    torch.manual_seed(train_config.seed)
    train_config.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = setup_logging(train_config.output_dir)
    logging.info("Logging to %s", log_file)
    logging.info("Training config: %s", train_config)

    tokenizer = AutoTokenizer.from_pretrained(
        train_config.model_name, trust_remote_code=train_config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        train_config.model_name,
        dtype=torch.bfloat16 if train_config.device.startswith("cuda") else torch.float32,
        trust_remote_code=train_config.trust_remote_code,
    ).to(train_config.device)

    # FIX 2: Do NOT call model.resize_token_embeddings() unless you actually
    # added new tokens. Llama-3.2 tokenizer and model vocab are already aligned.
    # Calling resize re-initializes embedding rows and corrupts weights.
    # Only uncomment if you explicitly added tokens to the tokenizer:
    # if len(tokenizer) != model.config.vocab_size:
    #     model.resize_token_embeddings(len(tokenizer))

    model.train()

    optimizer = AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)

    grpo_config = CreativeGRPOConfig(
        clip_eps=0.2,
        lambda_entropy=0.01,
        confidence_threshold=-1.5,
    )

    curriculum = build_curriculum(train_config)

    for step in range(1, train_config.num_steps + 1):
        active_stage = stage_for_step(curriculum, step)
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_reward = 0.0

        for micro_step in range(train_config.gradient_accumulation_steps):
            offset = (step - 1) * train_config.gradient_accumulation_steps + micro_step
            batch = cycle_batch(active_stage.examples, train_config.batch_size, offset)
            prompts = [ex["prompt"] for ex in batch]

            prompt_inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=train_config.max_prompt_length,
            )
            prompt_inputs = {k: v.to(train_config.device) for k, v in prompt_inputs.items()}

            # Rollout → now returns 4 values including proper full_attn_mask
            sequences, old_log_probs, full_attn_mask, answer_mask = rollout_from_policy(
                model, tokenizer, prompt_inputs,
                train_config.max_new_tokens, train_config.temperature, train_config.top_p,
            )

            # FIX 1 applied: forward pass gets full_attn_mask (prompt + completion context)
            # NOT answer_mask (which zeros the prompt and breaks attention)
            outputs = model(input_ids=sequences, attention_mask=full_attn_mask)

            completions = decode_completions(tokenizer, sequences, prompt_inputs["input_ids"])
            rewards = compute_rewards(batch, completions).to(train_config.device)
            novelty = compute_novelty(completions).to(train_config.device)

            # FIX 3: pass answer_mask as both attention_mask and answer_mask to the loss.
            # The loss uses attention_mask to know which positions to include in entropy/ratio,
            # and answer_mask to know which positions to use for confidence (seq log-prob).
            # Since we want confidence only over completion tokens, both are answer_mask here.
            # The log-ratio is also gated by answer_mask — prompt old_log_probs are 0 anyway
            # but the mask makes the intent explicit and avoids any numerical edge-cases.
            loss_dict = creative_grpo_loss(
                new_logits=outputs.logits,
                old_log_probs=old_log_probs,
                input_ids=sequences,
                attention_mask=answer_mask,   # gates entropy + PPO ratio to completion tokens
                answer_mask=answer_mask,      # gates confidence (seq log-prob) to answer span
                rewards=rewards,
                novelty=novelty,
                config=grpo_config,
            )

            if not torch.isfinite(loss_dict["loss"]):
                raise RuntimeError(
                    "Non-finite Creative GRPO loss detected: "
                    f"loss={loss_dict['loss'].item()} "
                    f"clip={loss_dict['clip_loss'].item()} "
                    f"entropy={loss_dict['entropy_reg'].item()} "
                    f"seq_logprob={loss_dict['seq_logprob'].item()} "
                    f"reward_mean={rewards.mean().item()} "
                    f"novelty_mean={novelty.mean().item()}"
                )

            (loss_dict["loss"] / train_config.gradient_accumulation_steps).backward()
            running_loss += loss_dict["loss"].item()
            running_reward += rewards.mean().item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % train_config.print_every == 0 or step == 1:
            avg = train_config.gradient_accumulation_steps
            logging.info(
                "step=%04d stage=%s loss=%.4f reward=%.4f clip=%.4f entropy=%.4f seq_logprob=%.4f sign_mean=%.4f",
                step,
                active_stage.name,
                running_loss / avg,
                running_reward / avg,
                loss_dict["clip_loss"].item(),
                loss_dict["entropy_reg"].item(),
                loss_dict["seq_logprob"].item(),
                loss_dict["sign_mean"].item(),
            )

        if step % train_config.save_every == 0:
            ckpt_dir = train_config.output_dir / f"step_{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            logging.info("Saved checkpoint to %s", ckpt_dir)

    final_dir = train_config.output_dir / f"step_{train_config.num_steps}"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logging.info("Saved final model to %s", final_dir)


if __name__ == "__main__":
    main()
