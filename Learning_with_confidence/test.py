import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top-p", default=0.95, type=float)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-every", default=50, type=int)
    parser.add_argument("--print-every", default=5, type=int)
    parser.add_argument("--gsm8k-start-step", default=1, type=int)
    parser.add_argument("--math-start-step", default=101, type=int)
    parser.add_argument(
        "--math-config-name",
        default="algebra",
        help="Subset of EleutherAI/hendrycks_math to use. Example: algebra, geometry, number_theory.",
    )
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
    target_final = extract_final_answer(target_text)
    prediction_final = extract_final_answer(prediction_text)

    target_numeric = extract_numeric_answer(target_text)
    prediction_numeric = extract_numeric_answer(prediction_text)
    if target_numeric is not None and prediction_numeric is not None:
        try:
            return abs(float(target_numeric) - float(prediction_numeric)) < 1e-6
        except ValueError:
            pass

    normalized_target = normalize_numeric_string(target_final.lower())
    normalized_prediction = normalize_numeric_string(prediction_final.lower())
    return normalized_target == normalized_prediction


def format_prompt(question: str) -> str:
    return (
        "Solve the following math problem. Show your reasoning briefly, then end with a line "
        "formatted exactly as 'Final answer: <answer>'.\n\n"
        f"Problem: {question}\n"
    )


def load_gsm8k_examples(split: str) -> list[dict[str, str]]:
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    return [
        {
            "prompt": format_prompt(example["question"]),
            "answer": extract_final_answer(example["answer"]),
            "dataset": "gsm8k",
        }
        for example in dataset
    ]


def load_math_examples(split: str, config_name: str) -> list[dict[str, str]]:
    dataset = load_dataset("EleutherAI/hendrycks_math", config_name, split=split)
    return [
        {
            "prompt": format_prompt(example["problem"]),
            "answer": extract_final_answer(example["solution"]),
            "dataset": f"math/{config_name}",
        }
        for example in dataset
    ]


def build_curriculum(config: TrainConfig) -> list[DatasetStage]:
    stages = [
        DatasetStage(
            name="gsm8k",
            start_step=config.gsm8k_start_step,
            examples=load_gsm8k_examples(config.train_split),
        ),
        DatasetStage(
            name=f"math/{config.math_config_name}",
            start_step=config.math_start_step,
            examples=load_math_examples(config.train_split, config.math_config_name),
        ),
    ]
    return sorted(stages, key=lambda stage: stage.start_step)


def stage_for_step(stages: list[DatasetStage], step: int) -> DatasetStage:
    active_stage = stages[0]
    for stage in stages:
        if step >= stage.start_step:
            active_stage = stage
        else:
            break
    return active_stage


def cycle_batch(examples: list[dict[str, str]], batch_size: int, offset: int) -> list[dict[str, str]]:
    start = (offset * batch_size) % len(examples)
    return [examples[(start + index) % len(examples)] for index in range(batch_size)]


def tokenize_prompts(
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_prompt_length: int,
    device: str,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )
    return {key: value.to(device) for key, value in encoded.items()}


@torch.no_grad()
def rollout_from_policy(
    policy: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    sequences = generated.sequences
    prompt_length = prompt_inputs["input_ids"].shape[1]
    completion_ids = sequences[:, prompt_length:]

    old_log_probs = []
    for score_t, token_t in zip(generated.scores, completion_ids.T):
        score_log_probs = torch.log_softmax(score_t, dim=-1)
        old_log_probs.append(score_log_probs.gather(1, token_t.unsqueeze(1)).squeeze(1))

    if old_log_probs:
        old_log_probs_tensor = torch.stack(old_log_probs, dim=1)
    else:
        old_log_probs_tensor = torch.zeros(
            sequences.shape[0],
            0,
            device=sequences.device,
            dtype=torch.float32,
        )

    completion_mask = (completion_ids != tokenizer.pad_token_id).float()
    answer_mask = torch.cat(
        [torch.zeros_like(prompt_inputs["attention_mask"], dtype=torch.float32), completion_mask],
        dim=1,
    )

    return sequences, old_log_probs_tensor, answer_mask


def decode_completions(
    tokenizer: AutoTokenizer,
    sequences: torch.Tensor,
    prompt_input_ids: torch.Tensor,
) -> list[str]:
    prompt_length = prompt_input_ids.shape[1]
    completion_ids = sequences[:, prompt_length:]
    return tokenizer.batch_decode(completion_ids, skip_special_tokens=True)


def compute_rewards(batch: list[dict[str, str]], completions: list[str]) -> torch.Tensor:
    rewards = []
    for example, completion in zip(batch, completions):
        rewards.append(1.0 if answers_match(example["answer"], completion) else -1.0)
    return torch.tensor(rewards, dtype=torch.float32)


def compute_novelty(completions: list[str]) -> torch.Tensor:
    novelty_scores = []
    for completion in completions:
        tokens = completion.split()
        if not tokens:
            novelty_scores.append(0.25)
            continue
        unique_ratio = len(set(tokens)) / len(tokens)
        novelty_scores.append(0.25 + 0.75 * min(1.0, unique_ratio))
    return torch.tensor(novelty_scores, dtype=torch.float32)


def ensure_tokenizer_padding(tokenizer: AutoTokenizer) -> AutoTokenizer:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def save_checkpoint(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: Path,
    step: int,
) -> None:
    checkpoint_dir = output_dir / f"step_{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)


def main() -> None:
    train_config = parse_args()
    torch.manual_seed(train_config.seed)
    train_config.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        train_config.model_name,
        trust_remote_code=train_config.trust_remote_code,
    )
    tokenizer = ensure_tokenizer_padding(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        train_config.model_name,
        torch_dtype=torch.bfloat16 if train_config.device.startswith("cuda") else torch.float32,
        trust_remote_code=train_config.trust_remote_code,
    ).to(train_config.device)
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

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
            prompts = [example["prompt"] for example in batch]
            prompt_inputs = tokenize_prompts(
                tokenizer,
                prompts,
                train_config.max_prompt_length,
                train_config.device,
            )

            sequences, old_log_probs, answer_mask = rollout_from_policy(
                model,
                tokenizer,
                prompt_inputs,
                train_config.max_new_tokens,
                train_config.temperature,
                train_config.top_p,
            )

            outputs = model(input_ids=sequences, attention_mask=(sequences != tokenizer.pad_token_id))
            completions = decode_completions(tokenizer, sequences, prompt_inputs["input_ids"])
            rewards = compute_rewards(batch, completions).to(train_config.device)
            novelty = compute_novelty(completions).to(train_config.device)

            padded_old_log_probs = torch.cat(
                [
                    torch.zeros(
                        old_log_probs.shape[0],
                        prompt_inputs["input_ids"].shape[1],
                        device=train_config.device,
                        dtype=old_log_probs.dtype,
                    ),
                    old_log_probs,
                ],
                dim=1,
            )

            loss_dict = creative_grpo_loss(
                new_logits=outputs.logits,
                old_log_probs=padded_old_log_probs,
                input_ids=sequences,
                attention_mask=answer_mask,
                answer_mask=answer_mask,
                rewards=rewards,
                novelty=novelty,
                config=grpo_config,
            )

            (loss_dict["loss"] / train_config.gradient_accumulation_steps).backward()
            running_loss += loss_dict["loss"].item()
            running_reward += rewards.mean().item()

        optimizer.step()

        if step % train_config.print_every == 0 or step == 1:
            print(
                f"step={step:04d} "
                f"stage={active_stage.name} "
                f"loss={running_loss / train_config.gradient_accumulation_steps:.4f} "
                f"reward={running_reward / train_config.gradient_accumulation_steps:.4f} "
                f"clip={loss_dict['clip_loss'].item():.4f} "
                f"entropy={loss_dict['entropy_reg'].item():.4f} "
                f"seq_logprob={loss_dict['seq_logprob'].item():.4f} "
                f"sign_mean={loss_dict['sign_mean'].item():.4f}"
            )
            print(f"target={batch[0]['answer']!r}")
            print(f"sample_completion={completions[0]!r}")

        if step % train_config.save_every == 0:
            save_checkpoint(model, tokenizer, train_config.output_dir, step)

    save_checkpoint(model, tokenizer, train_config.output_dir, train_config.num_steps)


if __name__ == "__main__":
    main()
