# Learning with Confidence

This repository implements a custom GRPO-style training loop for math reasoning models.

The current codebase has two parts:

- `main.py` contains the `CreativeGRPOConfig` dataclass and the `creative_grpo_loss` objective.
- `test.py` is the actual training script. It loads a causal language model, builds a curriculum over GSM8K and MATH, runs rollouts, computes rewards, and updates the model with the creative GRPO loss.

## What the implementation does

The training loop currently:

- prompts the model with math questions
- samples completions with `transformers.generate`
- scores each completion with exact-answer matching
- computes a novelty score from completion diversity
- computes a confidence signal from answer-span log-probability
- applies a PPO-style clipped update with a creativity-aware advantage
- adds a token-level entropy regularizer
- saves checkpoints during training and at the end

## Data

The script uses Hugging Face datasets:

- `openai/gsm8k` with the `main` config
- `EleutherAI/hendrycks_math` with a configurable subset, defaulting to `algebra`

The default curriculum starts with GSM8K and switches to MATH later in training.

## Model

The default model in `test.py` is:

- `meta-llama/Llama-3.2-3B-Instruct`

If you use this model, you may need to authenticate with Hugging Face before downloading it.

## Running training

Example:

```bash
python test.py
```

Useful flags:

- `--model-name`
- `--output-dir`
- `--batch-size`
- `--max-prompt-length`
- `--max-new-tokens`
- `--learning-rate`
- `--num-steps`
- `--gradient-accumulation-steps`
- `--temperature`
- `--top-p`
- `--gsm8k-start-step`
- `--math-start-step`
- `--math-config-name`
- `--train-split`
- `--trust-remote-code`

Example with a smaller run:

```bash
python test.py \
  --num-steps 20 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --max-new-tokens 64
```

## Loss details

`creative_grpo_loss` combines:

- sequence-level confidence from answer-span log-probability
- a sign function that penalizes confident correct answers and rewards uncertain correct answers
- a novelty-weighted advantage
- PPO clipping
- entropy regularization

The implementation uses shifted causal-LM logits so token scores line up with next-token prediction.

## Outputs

Training writes checkpoints to:

- `Learning_with_confidence/checkpoints/step_<N>`

The final model is saved to:

- `Learning_with_confidence/checkpoints/step_<num_steps>`

## Notes

- `torch_dtype` has been replaced with `dtype` in the current implementation.
- The training script uses `bfloat16` on CUDA and `float32` on CPU.
- `test.py` is a training entry point, not a unit test.
