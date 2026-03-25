# dataset_download.py
from datasets import load_dataset
import json
import os
import random

random.seed(42)
SAVE_DIR = "data/seen"
os.makedirs(SAVE_DIR, exist_ok=True)

N = 200  # questions per dataset

# ── 1. GSM8K ─────────────────────────────────────────────────────────────────
print("Downloading GSM8K...")
gsm8k = load_dataset("gsm8k", "main", split="test")
gsm8k_samples = random.sample(list(gsm8k), N)
gsm8k_out = [
    {
        "id":       f"gsm8k_{i}",
        "category": "math",
        "question": item["question"],
        "answer":   item["answer"],
        "hint":     None   # filled in next step
    }
    for i, item in enumerate(gsm8k_samples)
]

# ── 2. TriviaQA ───────────────────────────────────────────────────────────────
print("Downloading TriviaQA...")
trivia = load_dataset("trivia_qa", "rc.nocontext", split="validation")
trivia_samples = random.sample(list(trivia), N)
trivia_out = [
    {
        "id":       f"trivia_{i}",
        "category": "factual",
        "question": item["question"],
        "answer":   item["answer"]["value"],
        "hint":     None
    }
    for i, item in enumerate(trivia_samples)
]

# ── 3. MMLU ───────────────────────────────────────────────────────────────────
print("Downloading MMLU...")
mmlu = load_dataset("cais/mmlu", "all", split="test")
mmlu_samples = random.sample(list(mmlu), N)
choices_map = {0: "A", 1: "B", 2: "C", 3: "D"}
mmlu_out = [
    {
        "id":       f"mmlu_{i}",
        "category": "science",
        "question": item["question"],
        "choices":  item["choices"],
        "answer":   choices_map[item["answer"]],
        "hint":     None
    }
    for i, item in enumerate(mmlu_samples)
]

# ── 4. HumanEval (coding) ─────────────────────────────────────────────────────
print("Downloading HumanEval...")
humaneval = load_dataset("openai_humaneval", split="test")
# HumanEval only has 164 problems, take all
humaneval_out = [
    {
        "id":           f"humaneval_{i}",
        "category":     "coding",
        "question":     item["prompt"],
        "answer":       item["canonical_solution"],
        "test":         item["test"],
        "entry_point":  item["entry_point"],
        "hint":         None
    }
    for i, item in enumerate(humaneval)
]

# ── Save ──────────────────────────────────────────────────────────────────────
datasets = {
    "gsm8k":     gsm8k_out,
    "trivia":    trivia_out,
    "mmlu":      mmlu_out,
    "humaneval": humaneval_out,
}

for name, data in datasets.items():
    path = os.path.join(SAVE_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} questions → {path}")

print("\nDataset download complete.")