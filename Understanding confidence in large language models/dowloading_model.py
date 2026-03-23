from huggingface_hub import snapshot_download
import os

models = {
    "base":     "Qwen/Qwen3-4B",
    "instruct": "Qwen/Qwen3-4B-Instruct-2507",
    "thinking": "Qwen/Qwen3-4B-Thinking-2507",
}

for tier, repo_id in models.items():
    print(f"\nDownloading {tier}: {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        ignore_patterns=["*.bin", "original/*"],
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"Done: {tier}")

print("\nAll three models downloaded to ~/.cache/huggingface/hub/")