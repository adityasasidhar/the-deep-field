import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer

class Hyperparameters:
    tokenizer_name = "HuggingFaceTB/SmolLM3-3B"
    seed=42
    vocab_size = 128256
    max_seq_len = 32768
    dim = 768
    hidden_dim = 2304
    n_layers = 12
    recurrent_steps = 3
    n_heads = 12
    n_kv_heads = 3
    local_window = 1024
    global_every = 4
    rope_theta = 500000
    dropout = 0.0
    gradient_checkpointing = True


cfg = Hyperparameters()
tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base=10000):
        super().__init__()

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float() / dim)
        )

        t = torch.arange(max_position_embeddings).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len):
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len]
        )


def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class GQAAttention(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()

        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.local_window = cfg.local_window
        self.is_global = (layer_idx % cfg.global_every == 0)

        self.q_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)

        kv_dim = self.n_kv_heads * self.head_dim
        self.k_proj = nn.Linear(cfg.dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.rope = RotaryEmbedding(
            self.head_dim,
            cfg.max_seq_len,
            cfg.rope_theta
        )

        max_T = cfg.max_seq_len
        causal = torch.triu(
            torch.ones(max_T, max_T, dtype=torch.bool),
            diagonal=1
        )
        self.register_buffer("causal_mask_cache", causal, persistent=False)

        if not self.is_global:
            i = torch.arange(max_T)
            dist = i[:, None] - i[None, :]
            local = (dist < 0) | (dist > cfg.local_window)
            combined = causal | local
            self.register_buffer("local_mask_cache", combined, persistent=False)

    def repeat_kv(self, x):
        repeat = self.n_heads // self.n_kv_heads
        return x.repeat_interleave(repeat, dim=1)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = self.rope(T)
        q, k = apply_rope(q, k, cos, sin)
        k = self.repeat_kv(k)
        v = self.repeat_kv(v)
        if self.is_global:
            mask = self.causal_mask_cache[:T, :T]
        else:
            mask = self.local_mask_cache[:T, :T]

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=~mask,
            dropout_p=0.0,
            is_causal=False
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()

        self.attn_norm = RMSNorm(cfg.dim)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.attn = GQAAttention(cfg, layer_idx)
        self.ffn = SwiGLU(cfg.dim, cfg.hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class RecurrentTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg, i)
            for i in range(cfg.n_layers)
        ])

        self.depth_embeddings = nn.Parameter(torch.zeros(cfg.recurrent_steps, cfg.dim))
        self.final_norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids, targets=None):
        x = self.embed_tokens(input_ids)

        for step in range(self.cfg.recurrent_steps):
            x = x + self.depth_embeddings[step]

            for block in self.blocks:
                if self.cfg.gradient_checkpointing and self.training:
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits

        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, self.cfg.vocab_size),
            targets[:, 1:].reshape(-1)
        )
        return loss


device = "cuda" if torch.cuda.is_available() else "cpu"

model = RecurrentTransformer(cfg).to(device)

total = sum(p.numel() for p in model.parameters())
tied = cfg.vocab_size * cfg.dim

unique = total - tied

print(f"\nTotal Parameters (with ties): {total / 1e6:.2f}M")
print(f"Unique Parameters:            {unique / 1e6:.2f}M")
text = "Artificial intelligence is transforming the future of computing."

tokens = tokenizer(text, return_tensors="pt").input_ids.to(device)

with torch.no_grad():
    logits = model(tokens)

print(f"\nInput shape:  {tokens.shape}")
print(f"Output shape: {logits.shape}")

model.train()
input_ids = tokens
targets = tokens  

loss = model(input_ids, targets=targets)
loss.backward()

print(f"\nTest loss (untrained): {loss.item():.4f}")
print("Backward pass: OK")

model.zero_grad()
model.eval()

# =========================================================
# GENERATION
# =========================================================

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50
):
    model.eval()

    input_ids = tokenizer(
        prompt,
        return_tensors="pt"
    ).input_ids.to(device)

    for _ in range(max_new_tokens):

        if input_ids.shape[1] > cfg.max_seq_len:
            input_ids = input_ids[:, -cfg.max_seq_len:]

        logits = model(input_ids)
        logits = logits[:, -1] / temperature 

        if top_k is not None:
            vals, _ = torch.topk(logits, top_k)
            min_val = vals[:, -1, None]
            logits = logits.masked_fill(logits < min_val, -float("inf"))

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

sample = generate(
    model,
    tokenizer,
    "The future of compact AI models",
    max_new_tokens=80
)

print("\nGenerated:\n")
print(sample)