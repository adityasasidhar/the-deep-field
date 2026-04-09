import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) architecture.
    Focuses on KV caching compression and decoupled RoPE.
    """
    def __init__(self, d_model, num_heads, q_latent_dim, kv_latent_dim, rope_dim=16):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.q_latent_dim = q_latent_dim
        self.kv_latent_dim = kv_latent_dim
        self.rope_dim = rope_dim
        
        # 1. KV Compression (Down-projection to Latent Space)
        # This is the core innovation: we only cache the output of this layer!
        self.w_down_kv = nn.Linear(d_model, kv_latent_dim, bias=False)
        
        # 2. KV Up-projection (Decompresses back to K and V)
        self.w_up_k = nn.Linear(kv_latent_dim, num_heads * self.d_head, bias=False)
        self.w_up_v = nn.Linear(kv_latent_dim, num_heads * self.d_head, bias=False)
        
        # 3. Query Compression (Down-projection)
        self.w_down_q = nn.Linear(d_model, q_latent_dim, bias=False)
        
        # 4. Query Up-projection
        self.w_up_q = nn.Linear(q_latent_dim, num_heads * self.d_head, bias=False)
        
        # 5. Decoupled RoPE Projections 
        # (Rotary embeddings cannot be compressed, so they are handled separately)
        self.w_rope_q = nn.Linear(d_model, num_heads * rope_dim, bias=False)
        self.w_rope_k = nn.Linear(d_model, num_heads * rope_dim, bias=False)
        
        # 6. Final Output Projection
        self.w_out = nn.Linear(num_heads * self.d_head, d_model, bias=False)

    def apply_rotary_pos_emb(self, x):
        # Placeholder for Rotary Position Embedding (RoPE) implementation.
        # In practice, this applies standard rotational matrices to the sequence.
        return x

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # --- Query Processing ---
        # Compress and Up-project Queries
        c_q = self.w_down_q(x)  # (batch, seq, q_latent_dim)
        q_base = self.w_up_q(c_q).view(batch_size, seq_len, self.num_heads, self.d_head)
        
        # Decoupled RoPE for Queries
        q_rope = self.w_rope_q(x).view(batch_size, seq_len, self.num_heads, self.rope_dim)
        q_rope = self.apply_rotary_pos_emb(q_rope)
        
        # Combine Base Queries and RoPE Queries
        q = torch.cat([q_base, q_rope], dim=-1) # (batch, seq, num_heads, d_head + rope_dim)
        q = q.transpose(1, 2) # (batch, num_heads, seq, d_head + rope_dim)
        
        # --- Key/Value Processing (The Latent Bottleneck) ---
        # Compress to Latent KV 
        c_kv = self.w_down_kv(x) # (batch, seq, kv_latent_dim)
        
        # Up-project to Keys and Values
        k_base = self.w_up_k(c_kv).view(batch_size, seq_len, self.num_heads, self.d_head)
        v = self.w_up_v(c_kv).view(batch_size, seq_len, self.num_heads, self.d_head)
        v = v.transpose(1, 2) # (batch, num_heads, seq, d_head)
        
        # Decoupled RoPE for Keys
        k_rope = self.w_rope_k(x).view(batch_size, seq_len, self.num_heads, self.rope_dim)
        k_rope = self.apply_rotary_pos_emb(k_rope)
        
        # Combine Base Keys and RoPE Keys
        k = torch.cat([k_base, k_rope], dim=-1) # (batch, seq, num_heads, d_head + rope_dim)
        k = k.transpose(1, 2) # (batch, num_heads, seq, d_head + rope_dim)
        
        # --- Attention Computation ---
        # Scaled Dot-Product Attention
        scale = 1.0 / math.sqrt(self.d_head + self.rope_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale # (batch, num_heads, seq, seq)
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # Multiply by Values
        out = torch.matmul(attn_weights, v) # (batch, num_heads, seq, d_head)
        
        # --- Output Projection ---
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.w_out(out)

# Example Usage
if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    
    # DeepSeek compresses Q, K, and V drastically to save cache memory
    model = MultiHeadLatentAttention(
        d_model=d_model, 
        num_heads=num_heads, 
        q_latent_dim=128, 
        kv_latent_dim=256
    )
    
    # Dummy input sequence
    x = torch.randn(2, 64, d_model) # (batch_size, seq_len, d_model)
    output = model(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)