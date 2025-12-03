
# Minimal illustrative PyTorch components for an autoregressive transformer.
# This is intentionally compact and educational rather than production-ready.

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class MinimalTransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = SimpleFeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        res = x
        x = self.ln1(x)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = res + self.dropout(attn_out)
        res = x
        x = self.ln2(x)
        x = res + self.dropout(self.ffn(x))
        return x

class MinimalLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=4, n_head=8, d_ff=2048, max_seq=2048):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.layers = nn.ModuleList([MinimalTransformerBlock(d_model, n_head, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        b, seq = input_ids.shape
        pos = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(b, -1)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        causal_mask = None
        for l in self.layers:
            x = l(x, attn_mask=causal_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
