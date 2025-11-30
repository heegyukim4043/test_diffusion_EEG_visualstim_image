# dc_ldm/modules/diffusionmodules/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.heads, -1, h * w), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhdn,bhde->bhne', q, context)
        out = out.reshape(b, -1, h, w)
        return self.to_out(out)


class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.0, context_dim=None, cond_scale=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.d_head = d_head
        self.depth = depth
        self.dropout = dropout
        self.context_dim = context_dim  # <-- 추가
        self.cond_scale = cond_scale    # <-- cond_scale 유지

        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        inner_dim = n_heads * d_head
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1)

        # context_dim을 BasicTransformerBlock에 넘겨줄 수 있도록 확장
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim) for _ in range(depth)]
        )

        self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1)

    def forward(self, x, context=None):
        h = self.norm(x)
        h = self.proj_in(h)

        b, c, h_, w_ = h.shape
        h = h.reshape(b, c, h_ * w_).permute(0, 2, 1)  # [b, hw, c]

        for block in self.transformer_blocks:
            h = block(h, context=context)

        h = h.permute(0, 2, 1).reshape(b, c, h_, w_)
        h = self.proj_out(h)

        return x + self.cond_scale * h




class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0.0, context_dim=None):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True,kdim=context_dim, vdim=context_dim)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        # Self-attention
        h = self.norm1(x)
        x = x + self.attn1(h, h, h)[0]
        # Cross-attention
        h = self.norm2(x)
        if context is None:
            context = x
        x = x + self.attn2(h, context, context)[0]
        # Feedforward
        h = self.norm3(x)
        x = x + self.ff(h)
        return x
