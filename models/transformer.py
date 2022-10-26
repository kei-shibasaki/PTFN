from turtle import forward
import torch 
from torch import nn

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads = None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias = False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias = False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv = None):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio):
        super().__init__()
        self.attn = PreNorm(dim, SelfAttention(dim, heads))
        self.ffn = PreNorm(dim, FeedForward(dim, mlp_ratio*dim))
    
    def forward(self, x):
        B, C, H, W = x.shape
        # (B,C,H,W) -> (B,HW,C)
        x = x.reshape(B,H*W,C)
        x = self.attn(x)
        x = self.ffn(x)
        x = x.reshape(B,C,H,W)
        return x

class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio):
        super().__init__()
        self.attn = PreNorm(dim, SelfAttention(dim, heads))
        self.ffn = PreNorm(dim, FeedForward(dim, mlp_ratio*dim))
    
    def forward(self, x):
        B, C, H, W = x.shape
        # (B,C,H,W) -> (B,HW,C)
        x = x.reshape(B,H*W,C)
        x = self.attn(x)
        x = self.ffn(x)
        x = x.reshape(B,C,H,W)
        return x