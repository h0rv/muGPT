import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """
    Single head of self-attention
    """

    def __init__(self, head_size, num_embds, block_size, dropout):
        super().__init__()

        self.key = nn.Linear(num_embds, head_size, bias=False)
        self.query = nn.Linear(num_embds, head_size, bias=False)
        self.value = nn.Linear(num_embds, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k, q = self.key(x), self.query(x)  # ((B, T, C), (B, T, C))

        # compute attention scores ("affinities")

        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values

        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) --> (B, T, C)

        return out


class MultiHead(nn.Module):
    """
    Multiple heads of self-attention in parallel
    """

    def __init__(self, num_heads, head_size, num_embds, block_size, dropout):
        super().__init__()

        self.heads = nn.ModuleList(
            Head(head_size, num_embds, block_size, dropout) for _ in range(num_heads)
        )
        self.projection = nn.Linear(num_heads * head_size, num_embds)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        out = self.dropout(out)

        return out
