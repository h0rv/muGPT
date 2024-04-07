import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """
    Single head of self-attention
    """

    def __init__(self, head_size, n_embd, block_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        k, q = self.key(x), self.query(x)  # ((B, T, C), (B, T, C))

        # compute attention scores ("affinities")

        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # perform the weighted aggregation of the values

        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) --> (B, T, C)

        return out


class MultiHead(nn.Module):
    """
    Multiple heads of self-attention in parallel
    """

    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()

        self.heads = nn.ModuleList(
            Head(head_size, n_embd, block_size) for _ in range(num_heads)
        )

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)
