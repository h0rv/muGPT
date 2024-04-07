import torch.nn as nn

from head import MultiHead
from ffwd import FeedForward


class Block(nn.Module):
    """
    Transformer block: communication followed by computation
    """

    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()

        head_size = n_embd // num_heads
        self.sa = MultiHead(num_heads, head_size, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x
