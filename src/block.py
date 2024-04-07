import torch.nn as nn

from head import MultiHead
from ffwd import FeedForward


class Block(nn.Module):
    """
    Transformer block: communication followed by computation
    """

    def __init__(self, num_embds, num_heads, block_size, dropout):
        super().__init__()

        head_size = num_embds // num_heads
        self.sa = MultiHead(num_heads, head_size, num_embds, block_size, dropout)
        self.ffwd = FeedForward(num_embds, dropout)
        self.ln1 = nn.LayerNorm(num_embds)
        self.ln2 = nn.LayerNorm(num_embds)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.sa(x)

        x = self.ln2(x)
        x = x + self.ffwd(x)

        return x
