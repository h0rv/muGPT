import torch.nn as nn


class FeedForward(nn.Module):
    """
    Simple linear layer followed by a non-linearity
    """

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # projection layer
        )

    def forward(self, x):
        return self.net(x)
