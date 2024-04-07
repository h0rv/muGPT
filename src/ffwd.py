import torch.nn as nn


class FeedForward(nn.Module):
    """
    Simple linear layer followed by a non-linearity
    """

    def __init__(self, num_embds, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(num_embds, 4 * num_embds),
            nn.ReLU(),
            nn.Linear(4 * num_embds, num_embds),  # projection layer
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
