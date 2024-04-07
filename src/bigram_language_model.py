import torch
import torch.nn as nn
from torch.nn import functional as F

from block import Block


class BigramLanguageModel(nn.Module):
    def __init__(
        self, vocab_size, num_embds, block_size, num_layers, num_heads, dropout, device
    ):
        super().__init__()
        # each token reads the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, num_embds)
        self.position_embedding_table = nn.Embedding(block_size, num_embds)

        self.blocks = nn.Sequential(
            *[Block(num_embds, num_heads, block_size, dropout) for _ in range(num_layers)]
        )

        self.ln_f = nn.LayerNorm(num_embds)  # final layer norm

        self.lm_head = nn.Linear(num_embds, vocab_size)

        self.block_size = block_size
        self.device = device

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensors of ints
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)  # (B, C, T)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context

        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size :]
            # get predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # ==> (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx
