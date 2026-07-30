import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, n_embed):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            n_embed
        )

    def forward(self, idx):
        return self.embedding(idx)
    

class PositionalEmbedding(nn.Module):

    def __init__(self, block_size, n_embed):
        super().__init__()

        self.embedding = nn.Embedding(
            block_size,
            n_embed
        )

    def forward(self, T):

        positions = torch.arange(T)

        return self.embedding(positions)