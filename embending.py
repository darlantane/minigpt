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