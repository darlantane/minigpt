import torch
import torch.nn as nn


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.embedding_table = nn.Embedding(
            vocab_size,
            vocab_size
        )


    def forward(self, idx, targets=None):

        # idx : (batch, time)
        logits = self.embedding_table(idx)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.view(B*T, C)

            targets = targets.view(B*T)

            loss = nn.functional.cross_entropy(
                logits,
                targets
            )

        return logits, loss


    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            logits, _ = self(idx)

            logits = logits[:, -1, :]

            probs = torch.softmax(
                logits,
                dim=-1
            )

            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat(
                (idx, idx_next),
                dim=1
            )

        return idx