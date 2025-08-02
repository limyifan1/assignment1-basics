import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.embeddings = nn.Parameter(
            torch.empty(
                size=(num_embeddings, embedding_dim), device=device, dtype=dtype
            )
        )
        torch.nn.init.trunc_normal_(self.embeddings, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor):
        return self.embeddings[token_ids]
