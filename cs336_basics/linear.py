import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weights = nn.Parameter(
            torch.empty(size=(out_features, in_features), device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(
            self.weights,
            mean=0,
            std=(2 / (in_features + out_features)) ** 0.5,
            a=-((2 / (in_features + out_features)) ** 0.5),
            b=(2 / (in_features + out_features)) ** 0.5,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = einsum(
            x,
            self.weights,
            "batch sequence d_in, d_out d_in -> batch sequence d_out",
        )
        return output
