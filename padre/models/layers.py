"""Shared Transformer and normalization helpers."""

from typing import Tuple

import torch
import torch.nn as nn

def build_transformer_encoder(d_model: int, n_heads: int, dim_ff: int, layers: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=dim_ff,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers)


def normalize_target_window(x_target: torch.Tensor, use_revin: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if use_revin:
        loc = x_target.mean(dim=1, keepdim=True).detach()
        scale = torch.sqrt(x_target.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        return (x_target - loc) / scale, loc, scale
    loc = x_target[:, -1:, :].detach()
    scale = torch.ones_like(loc)
    return x_target - loc, loc, scale
