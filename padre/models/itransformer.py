"""iTransformer-style inverted-attention backbone."""

import torch
import torch.nn as nn

from .layers import build_transformer_encoder, normalize_target_window

class ITransformerForecaster(nn.Module):
    """iTransformer-style inverted attention over variate tokens."""

    def __init__(
        self,
        lookback: int,
        horizon: int,
        input_dim: int,
        target_dim: int,
        d_model: int,
        n_heads: int,
        layers: int,
        dim_ff: int,
        dropout: float,
        revin: bool,
        token_mode: str,
        linear_skip: bool,
    ):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.cov_dim = input_dim - target_dim
        self.revin = revin
        self.token_mode = token_mode
        self.linear_skip = linear_skip
        if token_mode not in {"all", "target_context"}:
            raise ValueError(f"Unsupported iTransformer token mode: {token_mode}")
        self.value_embedding = nn.Linear(lookback, d_model)
        self.cov_embedding = (
            nn.Linear(lookback * self.cov_dim, d_model)
            if token_mode == "target_context" and self.cov_dim > 0
            else None
        )
        token_count = input_dim if token_mode == "all" else target_dim
        self.var_embed = nn.Parameter(torch.zeros(1, token_count, d_model))
        self.encoder = build_transformer_encoder(d_model, n_heads, dim_ff, layers, dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, horizon)
        self.skip_linears = nn.ModuleList([nn.Linear(lookback, horizon) for _ in range(target_dim)]) if linear_skip else None
        if self.skip_linears is not None:
            for layer in self.skip_linears:
                nn.init.constant_(layer.weight, 1.0 / lookback)
                nn.init.zeros_(layer.bias)
        nn.init.trunc_normal_(self.var_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_target = x[:, :, : self.target_dim]
        x_cov = x[:, :, self.target_dim :]
        z_target, loc, scale = normalize_target_window(x_target, self.revin)
        if self.token_mode == "all":
            token_values = torch.cat([z_target, x_cov], dim=-1) if self.cov_dim > 0 else z_target
            # [B, V, L] -> each target or covariate variate is one token.
            tokens = self.value_embedding(token_values.transpose(1, 2))
            tokens = tokens + self.var_embed[:, : token_values.shape[-1]]
        else:
            # Compatibility mode for datasets where using covariates as global
            # context is more stable than attending over them as extra tokens.
            tokens = self.value_embedding(z_target.transpose(1, 2))
            tokens = tokens + self.var_embed
            if self.cov_embedding is not None:
                tokens = tokens + self.cov_embedding(x_cov.flatten(1)).unsqueeze(1)
        tokens = self.dropout(tokens)
        z = self.encoder(tokens)
        out = self.head(z[:, : self.target_dim]).permute(0, 2, 1).contiguous()
        if self.skip_linears is not None:
            skip = torch.cat(
                [layer(z_target[:, :, idx]).unsqueeze(-1) for idx, layer in enumerate(self.skip_linears)],
                dim=-1,
            )
            out = out + skip
        return out * scale + loc
