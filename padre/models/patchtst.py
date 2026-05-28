"""PatchTST-style channel-independent Transformer backbone."""

import torch
import torch.nn as nn

from .layers import build_transformer_encoder, normalize_target_window

class PatchTSTForecaster(nn.Module):
    """PatchTST-style channel-independent patch transformer.

    Target variables share the patch projection, Transformer encoder, and head.
    Optional time/covariate features are repeated for every target channel inside
    each patch, keeping the channel-independent target path intact.
    """

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
        patch_len: int,
        stride: int,
        dropout: float,
        revin: bool,
        individual_heads: bool,
    ):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.cov_dim = input_dim - target_dim
        self.patch_len = patch_len
        self.stride = stride
        self.revin = revin
        self.individual_heads = individual_heads
        self.num_patches = 1 + max(0, (lookback - patch_len) // stride)
        patch_dim = patch_len * (1 + self.cov_dim)
        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        self.encoder = build_transformer_encoder(d_model, n_heads, dim_ff, layers, dropout)
        self.dropout = nn.Dropout(dropout)
        if individual_heads:
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Flatten(start_dim=1),
                        nn.Linear(self.num_patches * d_model, horizon),
                    )
                    for _ in range(target_dim)
                ]
            )
            self.head = None
        else:
            self.head = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(self.num_patches * d_model, horizon),
            )
            self.heads = None
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_target = x[:, :, : self.target_dim]
        x_cov = x[:, :, self.target_dim :]
        z_target, loc, scale = normalize_target_window(x_target, self.revin)

        target_patches = z_target.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # [B, P, C, patch_len]
        bsz, n_patches, channels, _ = target_patches.shape
        target_patches = target_patches.permute(0, 2, 1, 3).reshape(bsz * channels, n_patches, self.patch_len)

        if self.cov_dim > 0:
            cov_patches = x_cov.unfold(dimension=1, size=self.patch_len, step=self.stride)
            # [B, P, cov_dim, patch_len] -> [B, P, patch_len * cov_dim]
            cov_patches = cov_patches.permute(0, 1, 3, 2).reshape(bsz, n_patches, self.patch_len * self.cov_dim)
            cov_patches = cov_patches.unsqueeze(1).expand(bsz, channels, n_patches, -1)
            cov_patches = cov_patches.reshape(bsz * channels, n_patches, self.patch_len * self.cov_dim)
            patches = torch.cat([target_patches, cov_patches], dim=-1)
        else:
            patches = target_patches

        z = self.patch_proj(patches)
        z = self.dropout(z + self.pos_embed[:, :n_patches])
        z = self.encoder(z)
        if self.individual_heads:
            z_by_channel = z.view(bsz, channels, n_patches, -1)
            outs = [head(z_by_channel[:, idx]).unsqueeze(-1) for idx, head in enumerate(self.heads)]
            out = torch.cat(outs, dim=-1)
        else:
            out = self.head(z).view(bsz, channels, self.horizon).permute(0, 2, 1).contiguous()
        return out * scale + loc
