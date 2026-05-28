"""Temporal convolutional backbone used by PADRE."""

from typing import List

import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.norm = nn.GroupNorm(1, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        out = out + self.downsample(x)
        return self.norm(out)


class TCNForecaster(nn.Module):
    def __init__(
        self,
        lookback: int,
        input_dim: int,
        target_dim: int,
        horizon: int,
        hidden: int = 32,
        levels: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
        revin: bool = True,
        head_mode: str = "last",
    ):
        super().__init__()
        if head_mode not in {"last", "summary", "flatten"}:
            raise ValueError(f"Unsupported TCN head mode: {head_mode}")
        layers: List[nn.Module] = []
        in_ch = input_dim
        for level in range(levels):
            out_ch = hidden
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=2**level,
                    dropout=dropout,
                )
            )
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.head_mode = head_mode
        if head_mode == "last":
            head_input = hidden
        elif head_mode == "summary":
            head_input = hidden * 3
        else:
            head_input = hidden * lookback
        head_hidden = hidden if head_mode == "last" else max(hidden, min(512, head_input // 2))
        self.head = nn.Sequential(
            nn.Linear(head_input, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, horizon * target_dim),
        )
        self.horizon = horizon
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.revin = revin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, V]
        if self.revin:
            x_target = x[:, :, : self.target_dim]
            x_cov = x[:, :, self.target_dim :]
            loc = x_target.mean(dim=1, keepdim=True)
            scale = torch.sqrt(x_target.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
            z_target = (x_target - loc) / scale
            z_in = torch.cat([z_target, x_cov], dim=-1) if x_cov.numel() else z_target
        else:
            loc = None
            scale = None
            z_in = x
        z = z_in.transpose(1, 2)
        z = self.tcn(z)
        if self.head_mode == "last":
            features = z[:, :, -1]
        elif self.head_mode == "summary":
            features = torch.cat([z[:, :, -1], z.mean(dim=-1), z.amax(dim=-1)], dim=-1)
        else:
            features = z.flatten(1)
        out = self.head(features)
        out = out.view(x.shape[0], self.horizon, self.target_dim)
        if self.revin:
            out = out * scale + loc
        return out
