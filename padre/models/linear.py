

import torch
import torch.nn as nn

class NLinearForecaster(nn.Module):
    """A fast per-variable normalized linear backbone for online runs."""

    def __init__(self, lookback: int, horizon: int, input_dim: int, target_dim: int):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.linears = nn.ModuleList([nn.Linear(lookback * input_dim, horizon) for _ in range(target_dim)])
        for layer in self.linears:
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, V]. The last-value subtraction follows NLinear-style
        # normalization and keeps the online model robust to level shifts.
        x_target = x[:, :, : self.target_dim]
        x_cov = x[:, :, self.target_dim :]
        seq_last = x_target[:, -1:, :].detach()
        z_target = x_target - seq_last
        z = torch.cat([z_target, x_cov], dim=-1) if x_cov.numel() else z_target
        flat = z.flatten(1)
        outs = []
        for i, layer in enumerate(self.linears):
            outs.append(layer(flat).unsqueeze(-1))
        out = torch.cat(outs, dim=-1)
        return out + seq_last


class MLPForecaster(nn.Module):
    def __init__(self, lookback: int, horizon: int, input_dim: int, target_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.net = nn.Sequential(
            nn.Linear(lookback * input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon * target_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_target = x[:, :, : self.target_dim]
        x_cov = x[:, :, self.target_dim :]
        seq_last = x_target[:, -1:, :].detach()
        z_target = x_target - seq_last
        z = torch.cat([z_target, x_cov], dim=-1) if x_cov.numel() else z_target
        out = self.net(z.flatten(1)).view(x.shape[0], self.horizon, self.target_dim)
        return out + seq_last


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, V]
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        front = x[:, :1, :].repeat(1, pad_left, 1)
        end = x[:, -1:, :].repeat(1, pad_right, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        return self.avg(x_pad.transpose(1, 2)).transpose(1, 2)


class DLinearForecaster(nn.Module):
    def __init__(self, lookback: int, horizon: int, input_dim: int, target_dim: int, kernel_size: int = 25):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.moving_avg = MovingAverage(kernel_size)
        self.seasonal = nn.ModuleList([nn.Linear(lookback, horizon) for _ in range(target_dim)])
        self.trend = nn.ModuleList([nn.Linear(lookback, horizon) for _ in range(target_dim)])
        for layers in (self.seasonal, self.trend):
            for layer in layers:
                nn.init.constant_(layer.weight, 1.0 / lookback)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_target = x[:, :, : self.target_dim]
        trend = self.moving_avg(x_target)
        seasonal = x_target - trend
        outs = []
        for i in range(self.target_dim):
            out = self.seasonal[i](seasonal[:, :, i]) + self.trend[i](trend[:, :, i])
            outs.append(out.unsqueeze(-1))
        return torch.cat(outs, dim=-1)
