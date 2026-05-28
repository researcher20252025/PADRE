"""PADRE pattern memory, tri-view drift controller, and residual adapter."""

from collections import deque
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader

from .config import ExperimentConfig

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualAdapter(nn.Module):
    """Online residual head trained only from delayed feedback."""

    def __init__(self, input_dim: int, horizon: int, target_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.horizon = horizon
        self.target_dim = target_dim
        feature_dim = 4 * input_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon * target_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1)
        std = torch.sqrt(x.var(dim=1, unbiased=False) + 1e-5)
        last = x[:, -1, :]
        trend = x[:, -1, :] - x[:, 0, :]
        feats = torch.cat([mean, std, last, trend], dim=-1)
        return self.net(feats).view(x.shape[0], self.horizon, self.target_dim)


class PADREController(nn.Module):
    def __init__(self, lookback: int, horizon: int, input_dim: int, target_dim: int, cfg: ExperimentConfig):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.emb_dim = cfg.emb_dim
        self.drift_dim = cfg.drift_dim
        self.residual_memory = cfg.residual_memory
        self.episodic_memory_size = cfg.episodic_memory
        x_dim = lookback * input_dim
        r_dim = horizon * target_dim
        back_dim = cfg.residual_memory * r_dim
        self.f_reg = MLP(x_dim, max(cfg.policy_hidden, cfg.emb_dim), cfg.emb_dim, cfg.dropout)
        self.f_res = MLP(r_dim, max(cfg.policy_hidden, cfg.emb_dim), cfg.emb_dim, cfg.dropout)
        self.g_on = MLP(x_dim, max(cfg.policy_hidden, cfg.drift_dim), cfg.drift_dim, cfg.dropout)
        self.g_back = MLP(back_dim, max(cfg.policy_hidden, cfg.drift_dim), cfg.drift_dim, cfg.dropout)
        self.g_mem = MLP(cfg.drift_dim, max(cfg.policy_hidden, cfg.drift_dim), cfg.drift_dim, cfg.dropout)
        state_dim = 2 * cfg.emb_dim + cfg.drift_dim + 1
        self.policy = nn.Sequential(
            nn.Linear(state_dim, cfg.policy_hidden),
            nn.GELU(),
            nn.Linear(cfg.policy_hidden, cfg.policy_hidden // 2),
            nn.GELU(),
            nn.Linear(cfg.policy_hidden // 2, 1),
        )
        # Start from moderate updates; online policy training will move it.
        nn.init.constant_(self.policy[-1].bias, 0.0)
        self.pattern_temperature = cfg.pattern_temperature
        self.pattern_residual_feature_mode = cfg.pattern_residual_feature_mode
        self.lambda_tri = cfg.lambda_tri
        self.memory_threshold = cfg.memory_threshold
        self.gate_floor = cfg.gate_floor
        self.gate_cap = cfg.gate_cap
        self.area_gate_scale = cfg.area_gate_scale
        self.register_buffer("pattern_bank", torch.empty(0, 2 * cfg.emb_dim))
        self.register_buffer("regime_centers", torch.empty(0, cfg.emb_dim))
        self.register_buffer("residual_centers", torch.empty(0, cfg.emb_dim))
        self.register_buffer("correction_regime_centers", torch.empty(0, cfg.emb_dim))
        self.register_buffer("output_residual_centers", torch.empty(0, r_dim))
        self.residual_buffer: deque[torch.Tensor] = deque(maxlen=cfg.residual_memory)
        self.episodic_memory: deque[torch.Tensor] = deque(maxlen=cfg.episodic_memory)

    def reset_online_state(self) -> None:
        self.residual_buffer.clear()
        self.episodic_memory.clear()

    def freeze_pattern_encoders(self) -> None:
        for module in (self.f_reg, self.f_res):
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    def policy_parameters(self) -> Iterable[nn.Parameter]:
        modules = [self.g_on, self.g_back, self.g_mem, self.policy]
        for module in modules:
            yield from module.parameters()

    @torch.no_grad()
    def build_pattern_bank(
        self,
        model: nn.Module,
        loader: DataLoader,
        k_clusters: int,
        device: torch.device,
    ) -> None:
        self.eval()
        model.eval()
        regs: List[np.ndarray] = []
        ress: List[np.ndarray] = []
        correction_regs: List[np.ndarray] = []
        residual_outputs: List[np.ndarray] = []
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            pred = model(x)
            residual = pred - y
            q_reg = self.f_reg(x.flatten(1))
            q_res = self.f_res(residual.flatten(1))
            regs.append(q_reg.detach().cpu().numpy())
            ress.append(q_res.detach().cpu().numpy())
            correction_regs.append(self._correction_regime_features(x).detach().cpu().numpy())
            residual_outputs.append(residual.flatten(1).detach().cpu().numpy())
        reg_np = np.concatenate(regs, axis=0)
        res_np = np.concatenate(ress, axis=0)
        correction_reg_np = np.concatenate(correction_regs, axis=0)
        residual_out_np = np.concatenate(residual_outputs, axis=0)
        k = max(1, min(k_clusters, len(reg_np)))
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=min(2048, max(64, len(reg_np))),
            n_init=10,
            random_state=0,
        )
        labels = kmeans.fit_predict(reg_np)
        reg_centers = kmeans.cluster_centers_.astype(np.float32)
        res_centers = np.zeros((k, res_np.shape[1]), dtype=np.float32)
        for cluster in range(k):
            mask = labels == cluster
            if np.any(mask):
                res_centers[cluster] = res_np[mask].mean(axis=0)
            else:
                res_centers[cluster] = res_np.mean(axis=0)
        pattern_bank = np.concatenate([reg_centers, res_centers], axis=1)
        self.pattern_bank = torch.from_numpy(pattern_bank).to(device)
        self.regime_centers = torch.from_numpy(reg_centers).to(device)
        self.residual_centers = torch.from_numpy(res_centers).to(device)

        correction_kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=min(2048, max(64, len(correction_reg_np))),
            n_init=10,
            random_state=1,
        )
        correction_labels = correction_kmeans.fit_predict(correction_reg_np)
        correction_centers = correction_kmeans.cluster_centers_.astype(np.float32)
        output_res_centers = np.zeros((k, residual_out_np.shape[1]), dtype=np.float32)
        for cluster in range(k):
            mask = correction_labels == cluster
            if np.any(mask):
                output_res_centers[cluster] = residual_out_np[mask].mean(axis=0)
            else:
                output_res_centers[cluster] = residual_out_np.mean(axis=0)
        self.correction_regime_centers = torch.from_numpy(correction_centers).to(device)
        self.output_residual_centers = torch.from_numpy(output_res_centers).to(device)

    def _backward_input(self, residual: torch.Tensor) -> torch.Tensor:
        device = residual.device
        pieces: List[torch.Tensor] = [p.to(device) for p in self.residual_buffer] + [residual.detach()]
        if len(pieces) < self.residual_memory:
            pad = torch.zeros_like(residual)
            pieces = [pad for _ in range(self.residual_memory - len(pieces))] + pieces
        else:
            pieces = pieces[-self.residual_memory :]
        return torch.cat([p.flatten() for p in pieces], dim=0).unsqueeze(0)

    def _memory_view(self, z_back: torch.Tensor) -> torch.Tensor:
        if not self.episodic_memory:
            return z_back
        mem = torch.stack([m.to(z_back.device) for m in self.episodic_memory], dim=0)
        z_norm = F.normalize(z_back.detach(), dim=-1)
        mem_norm = F.normalize(mem, dim=-1)
        idx = torch.matmul(mem_norm, z_norm.squeeze(0)).argmax()
        return self.g_mem(mem[idx].unsqueeze(0))

    def residual_correction_value(self, device: torch.device) -> Optional[torch.Tensor]:
        if not self.residual_buffer:
            return None
        residuals = [r.to(device).view(self.horizon, self.target_dim) for r in self.residual_buffer]
        return torch.stack(residuals, dim=0).mean(dim=0)

    def _correction_regime_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.pattern_residual_feature_mode == "encoded":
            return self.f_reg(x.flatten(1))
        if self.pattern_residual_feature_mode == "stats":
            mean = x.mean(dim=1)
            std = torch.sqrt(x.var(dim=1, unbiased=False) + 1e-5)
            last = x[:, -1, :]
            trend = x[:, -1, :] - x[:, 0, :]
            return torch.cat([mean, std, last, trend], dim=-1)
        raise ValueError(f"Unsupported pattern residual feature mode: {self.pattern_residual_feature_mode}")

    def pattern_residual(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if self.correction_regime_centers.numel() == 0 or self.output_residual_centers.numel() == 0:
            return None
        q_reg = F.normalize(self._correction_regime_features(x), dim=-1)
        centers = F.normalize(self.correction_regime_centers, dim=-1)
        logits = torch.matmul(q_reg, centers.t()) / max(self.pattern_temperature, 1e-4)
        weights = torch.softmax(logits, dim=-1)
        residual = torch.matmul(weights, self.output_residual_centers)
        return residual.view(x.shape[0], self.horizon, self.target_dim)

    def drift_state(self, x: torch.Tensor, residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_on = self.g_on(x.flatten(1))
        back_input = self._backward_input(residual)
        z_back = self.g_back(back_input)
        z_mem = self._memory_view(z_back)
        # Normalize the drift views so the triangle area is scale-stable.
        z_on_n = F.normalize(z_on, dim=-1)
        z_back_n = F.normalize(z_back, dim=-1)
        z_mem_n = F.normalize(z_mem, dim=-1)
        u = z_back_n - z_on_n
        v = z_back_n - z_mem_n
        gram = (u.square().sum(dim=-1) * v.square().sum(dim=-1)) - (u * v).sum(dim=-1).square()
        area = 0.5 * torch.sqrt(torch.clamp(gram, min=1e-12)).unsqueeze(-1)
        return z_on_n, z_back_n, z_mem_n, area

    def pattern_prompt(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if self.pattern_bank.numel() == 0:
            return torch.zeros(x.shape[0], 2 * self.emb_dim, device=x.device)
        q = torch.cat([self.f_reg(x.flatten(1)), self.f_res(residual.flatten(1))], dim=-1)
        q = F.normalize(q, dim=-1)
        bank = F.normalize(self.pattern_bank, dim=-1)
        logits = torch.matmul(q, bank.t()) / max(self.pattern_temperature, 1e-4)
        weights = torch.softmax(logits, dim=-1)
        return torch.matmul(weights, self.pattern_bank)

    def gate(self, x: torch.Tensor, residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt = self.pattern_prompt(x, residual)
        z_on, z_back, _, area = self.drift_state(x, residual)
        state = torch.cat([prompt, z_on, area], dim=-1)
        raw_gate = torch.sigmoid(self.policy(state))
        gate = self.gate_floor + (1.0 - self.gate_floor) * raw_gate
        if self.area_gate_scale > 0:
            gate = gate * torch.exp(-self.area_gate_scale * area.detach())
        if self.gate_cap < 1.0:
            gate = torch.clamp(gate, max=self.gate_cap)
        return gate, area, z_back

    def push_feedback(self, residual: torch.Tensor, z_back: torch.Tensor, gate_value: float) -> None:
        self.residual_buffer.append(residual.detach().flatten().cpu())
        if gate_value > self.memory_threshold:
            self.episodic_memory.append(z_back.detach().flatten().cpu())
