"""Offline pretraining and delayed-feedback online evaluation loops."""

import copy
import math
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .controller import PADREController, ResidualAdapter

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    loss_name: str,
) -> float:
    model.train()
    losses = []
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = online_loss(pred, y, loss_name)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else math.nan


@torch.no_grad()
def evaluate_static(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        pred = model(x)
        total += F.mse_loss(pred, y, reduction="sum").item()
        count += int(np.prod(y.shape))
    return total / max(count, 1)


def online_loss(pred: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "mse":
        return F.mse_loss(pred, target)
    if loss_name == "smooth_l1":
        return F.smooth_l1_loss(pred, target)
    raise ValueError(f"Unsupported online loss: {loss_name}")


def online_trainable_parameters(model: nn.Module, scope: str) -> List[nn.Parameter]:
    if scope == "all":
        return list(model.parameters())
    if scope == "head":
        params = [param for name, param in model.named_parameters() if "head" in name]
        if params:
            return params
        return list(model.parameters())
    raise ValueError(f"Unsupported online train scope: {scope}")


@torch.no_grad()
def update_ema_model(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def apply_prediction_adjustments(
    pred: torch.Tensor,
    x: torch.Tensor,
    controller: PADREController,
    cfg: ExperimentConfig,
    adapter: Optional[ResidualAdapter] = None,
    detach_adapter: bool = False,
) -> torch.Tensor:
    if cfg.pattern_residual_correction > 0:
        pattern_residual = controller.pattern_residual(x)
        if pattern_residual is not None:
            pred = pred - cfg.pattern_residual_correction * pattern_residual
    if adapter is not None and cfg.adapter_weight > 0:
        adapter_residual = adapter(x)
        if detach_adapter:
            adapter_residual = adapter_residual.detach()
        pred = pred - cfg.adapter_weight * adapter_residual
    if cfg.residual_correction > 0:
        correction = controller.residual_correction_value(pred.device)
        if correction is not None:
            pred = pred - cfg.residual_correction * correction.unsqueeze(0)
    return pred


def pretrain_backbone(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: ExperimentConfig,
    device: torch.device,
) -> Tuple[float, float]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.pretrain_lr,
        weight_decay=cfg.weight_decay,
    )
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = float("inf")
    best_train = float("inf")
    patience = max(3, min(8, cfg.pretrain_epochs // 2))
    stale = 0
    for epoch in range(1, cfg.pretrain_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, cfg.grad_clip, cfg.pretrain_loss)
        val_loss = evaluate_static(model, val_loader, device) if len(val_loader.dataset) else train_loss
        if val_loss < best_val:
            best_val = val_loss
            best_train = train_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        print(f"pretrain epoch {epoch:03d} train={train_loss:.6f} val={val_loss:.6f}", flush=True)
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_train, best_val


def train_pattern_encoders(
    model: nn.Module,
    controller: PADREController,
    loader: DataLoader,
    cfg: ExperimentConfig,
    device: torch.device,
) -> float:
    if cfg.pattern_epochs <= 0:
        return math.nan
    model.eval()
    reg_decoder = nn.Linear(cfg.emb_dim, cfg.horizon * controller.target_dim).to(device)
    res_decoder = nn.Linear(cfg.emb_dim, cfg.horizon * controller.target_dim).to(device)
    params = list(controller.f_reg.parameters()) + list(controller.f_res.parameters())
    optimizer = torch.optim.AdamW(params + list(reg_decoder.parameters()) + list(res_decoder.parameters()), lr=cfg.pattern_lr)
    last_loss = math.nan
    for epoch in range(1, cfg.pattern_epochs + 1):
        losses = []
        controller.f_reg.train()
        controller.f_res.train()
        reg_decoder.train()
        res_decoder.train()
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                pred = model(x)
                residual = pred - y
            q_reg = controller.f_reg(x.flatten(1))
            q_res = controller.f_res(residual.flatten(1))
            target_flat = y.flatten(1)
            residual_flat = residual.flatten(1)
            loss = F.mse_loss(reg_decoder(q_reg), target_flat) + F.mse_loss(res_decoder(q_res), residual_flat)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params + list(reg_decoder.parameters()) + list(res_decoder.parameters()), cfg.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        last_loss = float(np.mean(losses)) if losses else math.nan
        print(f"pattern epoch {epoch:03d} loss={last_loss:.6f}", flush=True)
    return last_loss


def online_evaluate(
    model: nn.Module,
    controller: PADREController,
    adapter: Optional[ResidualAdapter],
    input_values: np.ndarray,
    target_values: np.ndarray,
    starts: Sequence[int],
    cfg: ExperimentConfig,
    device: torch.device,
    score: bool = True,
    reset_state: bool = True,
) -> Dict[str, float]:
    model.train()
    controller.train()
    if adapter is not None:
        adapter.train()
    controller.f_reg.eval()
    controller.f_res.eval()
    if reset_state:
        controller.reset_online_state()
    online_params = online_trainable_parameters(model, cfg.online_train_scope)
    backbone_optimizer = torch.optim.AdamW(online_params, lr=cfg.online_lr, weight_decay=cfg.online_weight_decay)
    ema_model: Optional[nn.Module] = None
    if cfg.prediction_ema_decay > 0:
        ema_model = copy.deepcopy(model).to(device)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad_(False)
    adapter_optimizer = (
        torch.optim.AdamW(adapter.parameters(), lr=cfg.adapter_lr, weight_decay=0.0)
        if adapter is not None and cfg.adapter_weight > 0
        else None
    )
    policy_optimizer = torch.optim.AdamW(
        controller.policy_parameters(),
        lr=cfg.policy_lr,
        weight_decay=cfg.policy_weight_decay,
    )
    total_sse = 0.0
    total_count = 0
    gate_sum = 0.0
    area_sum = 0.0
    updates = 0
    forecasts = 0
    pending: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    replay_x: deque[np.ndarray] = deque(maxlen=max(1, cfg.replay_size))
    replay_y: deque[np.ndarray] = deque(maxlen=max(1, cfg.replay_size))
    starts_arr = list(starts)
    if cfg.limit_online and cfg.limit_online > 0:
        starts_arr = starts_arr[: cfg.limit_online]

    for step, start in enumerate(starts_arr):
        x_np = input_values[start : start + cfg.lookback]
        y_np = target_values[start + cfg.lookback : start + cfg.lookback + cfg.horizon]
        x = torch.from_numpy(x_np).unsqueeze(0).to(device=device, dtype=torch.float32)
        y = torch.from_numpy(y_np).unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            predict_model = ema_model if ema_model is not None else model
            predict_model.eval()
            if adapter is not None:
                adapter.eval()
            pred = predict_model(x)
            pred = apply_prediction_adjustments(pred, x, controller, cfg, adapter)
            model.train()
            if adapter is not None:
                adapter.train()
            if score:
                total_sse += F.mse_loss(pred, y, reduction="sum").item()
                total_count += int(np.prod(y.shape))
            pending.append((x_np.copy(), y_np.copy(), pred.squeeze(0).detach().cpu().numpy()))
            forecasts += 1

        delayed_idx = step - cfg.horizon
        if delayed_idx >= 0 and (step % max(1, cfg.online_update_stride) == 0):
            dx_np, dy_np, cached_pred_np = pending[delayed_idx]
            dx = torch.from_numpy(dx_np).unsqueeze(0).to(device=device, dtype=torch.float32)
            dy = torch.from_numpy(dy_np).unsqueeze(0).to(device=device, dtype=torch.float32)
            cached_pred = torch.from_numpy(cached_pred_np).unsqueeze(0).to(device=device, dtype=torch.float32)
            residual = cached_pred - dy
            replay_x.append(dx_np.copy())
            replay_y.append(dy_np.copy())

            # Train the gate by the paper's first-order one-step look-ahead surrogate.
            if cfg.policy_update_interval > 0 and updates % cfg.policy_update_interval == 0:
                policy_optimizer.zero_grad(set_to_none=True)
                pred_now = apply_prediction_adjustments(model(dx), dx, controller, cfg, adapter, detach_adapter=True)
                pred_loss = online_loss(pred_now, dy, cfg.online_loss)
                grads = torch.autograd.grad(
                    pred_loss,
                    tuple(online_params),
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                grad_norm_sq = torch.zeros((), device=device)
                for grad in grads:
                    if grad is not None:
                        grad_norm_sq = grad_norm_sq + grad.detach().square().sum()
                gate, area, _ = controller.gate(dx, residual)
                improve = cfg.policy_gain_scale * cfg.online_lr * gate.mean() * grad_norm_sq
                penalty = cfg.lambda_tri * gate.mean() * area.detach().mean()
                policy_loss = -improve + penalty
                policy_loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(list(controller.policy_parameters()), cfg.grad_clip)
                policy_optimizer.step()

            if adapter_optimizer is not None and adapter is not None:
                adapter_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    base_pred = apply_prediction_adjustments(model(dx), dx, controller, cfg, None)
                    target_residual = base_pred - dy
                adapter_loss = F.mse_loss(adapter(dx), target_residual)
                adapter_loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), cfg.grad_clip)
                adapter_optimizer.step()

            # Recompute gate after the policy step and update the backbone.
            with torch.no_grad():
                gate_detached, area_detached, z_back_detached = controller.gate(dx, residual)
                gate_value = float(gate_detached.item())
                area_value = float(area_detached.item())
            backbone_optimizer.zero_grad(set_to_none=True)
            batch_n = min(max(1, cfg.online_batch_size), len(replay_x))
            if batch_n == 1:
                bx = dx
                by = dy
            else:
                bx_np = np.stack(list(replay_x)[-batch_n:], axis=0)
                by_np = np.stack(list(replay_y)[-batch_n:], axis=0)
                bx = torch.from_numpy(bx_np).to(device=device, dtype=torch.float32)
                by = torch.from_numpy(by_np).to(device=device, dtype=torch.float32)
            pred_now = apply_prediction_adjustments(model(bx), bx, controller, cfg, adapter, detach_adapter=True)
            loss = online_loss(pred_now, by, cfg.online_loss)
            gated_loss = gate_value * loss
            gated_loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(online_params, cfg.grad_clip)
            backbone_optimizer.step()
            if ema_model is not None:
                update_ema_model(ema_model, model, cfg.prediction_ema_decay)
            controller.push_feedback(residual.squeeze(0), z_back_detached.squeeze(0), gate_value)
            gate_sum += gate_value
            area_sum += area_value
            updates += 1

        if score and (step + 1) % 5000 == 0:
            mse_so_far = total_sse / max(total_count, 1)
            avg_gate = gate_sum / max(updates, 1)
            print(f"online step {step + 1}/{len(starts_arr)} mse={mse_so_far:.6f} gate={avg_gate:.3f}", flush=True)

    return {
        "mse": total_sse / max(total_count, 1) if score else math.nan,
        "forecasts": float(forecasts),
        "updates": float(updates),
        "avg_gate": gate_sum / max(updates, 1),
        "avg_area": area_sum / max(updates, 1),
    }
