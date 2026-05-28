"""End-to-end PADRE experiment orchestration."""

import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .controller import PADREController, ResidualAdapter
from .data import WindowDataset, chronological_split, load_target_and_time, segment_indices
from .models import DLinearForecaster, ITransformerForecaster, MLPForecaster, NLinearForecaster, PatchTSTForecaster, TCNForecaster
from .training import evaluate_static, online_evaluate, pretrain_backbone, train_pattern_encoders
from .utils import choose_device, set_seed

def append_csv(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_experiment(cfg: ExperimentConfig) -> Dict[str, object]:
    set_seed(cfg.seed)
    device = choose_device(cfg.device)
    data_path = Path(cfg.data_path)
    raw_values, time_feats = load_target_and_time(str(data_path), cfg.time_features)
    n, target_dim = raw_values.shape
    train_end, val_end = chronological_split(n)
    mean = raw_values[:train_end].mean(axis=0, keepdims=True)
    std = raw_values[:train_end].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    target_values = ((raw_values - mean) / std).astype(np.float32)
    if time_feats is not None:
        input_values = np.concatenate([target_values, time_feats], axis=1).astype(np.float32)
    else:
        input_values = target_values
    input_dim = input_values.shape[1]

    train_starts = segment_indices(cfg.lookback, cfg.horizon, cfg.lookback, train_end)
    val_starts = segment_indices(cfg.lookback, cfg.horizon, train_end, val_end)
    test_starts = segment_indices(cfg.lookback, cfg.horizon, val_end, n)
    if len(train_starts) == 0 or len(test_starts) == 0:
        raise ValueError("Not enough rows for the configured lookback/horizon.")

    train_ds = WindowDataset(input_values, target_values, train_starts, cfg.lookback, cfg.horizon)
    val_ds = WindowDataset(input_values, target_values, val_starts, cfg.lookback, cfg.horizon)
    test_ds = WindowDataset(input_values, target_values, test_starts, cfg.lookback, cfg.horizon)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    bank_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    static_eval_starts = test_starts
    if cfg.limit_online and cfg.limit_online > 0:
        static_eval_starts = test_starts[: cfg.limit_online]
    static_test_ds = WindowDataset(input_values, target_values, static_eval_starts, cfg.lookback, cfg.horizon)
    static_test_loader = DataLoader(
        static_test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    if cfg.backbone == "tcn":
        model = TCNForecaster(
            lookback=cfg.lookback,
            input_dim=input_dim,
            target_dim=target_dim,
            horizon=cfg.horizon,
            hidden=cfg.hidden,
            levels=cfg.levels,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
            revin=cfg.revin,
            head_mode=cfg.tcn_head,
        ).to(device)
    elif cfg.backbone == "nlinear":
        model = NLinearForecaster(cfg.lookback, cfg.horizon, input_dim, target_dim).to(device)
    elif cfg.backbone == "mlp":
        model = MLPForecaster(cfg.lookback, cfg.horizon, input_dim, target_dim, cfg.hidden, cfg.dropout).to(device)
    elif cfg.backbone == "dlinear":
        model = DLinearForecaster(cfg.lookback, cfg.horizon, input_dim, target_dim).to(device)
    elif cfg.backbone == "patchtst":
        model = PatchTSTForecaster(
            cfg.lookback,
            cfg.horizon,
            input_dim,
            target_dim,
            cfg.d_model,
            cfg.n_heads,
            cfg.transformer_layers,
            cfg.transformer_ff,
            cfg.patch_len,
            cfg.patch_stride,
            cfg.dropout,
            cfg.revin,
            cfg.patch_individual_heads,
        ).to(device)
    elif cfg.backbone == "itransformer":
        model = ITransformerForecaster(
            cfg.lookback,
            cfg.horizon,
            input_dim,
            target_dim,
            cfg.d_model,
            cfg.n_heads,
            cfg.transformer_layers,
            cfg.transformer_ff,
            cfg.dropout,
            cfg.revin,
            cfg.itransformer_token_mode,
            cfg.linear_skip,
        ).to(device)
    else:
        raise ValueError(f"Unsupported backbone: {cfg.backbone}")
    controller = PADREController(cfg.lookback, cfg.horizon, input_dim, target_dim, cfg).to(device)
    adapter = (
        ResidualAdapter(input_dim, cfg.horizon, target_dim, cfg.adapter_hidden, cfg.dropout).to(device)
        if cfg.adapter_weight > 0
        else None
    )

    print(
        json.dumps(
            {
                "dataset": data_path.name,
                "rows": n,
                "target_vars": target_dim,
                "input_dim": input_dim,
                "train_windows": len(train_ds),
                "val_windows": len(val_ds),
                "test_windows": len(test_ds),
                "device": str(device),
                "config": asdict(cfg),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    started = time.time()
    pretrain_train, pretrain_val = pretrain_backbone(model, train_loader, val_loader, cfg, device)
    static_test_mse = evaluate_static(model, static_test_loader, device)
    print(f"static test mse before online={static_test_mse:.6f}", flush=True)
    if cfg.offline_only:
        elapsed = time.time() - started
        result: Dict[str, object] = {
            "dataset": data_path.stem,
            "backbone": cfg.backbone,
            "horizon": cfg.horizon,
            "lookback": cfg.lookback,
            "seed": cfg.seed,
            "target_vars": target_dim,
            "input_dim": input_dim,
            "time_features": cfg.time_features,
            "train_rows": train_end,
            "val_rows": val_end - train_end,
            "test_rows": n - val_end,
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
            "test_windows": len(static_test_ds),
            "pretrain_train_mse": pretrain_train,
            "pretrain_val_mse": pretrain_val,
            "static_test_mse": static_test_mse,
            "pattern_train_loss": math.nan,
            "padre_mse": math.nan,
            "avg_gate": math.nan,
            "avg_area": math.nan,
            "updates": 0.0,
            "warmup_val_online": cfg.warmup_val_online,
            "warmup_updates": 0.0,
            "elapsed_sec": elapsed,
            "hidden": cfg.hidden,
            "levels": cfg.levels,
            "tcn_head": cfg.tcn_head,
            "dropout": cfg.dropout,
            "revin": cfg.revin,
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "transformer_layers": cfg.transformer_layers,
            "transformer_ff": cfg.transformer_ff,
            "itransformer_token_mode": cfg.itransformer_token_mode,
            "linear_skip": cfg.linear_skip,
            "patch_len": cfg.patch_len,
            "patch_stride": cfg.patch_stride,
            "patch_individual_heads": cfg.patch_individual_heads,
            "pretrain_epochs": cfg.pretrain_epochs,
            "pretrain_lr": cfg.pretrain_lr,
            "pretrain_loss": cfg.pretrain_loss,
            "online_lr": cfg.online_lr,
            "online_weight_decay": cfg.online_weight_decay,
            "policy_lr": cfg.policy_lr,
            "policy_update_interval": cfg.policy_update_interval,
            "policy_gain_scale": cfg.policy_gain_scale,
            "gate_floor": cfg.gate_floor,
            "gate_cap": cfg.gate_cap,
            "area_gate_scale": cfg.area_gate_scale,
            "online_update_stride": cfg.online_update_stride,
            "online_batch_size": cfg.online_batch_size,
            "online_loss": cfg.online_loss,
            "residual_correction": cfg.residual_correction,
            "replay_size": cfg.replay_size,
            "lambda_tri": cfg.lambda_tri,
            "k_clusters": cfg.k_clusters,
            "temperature": cfg.pattern_temperature,
            "pattern_residual_correction": cfg.pattern_residual_correction,
            "pattern_residual_feature_mode": cfg.pattern_residual_feature_mode,
            "adapter_weight": cfg.adapter_weight,
            "adapter_lr": cfg.adapter_lr,
            "adapter_hidden": cfg.adapter_hidden,
            "prediction_ema_decay": cfg.prediction_ema_decay,
            "offline_only": True,
        }
        results_dir = Path(cfg.results_dir)
        append_csv(results_dir / "padre_results.csv", result)
        json_path = results_dir / f"{data_path.stem}_{cfg.backbone}_H{cfg.horizon}_seed{cfg.seed}_offline.json"
        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return result
    pattern_train_loss = train_pattern_encoders(model, controller, bank_loader, cfg, device)
    controller.build_pattern_bank(model, bank_loader, cfg.k_clusters, device)
    controller.freeze_pattern_encoders()
    warmup_metrics = None
    if cfg.warmup_val_online and len(val_starts) > 0:
        print("warming up online state on validation stream", flush=True)
        warmup_metrics = online_evaluate(
            model,
            controller,
            adapter,
            input_values,
            target_values,
            val_starts,
            cfg,
            device,
            score=False,
            reset_state=True,
        )
    online_metrics = online_evaluate(
        model,
        controller,
        adapter,
        input_values,
        target_values,
        test_starts,
        cfg,
        device,
        score=True,
        reset_state=not cfg.warmup_val_online,
    )
    elapsed = time.time() - started
    result: Dict[str, object] = {
        "dataset": data_path.stem,
        "backbone": cfg.backbone,
        "horizon": cfg.horizon,
        "lookback": cfg.lookback,
        "seed": cfg.seed,
        "target_vars": target_dim,
        "input_dim": input_dim,
        "time_features": cfg.time_features,
        "train_rows": train_end,
        "val_rows": val_end - train_end,
        "test_rows": n - val_end,
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "test_windows": len(test_ds) if cfg.limit_online == 0 else min(len(test_ds), cfg.limit_online),
        "pretrain_train_mse": pretrain_train,
        "pretrain_val_mse": pretrain_val,
        "static_test_mse": static_test_mse,
        "pattern_train_loss": pattern_train_loss,
        "padre_mse": online_metrics["mse"],
        "avg_gate": online_metrics["avg_gate"],
        "avg_area": online_metrics["avg_area"],
        "updates": online_metrics["updates"],
        "warmup_val_online": cfg.warmup_val_online,
        "warmup_updates": 0.0 if warmup_metrics is None else warmup_metrics["updates"],
        "elapsed_sec": elapsed,
        "hidden": cfg.hidden,
        "levels": cfg.levels,
        "tcn_head": cfg.tcn_head,
        "dropout": cfg.dropout,
        "revin": cfg.revin,
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "transformer_layers": cfg.transformer_layers,
        "transformer_ff": cfg.transformer_ff,
        "itransformer_token_mode": cfg.itransformer_token_mode,
        "linear_skip": cfg.linear_skip,
        "patch_len": cfg.patch_len,
        "patch_stride": cfg.patch_stride,
        "patch_individual_heads": cfg.patch_individual_heads,
        "pretrain_epochs": cfg.pretrain_epochs,
        "pretrain_lr": cfg.pretrain_lr,
        "pretrain_loss": cfg.pretrain_loss,
        "online_lr": cfg.online_lr,
        "online_weight_decay": cfg.online_weight_decay,
        "policy_lr": cfg.policy_lr,
        "policy_update_interval": cfg.policy_update_interval,
        "policy_gain_scale": cfg.policy_gain_scale,
        "gate_floor": cfg.gate_floor,
        "gate_cap": cfg.gate_cap,
        "area_gate_scale": cfg.area_gate_scale,
        "online_update_stride": cfg.online_update_stride,
        "online_batch_size": cfg.online_batch_size,
        "online_loss": cfg.online_loss,
        "online_train_scope": cfg.online_train_scope,
        "residual_correction": cfg.residual_correction,
        "replay_size": cfg.replay_size,
        "warmup_val_online": cfg.warmup_val_online,
        "lambda_tri": cfg.lambda_tri,
        "k_clusters": cfg.k_clusters,
        "temperature": cfg.pattern_temperature,
        "pattern_residual_correction": cfg.pattern_residual_correction,
        "pattern_residual_feature_mode": cfg.pattern_residual_feature_mode,
        "adapter_weight": cfg.adapter_weight,
        "adapter_lr": cfg.adapter_lr,
        "adapter_hidden": cfg.adapter_hidden,
        "prediction_ema_decay": cfg.prediction_ema_decay,
    }
    results_dir = Path(cfg.results_dir)
    append_csv(results_dir / "padre_results.csv", result)
    json_path = results_dir / f"{data_path.stem}_{cfg.backbone}_H{cfg.horizon}_seed{cfg.seed}.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if cfg.save_model:
        ckpt_path = results_dir / f"{data_path.stem}_{cfg.backbone}_H{cfg.horizon}_seed{cfg.seed}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "controller": controller.state_dict(),
                "adapter": None if adapter is None else adapter.state_dict(),
                "config": asdict(cfg),
            },
            ckpt_path,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result
