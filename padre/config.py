"""Configuration dataclass for PADRE experiments."""

from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    data_path: str
    horizon: int
    lookback: int = 60
    seed: int = 2026
    backbone: str = "tcn"
    time_features: bool = True
    hidden: int = 32
    levels: int = 3
    kernel_size: int = 3
    tcn_head: str = "last"
    dropout: float = 0.1
    revin: bool = True
    d_model: int = 64
    n_heads: int = 4
    transformer_layers: int = 2
    transformer_ff: int = 128
    itransformer_token_mode: str = "all"
    linear_skip: bool = False
    patch_len: int = 16
    patch_stride: int = 8
    patch_individual_heads: bool = False
    batch_size: int = 64
    pretrain_epochs: int = 12
    pretrain_lr: float = 1e-3
    pretrain_loss: str = "mse"
    weight_decay: float = 1e-4
    online_lr: float = 2e-4
    online_weight_decay: float = 0.0
    policy_lr: float = 1e-3
    policy_weight_decay: float = 1e-5
    policy_update_interval: int = 1
    policy_gain_scale: float = 1.0
    gate_floor: float = 0.0
    gate_cap: float = 1.0
    area_gate_scale: float = 0.0
    online_update_stride: int = 1
    online_batch_size: int = 1
    online_loss: str = "mse"
    online_train_scope: str = "all"
    replay_size: int = 256
    warmup_val_online: bool = False
    emb_dim: int = 32
    drift_dim: int = 32
    policy_hidden: int = 64
    pattern_epochs: int = 0
    pattern_lr: float = 1e-3
    k_clusters: int = 4
    pattern_temperature: float = 0.5
    pattern_residual_correction: float = 0.0
    pattern_residual_feature_mode: str = "encoded"
    adapter_weight: float = 0.0
    adapter_lr: float = 1e-3
    adapter_hidden: int = 64
    prediction_ema_decay: float = 0.0
    residual_memory: int = 4
    residual_correction: float = 0.0
    episodic_memory: int = 128
    memory_threshold: float = 0.65
    lambda_tri: float = 0.02
    grad_clip: float = 1.0
    device: str = "auto"
    limit_online: int = 0
    offline_only: bool = False
    num_workers: int = 0
    results_dir: str = "results"
    save_model: bool = False
