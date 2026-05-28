

import argparse
import itertools
import json
from pathlib import Path

from .config import ExperimentConfig
from .experiment import run_experiment
from .targets import PAPER_TARGETS

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PADRE TCN experiments for the three bundled datasets.")
    parser.add_argument("--datasets", nargs="+", default=["ETTh2.csv", "ETTm1.csv", "weather.csv"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[24, 48, 96])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--backbone",
        choices=["tcn", "nlinear", "mlp", "dlinear", "patchtst", "itransformer"],
        default="tcn",
    )
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--no-time-features", action="store_false", dest="time_features")
    parser.set_defaults(time_features=True)
    parser.add_argument("--pretrain-epochs", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--tcn-head", choices=["last", "summary", "flatten"], default="last")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-ff", type=int, default=128)
    parser.add_argument("--itransformer-token-mode", choices=["all", "target_context"], default="all")
    parser.add_argument("--linear-skip", action="store_true")
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--patch-individual-heads", action="store_true")
    parser.add_argument("--no-revin", action="store_false", dest="revin")
    parser.set_defaults(revin=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--pretrain-loss", choices=["mse", "smooth_l1"], default="mse")
    parser.add_argument("--online-lr", type=float, default=2e-4)
    parser.add_argument("--online-weight-decay", type=float, default=0.0)
    parser.add_argument("--policy-lr", type=float, default=1e-3)
    parser.add_argument("--policy-update-interval", type=int, default=1)
    parser.add_argument("--policy-gain-scale", type=float, default=1.0)
    parser.add_argument("--gate-floor", type=float, default=0.0)
    parser.add_argument("--gate-cap", type=float, default=1.0)
    parser.add_argument("--area-gate-scale", type=float, default=0.0)
    parser.add_argument("--online-update-stride", type=int, default=1)
    parser.add_argument("--online-batch-size", type=int, default=1)
    parser.add_argument("--online-loss", choices=["mse", "smooth_l1"], default="mse")
    parser.add_argument("--online-train-scope", choices=["all", "head"], default="all")
    parser.add_argument("--residual-correction", type=float, default=0.0)
    parser.add_argument("--replay-size", type=int, default=256)
    parser.add_argument("--warmup-val-online", action="store_true")
    parser.add_argument("--lambda-tri", type=float, default=0.02)
    parser.add_argument("--memory-threshold", type=float, default=0.65)
    parser.add_argument("--pattern-epochs", type=int, default=0)
    parser.add_argument("--pattern-lr", type=float, default=1e-3)
    parser.add_argument("--k-clusters", type=int, default=4)
    parser.add_argument("--pattern-residual-correction", type=float, default=0.0)
    parser.add_argument("--pattern-residual-feature-mode", choices=["encoded", "stats"], default="encoded")
    parser.add_argument("--adapter-weight", type=float, default=0.0)
    parser.add_argument("--adapter-lr", type=float, default=1e-3)
    parser.add_argument("--adapter-hidden", type=int, default=64)
    parser.add_argument("--prediction-ema-decay", type=float, default=0.0)
    parser.add_argument("--limit-online", type=int, default=0)
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_results = []
    for data_path, horizon, seed in itertools.product(args.datasets, args.horizons, args.seeds):
        cfg = ExperimentConfig(
            data_path=data_path,
            horizon=horizon,
            seed=seed,
            backbone=args.backbone,
            time_features=args.time_features,
            lookback=args.lookback,
            results_dir=args.results_dir,
            pretrain_epochs=args.pretrain_epochs,
            hidden=args.hidden,
            levels=args.levels,
            tcn_head=args.tcn_head,
            dropout=args.dropout,
            d_model=args.d_model,
            n_heads=args.n_heads,
            transformer_layers=args.transformer_layers,
            transformer_ff=args.transformer_ff,
            itransformer_token_mode=args.itransformer_token_mode,
            linear_skip=args.linear_skip,
            patch_len=args.patch_len,
            patch_stride=args.patch_stride,
            patch_individual_heads=args.patch_individual_heads,
            revin=args.revin,
            batch_size=args.batch_size,
            pretrain_lr=args.pretrain_lr,
            pretrain_loss=args.pretrain_loss,
            online_lr=args.online_lr,
            online_weight_decay=args.online_weight_decay,
            policy_lr=args.policy_lr,
            policy_update_interval=args.policy_update_interval,
            policy_gain_scale=args.policy_gain_scale,
            gate_floor=args.gate_floor,
            gate_cap=args.gate_cap,
            area_gate_scale=args.area_gate_scale,
            online_update_stride=args.online_update_stride,
            online_batch_size=args.online_batch_size,
            online_loss=args.online_loss,
            online_train_scope=args.online_train_scope,
            residual_correction=args.residual_correction,
            replay_size=args.replay_size,
            warmup_val_online=args.warmup_val_online,
            lambda_tri=args.lambda_tri,
            memory_threshold=args.memory_threshold,
            pattern_epochs=args.pattern_epochs,
            pattern_lr=args.pattern_lr,
            k_clusters=args.k_clusters,
            pattern_residual_correction=args.pattern_residual_correction,
            pattern_residual_feature_mode=args.pattern_residual_feature_mode,
            adapter_weight=args.adapter_weight,
            adapter_lr=args.adapter_lr,
            adapter_hidden=args.adapter_hidden,
            prediction_ema_decay=args.prediction_ema_decay,
            limit_online=args.limit_online,
            offline_only=args.offline_only,
            device=args.device,
        )
        result = run_experiment(cfg)
        target = PAPER_TARGETS.get(args.backbone, {}).get(Path(data_path).stem, {}).get(horizon)
        if target is not None:
            result["paper_target_backbone"] = args.backbone
            result["paper_target"] = target
            result["abs_gap"] = abs(float(result["padre_mse"]) - target)
            print(
                f"target gap {Path(data_path).stem} H={horizon}: "
                f"ours={float(result['padre_mse']):.6f}, paper={target:.6f}, "
                f"gap={float(result['abs_gap']):.6f}",
                flush=True,
            )
        all_results.append(result)
    summary_path = Path(args.results_dir) / "table_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(all_results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
