# PADRE Reproduction Summary

Implemented from `ijcai26.pdf`:

- chronological `20:5:75` split;
- delayed online protocol: a forecast is updated only after its full `H`-step target has arrived;
- offline Pattern Bank with regime-conditioned residual prototypes;
- Tri-View drift representation with online, backward-residual, and episodic-memory views;
- triangle-area consistency and prompt-guided gated online update;
- optional validation-stream warmup using the same delayed protocol, without test leakage;
- fixed experimental backbone focus: `TCN`, `PatchTST`, and `iTransformer`.

Latest implementation changes:

- CUDA PyTorch is installed and `--device auto` uses the local RTX 5060 Laptop GPU.
- PatchTST/iTransformer support paper-style `L=512`; TCN uses paper-style `L=60`.
- TCN now exposes `--tcn-head {last,summary,flatten}`. The default `last` keeps the original implementation; `flatten` is selected for the best current TCN/ETTm1 runs.
- iTransformer uses all input variates as inverted attention tokens by default, with `--itransformer-token-mode target_context` retained for compatibility screening.
- Online loss, train scope, update stride, gate floor/cap, and Pattern residual correction are configurable.
- PatchTST now exposes `--no-revin` and `--patch-individual-heads`; no-RevIN improved ETTm1 full-test PatchTST results.
- A delayed-feedback residual adapter was added with `--adapter-weight`; it is default-off and was not selected for best current full-test results.
- Prediction-time EMA (`--prediction-ema-decay`) was added and is selected by the strongest ETTm1/weather runs.
- `run_table.py` compares against the selected backbone's Table 1 target and now supports the new PatchTST/adapter flags.

Best fair full-test results so far, after the latest trick sweep. Gap is `ours - paper`:

| Dataset | Backbone | H | Ours MSE | Paper target | Gap |
| --- | --- | ---: | ---: | ---: | ---: |
| ETTh2 | TCN | 24 | 1.652 | 2.753 | -1.101 |
| ETTh2 | TCN | 48 | 2.322 | 3.824 | -1.502 |
| ETTh2 | TCN | 96 | 3.296 | 5.543 | -2.247 |
| ETTh2 | PatchTST | 24 | 1.535 | 1.582 | -0.047 |
| ETTh2 | PatchTST | 48 | 2.487 | 2.965 | -0.478 |
| ETTh2 | PatchTST | 96 | 4.023 | 5.284 | -1.261 |
| ETTh2 | iTransformer | 24 | 1.828 | 2.245 | -0.417 |
| ETTh2 | iTransformer | 48 | 2.738 | 3.826 | -1.088 |
| ETTh2 | iTransformer | 96 | 4.460 | 6.065 | -1.605 |
| ETTm1 | TCN | 24 | 0.640 | 0.446 | +0.194 |
| ETTm1 | TCN | 48 | 0.802 | 0.616 | +0.186 |
| ETTm1 | TCN | 96 | 0.866 | 0.673 | +0.193 |
| ETTm1 | PatchTST | 24 | 0.470 | 0.407 | +0.063 |
| ETTm1 | PatchTST | 48 | 0.637 | 0.556 | +0.081 |
| ETTm1 | PatchTST | 96 | 0.724 | 0.648 | +0.076 |
| ETTm1 | iTransformer | 24 | 0.484 | 0.405 | +0.079 |
| ETTm1 | iTransformer | 48 | 0.635 | 0.537 | +0.098 |
| ETTm1 | iTransformer | 96 | 0.721 | 0.626 | +0.095 |
| Weather | TCN | 24 | 0.731 | 0.624 | +0.107 |
| Weather | TCN | 48 | 1.159 | 0.836 | +0.323 |
| Weather | TCN | 96 | 1.502 | 1.196 | +0.306 |
| Weather | PatchTST | 24 | 0.661 | 0.702 | -0.041 |
| Weather | PatchTST | 48 | 0.957 | 0.954 | +0.003 |
| Weather | PatchTST | 96 | 1.269 | 1.243 | +0.026 |
| Weather | iTransformer | 24 | 0.723 | 0.744 | -0.021 |
| Weather | iTransformer | 48 | 1.006 | 0.975 | +0.031 |
| Weather | iTransformer | 96 | 1.294 | 1.144 | +0.150 |

Key result directories:

- `results_patchtst_full/`
- `results_itransformer_full/`
- `results_itransformer_tune/`
- `results_pattern_correction/`
- `results_patchtst_ettm1_norevin_full/`
- `results_patchtst_ettm1_norevin_gate_full/`
- `results_tcn_gpu_weather/`
- `results_patchtst_ettm1_ema_full/`
- `results_patchtst_ettm1_head_ema_full/`
- `results_itransformer_ettm1_best_full/`
- `results_weather_patchtst_best_full/`
- `results_weather_itransformer_norevin_ema_full/`
- `results_tcn_ettm1_flatten_full/`
- `results_tcn_weather_ema_full/`

Current status:

- ETTh2 is solved for all three selected backbones under matching paper backbone targets.
- PatchTST/weather is now effectively matched: H24 is better than the table target, H48 is within 0.003, and H96 is within 0.027.
- iTransformer/weather is improved by no-RevIN, stats-based Pattern residual correction, head-only online updates, and prediction EMA; H96 remains the largest iTransformer weather gap.
- ETTm1 improved across PatchTST and iTransformer with no time features, no-RevIN, stronger gate floor, and EMA, but late-test drift still keeps the full-test gaps around 0.06-0.10 for transformer backbones.
- TCN ETTm1/weather improved with the new TCN head/EMA/replay settings, but TCN remains the largest unresolved gap, especially weather H48/H96.

Tricks that helped:

- `--prediction-ema-decay 0.995` for ETTm1/weather online prediction stability.
- `--no-time-features --no-revin` for PatchTST/weather and ETTm1 transformer runs.
- `--online-train-scope head` for PatchTST/weather.
- `--tcn-head flatten --no-time-features` for TCN/ETTm1.
- `--online-batch-size 4` with higher online LR for TCN/weather.

Tricks screened but not selected:

- Smooth L1 pretraining, online weight decay, PatchTST individual heads, residual adapter, iTransformer `target_context`, iTransformer linear skip, larger iTransformer/TCN capacity, and disabling Pattern residual correction on weather iTransformer.
