# Reproduction Notes

## Protocol

- Split: chronological `20:5:75` train/validation/test.
- Horizons: `H in {24, 48, 96}`.
- Lookback: `L=60` for TCN; `L=512` for PatchTST and iTransformer.
- Online feedback: a prediction emitted at step `t` is only used after the full
  target window arrives at step `t + H`.
- Optional validation warmup uses the same delayed-feedback protocol and does
  not score or leak test samples.

## Selected Backbone Settings

The code keeps one core PADRE method and exposes backbone-specific stability
settings discovered during reproduction:

- TCN/ETTm1: `--tcn-head flatten --no-time-features --prediction-ema-decay 0.995`.
- TCN/weather: `--prediction-ema-decay 0.995 --online-batch-size 4`.
- PatchTST/weather: `--no-time-features --no-revin --online-train-scope head --prediction-ema-decay 0.995`.
- iTransformer/weather: `--no-revin --online-train-scope head --pattern-residual-correction 1.0 --pattern-residual-feature-mode stats --prediction-ema-decay 0.995`.

PADRE modules: Pattern Bank, Tri-View drift evidence, triangle-area consistency,
prompt-guided gate, delayed online update, and episodic memory.

## QuickTest

```powershell
python padre_experiment.py --data-path ETTh2.csv --horizon 24 --backbone tcn --lookback 60 --hidden 8 --levels 1 --pretrain-epochs 1 --limit-online 16 --results-dir results_smoke_refactor
```

## Full   Sweep

```powershell
python run_table.py --backbone patchtst --datasets ETTh2.csv ETTm1.csv weather.csv --horizons 24 48 96 --lookback 512
```
