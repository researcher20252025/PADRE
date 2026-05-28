# PADRE PyTorch Reproduction

This folder implements the PADRE method from `""Beyond Uniform Updates: Drift Pattern Aware Online Time Series Forecasting under Delayed Feedback""` in Python/PyTorch. The active reproduction focus is fixed to the three paper backbones requested by the experiments:

- `tcn`
- `patchtst`
- `itransformer`

Implemented protocol:

- chronological train/validation/test split of `20:5:75`;
- lookback `L=60` for TCN and `L=512` for PatchTST/iTransformer;
- forecast horizons `H in {24, 48, 96}`;
- delayed online protocol: a prediction emitted at step `t` is used for update only at step `t + H`;
- PADRE modules: offline Pattern Bank, Tri-View drift evidence, triangle-area consistency, prompt-guided gated online update, and episodic drift memory.

Code layout:

- `padre/config.py`: experiment configuration dataclass;
- `padre/data.py`: CSV loading, time features, chronological split, and windows;
- `padre/models/`: TCN, PatchTST, iTransformer, and compatibility linear baselines;
- `padre/controller.py`: Pattern Bank, Tri-View drift controller, gate, and residual adapter;
- `padre/training.py`: offline pretraining and delayed-feedback online update loop;
- `padre/experiment.py`: end-to-end experiment orchestration;
- `padre_experiment.py` and `run_table.py`: backward-compatible CLI wrappers.

Run one experiment:
```powershell
python padre_experiment.py --backbone tcn --data-path ETTh2.csv --horizon 24 --lookback 60
```
GPU note: the local environment has `torch 2.12.0+cu130`; `--device auto` uses the RTX 5060  GPU.



```powershell

```
