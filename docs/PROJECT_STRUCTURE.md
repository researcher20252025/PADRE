# Project Structure

The implementation is organized as a small Python package while preserving the
original command-line entrypoints:

```text
padre/
  config.py          ExperimentConfig dataclass
  data.py            CSV loading, time features, chronological split, windows
  models/            TCN, PatchTST, iTransformer, and compatibility baselines
  controller.py      Pattern Bank, Tri-View drift state, gate, residual adapter
  training.py        Offline pretraining and delayed-feedback online updates
  experiment.py      End-to-end single experiment orchestration
  cli.py             Single-run CLI parser
  table.py           Multi-dataset/multi-horizon sweep runner
  targets.py         Paper Table 1 targets for gap reporting

padre_experiment.py  Backward-compatible wrapper for single runs
run_table.py         Backward-compatible wrapper for table sweeps
```

The code path for a normal run is:

1. `padre.cli.parse_args` builds an `ExperimentConfig`.
2. `padre.experiment.run_experiment` loads and normalizes a dataset with the
   `20:5:75` chronological split.
3. A selected backbone is built from `padre.models`.
4. `padre.controller.PADREController` builds the offline Pattern Bank.
5. `padre.training.online_evaluate` runs delayed-feedback online adaptation.

The wrapper scripts are intentionally thin so existing commands such as
`python padre_experiment.py ...` and `python run_table.py ...` keep working.
