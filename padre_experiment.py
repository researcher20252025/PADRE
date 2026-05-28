"""Backward-compatible wrapper for the PADRE single-run CLI.

The implementation lives under the `padre` package. This file is kept so the
existing reproduction commands and external imports continue to work.
"""

from padre import ExperimentConfig, run_experiment
from padre.cli import main, parse_args

__all__ = ["ExperimentConfig", "main", "parse_args", "run_experiment"]


if __name__ == "__main__":
    main()
