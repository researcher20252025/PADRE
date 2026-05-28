"""Backward-compatible wrapper for table-style PADRE sweeps."""

from padre.table import main, parse_args
from padre.targets import PAPER_TARGETS

__all__ = ["PAPER_TARGETS", "main", "parse_args"]


if __name__ == "__main__":
    main()
