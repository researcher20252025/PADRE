"""Dataset loading, time features, and chronological windowing."""

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def load_values(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    values = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    # The bundled Weather file contains a single -9999 sentinel in wind speed.
    values = values.mask(values <= -999)
    values = values.interpolate(limit_direction="both").ffill().bfill()
    return values.to_numpy(dtype=np.float32)


def make_time_features(dates: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(dates, errors="coerce")
    dt = dt.ffill().bfill()
    minute_of_day = (dt.dt.hour.to_numpy() * 60 + dt.dt.minute.to_numpy()).astype(np.float32)
    day_of_week = dt.dt.dayofweek.to_numpy(dtype=np.float32)
    day_of_year = dt.dt.dayofyear.to_numpy(dtype=np.float32)
    month = (dt.dt.month.to_numpy(dtype=np.float32) - 1.0)
    feats = [
        np.sin(2 * np.pi * minute_of_day / 1440.0),
        np.cos(2 * np.pi * minute_of_day / 1440.0),
        np.sin(2 * np.pi * day_of_week / 7.0),
        np.cos(2 * np.pi * day_of_week / 7.0),
        np.sin(2 * np.pi * day_of_year / 366.0),
        np.cos(2 * np.pi * day_of_year / 366.0),
        np.sin(2 * np.pi * month / 12.0),
        np.cos(2 * np.pi * month / 12.0),
    ]
    return np.stack(feats, axis=1).astype(np.float32)


def load_target_and_time(path: str, use_time_features: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    df = pd.read_csv(path)
    values = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    values = values.mask(values <= -999)
    values = values.interpolate(limit_direction="both").ffill().bfill()
    time_feats = make_time_features(df.iloc[:, 0]) if use_time_features else None
    return values.to_numpy(dtype=np.float32), time_feats


def chronological_split(n: int) -> Tuple[int, int]:
    train_end = int(n * 0.20)
    val_end = int(n * 0.25)
    return train_end, val_end


def segment_indices(
    lookback: int,
    horizon: int,
    start_target: int,
    end_target: int,
) -> np.ndarray:
    """Return input starts whose forecast targets are fully inside a segment."""
    first_target = max(start_target, lookback)
    last_target = end_target - horizon
    if last_target < first_target:
        return np.empty(0, dtype=np.int64)
    target_starts = np.arange(first_target, last_target + 1, dtype=np.int64)
    return target_starts - lookback


class WindowDataset(Dataset):
    def __init__(
        self,
        input_values: np.ndarray,
        target_values: np.ndarray,
        starts: Sequence[int],
        lookback: int,
        horizon: int,
    ):
        self.input_values = input_values
        self.target_values = target_values
        self.starts = np.asarray(starts, dtype=np.int64)
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self) -> int:
        return int(len(self.starts))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[idx])
        x = self.input_values[start : start + self.lookback]
        y = self.target_values[start + self.lookback : start + self.lookback + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)
