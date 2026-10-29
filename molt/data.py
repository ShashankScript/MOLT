"""Credit-card dataset loading, temporal partitions, and matched stream sampling."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
TARGET = "Class"
TIME = "Time"


def _boundary_index(ordered: pd.DataFrame, target_index: int) -> int:
    # same-Time rows must stay on one side of a split
    if target_index <= 0:
        return 0
    if target_index >= len(ordered):
        return len(ordered)
    return int(np.searchsorted(ordered[TIME].to_numpy(), ordered[TIME].iloc[target_index], side="left"))


def load_transactions(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Put creditcard.csv in data/raw/.")
    df = pd.read_csv(path)
    required = {TIME, TARGET, "Amount"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if df[TARGET].isna().any() or df[TIME].isna().any():
        raise ValueError("Time and Class must not contain missing values.")
    if set(df[TARGET].unique()) - {0, 1}:
        raise ValueError("Class must be binary with 0=legitimate and 1=fraud.")
    return df.sort_values(TIME, kind="mergesort").reset_index(drop=True)


def temporal_split(
    df: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # chronological split, never shuffled
    if TIME not in df:
        raise ValueError(f"Temporal split requires the {TIME!r} column.")
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1):
        raise ValueError("Split fractions must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than 1.")
    ordered = df.sort_values(TIME, kind="mergesort").reset_index(drop=True)
    a = _boundary_index(ordered, int(len(ordered) * train_fraction))
    b = _boundary_index(ordered, int(len(ordered) * (train_fraction + validation_fraction)))
    if a == 0 or b <= a or b >= len(ordered):
        raise ValueError("Dataset is too small for the requested temporal split.")
    return ordered.iloc[:a].copy(), ordered.iloc[a:b].copy(), ordered.iloc[b:].copy()


def features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET not in df:
        raise ValueError("Class target column is required.")
    internal = [column for column in df.columns if str(column).startswith("_")]
    features = df.drop(columns=[TARGET, TIME, *internal], errors="ignore").copy()
    if features.empty:
        raise ValueError("No model features remain after excluding Time and Class.")
    return features, df[TARGET].astype(int).copy()


def _sample_pool(pool: pd.DataFrame, seed: int, keep_fraction: float, pool_id: int) -> pd.DataFrame:
    # keep every fraud row, subsample the legit ones
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(pool_id)]))
    fraud = pool[TARGET].to_numpy(dtype=int) == 1
    keep = fraud | (rng.random(len(pool)) < keep_fraction)
    sampled = pool.loc[keep].copy()
    sampled["_source_index"] = sampled.index.astype(int)
    return sampled.sort_values(TIME, kind="mergesort").reset_index(drop=True)


def build_stream_pools(
    df: pd.DataFrame,
    seed: int,
    initial_fraction: float = 0.20,
    calibration_fraction: float = 0.05,
    legitimate_keep_fraction: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create timestamp-disjoint pools with a matched class-retention protocol.

    All fraud examples are retained in each pool; legitimate rows are sampled
    without replacement at the same rate. This makes the unlabeled feature
    reference and later stream share the same sampling-induced class mixture.
    """
    if initial_fraction <= 0 or calibration_fraction <= 0:
        raise ValueError("Initial and calibration fractions must be positive.")
    if initial_fraction + calibration_fraction >= 1:
        raise ValueError("Initial and calibration fractions must sum to less than 1.")
    if not 0 < legitimate_keep_fraction <= 1:
        raise ValueError("legitimate_keep_fraction must be in (0, 1].")
    ordered = df.sort_values(TIME, kind="mergesort").reset_index(drop=True).copy()
    train_end = _boundary_index(ordered, int(len(ordered) * initial_fraction))
    calibration_end = _boundary_index(ordered, int(len(ordered) * (initial_fraction + calibration_fraction)))
    initial_raw = ordered.iloc[:train_end].copy()
    calibration_raw = ordered.iloc[train_end:calibration_end].copy()
    stream_raw = ordered.iloc[calibration_end:].copy()
    initial = _sample_pool(initial_raw, seed, legitimate_keep_fraction, 0)
    calibration = _sample_pool(calibration_raw, seed, legitimate_keep_fraction, 1)
    stream = _sample_pool(stream_raw, seed, legitimate_keep_fraction, 2)
    stream["_stream_index"] = np.arange(len(stream), dtype=np.int32)
    return initial, calibration, stream


def split_windows(
    stream: pd.DataFrame,
    window_size: int = 4000,
    min_tail_fraction: float = 0.5,
) -> list[pd.DataFrame]:
    """Split chronologically and merge a short final remainder by row count.

    The merge decision is label-blind: it depends only on the number of rows.
    """
    if window_size < 1:
        raise ValueError("window_size must be positive.")
    if not 0 <= min_tail_fraction < 1:
        raise ValueError("min_tail_fraction must be in [0, 1).")
    windows = [stream.iloc[i : i + window_size].copy() for i in range(0, len(stream), window_size)]
    if len(windows) > 1 and len(windows[-1]) < window_size * min_tail_fraction:
        windows[-2] = pd.concat([windows[-2], windows[-1]], axis=0).copy()
        windows.pop()
    return windows
