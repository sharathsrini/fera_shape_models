"""Dataset, transforms and splits for gas forward curves.

Key design choices
------------------
1. Curves are the unit of analysis: each (TRADE_DATE, MARKET) -> 36 prices is one sample.
2. We focus on SHAPE — so curves are *level-normalized* before model ingestion:
       y_i = x_i / mean(x)                          (multiplicative normalization)
   Then we standardise across the training set per market.
3. Time-aware split: we split chronologically, NOT randomly, to avoid leakage.
4. Market is encoded as a one-hot conditioning vector so a single model can serve all three.
5. We expose two main "views":
    - Raw price curve (for level/scale studies)
    - Shape curve (level-normalized) for AE training
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

TENORS = [f"M{i}" for i in range(1, 37)]
MARKETS = ("TTF", "THE", "JKM")


# ----------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------
def level_normalize(curve: np.ndarray) -> tuple[np.ndarray, float]:
    """Divide curve by its mean so the SHAPE is preserved but level is removed.

    Returns (shape_curve, level) where shape_curve has mean 1.
    """
    level = float(np.mean(curve))
    if level <= 0:
        return curve.astype(np.float32), level
    return (curve / level).astype(np.float32), level


def log_transform(curve: np.ndarray) -> np.ndarray:
    """Optional log transform (price > 0)."""
    return np.log(np.clip(curve, 1e-6, None)).astype(np.float32)


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
@dataclass
class CurveSample:
    trade_date: pd.Timestamp
    market: str
    shape_curve: np.ndarray         # (36,)  mean-1
    level: float                    # original mean price
    market_onehot: np.ndarray       # (3,)
    raw_curve: np.ndarray           # (36,)  original prices


class GasCurveDataset(Dataset):
    """Returns (shape_curve, market_onehot, level, raw_curve, meta_idx).

    Parameters
    ----------
    df : DataFrame with columns TRADE_DATE, MARKET, M1..M36 (wide format).
    normalize : how to scale the SHAPE input fed to the model.
        "level"          - divide by mean (default; preserves shape)
        "level_std"      - level-normalize, then z-score per tenor (training stats only)
        "log_level"      - log + divide by mean
        "raw"            - identity (for level-aware models)
    train_stats : dict {"mean": (36,), "std": (36,)} fit on TRAIN ONLY.
                  If supplied with normalize="level_std", will be used; else computed.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        normalize: str = "level_std",
        train_stats: dict | None = None,
    ):
        assert normalize in ("level", "level_std", "log_level", "raw")
        self.df = df.reset_index(drop=True)
        self.normalize = normalize
        self.market_to_idx = {m: i for i, m in enumerate(MARKETS)}

        # Pre-compute raw + shape arrays
        raw = self.df[TENORS].values.astype(np.float32)
        levels = raw.mean(axis=1)
        levels = np.where(levels <= 0, 1.0, levels)
        shape = raw / levels[:, None]
        if normalize == "log_level":
            shape = np.log(np.clip(raw, 1e-6, None)) - np.log(levels)[:, None]
        elif normalize == "raw":
            shape = raw

        if normalize == "level_std":
            if train_stats is None:
                mu = shape.mean(axis=0)
                sd = shape.std(axis=0) + 1e-6
                self.train_stats = {"mean": mu, "std": sd}
            else:
                self.train_stats = train_stats
            shape = (shape - self.train_stats["mean"]) / self.train_stats["std"]
        else:
            self.train_stats = train_stats or {}

        self._raw = raw
        self._shape = shape.astype(np.float32)
        self._levels = levels.astype(np.float32)

        onehots = np.zeros((len(self.df), len(MARKETS)), dtype=np.float32)
        for i, m in enumerate(self.df["MARKET"].values):
            onehots[i, self.market_to_idx[m]] = 1.0
        self._oh = onehots

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self._shape[idx]),
            torch.from_numpy(self._oh[idx]),
            torch.tensor(self._levels[idx], dtype=torch.float32),
            torch.from_numpy(self._raw[idx]),
            idx,
        )


# ----------------------------------------------------------------------
# Known event windows (these are TAGGED, optionally EXCLUDED from training)
# ----------------------------------------------------------------------
# Each entry: (label, start_date, end_date inclusive). Add freely.
KNOWN_EVENT_WINDOWS = [
    ("JKM cold-snap 2021Q1",   "2021-01-08", "2021-02-15"),
    ("EU storage scare 2021",  "2021-09-15", "2021-10-15"),
    ("EU pre-invasion spike",  "2021-12-01", "2022-02-23"),
    ("Russian invasion shock", "2022-02-24", "2022-04-30"),
    ("Nord Stream halt",       "2022-08-15", "2022-09-30"),
    ("Dec 2022 cold/holidays", "2022-12-15", "2023-01-10"),
]


def tag_known_events(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with an EVENT_LABEL column."""
    out = df.copy()
    out["EVENT_LABEL"] = ""
    for label, s, e in KNOWN_EVENT_WINDOWS:
        m = (out["TRADE_DATE"] >= pd.Timestamp(s)) & (out["TRADE_DATE"] <= pd.Timestamp(e))
        out.loc[m, "EVENT_LABEL"] = label
    return out


def _filter_excluded(df: pd.DataFrame, exclude_windows: list[tuple[str, str]] | None) -> pd.DataFrame:
    if not exclude_windows:
        return df
    mask = pd.Series(False, index=df.index)
    for s, e in exclude_windows:
        mask |= (df["TRADE_DATE"] >= pd.Timestamp(s)) & (df["TRADE_DATE"] <= pd.Timestamp(e))
    return df.loc[~mask].reset_index(drop=True)


# ----------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------
def time_aware_split(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split per market then concatenate."""
    parts = {"train": [], "val": [], "test": []}
    for mkt, g in df.groupby("MARKET"):
        g = g.sort_values("TRADE_DATE").reset_index(drop=True)
        n = len(g)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        n_train = n - n_val - n_test
        parts["train"].append(g.iloc[:n_train])
        parts["val"].append(g.iloc[n_train : n_train + n_val])
        parts["test"].append(g.iloc[n_train + n_val :])
    out = {k: pd.concat(v).sort_values("TRADE_DATE").reset_index(drop=True) for k, v in parts.items()}
    return out["train"], out["val"], out["test"]


def load_wide(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["TRADE_DATE"])
    return df[["TRADE_DATE", "MARKET"] + TENORS].copy()


def build_loaders(
    csv_path: str | Path,
    batch_size: int = 64,
    normalize: str = "level_std",
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    num_workers: int = 0,
    exclude_windows: list[tuple[str, str]] | None = None,
    train_stats_override: dict | None = None,
):
    """Build train/val/test data loaders.

    Parameters
    ----------
    exclude_windows : optional list of (start, end) date strings. Curves whose
        TRADE_DATE falls inside any window are REMOVED FROM TRAIN ONLY (val and
        test still see them). Use to learn "normal" without crisis contamination.
        Pass `KNOWN_EVENT_WINDOWS` (re-mapped to (s, e) tuples) to exclude all
        tagged historical events.
    train_stats_override : dict from a saved checkpoint's `preproc` payload.
        If supplied, val/test (and the recomputed train ds) reuse these stats
        verbatim — guarantees inference uses the SAME normalization that was
        applied during training.
    """
    df = load_wide(csv_path)
    tr, va, te = time_aware_split(df, val_frac, test_frac)
    tr_filtered = _filter_excluded(tr, exclude_windows)
    train_ds = GasCurveDataset(tr_filtered, normalize=normalize, train_stats=train_stats_override)
    ts = train_stats_override or train_ds.train_stats
    val_ds = GasCurveDataset(va, normalize=normalize, train_stats=ts)
    test_ds = GasCurveDataset(te, normalize=normalize, train_stats=ts)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return {
        "df": df,
        "train_df_raw": tr,
        "train_df": tr_filtered, "val_df": va, "test_df": te,
        "train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds,
        "train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader,
        "train_stats": ts,
        "exclude_windows": exclude_windows or [],
    }


def loaders_for_inference(csv_path: str | Path, preproc: dict, batch_size: int = 128):
    """Build loaders that reuse a CHECKPOINTED preproc (mean/std), so scoring is
    invariant to changes in the input CSV.
    """
    return build_loaders(csv_path, batch_size=batch_size,
                         normalize=preproc.get("normalize", "level_std"),
                         train_stats_override={
                             "mean": np.asarray(preproc["mean"], dtype=np.float32),
                             "std":  np.asarray(preproc["std"],  dtype=np.float32),
                         })
