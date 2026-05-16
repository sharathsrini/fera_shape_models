"""Evaluation utilities: reconstruction error, anomaly scoring,
ranking shape anomalies and per-tenor kink detection.

Calibration philosophy
----------------------
Anomaly scores must be calibrated against a reference distribution that is
itself "normal". We use the TRAINING residuals (per market and per tenor) as
that reference. Then every scored curve is compared back through:

    curve_z      = (curve_mse - mu_curve[market]) / sigma_curve[market]
    tenor_z[t]   = (|resid[t]| - mu_t[market, t]) / sigma_t[market, t]

This makes scores comparable across markets and gives kink z-scores that mean
"how unusual is this residual compared to the model's typical training
residual for THIS market at THIS tenor".
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .utils import get_device
from .data import TENORS, MARKETS


@torch.no_grad()
def collect_reconstructions(model, loader, device=None):
    """Iterate `loader` once and collect (orig, recon, market_oh, level, idx)
    arrays in iteration order. Model is set to eval(), and VAEs return
    deterministic mu-based reconstructions.
    """
    device = device or get_device()
    model = model.to(device).eval()
    recs, originals, ids, levels, markets = [], [], [], [], []
    for shape, market, level, raw, idx in loader:
        shape = shape.to(device); market = market.to(device)
        out = model(shape, market)
        recs.append(out["recon"].detach().cpu().numpy())
        originals.append(shape.detach().cpu().numpy())
        markets.append(market.detach().cpu().numpy())
        levels.append(level.numpy())
        ids.append(idx.numpy())
    return {
        "recon": np.concatenate(recs),
        "orig": np.concatenate(originals),
        "market_oh": np.concatenate(markets),
        "level": np.concatenate(levels),
        "idx": np.concatenate(ids),
    }


def per_curve_score(orig, recon):
    """Mean-squared residual per curve  →  raw curve-level shape anomaly score."""
    return np.mean((orig - recon) ** 2, axis=1)


def per_tenor_residual(orig, recon):
    """Signed residual per tenor (B, 36)."""
    return orig - recon


def build_residual_baseline(orig: np.ndarray, recon: np.ndarray,
                            markets: list[str]) -> dict:
    """Fit per-market mean/std of curve MSE and per-(market, tenor) mean/std of
    |residual|. Pass training-set reconstructions here.

    Returns
    -------
    {
        "per_market": {market: {"mu": float, "sigma": float}},     # for curve_z
        "per_tenor":  {market: {"mu": (36,), "sigma": (36,)}},     # for tenor_z
        "global":     {"mu": float, "sigma": float},
    }
    """
    res_abs = np.abs(orig - recon)
    mse = np.mean((orig - recon) ** 2, axis=1)
    mkt_arr = np.asarray(markets)

    per_market = {}
    per_tenor = {}
    for m in np.unique(mkt_arr):
        sel = mkt_arr == m
        per_market[m] = {
            "mu": float(np.mean(mse[sel])),
            "sigma": float(np.std(mse[sel]) + 1e-9),
        }
        per_tenor[m] = {
            "mu": res_abs[sel].mean(axis=0),
            "sigma": res_abs[sel].std(axis=0) + 1e-9,
        }
    return {
        "per_market": per_market,
        "per_tenor": per_tenor,
        "global": {"mu": float(mse.mean()), "sigma": float(mse.std() + 1e-9)},
    }


def calibrated_curve_z(scores: np.ndarray, markets: list[str], baseline: dict) -> np.ndarray:
    """z-score curve MSE using per-market baseline."""
    out = np.empty_like(scores, dtype=np.float64)
    mkt = np.asarray(markets)
    for m, stat in baseline["per_market"].items():
        sel = mkt == m
        out[sel] = (scores[sel] - stat["mu"]) / stat["sigma"]
    return out


def calibrated_tenor_z(orig: np.ndarray, recon: np.ndarray,
                       markets: list[str], baseline: dict) -> np.ndarray:
    """Per-(curve, tenor) z-score of |residual| against per-market-tenor baseline."""
    res_abs = np.abs(orig - recon)
    out = np.empty_like(res_abs, dtype=np.float64)
    mkt = np.asarray(markets)
    for m, stat in baseline["per_tenor"].items():
        sel = mkt == m
        out[sel] = (res_abs[sel] - stat["mu"]) / stat["sigma"]
    return out


def detect_kinks_calibrated(orig: np.ndarray, recon: np.ndarray,
                            markets: list[str], baseline: dict,
                            z_thresh: float = 3.0) -> list[dict]:
    """Flag tenors with |z| > z_thresh using the calibrated tenor baseline."""
    z = calibrated_tenor_z(orig, recon, markets, baseline)
    flagged = []
    for i in range(z.shape[0]):
        bad = np.where(np.abs(z[i]) > z_thresh)[0]
        if len(bad):
            flagged.append({
                "row_idx": int(i),
                "market": markets[i],
                "bad_tenors": [int(t) + 1 for t in bad],
                "residual_z": [float(z[i, t]) for t in bad],
            })
    return flagged


# ----------------------------------------------------------------------
# Back-compat shims (old names still importable)
# ----------------------------------------------------------------------
def per_tenor_score(orig, recon):
    return np.abs(orig - recon)


def detect_kinks(orig, recon, residual_z_thresh: float = 3.0):
    """Legacy: z-scored on the SCORED set itself. Prefer detect_kinks_calibrated."""
    res = orig - recon
    mu = res.mean(axis=0, keepdims=True)
    sd = res.std(axis=0, keepdims=True) + 1e-9
    z = (res - mu) / sd
    flagged = []
    for i in range(z.shape[0]):
        bad = np.where(np.abs(z[i]) > residual_z_thresh)[0]
        if len(bad):
            flagged.append({"row_idx": int(i), "bad_tenors": [int(t) + 1 for t in bad],
                            "residual_z": [float(z[i, t]) for t in bad]})
    return flagged


def rank_anomalies(df_meta: pd.DataFrame, scores: np.ndarray, top_k: int = 20) -> pd.DataFrame:
    out = df_meta.copy().reset_index(drop=True)
    out["score"] = scores
    return out.sort_values("score", ascending=False).head(top_k)
