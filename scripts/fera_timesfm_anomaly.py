"""
FERA · TimesFM residual-based anomaly detection for gas forward curves
======================================================================
Turns Google's TimesFM (a zero-shot *forecaster*) into an anomaly *detector*
via one-step-ahead forecast residuals, conformal-calibrated per series.

Input  : ml_wide.csv  (S_NO, TRADE_DATE, MARKET, M1..M36)
Output : per-(date, market, tenor) anomaly scores + per-curve aggregate score

Designed to run as ONE more scorer feeding FERA's existing ensemble
(Isolation Forest / PCA-shape / autoencoder). It adds the *temporal* axis
those cross-sectional detectors miss.

Env: pip install timesfm[torch]  (needs HuggingFace access; GPU recommended).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 1. Build the 108 constant-tenor series (3 markets x 36 tenors)
# ----------------------------------------------------------------------------
TENORS = [f"M{i}" for i in range(1, 37)]

def build_series(wide_path: str) -> dict[tuple[str, str], pd.Series]:
    """Return {(market, tenor): daily price Series indexed by trade date}."""
    w = pd.read_csv(wide_path, parse_dates=["TRADE_DATE"]).sort_values("TRADE_DATE")
    series = {}
    for mkt, g in w.groupby("MARKET"):
        g = g.set_index("TRADE_DATE")
        for t in TENORS:
            series[(mkt, t)] = g[t].asfreq("B").interpolate(limit=3)  # align to business days
    return series

# ----------------------------------------------------------------------------
# 2. Stationarise. Detect on LOG-RETURNS, not levels.
#    Levels are non-stationary (2022 crisis: M1 ranged ~14 -> 319). Residuals
#    on levels would be dominated by trend; returns make "surprise" meaningful.
#    NOTE on the roll: M1..M36 are *constant-tenor*, so the underlying contract
#    rolls monthly. Gas rolls are smooth enough to leave in; if a tenor shows a
#    recurring month-boundary blip, mask roll days or detect on constant-delivery
#    series rebuilt from ml_long.csv (DELIVERY_MONTH) instead.
# ----------------------------------------------------------------------------
def to_returns(s: pd.Series) -> pd.Series:
    return np.log(s.where(s > 0)).diff().dropna()

# ----------------------------------------------------------------------------
# 3. TimesFM one-step-ahead, rolling, batched across all series per anchor day
# ----------------------------------------------------------------------------
CONTEXT_LEN = 512   # TimesFM 1.0 supports up to 512; use as much history as you have
HORIZON_LEN = 1     # one-step-ahead surprise

def load_timesfm():
    import timesfm
    return timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu",                 # GPU not available in this env
            per_core_batch_size=32,
            horizon_len=HORIZON_LEN,
            context_len=CONTEXT_LEN,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        ),
    )

def rolling_forecasts(tfm, ret: dict, dates: pd.DatetimeIndex, burn_in: int):
    """For each anchor date >= burn_in, batch-forecast next-day return for every
    series. Returns long df: [date, key, y_true, y_hat, q_lo, q_hi]."""
    keys = list(ret.keys())
    rows = []
    for i in range(burn_in, len(dates) - 1):
        ctx, tgt_keys = [], []
        for k in keys:
            hist = ret[k].loc[:dates[i]].values
            if len(hist) >= 32:                        # min usable context
                ctx.append(hist[-CONTEXT_LEN:])
                tgt_keys.append(k)
        if not ctx:
            continue
        point, quantiles = tfm.forecast(inputs=ctx, freq=[0] * len(ctx))  # 0 = high-freq
        # quantiles[:, h, :] -> [mean, q0.1 ... q0.9]; use 0.1/0.9 as the band
        for j, k in enumerate(tgt_keys):
            nxt = dates[i + 1]
            if nxt in ret[k].index:
                rows.append({
                    "date": nxt, "key": k,
                    "y_true": ret[k].loc[nxt],
                    "y_hat":  point[j, 0],
                    "q_lo":   quantiles[j, 0, 1],
                    "q_hi":   quantiles[j, 0, 9],
                })
    return pd.DataFrame(rows)

# ----------------------------------------------------------------------------
# 4. Score. Two complementary signals, both per series.
#    (a) Conformal score: |residual| / calibrated MAD  -> threshold-free, robust.
#    (b) Quantile-band breach: how far outside the predicted [q_lo, q_hi] band.
#    Conformal wrapping matters because TimesFM quantiles are NOT guaranteed
#    calibrated on gas forwards (out-of-domain vs its pretraining corpus).
# ----------------------------------------------------------------------------
def score(fc: pd.DataFrame, calib_frac: float = 0.5) -> pd.DataFrame:
    out = []
    for k, g in fc.groupby("key"):
        g = g.sort_values("date").copy()
        g["resid"] = g["y_true"] - g["y_hat"]
        n_cal = int(len(g) * calib_frac)
        mad = np.median(np.abs(g["resid"].iloc[:n_cal])) * 1.4826 + 1e-9
        g["conformal_z"] = (g["resid"] / mad).abs()
        band = (g["q_hi"] - g["q_lo"]).clip(lower=1e-9)
        breach = np.maximum(g["q_lo"] - g["y_true"], g["y_true"] - g["q_hi"])
        g["band_breach"] = (breach / band).clip(lower=0)
        out.append(g)
    return pd.concat(out, ignore_index=True)

# ----------------------------------------------------------------------------
# 5. Aggregate per-tenor scores up to a per-curve (date, market) score.
#    Max -> "any tenor anomalous"; mean -> "whole-curve drift". Keep both.
# ----------------------------------------------------------------------------
def curve_scores(scored: pd.DataFrame) -> pd.DataFrame:
    scored = scored.copy()
    scored["market"] = scored["key"].map(lambda k: k[0])
    return (scored.groupby(["date", "market"])["conformal_z"]
            .agg(curve_max="max", curve_mean="mean")
            .reset_index())

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    series  = build_series("ml_wide.csv")
    returns = {k: to_returns(s) for k, s in series.items()}
    dates   = sorted(set().union(*[r.index for r in returns.values()]))
    dates   = pd.DatetimeIndex(dates)

    tfm     = load_timesfm()
    fc      = rolling_forecasts(tfm, returns, dates, burn_in=252)  # ~1y warm-up
    scored  = score(fc)
    curves  = curve_scores(scored)

    scored.to_parquet("fera_timesfm_tenor_scores.parquet")
    curves.to_parquet("fera_timesfm_curve_scores.parquet")
    print(curves.sort_values("curve_max", ascending=False).head(20))
