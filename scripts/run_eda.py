"""
Exploratory Data Analysis for gas forward curves.

Run with:
    uv run --python /usr/bin/python3 python scripts/run_eda.py
or simply:
    python scripts/run_eda.py

Outputs are saved to ./eda_outputs/.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eda_outputs"
OUT.mkdir(exist_ok=True, parents=True)
sns.set_theme(style="whitegrid", context="talk")

# ----------------------------------------------------------------------
# 1. Load
# ----------------------------------------------------------------------
wide = pd.read_csv(ROOT / "ml_wide.csv", parse_dates=["TRADE_DATE"])
long = pd.read_csv(ROOT / "ml_long.csv", parse_dates=["TRADE_DATE", "DELIVERY_MONTH"])
TENORS = [f"M{i}" for i in range(1, 37)]

print(f"WIDE: {wide.shape},  LONG: {long.shape}")
print(f"Markets:  {sorted(wide['MARKET'].unique().tolist())}")
print(f"Date range: {wide['TRADE_DATE'].min().date()}  →  {wide['TRADE_DATE'].max().date()}")
print(f"Trade dates: {wide['TRADE_DATE'].nunique()}")

# ----------------------------------------------------------------------
# 2. Per-market summary  →  CSV + simple bar plot
# ----------------------------------------------------------------------
mkt_summary = (
    wide.groupby("MARKET")[TENORS]
    .agg(["mean", "std", "min", "max"])
    .round(3)
)
mkt_summary.to_csv(OUT / "market_summary.csv")

# Mean curve per market
mean_curves = wide.groupby("MARKET")[TENORS].mean()
fig, ax = plt.subplots(figsize=(12, 5))
for m in mean_curves.index:
    ax.plot(range(1, 37), mean_curves.loc[m].values, marker="o", label=m, linewidth=2)
ax.set_xlabel("Forward Month (tenor)")
ax.set_ylabel("Mean Price (EUR / MWh or $/MMBtu)")
ax.set_title("Average forward curve by market (2021-01 to 2026-05)")
ax.legend(title="Market")
fig.tight_layout()
fig.savefig(OUT / "01_mean_curve_by_market.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------
# 3. Front-month time series (M1)  — gives macro context (2022 spike etc.)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 5))
for m, g in wide.groupby("MARKET"):
    g = g.sort_values("TRADE_DATE")
    ax.plot(g["TRADE_DATE"], g["M1"], label=m, linewidth=1.2)
ax.set_title("Front-month price (M1) across time — captures 2022 EU gas crisis")
ax.set_ylabel("Price")
ax.legend(title="Market")
ax.xaxis.set_major_locator(mdates.YearLocator())
fig.tight_layout()
fig.savefig(OUT / "02_m1_timeseries.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------
# 4. Heatmap of one market's curves over time (THE)  — tenor × date
# ----------------------------------------------------------------------
for mkt in ["TTF", "THE", "JKM"]:
    sub = wide[wide["MARKET"] == mkt].sort_values("TRADE_DATE")
    Z = sub[TENORS].values.T              # 36 × N
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(
        Z, aspect="auto", origin="lower",
        extent=[mdates.date2num(sub["TRADE_DATE"].iloc[0]),
                mdates.date2num(sub["TRADE_DATE"].iloc[-1]),
                1, 36],
        cmap="viridis",
    )
    ax.xaxis_date()
    ax.set_xlabel("Trade Date")
    ax.set_ylabel("Tenor (months ahead)")
    ax.set_title(f"{mkt}: forward price heatmap")
    fig.colorbar(im, ax=ax, label="Price")
    fig.tight_layout()
    fig.savefig(OUT / f"03_heatmap_{mkt}.png", dpi=150)
    plt.close(fig)

# ----------------------------------------------------------------------
# 5. Slope and curvature distributions
#    slope     = M12 - M1                  (1y slope)
#    curvature = (M1 + M24 - 2*M12) / 2    (2nd diff at 12m)
# ----------------------------------------------------------------------
shape = wide[["TRADE_DATE", "MARKET"]].copy()
shape["LEVEL"] = wide[TENORS].mean(axis=1)
shape["SLOPE_1Y"] = wide["M12"] - wide["M1"]
shape["SLOPE_3Y"] = wide["M36"] - wide["M1"]
shape["CURVATURE"] = (wide["M1"] + wide["M24"] - 2 * wide["M12"]) / 2
shape["RANGE"] = wide[TENORS].max(axis=1) - wide[TENORS].min(axis=1)
shape["REGIME"] = np.where(shape["SLOPE_1Y"] >= 0, "Contango", "Backwardation")

shape.to_csv(OUT / "shape_features.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, col, title in zip(
    axes,
    ["SLOPE_1Y", "CURVATURE", "LEVEL"],
    ["1-year slope (M12−M1)", "Curvature ((M1+M24)/2 − M12)", "Curve level (mean)"],
):
    for mkt in shape["MARKET"].unique():
        sns.kdeplot(shape.loc[shape["MARKET"] == mkt, col], ax=ax, label=mkt, fill=True, alpha=0.3)
    ax.set_title(title)
    ax.legend()
fig.tight_layout()
fig.savefig(OUT / "04_shape_feature_distributions.png", dpi=150)
plt.close(fig)

# Regime distribution
fig, ax = plt.subplots(figsize=(8, 4))
regime_ct = (shape.groupby(["MARKET", "REGIME"]).size().unstack(fill_value=0))
regime_ct.plot(kind="bar", stacked=True, ax=ax, color=["#d9534f", "#5cb85c"])
ax.set_title("Contango vs Backwardation count by market")
ax.set_ylabel("# trade dates")
fig.tight_layout()
fig.savefig(OUT / "05_regime_counts.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------
# 6. PCA baseline (sanity reference for AE latent dim choice)
# ----------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# work per market - shapes vary in scale
pca_explained = {}
for mkt in shape["MARKET"].unique():
    X = wide.loc[wide["MARKET"] == mkt, TENORS].values
    # log-return-style normalization to focus on SHAPE not level: divide by level
    Xn = X / X.mean(axis=1, keepdims=True)
    Xn = StandardScaler().fit_transform(Xn)
    p = PCA().fit(Xn)
    pca_explained[mkt] = np.cumsum(p.explained_variance_ratio_)

fig, ax = plt.subplots(figsize=(10, 5))
for mkt, ev in pca_explained.items():
    ax.plot(range(1, len(ev) + 1), ev, marker="o", label=mkt)
ax.axhline(0.95, color="grey", linestyle="--", label="95% var")
ax.axhline(0.99, color="black", linestyle=":", label="99% var")
ax.set_xlim(0, 12)
ax.set_xlabel("# Principal Components")
ax.set_ylabel("Cumulative explained variance")
ax.set_title("PCA on level-normalized curves — informs AE latent dim")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "06_pca_explained_variance.png", dpi=150)
plt.close(fig)

with open(OUT / "pca_summary.txt", "w") as f:
    for mkt, ev in pca_explained.items():
        k95 = int(np.searchsorted(ev, 0.95) + 1)
        k99 = int(np.searchsorted(ev, 0.99) + 1)
        f.write(f"{mkt}: 95% var → {k95} PCs,  99% var → {k99} PCs\n")

# ----------------------------------------------------------------------
# 7. Show a handful of candidate anomalies based on shape features alone
#    (rough baseline — Mahalanobis on shape features per market)
# ----------------------------------------------------------------------
from scipy.spatial.distance import mahalanobis

anomalies = []
for mkt in shape["MARKET"].unique():
    s = shape[shape["MARKET"] == mkt][["LEVEL", "SLOPE_1Y", "SLOPE_3Y", "CURVATURE", "RANGE"]]
    mu = s.mean().values
    cov = np.cov(s.values, rowvar=False)
    inv = np.linalg.pinv(cov)
    d = np.array([mahalanobis(row, mu, inv) for row in s.values])
    sub = shape[shape["MARKET"] == mkt].copy()
    sub["maha"] = d
    top = sub.sort_values("maha", ascending=False).head(8)
    anomalies.append(top)
top_anom = pd.concat(anomalies)
top_anom.to_csv(OUT / "feature_based_top_anomalies.csv", index=False)

# Plot the top 6 anomalies overlaid with the median curve, per market
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, mkt in zip(axes, ["TTF", "THE", "JKM"]):
    median_curve = wide.loc[wide["MARKET"] == mkt, TENORS].median().values
    ax.plot(range(1, 37), median_curve, color="black", linewidth=2, label="median")
    sub_anom = top_anom[top_anom["MARKET"] == mkt].head(6)
    for _, row in sub_anom.iterrows():
        cd = row["TRADE_DATE"]
        cv = wide[(wide["MARKET"] == mkt) & (wide["TRADE_DATE"] == cd)][TENORS].values[0]
        ax.plot(range(1, 37), cv, alpha=0.7, label=str(pd.to_datetime(cd).date()))
    ax.set_title(f"{mkt} — top shape outliers (feature-based)")
    ax.set_xlabel("Tenor"); ax.set_ylabel("Price")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "07_feature_based_anomalies.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------
# 8. Save final EDA summary text
# ----------------------------------------------------------------------
with open(OUT / "eda_summary.txt", "w") as f:
    f.write("Gas Forward Curve — EDA summary\n")
    f.write("=" * 60 + "\n")
    f.write(f"Records (wide): {wide.shape}\n")
    f.write(f"Records (long): {long.shape}\n")
    f.write(f"Date range:     {wide['TRADE_DATE'].min().date()}  →  {wide['TRADE_DATE'].max().date()}\n")
    f.write(f"Trade dates:    {wide['TRADE_DATE'].nunique()}\n")
    f.write(f"Markets:        {sorted(wide['MARKET'].unique().tolist())}\n")
    f.write(f"Tenors:         M1 .. M36 (monthly forward)\n")
    f.write(f"NaNs:           {wide[TENORS].isna().sum().sum()}\n\n")
    f.write("Per-market record count:\n")
    f.write(wide["MARKET"].value_counts().to_string() + "\n\n")
    f.write("Front-month price by market:\n")
    f.write(wide.groupby("MARKET")["M1"].describe().to_string() + "\n\n")
    f.write("PCA — components needed for 95% / 99% variance on shape-normalized curves:\n")
    for mkt, ev in pca_explained.items():
        k95 = int(np.searchsorted(ev, 0.95) + 1)
        k99 = int(np.searchsorted(ev, 0.99) + 1)
        f.write(f"  {mkt}: 95% → {k95} PCs,  99% → {k99} PCs\n")

print("\nEDA done. Outputs in:", OUT)
