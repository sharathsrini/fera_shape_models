"""Benchmark existing anomaly score files against curated labels."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney ROC AUC. Returns nan if one class is absent."""
    y_true = np.asarray(y_true).astype(bool)
    score = np.asarray(score, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # Average ranks for ties.
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for group in np.where(counts > 1)[0]:
            idx = np.where(inv == group)[0]
            ranks[idx] = ranks[idx].mean()
    rank_sum_pos = ranks[y_true].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    """Average precision with higher score meaning more anomalous."""
    y_true = np.asarray(y_true).astype(bool)
    score = np.asarray(score, dtype=float)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(len(y_sorted)) + 1)
    return float((precision * y_sorted).sum() / n_pos)


def main() -> None:
    labels = pd.read_csv(ROOT / "labels/curve_labels.csv", parse_dates=["TRADE_DATE"])
    score_dir = ROOT / "results/anomalies"
    out_rows = []

    for score_path in sorted(score_dir.glob("*_curve_scores.csv")):
        model = score_path.name.replace("_curve_scores.csv", "")
        score_df = pd.read_csv(score_path, parse_dates=["TRADE_DATE"])
        # Prefer the train-calibrated z-score (comparable across markets/models).
        # Fall back to a unified `score` column only if a custom scorer wrote one.
        # `score_raw_mse` is the raw, uncalibrated MSE and is the last resort.
        candidate_order = ["score_curve_z", "score", "score_raw_mse"]
        score_col = next(col for col in candidate_order if col in score_df.columns)
        scores = score_df[["TRADE_DATE", "MARKET", score_col]].rename(
            columns={score_col: "score"}
        )
        data = labels.merge(scores, on=["TRADE_DATE", "MARKET"], how="inner")

        clean = data[
            ~data["benchmark_tier"].str.contains("exclude|review|data_quality", regex=True)
        ].copy()
        strict = clean[clean["benchmark_tier"].isin(["gold", "normal"])].copy()
        broad = clean[clean["benchmark_tier"].isin(["gold", "silver", "normal"])].copy()

        for target_name, target_df, positive_tiers in [
            ("strict_gold_vs_normal", strict, {"gold"}),
            ("broad_gold_silver_vs_normal", broad, {"gold", "silver"}),
        ]:
            for market in ["ALL", "TTF", "THE", "JKM"]:
                sub = target_df if market == "ALL" else target_df[target_df["MARKET"] == market]
                y = sub["benchmark_tier"].isin(positive_tiers).to_numpy()
                s = sub["score"].to_numpy()
                out_rows.append(
                    {
                        "model": model,
                        "score_col": score_col,
                        "target": target_name,
                        "market": market,
                        "n_rows": len(sub),
                        "n_positive": int(y.sum()),
                        "roc_auc": roc_auc(y, s),
                        "average_precision": average_precision(y, s),
                        "positive_rate": float(y.mean()) if len(y) else float("nan"),
                    }
                )

    out = pd.DataFrame(out_rows).sort_values(
        ["target", "market", "roc_auc", "average_precision"],
        ascending=[True, True, False, False],
    )
    out_path = ROOT / "labels/model_benchmark_against_labels.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
