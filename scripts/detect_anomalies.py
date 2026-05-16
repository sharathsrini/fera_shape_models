"""Score every curve with each trained AE and surface shape anomalies.

Key correctness properties (versus the v1 script the critique flagged):

  * Loads CHECKPOINTED preprocessing stats (mean/std) — does NOT recompute
    them from the input CSV, so scores are invariant to changes in the CSV.
  * Scores are CALIBRATED on TRAIN residuals (per market for curve scores,
    per (market, tenor) for kink scores). This gives a meaningful unit
    ("σ above the model's typical train residual for this market/tenor")
    rather than raw MSE.
  * Leaderboard is val-loss-aware and event-aware: we report
        - val recon error (lower = better reconstructor)
        - precision@K against KNOWN_EVENT_WINDOWS
        - calibrated tail (p99 of curve_z, not raw MSE).
  * Anomaly plots show BOTH the raw price curve AND the level-normalized
    shape so reviewers can tell a level shock from a real shape break.

Outputs in results/anomalies/:
    {model}_curve_scores.csv      ranked scores per curve (raw_mse, curve_z)
    {model}_top20.csv             per-market top-K
    {model}_top20.png             dual raw+shape plot, top 6 per market
    {model}_kinks.json            calibrated tenor-level kinks (z > 3)
    leaderboard.csv               cross-model comparison
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, ConcatDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ae_shape.data import (
    build_loaders, loaders_for_inference, tag_known_events,
    TENORS, MARKETS, KNOWN_EVENT_WINDOWS,
)
from ae_shape.models import build_model
from ae_shape.config import MODEL_REGISTRY
from ae_shape.evaluate import (
    collect_reconstructions, per_curve_score,
    calibrated_curve_z, calibrated_tenor_z, detect_kinks_calibrated,
)
from ae_shape.utils import get_device, save_json, seed_everything


# ----------------------------------------------------------------------
# Checkpoint loading (preproc-aware)
# ----------------------------------------------------------------------
def restore(name: str, ckpt_path: Path, device):
    kind, kwargs, _ = MODEL_REGISTRY[name]
    model = build_model(kind, **kwargs)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model = model.to(device).eval()
    preproc = state.get("preproc")
    if preproc is None or preproc.get("mean") is None:
        raise RuntimeError(
            f"Checkpoint {ckpt_path} has no preproc payload. "
            "Retrain with the updated train.py so stats are saved."
        )
    baseline_raw = state.get("baseline")
    if baseline_raw is None:
        raise RuntimeError(
            f"Checkpoint {ckpt_path} has no residual baseline. "
            "Retrain so the train-time baseline is persisted (eliminates "
            "the chance of contaminated calibration at scoring time)."
        )
    # rehydrate numpy arrays
    baseline = {
        "per_market": baseline_raw["per_market"],
        "per_tenor":  {m: {"mu": np.asarray(v["mu"], dtype=np.float32),
                           "sigma": np.asarray(v["sigma"], dtype=np.float32)}
                       for m, v in baseline_raw["per_tenor"].items()},
        "global":     baseline_raw["global"],
    }
    return model, preproc, baseline, state


def market_label_from_oh(oh: np.ndarray) -> list[str]:
    idx = oh.argmax(axis=1)
    return [MARKETS[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "ml_wide.csv"))
    ap.add_argument("--save-dir", default=str(ROOT / "results/anomalies"))
    ap.add_argument("--ckpt-dir", default=str(ROOT / "results/checkpoints"))
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    seed_everything(42)  # makes DataLoader shuffling reproducible across runs

    # Tag event windows (we'll use this for precision@K)
    raw_df = pd.read_csv(args.csv, parse_dates=["TRADE_DATE"])
    raw_df = raw_df[["TRADE_DATE", "MARKET"] + TENORS]
    tagged = tag_known_events(raw_df)

    leaderboard_rows = []
    for name in MODEL_REGISTRY:
        ckpt = Path(args.ckpt_dir) / f"{name}.pt"
        if not ckpt.exists():
            print(f"skip {name}: no checkpoint")
            continue
        print(f"\nScoring {name} ...")
        model, preproc, baseline, ckpt_state = restore(name, ckpt, device)

        # Re-build loaders with the EXACT preproc stats from the checkpoint
        loaders = loaders_for_inference(args.csv, preproc, batch_size=128)
        all_ds = ConcatDataset([loaders["train_ds"], loaders["val_ds"], loaders["test_ds"]])
        all_loader = DataLoader(all_ds, batch_size=128, shuffle=False)
        flat = pd.concat([loaders["train_df"], loaders["val_df"], loaders["test_df"]]).reset_index(drop=True)

        # Baseline now comes from the checkpoint (computed on the exact same
        # training curves the model was trained on, honoring --exclude-known-events).
        # No more risk of contamination from refitting on a different train split.

        # 2. score everything
        R_all = collect_reconstructions(model, all_loader, device=device)
        market_labels = market_label_from_oh(R_all["market_oh"])
        raw_mse = per_curve_score(R_all["orig"], R_all["recon"])
        curve_z = calibrated_curve_z(raw_mse, market_labels, baseline)
        kinks = detect_kinks_calibrated(R_all["orig"], R_all["recon"], market_labels,
                                        baseline, z_thresh=3.0)

        # 3. validation reconstruction error (for leaderboard "quality" column)
        R_val = collect_reconstructions(model, loaders["val_loader"], device=device)
        val_recon = float(np.mean((R_val["orig"] - R_val["recon"]) ** 2))

        # 4. assemble scored frame
        out = flat.copy()
        out["score_raw_mse"] = raw_mse
        out["score_curve_z"] = curve_z
        out["market_label"] = market_labels
        out = tag_known_events(out)
        out_path = save_dir / f"{name}_curve_scores.csv"
        out.to_csv(out_path, index=False)

        # 5. per-market top-K (rank by calibrated curve_z)
        top_rows = []
        for mkt in MARKETS:
            top_rows.append(
                out[out["MARKET"] == mkt].sort_values("score_curve_z", ascending=False).head(args.top_k)
            )
        topk = pd.concat(top_rows)
        topk.to_csv(save_dir / f"{name}_top{args.top_k}.csv", index=False)

        save_json(kinks[:300], save_dir / f"{name}_kinks.json")

        # 6. precision@K against known events (do top-K dates fall in tagged windows?)
        topk_event_hits = int((topk["EVENT_LABEL"] != "").sum())
        precision_at_topK_per_market = topk_event_hits / max(len(topk), 1)

        # 7. plots — dual raw + shape
        fig, axes = plt.subplots(2, 3, figsize=(18, 9))
        for col, mkt in enumerate(MARKETS):
            ax_raw, ax_shape = axes[0, col], axes[1, col]
            raw_med = flat.loc[flat["MARKET"] == mkt, TENORS].median().values
            ax_raw.plot(range(1, 37), raw_med, color="black", linewidth=2, label="median")
            shape_med = (flat.loc[flat["MARKET"] == mkt, TENORS].div(
                            flat.loc[flat["MARKET"] == mkt, TENORS].mean(axis=1), axis=0
                         )).median().values
            ax_shape.plot(range(1, 37), shape_med, color="black", linewidth=2, label="median")
            sub = topk[topk["MARKET"] == mkt].head(6)
            for _, row in sub.iterrows():
                curve = flat.loc[
                    (flat["MARKET"] == mkt) & (flat["TRADE_DATE"] == row["TRADE_DATE"]),
                    TENORS,
                ].values
                if len(curve):
                    shape = curve[0] / np.mean(curve[0])
                    lbl = pd.to_datetime(row["TRADE_DATE"]).date()
                    tag = row.get("EVENT_LABEL", "")
                    full_lbl = f"{lbl}" + (f" [{tag}]" if tag else "")
                    ax_raw.plot(range(1, 37), curve[0], alpha=0.85, label=full_lbl)
                    ax_shape.plot(range(1, 37), shape, alpha=0.85, label=full_lbl)
            ax_raw.set_title(f"{mkt} — RAW price (top 6 by curve_z)")
            ax_raw.set_xlabel("Tenor"); ax_raw.set_ylabel("Price")
            ax_raw.legend(fontsize=7)
            ax_shape.set_title(f"{mkt} — SHAPE = price / mean")
            ax_shape.set_xlabel("Tenor"); ax_shape.set_ylabel("Shape")
            ax_shape.legend(fontsize=7)
        fig.suptitle(f"{name}: top-{args.top_k} shape anomalies (raw vs shape view)", y=1.02)
        fig.tight_layout()
        fig.savefig(save_dir / f"{name}_top{args.top_k}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        leaderboard_rows.append({
            "model": name,
            "val_recon_mse":            round(val_recon, 6),
            "curve_z_p95":              float(np.quantile(curve_z, 0.95)),
            "curve_z_p99":              float(np.quantile(curve_z, 0.99)),
            "curve_z_max":              float(curve_z.max()),
            "n_kink_curves":            len(kinks),
            "precision_at_topK_events": round(precision_at_topK_per_market, 3),
        })

    if not leaderboard_rows:
        print("No checkpoints scored. Train models first.")
        return

    # Final leaderboard. Lower val_recon = better reconstructor.
    # Higher curve_z_p99 + precision_at_topK_events = better anomaly separation.
    lb = pd.DataFrame(leaderboard_rows).sort_values(["val_recon_mse"])
    lb.to_csv(save_dir / "leaderboard.csv", index=False)

    print("\nLeaderboard (sorted by val_recon_mse ↑ = better reconstructor)")
    print("Anomaly quality: higher curve_z_p99 AND higher precision_at_topK_events is better")
    print(lb.to_string(index=False))


if __name__ == "__main__":
    main()
