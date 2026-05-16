"""Train every model in MODEL_REGISTRY and dump a summary table.

Usage:
    python scripts/train_all.py [--epochs 40]
"""
from __future__ import annotations
import argparse, sys, json, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ae_shape.data import build_loaders, KNOWN_EVENT_WINDOWS
from ae_shape.models import build_model
from ae_shape.train import fit, TrainConfig
from ae_shape.config import MODEL_REGISTRY, get_train_config
from ae_shape.utils import seed_everything, get_device, count_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "ml_wide.csv"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-dir", default=str(ROOT / "results/checkpoints"))
    ap.add_argument("--summary", default=str(ROOT / "results/training_summary.csv"))
    ap.add_argument("--only", default=None,
                    help="Comma-separated list of registry keys to train.")
    ap.add_argument("--exclude-known-events", action="store_true",
                    help="Mask KNOWN_EVENT_WINDOWS from the training split so the AE "
                         "learns a cleaner 'normal' distribution. Val/test still see them.")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    print(f"Device: {device}")

    exclude_windows = [(s, e) for (_lbl, s, e) in KNOWN_EVENT_WINDOWS] if args.exclude_known_events else None
    loaders = build_loaders(args.csv, batch_size=args.batch_size, normalize="level_std",
                            exclude_windows=exclude_windows)
    if exclude_windows:
        print(f"Excluded {len(exclude_windows)} known-event windows from TRAIN "
              f"(raw train={len(loaders['train_df_raw'])}, filtered={len(loaders['train_df'])})")
    print(f"Train/val/test sizes: {len(loaders['train_ds'])} / "
          f"{len(loaders['val_ds'])} / {len(loaders['test_ds'])}")

    keys = list(MODEL_REGISTRY.keys()) if not args.only else args.only.split(",")
    rows = []
    for key in keys:
        kind, kwargs, overrides = MODEL_REGISTRY[key]
        cfg = get_train_config({"epochs": args.epochs, **overrides})
        model = build_model(kind, **kwargs)
        model.name = key
        print(f"\n=== Training {key}  (kind={kind}, params={count_params(model):,}) ===")
        t0 = time.time()
        info = fit(model, loaders, cfg=cfg, save_dir=args.save_dir, name=key, device=device)
        rows.append({
            "model": key,
            "kind": kind,
            "n_params": count_params(model),
            "best_epoch": info["best_epoch"],
            "best_val_loss": info["best_val"],
            "wall_time_s": round(time.time() - t0, 1),
            "ckpt": info["ckpt"],
        })

    df = pd.DataFrame(rows).sort_values("best_val_loss")
    df.to_csv(args.summary, index=False)
    print("\nSummary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
