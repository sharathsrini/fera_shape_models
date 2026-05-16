"""Assemble training_summary.csv from saved checkpoints + history JSONs."""
import json, sys, torch
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
ckpt_dir = ROOT / "results/checkpoints"
rows = []
for pt in sorted(ckpt_dir.glob("*.pt")):
    # `weights_only=False` required because checkpoints carry numpy-backed preproc
    # and python-dict baseline. Trusted local files only.
    state = torch.load(pt, map_location="cpu", weights_only=False)
    hist = ckpt_dir / f"{pt.stem}.history.json"
    h = json.loads(hist.read_text()) if hist.exists() else {}
    rows.append({
        "model": pt.stem,
        "n_params": state.get("n_params", 0),
        "best_epoch": state.get("epoch", 0),
        "best_val_loss": float(state.get("val_loss", float("nan"))),
        "wall_time_s": h.get("wall_time_s"),
        "ckpt": str(pt),
    })
df = pd.DataFrame(rows).sort_values("best_val_loss")
df.to_csv(ROOT / "results/training_summary.csv", index=False)
print(df.to_string(index=False))
