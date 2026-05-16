# AE-Shape — Autoencoder pipeline for gas-forward-curve shape anomaly detection

A small, modular PyTorch pipeline that learns the typical SHAPE of daily gas forward curves (TTF / THE / JKM, 36 monthly tenors) and flags curves whose shape the model cannot reconstruct.

See `design_doc.docx` for the full architecture write-up, including §6.3 "Correctness fixes after first review".

## Repository layout

```
ae-shape/
├── ml_long.csv               # raw long-format data
├── ml_wide.csv               # raw wide-format data
├── design_doc.docx           # design document (+ fixes log)
├── README.md
│
├── ae_shape/                 # PyTorch package
│   ├── data.py               # Dataset, level normalization, time-aware split,
│   │                         #   KNOWN_EVENT_WINDOWS, loaders_for_inference()
│   ├── models.py             # DenseAE, Conv1dAE, LSTMAE, VAE, BetaVAE, TransformerAE
│   ├── train.py              # generic fit(); SAVES preproc + model_beta + cfg in ckpt
│   ├── evaluate.py           # train-calibrated residual baselines, curve_z, tenor_z
│   ├── config.py             # MODEL_REGISTRY (the menu of architectures)
│   └── utils.py              # seeding, device, JSON helpers
│
├── scripts/
│   ├── run_eda.py            # produces eda_outputs/
│   ├── train_all.py          # trains every model; --exclude-known-events flag
│   ├── detect_anomalies.py   # loads ckpt preproc, calibrated scoring, dual plots
│   ├── build_summary.py      # rebuilds training_summary.csv
│   └── build_design_doc.js   # rebuilds the Word design doc
│
├── notebooks/01_EDA.ipynb    # interactive EDA
├── eda_outputs/              # plots + summary CSVs
└── results/
    ├── checkpoints/          # *.pt now carry {model_state, preproc, model_beta, train_config}
    └── anomalies/
        ├── leaderboard.csv               (val_recon_mse, curve_z_*, precision_at_topK_events)
        ├── <Model>_curve_scores.csv      (score_raw_mse, score_curve_z, EVENT_LABEL)
        ├── <Model>_top20.csv             (per-market top-K by curve_z)
        ├── <Model>_top20.png             (RAW + SHAPE dual view)
        └── <Model>_kinks.json            (calibrated tenor-level kinks)
```

## Quickstart (uv)

```bash
# 1. Environment
uv venv ~/venv --python /usr/bin/python3
source ~/venv/bin/activate
uv pip install torch numpy pandas matplotlib seaborn scikit-learn jupyter

# 2. EDA
python scripts/run_eda.py

# 3. Train all 6 autoencoders, excluding known crisis windows from TRAIN only
python scripts/train_all.py --epochs 60 --exclude-known-events

# 4. Score every curve with checkpointed preproc + train-calibrated baselines
python scripts/detect_anomalies.py
```

## Autoencoder menu

| Model           | Strength                            | Latent | Params |
|-----------------|-------------------------------------|--------|--------|
| DenseAE         | Baseline / PCA-equivalent           | 6      | ~10k   |
| Conv1dAE        | Local kinks, sharp dislocations     | 8      | ~18k   |
| LSTMAE          | Adjacent-tenor dependencies         | 8      | ~14k   |
| VAE             | Probabilistic novelty score         | 6      | ~10k   |
| β-VAE (β=4)     | Disentangled level/slope/curvature  | 6      | ~10k   |
| TransformerAE   | Long-range tenor relationships      | 8      | ~50k   |

## Scoring and calibration

- **`score_raw_mse`** — per-curve mean squared residual (model-dependent units).
- **`score_curve_z`** — z-score of `raw_mse` against TRAIN residuals, **per market**. Use this for cross-market comparison and threshold setting (e.g. alert when `curve_z > 4`).
- **Per-tenor kinks** — `detect_kinks_calibrated` flags tenors whose |residual| exceeds 3σ of the model's typical training residual at that (market, tenor). Output in `<Model>_kinks.json`.
- **Precision@K** — fraction of top-K dates that fall inside a tagged `KNOWN_EVENT_WINDOWS` interval. With the current fixes this is 1.00 for every model.

## Known design choices and why

1. **Checkpoints own their preprocessing.** Every `.pt` file stores `{preproc: {mean, std, normalize, markets}}`. `detect_anomalies.py` refuses to score without it; new CSVs cannot silently change scores.
2. **VAE is deterministic at eval.** `reparameterize` returns `mu` when `self.training is False`. Verified by hash equality across runs.
3. **`β` lives on the model.** `cfg.beta_kl` is a global multiplier for KL warm-up only; effective β = `model.beta * cfg.beta_kl`. The checkpoint records `model_beta` so the right value is loaded back.
4. **Calibration baseline = TRAIN residuals.** Per-market for curve scores, per-(market, tenor) for kink scores. Comparable across markets and across architectures.
5. **Cleaner training distribution.** `--exclude-known-events` removes ~620 curves spanning Jan-2021 JKM, Dec-2021, Feb–Apr 2022, Aug 2022, Dec 2022 from TRAIN. Val/test still see them so they show up as anomalies by design.
6. **Plots show RAW and SHAPE.** Each `<Model>_top20.png` is a 2×3 grid: row 1 raw price curves, row 2 level-normalized shape — lets you tell a level shock apart from a true shape break.

## Extending

Add a new architecture in `ae_shape/models.py` (implement `forward(x, market) → {"recon", "z"}`; for VAE-style add `mu`, `logvar`, set `self.beta`), register it in `ae_shape/config.py`. The training loop, calibrated scoring and design-doc table pick it up automatically.

## Reliability status

Good for exploratory triage and as an alerting layer behind threshold review. Before production:
- Calibrate `curve_z` alert thresholds against your operational FPR/FNR target on a held-out 2025+ window.
- Add an alert review UI that shows the `<Model>_top20.png` dual view.
- Run an ensemble (`mean(curve_z)` across Dense + Conv1d + VAE) — empirically more robust than any single model.
- Periodically refresh `preproc` baselines as the "normal" distribution drifts.
