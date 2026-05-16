# Curve Anomaly Labels

This folder contains source-backed labels for benchmarking the curve anomaly models.

## Files

- `event_catalog.csv` - hand-curated anomaly event windows, market scope, severity, confidence, affected tenors, and source URLs.
- `curve_labels.csv` - one row per `(TRADE_DATE, MARKET)` curve with labels only.
- `ml_wide_labelled.csv` - `ml_wide.csv` plus curve labels and derived shape diagnostics.
- `ml_long_labelled.csv` - `ml_long.csv` plus curve labels and `is_event_affected_tenor`.
- `label_summary.csv` - counts by `benchmark_tier` and market.
- `model_benchmark_against_labels.csv` - initial AUROC / average-precision sanity check for existing model score files.

Regenerate everything with:

```bash
python3 scripts/build_labels.py
python3 scripts/benchmark_against_labels.py
```

## Benchmark Tiers

- `gold` - high-confidence, source-backed core market anomaly. Best strict benchmark target.
- `silver` - source-backed shoulder window or plausible/recent event. Good broad benchmark target.
- `review` - useful analyst-review case, but not hard truth. Do not use as strict ground truth.
- `exclude` - data-quality/calendar carry-forward row without a market anomaly. Exclude from training and metrics.
- `*_with_data_quality_flag` - genuine/review market event, but the exact curve is duplicated from the previous available curve. Keep only if your benchmark explicitly allows carried prices.
- `normal` - no curated anomaly or data-quality flag.

Recommended binary targets:

- Strict: `benchmark_is_anomaly_strict == true`.
- Broad: `benchmark_is_anomaly_broad == true`.
- Exclude rows where `benchmark_tier` starts with `exclude` or contains `data_quality_flag` for cleanest metrics.

## Label Philosophy

The labels are event-backed and conservative. They are not exchange-certified ground truth. They are designed to test whether a model can recover known curve-shape stress regimes without treating pure price level alone as the target.

Important nuance:

- The AE inputs are shape-normalized, so level-only events can be under-labeled or less visible.
- `review` labels capture ambiguous cases, especially late Dec 2022, where real market repricing overlaps with holiday/liquidity and duplicate-curve behavior.
- `is_event_affected_tenor` in the long file marks the tenors expected to carry the event signature, such as `M1-M3` for the Jan 2021 JKM prompt spike or `M12-M13`/near-year regions for war/storage shocks.

## Source-Backed Events

- Jan 2021 JKM Asian LNG prompt spike.
- Oct 2021 European gas crunch.
- Dec 2021 winter gas crisis.
- Mar 2022 Russia invasion gas shock.
- Jul/Aug 2022 Nord Stream and storage-driven European gas crisis.
- JKM spillover from 2022 global LNG competition.
- Late Dec 2022 European repricing and price-cap review window.
- Mar 2026 LNG shipping/storage shock, marked silver because it is recent and should be independently settlement-validated.
