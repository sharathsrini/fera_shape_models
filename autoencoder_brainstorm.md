# Autoencoder ideas — TTF / THE / JKM forward curves

## What's in the data
- **Wide:** 4,097 rows × (S_NO, TRADE_DATE, MARKET, M1…M36). Three markets, ~1,360 daily snapshots each, 2021-01-04 → 2026-05-11. No missing tenors.
- **Long:** 147,492 rows, same data unstacked, plus PRICE_EUR / FX_RATE / IS_FX_FILLED (FX is only filled on 36 rows — effectively unused).
- Each row is a smooth 36-tenor curve with visible seasonality (winter hump around M11–M12, summer trough around M5–M6) and a clear level / slope / curvature decomposition.

## Idea 1 — Curve compression AE (baseline)
- **Input/output:** 36-d curve → encoder → 3-d latent → decoder → 36-d.
- **Why it fits:** PCA on Nelson–Siegel-style curves typically explains 95–99% of variance with 3 factors. An AE should match or beat that with a nonlinear bottleneck.
- **Baseline to beat:** PCA(3). Report explained-variance and per-tenor MAE vs PCA.
- **Effort:** half a day. This is your "is the framework working" check.

## Idea 2 — Cross-market joint AE
- **Input:** concatenated [TTF, THE, JKM] = 108-d (only on dates where all three trade). Latent ~6–8 d.
- **Why it fits:** TTF↔THE is a tight European basis; TTF↔JKM is the LNG arbitrage signal. A joint latent captures the spreads as continuous dimensions.
- **Use:** spread anomaly detection, basis forecasting, hedge-ratio learning.
- **Effort:** 1 day after Idea 1.

## Idea 3 — Variational AE for scenario generation
- **Model:** β-VAE, latent 3–5 d, KL-annealed.
- **Why it fits:** Risk teams need plausible alternative curves for VaR / stress tests. Sampling from the prior gives an unlimited supply of *realistic* curves — far better than parametric Nelson–Siegel sims.
- **Validation:** moment-matching on level/slope/curvature distributions; sampled curves must stay arbitrage-free (monotone-ish where expected, no negative prices, etc.).
- **Effort:** 2 days.

## Idea 4 — Conditional VAE
- **Conditions:** one-hot market, trade-month-of-year (seasonality), and a regime flag (e.g. spot level bucket).
- **Why it fits:** TTF in winter '22 vs TTF in summer '24 are different beasts. Conditioning lets one model handle all three markets and all regimes without mode collapse.
- **Use:** "give me 1,000 plausible TTF curves for a cold-winter regime."
- **Effort:** 3 days.

## Idea 5 — Denoising AE for cleaning + missing tenors
- **Setup:** randomly mask 4–8 of the 36 tenors at train time; learn to reconstruct.
- **Why it fits:** illiquid back-end tenors (M30–M36) often have stale or interpolated quotes. A DAE gives you a principled way to re-mark them and to flag suspicious points.
- **Bonus:** also handles the few rows with NaN in long form without ad-hoc filling.
- **Effort:** 1 day.

## Idea 6 — Anomaly detection on reconstruction error
- **Setup:** train AE on 2021–2022 calm regime → score every later day by reconstruction error.
- **Why it fits:** known stress events (Feb '22 invasion, Aug '22 European squeeze) should light up. You get a quantitative regime-shift detector with no labels.
- **Deliverable:** time-series of anomaly score per market, overlaid with known events.
- **Effort:** 1 day on top of Idea 1.

## Idea 7 — Sequence AE (curve dynamics, not just shape)
- **Input:** rolling 20-day window of curves = 20 × 36. Encoder is a 1-D CNN or small Transformer; decoder reconstructs the window.
- **Why it fits:** captures *how* the curve moves, not just what it looks like. Latent feeds naturally into a next-day-curve forecaster.
- **Effort:** 3–4 days; needs a held-out forward window for honest eval.

## Idea 8 — Disentangled β-VAE for risk attribution
- **Goal:** force latent dims to align with level / slope / winter-seasonal / summer-seasonal.
- **Why it fits:** if it works, P&L attribution becomes "Δ price = Δ level × ∂P/∂level + …" with the AE replacing PCA factors. Quant-risk-friendly.
- **Watch out:** disentanglement is fragile; report rotation-invariant metrics, not just visual.
- **Effort:** 2–3 days; mostly hyperparameter sweeps.

---

## Recommended ordering
1. **Idea 1** — proves the pipeline and gives you the PCA-beating baseline number you'll cite everywhere.
2. **Idea 6** — cheapest "wow" demo (anomaly chart with real events on it).
3. **Idea 2 or 5** — pick based on the actual business question (cross-market signal vs. curve cleaning).
4. **Idea 3/4** — once a stakeholder asks for scenarios.

## Architecture defaults to start with
- Encoder/decoder: 36 → 64 → 32 → 3, ReLU/GELU, mirror on the decoder.
- Loss: MSE on standardized prices (per-tenor z-score per market) — equal-weights every tenor.
- Hold out the last 6 months as a strict time-based test set; never random-shuffle.
- Always plot PCA(3) on the same axes — if your AE isn't beating it, the AE isn't earning its keep.

## Things that will bite you
- **Don't shuffle across dates** during training — curves are autocorrelated; random split leaks.
- **Standardize per-market** or the model will spend its capacity learning the JKM-vs-TTF level gap instead of the curve shape.
- **Winter '22 outliers** dominate variance — consider robust loss (Huber) or a regime flag.
- **Arbitrage-free constraints** are not free; if you generate curves for risk, add a calendar-spread monotonicity penalty.
