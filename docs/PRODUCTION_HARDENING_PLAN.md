# Production Hardening Plan — Gas Curve Shape Anomaly Detection

**Status:** Draft 1 — to be executed in phases.
**Audience:** Future Claude Code sessions and human reviewers.
**Source review date:** 2026-05-16 (branch `claude/review-pipeline-validation-BxV6y`).

---

## 0. How to read this document

This is a precise, ordered, executable plan. It is **not** a research summary;
the research is already done and condensed into Appendix A. Each phase is
sized to one or more pull requests. Each step within a phase has the same
seven-field structure so it can be picked up cold by a different operator.

### Per-step structure

1. **Goal** — one sentence, what done looks like.
2. **Why** — the literature or codebase fact that justifies the step.
3. **Pre-conditions** — what must already be true. If any is false, stop and
   complete it first.
4. **Files to touch** — exact paths (and line numbers where possible).
5. **Implementation outline** — concrete instructions, pseudocode, and bash
   commands. Not full code; enough that a competent operator can finish it.
6. **Validation** — numerical or behavioural criteria that prove it worked.
   If validation fails, do **not** merge.
7. **Failure modes** — specific things that have plausibly bitten this kind
   of pipeline before; check each one before claiming the step is done.
8. **Rollback** — how to revert without losing data or breaking downstream.

### Global rules

- **One change per PR.** Do not bundle steps; you will lose the ability to
  attribute regressions.
- **Branch per step.** Cut from `main` (or the latest merged step's branch).
  Name pattern: `claude/<phase>-<step>-<slug>`, e.g.
  `claude/phase2-step1-seasonal-estimate`.
- **Feature-flag every behavioural change.** Default to the old path for one
  week of overlap, then flip the default in a follow-up PR.
- **Never delete the current code path** until its replacement has shipped
  one full week without regression on the locked evaluation (Phase 0).
- **Numbers are evidence.** Every PR description must contain the
  walk-forward AUC / AP / range-F1 table from `scripts/eval_walk_forward.py`,
  compared against the locked baseline.
- **No silent retraining.** If a model has to be retrained as part of a step,
  say so explicitly and persist the new checkpoint under a new name.

### Phase ordering rationale

Phase 0 must come first because no later phase can be evaluated without it.
Phase 1 must come before Phase 2 because if shallow baselines already match
the AEs (ADBench, NeurIPS 2022), several later phases reduce to ensembling
those baselines instead of upgrading the AEs.
Phases 2 and 3 are independent and can run in parallel.
Phase 4 must complete before Phases 5–7 because drift monitoring and analyst
logs need a model registry to point at.

---

## 1. Repository snapshot at plan inception

### 1.1 Layout

```
ae_shape/
  __init__.py
  config.py        MODEL_REGISTRY — 6 architectures
  data.py          GasCurveDataset, time_aware_split, KNOWN_EVENT_WINDOWS (line 132)
  models.py        DenseAE, Conv1dAE, LSTMAE, VAE, BetaVAE, TransformerAE
  train.py         fit(); baseline computed at lines 166-188; torch.load at line 170
  evaluate.py      calibrated_curve_z at line 99; build_residual_baseline at line 63
  utils.py         seed, device, JSON helpers
scripts/
  run_eda.py
  train_all.py     CLI; --exclude-known-events flag (line 31)
  detect_anomalies.py
                   restore() at line 53; torch.load weights_only=False at line 56
  build_labels.py  EVENTS catalog with gold/silver/review/exclude tiers
  benchmark_against_labels.py
                   roc_auc + average_precision on labels
  build_summary.py torch.load weights_only=False at line 11
labels/
  curve_labels.csv, event_catalog.csv, ml_wide_labelled.csv, ml_long_labelled.csv
  model_benchmark_against_labels.csv
results/
  checkpoints/     6 .pt files, each with n_train_curves=2657 (= trained with --exclude-known-events)
  anomalies/       leaderboard.csv + per-model scores/top20/kinks/plots
```

### 1.2 Known facts that future steps depend on

- `n_train_curves=2657` in every saved baseline → the current checkpoints
  were trained with `--exclude-known-events` (raw train would be 3281).
- Six event windows in `KNOWN_EVENT_WINDOWS` exclude **624** curves: JKM
  cold-snap Q1-2021 (81), EU storage scare Q4-2021 (69), EU pre-invasion
  (183), Russian invasion (135), Nord Stream halt (104), Dec 2022 (52).
- The benchmark labels in `labels/EVENTS` are richer than
  `KNOWN_EVENT_WINDOWS` (different windows, plus events not in
  KNOWN_EVENT_WINDOWS, e.g. July-2022 Nord Stream 20% cut, March-2026 LNG
  shock). These two lists must be reconciled before Phase 6.
- `level_std` normalisation in `data.py:95-104` pools across markets; this
  contradicts the per-market spirit of the rest of the pipeline. Address in
  Phase 2.
- VAE determinism at eval is implemented (`models.py:165-171`) and verified
  via the baseline reproducibility.

### 1.3 Known correctness issues to fix opportunistically

| ID | File:line | Severity | Description |
| -- | --------- | -------- | ----------- |
| A  | `detect_anomalies.py:103` | low | `tagged` computed but never used — dead code |
| B  | `data.py:30, 41` | low | `level_normalize` and `log_transform` are dead helpers |
| C  | `detect_anomalies.py:158` | low | `precision_at_topK_per_market` is global, not per-market |
| D  | `data.py:95-104` | medium | Pooled mean/std vs per-market — fix in Phase 2 |
| E  | `data.py:132` vs `build_labels.py:29` | medium | Two event lists; reconcile in Phase 6 |
| F  | `train.py:170`, `detect_anomalies.py:56`, `build_summary.py:11` | high | `weights_only=False` on `torch.load` — CVE-2025-32434 attack surface; fix in Phase 4 |

---

## Phase 0: Lock the evaluation harness

You cannot improve what you cannot measure. The existing
`benchmark_against_labels.py` is a fine starting point but uses one fixed
split, point-wise F1, and reports a single number per model. We need a
walk-forward harness with bootstrap confidence intervals and a range-based
F1 to match the way the labels are actually defined (week-long events).

### Step 0.1 — Freeze the current numbers as the immovable baseline

- **Goal.** Capture today's per-market AUC / AP / precision@20 / range-F1 as
  the floor that no later change is allowed to drop below by more than
  1 point.
- **Why.** Without a captured baseline, "regression" looks like noise and
  ships silently.
- **Pre-conditions.**
  - Working tree clean on `main`.
  - `python scripts/benchmark_against_labels.py` runs end-to-end.
- **Files to touch.** New: `results/_baseline_2026_05_16/BASELINE.md` and a
  copy of the two leaderboard CSVs into the same directory.
- **Implementation outline.**
  ```bash
  mkdir -p results/_baseline_2026_05_16
  cp labels/model_benchmark_against_labels.csv \
     results/_baseline_2026_05_16/
  cp results/anomalies/leaderboard.csv \
     results/_baseline_2026_05_16/
  sha256sum ml_wide.csv ml_long.csv > results/_baseline_2026_05_16/data_hashes.txt
  git rev-parse HEAD > results/_baseline_2026_05_16/git_sha.txt
  ```
  Then write a `BASELINE.md` that lists, in a table: per-model, per-market,
  ROC-AUC, AP, precision@20. Mark this file `DO NOT EDIT`.
- **Validation.** A reviewer can `diff` future runs against this directory
  with a one-line command.
- **Failure modes.**
  - **Stale checkpoints.** If `results/checkpoints/` is out of date relative
    to the CSV files, the captured numbers are meaningless. Run a fresh
    `detect_anomalies.py` before the snapshot.
  - **Uncommitted changes in the tree.** The `git_sha.txt` will be wrong.
    `git status` must be clean.
- **Rollback.** `git rm -r results/_baseline_2026_05_16/`. Harmless.
- **Commit message.** `chore(eval): freeze 2026-05-16 baseline metrics`.

### Step 0.2 — Walk-forward evaluation script

- **Goal.** Replace the single time-aware split with an expanding-window
  backtest with 8 folds and bootstrap CIs.
- **Why.** Schmidl, Wenig & Papenbrock (PVLDB 2022, DOI
  10.14778/3538598.3538602): scoring protocol affects rankings more than
  algorithm choice. A single split makes one configuration look luckier than
  it is.
- **Pre-conditions.** Step 0.1 complete.
- **Files to touch.**
  - New: `scripts/eval_walk_forward.py`.
  - New: `ae_shape/evaluation_protocol.py` (reusable utilities).
- **Implementation outline.**
  ```python
  # ae_shape/evaluation_protocol.py
  from sklearn.model_selection import TimeSeriesSplit

  def walk_forward_indices(dates, n_splits=8, min_train_days=365, test_days=125):
      """Return list of (train_idx, calib_idx, test_idx).
      train block grows; calib is last 10% of train block; test is fixed-size.
      """
      ...
  ```
  ```python
  # scripts/eval_walk_forward.py
  # For each fold:
  #   1. Load wide CSV, restrict to dates ≤ fold_end.
  #   2. Build loaders with the fold's train/calib indices.
  #   3. Re-train each model in MODEL_REGISTRY (use --epochs from config).
  #   4. Score test block; compute ROC-AUC, AP, range-F1, precision@20.
  #   5. Append a row {fold, model, market, metric, value}.
  # Bootstrap 1000 resamples per (fold, model, market) for 95% CI.
  # Output: results/walk_forward/<model>.csv and aggregated _summary.csv.
  ```
- **Validation.**
  - Mean walk-forward AUC across folds is within ~0.05 of the single-split
    AUC for every model. If not, the original AUC is overstated; note this
    in the README before continuing to Phase 1.
  - Bootstrap CI width per fold should narrow as the test block grows.
- **Failure modes.**
  - **Label leakage by `KNOWN_EVENT_WINDOWS`.** Currently exclusion is
    applied globally. In walk-forward, fold 5's training block may contain
    fold 4's test block. Apply exclusion *per fold*, not globally.
  - **Calibration baseline contamination.** The residual baseline
    (`train.py:166-188`) is currently fit on the full filtered train. In
    walk-forward, fit per fold. If you forget, you are calibrating on the
    future.
  - **Random seed coupling.** Each fold should use the same seed; otherwise
    fold-to-fold variance includes initialisation variance and the CIs are
    too wide.
  - **GPU time.** 8 folds × 6 models × 60 epochs ≈ 2-3 hours on a CPU. Budget
    accordingly; this is the slowest single step in the plan.
- **Rollback.** Remove the new script. Nothing downstream depends on it
  until Phase 0.3.
- **Commit message.** `feat(eval): walk-forward harness with bootstrap CIs`.

### Step 0.3 — Range-based F1 alongside point-wise

- **Goal.** Adopt Tatbul et al. (NeurIPS 2018, arXiv:1803.03639) range-based
  precision/recall with existence-only matching.
- **Why.** Ground truth in `labels/curve_labels.csv` is week-long events,
  not single days. Point-wise F1 penalises "we flagged Wednesday, the storm
  peaked Thursday" the same as "we missed it entirely."
- **Pre-conditions.** Step 0.2 complete.
- **Files to touch.**
  - New: `ae_shape/metrics.py`.
  - Modified: `scripts/benchmark_against_labels.py` (add a column, do not
    replace existing numbers).
  - Modified: `scripts/eval_walk_forward.py` (add range-F1 to its outputs).
- **Implementation outline.**
  ```python
  # ae_shape/metrics.py
  def range_based_f1(pred_intervals, true_intervals,
                     alpha_recall=0.0,         # existence-only
                     positional_bias="flat",   # uniform weighting
                     gamma_cardinality=1.0):
      """Tatbul et al. NeurIPS 2018. Returns (precision, recall, f1).
      Each interval is a half-open (start_inclusive, end_exclusive) day pair.
      """
      ...
  ```
  Convert per-day predictions to intervals by merging consecutive flagged
  days within a market. Convert labels to intervals by collapsing
  `event_id` runs.
- **Validation.**
  - Range-F1 ≥ point-wise F1 for every model on every market. If any is
    lower, your interval merging has a bug.
  - On a fabricated 5-day event with single-day predictions at each end
    (existence-only), recall should be exactly 1.0.
- **Failure modes.**
  - **Off-by-one on inclusivity.** Half-open vs closed intervals silently
    inflate recall. Write a unit test on a single-day event before trusting.
  - **Cross-market interval merging.** A TTF event interval and a JKM event
    interval on the same day are different intervals. Group by market.
- **Rollback.** Drop the new column; previous CSVs unchanged.
- **Commit message.** `feat(metrics): range-based F1 per Tatbul NeurIPS 2018`.

---

## Phase 1: Shallow baselines (ADBench sanity check)

### Step 1.1 — Add PCA, IsolationForest, ECOD, Mahalanobis baselines

- **Goal.** Find out whether the autoencoder zoo actually beats one-line
  sklearn methods. If it does not, Phases 2+ change shape (we drop the AE
  upgrade work and focus on baselines + thresholds).
- **Why.** Han et al., NeurIPS 2022 (ADBench, arXiv:2206.09426): on small
  tabular problems, deep AEs are not statistically better than
  IsolationForest, ECOD, COPOD, or kNN-distance; several deep methods are
  worse. Your input is 36-dimensional. The result almost certainly
  transfers.
- **Pre-conditions.** Phase 0 complete; walk-forward harness operational.
- **Files to touch.**
  - New: `ae_shape/baselines.py`.
  - Modified: `scripts/eval_walk_forward.py` (register new scorers).
- **Implementation outline.**
  ```python
  # ae_shape/baselines.py
  from sklearn.ensemble import IsolationForest
  from sklearn.covariance import LedoitWolf
  from sklearn.decomposition import PCA
  # pyod for ECOD: pip install pyod
  from pyod.models.ecod import ECOD

  class PerMarketScorer:
      """Holds one scorer per market. Fit/score interface identical to AE."""
      def __init__(self, factory):
          self.factory = factory
          self.per_market = {}

      def fit(self, shape_curves_by_market: dict[str, np.ndarray]):
          for mkt, X in shape_curves_by_market.items():
              s = self.factory(); s.fit(X); self.per_market[mkt] = s
          return self

      def score(self, shape_curve, market):
          # higher = more anomalous
          ...

  # Factories:
  def make_iforest():
      return IsolationForest(n_estimators=200, contamination='auto',
                             random_state=42)
  def make_ecod():
      return ECOD()
  def make_mahalanobis():
      # Wrap LedoitWolf into an .decision_function returning Mahalanobis dist
      ...
  def make_pca_recon(var_kept=0.95):
      # Returns reconstruction MSE on the held-out residual subspace
      ...
  ```
  Score on the level-normalised, seasonal-stripped curve once Phase 2 lands;
  until then, score on the same `level_std` input the AEs see.
- **Validation.**
  - Run `scripts/eval_walk_forward.py` with the four new scorers added to
    the model registry. Output: a leaderboard with rows for each model and
    market.
  - **Decision rule:**
    - If PCA-recon ≥ best AE on every market within 0.01 AUC: declare a
      truth ("AEs do not add value here") in `BASELINE.md`; the rest of the
      plan becomes "make the shallow baseline production-grade."
    - If best AE beats all four baselines by ≥0.02 AUC on at least two
      markets: AEs earn their keep; proceed with Phase 2.
    - Anything in between: keep AEs as part of the ensemble in Phase 6 but
      do not assume superiority.
- **Failure modes.**
  - **Pooled scaling.** Each market has a different residual scale. The four
    scorers must be fit per-market (the `PerMarketScorer` wrapper above).
  - **Crisis-curve contamination.** Use the same `--exclude-known-events`
    policy. If you forget, the shallow models look artificially worse.
  - **IsolationForest seed instability.** Fix `random_state=42`. Without
    this, fold-to-fold variance is dominated by seed noise.
  - **ECOD ties on identical curves.** The duplicate-curve flag from
    `build_labels.py:add_duplicate_flags` already marks these. Drop
    duplicates before fitting, otherwise ECOD's empirical CDFs are
    degenerate.
- **Rollback.** Delete `ae_shape/baselines.py`; remove registry entries.
- **Commit message.** `feat(baselines): IsolationForest, ECOD, PCA, Mahalanobis per market`.

### Step 1.2 — Decision point: do we still need deep AEs?

- **Goal.** Document the answer to the decision rule in Step 1.1 and amend
  the rest of the plan if needed.
- **Files to touch.** `docs/PRODUCTION_HARDENING_PLAN.md` (this file) —
  insert a "Phase 1.2 outcome" section noting which branch was taken.
- **Validation.** A reader can tell, from this document alone, why later
  phases either include or skip the "retrain the AE zoo" step.
- **Commit message.** `docs: record Phase 1 outcome`.

---

## Phase 2: Strip seasonality

The literature is unanimous: gas curves have a strong annual cycle (winter
premium in Europe; summer cooling demand in Asia). The AE is currently
spending latent budget memorising this. Subtract it first.

### Step 2.1 — Estimate a deterministic seasonal premium

- **Goal.** Compute a per-market `seasonal[12]` vector that captures the
  level-normalised price typical of each delivery-month bucket.
- **Why.** Borovkova & Geman (2006, DOI 10.1007/s11147-007-9008-4) and
  Mirantes et al. (2012, *European Financial Management*) — the standard
  decomposition for gas is `F(t, M_k) = level(t) + seasonal(month_k) +
  residual(t,k)`.
- **Pre-conditions.** Phase 0 complete; clean (event-excluded) training
  data identifiable.
- **Files to touch.**
  - New: `ae_shape/seasonality.py`.
  - Modified: `ae_shape/data.py` — read seasonal vectors during dataset
    construction.
- **Implementation outline.**
  ```python
  # ae_shape/seasonality.py
  def fit_seasonal(wide_df: pd.DataFrame,
                   exclude_windows=None) -> dict[str, np.ndarray]:
      """For each market, return a length-12 array of average level-
      normalised price by delivery-month-of-year.
      """
      df = _filter_excluded(wide_df, exclude_windows)
      out = {}
      for mkt, g in df.groupby("MARKET"):
          raw = g[TENORS].values
          level = raw.mean(axis=1, keepdims=True)
          shape = raw / level                            # (N, 36)
          # delivery month for tenor M_k depends on TRADE_DATE
          trade_months = pd.to_datetime(g["TRADE_DATE"]).dt.month.values
          delivery_months = ((trade_months[:, None] + np.arange(36)) - 1) % 12 + 1
          seasonal = np.zeros(12)
          counts = np.zeros(12)
          for m in range(1, 13):
              mask = delivery_months == m
              if mask.any():
                  seasonal[m-1] = shape[mask].mean()
                  counts[m-1] = mask.sum()
          out[mkt] = seasonal
      return out
  ```
- **Validation.**
  - TTF/THE: clear winter peak (months 12, 1, 2 are highest).
  - JKM: shape is similar to TTF/THE but typically less pronounced.
  - All vectors sum to ≈ 12.0 (sanity, since the underlying is mean-1).
- **Failure modes.**
  - **Crisis contamination of the seasonal estimate.** Fit on
    event-excluded training data only. If 2021-12 is in the input, December
    seasonal is inflated and every future December looks normal — the
    *opposite* of what you want.
  - **Wrong delivery-month convention.** JKM contracts roll on a different
    calendar than TTF. Confirm against `ml_long.csv`'s `DELIVERY_MONTH`
    column for at least one date per market.
  - **Insufficient sample.** If any month has <30 observations after
    filtering, fall back to a smoothed interpolation across adjacent
    months rather than a noisy mean.
- **Rollback.** Delete `seasonality.py`; no downstream dependency yet.
- **Commit message.** `feat(seasonal): per-market delivery-month premium estimator`.

### Step 2.2 — Apply the residual transform in the dataset

- **Goal.** Add a new normalisation mode `"seasonal_level_std"` to
  `GasCurveDataset`, and persist the seasonal vector in the checkpoint.
- **Why.** Same as Step 2.1.
- **Pre-conditions.** Step 2.1 produces sensible seasonal vectors.
- **Files to touch.**
  - `ae_shape/data.py:74-104` — add the new mode.
  - `ae_shape/train.py:122-131` — extend `preproc` payload with `seasonal`.
  - `scripts/detect_anomalies.py:53-80` — extend `restore()` to require
    `preproc["seasonal"]` for the new mode.
- **Implementation outline.**
  - In `GasCurveDataset.__init__`, after computing `shape = raw / levels`,
    if `normalize == "seasonal_level_std"`: subtract the per-(market, tenor)
    seasonal contribution, then apply per-tenor z-scoring as today.
  - The seasonal subtraction maps each row's 36 tenors to delivery months
    using the same `(trade_month + k - 1) % 12 + 1` rule as Step 2.1.
  - In `train.fit`, when this mode is active, include
    `preproc["seasonal"] = {market: list(seasonal_vec)}` in the saved
    payload.
  - In `loaders_for_inference`, pass `seasonal` through to the new mode.
  - In `restore()`, if the active mode requires `seasonal` but the
    checkpoint lacks it, raise a clear error (mirror the existing
    `mean`/`std` guard).
- **Validation.**
  - Train one cheap model (`Dense_lat6`) with the new mode for 30 epochs.
  - Walk-forward AUC for that model should improve by >0.02 on at least
    two markets. If not, the seasonal vector is wrong — go back to 2.1.
  - Recovered residual mean per tenor on the training set should be ≈ 0;
    residual std per tenor should be of similar magnitude across tenors.
- **Failure modes.**
  - **Forgotten persistence.** If `seasonal` is not in `preproc`, inference
    silently zeroes it and scores drift. The guard in `restore()` must
    raise.
  - **Mismatched market keys.** `preproc["seasonal"]` keys must include all
    three markets; missing one silently breaks one market.
  - **Per-market mean/std interaction.** Once seasonality is out, the
    pooled mean/std (data.py:95-104) is even more wrong. Switch to
    per-market mean/std in the same PR. This fixes correctness issue D
    from §1.3.
- **Rollback.** Leave the new mode in but flip the default in `train_all.py`
  back to `level_std`. The new mode is dormant.
- **Commit message.** `feat(data): seasonal_level_std mode + per-market stats`.

### Step 2.3 — Retrain the AE zoo on the residuals

- **Goal.** Produce a parallel set of checkpoints
  `results/checkpoints_seasonal/<model>.pt`.
- **Pre-conditions.** Steps 2.1 and 2.2 merged; spot-check passed.
- **Files to touch.**
  - `scripts/train_all.py` — accept `--normalize {level_std,seasonal_level_std}`
    and `--save-dir` (already supports the latter).
- **Implementation outline.**
  ```bash
  python scripts/train_all.py \
      --epochs 60 \
      --exclude-known-events \
      --normalize seasonal_level_std \
      --save-dir results/checkpoints_seasonal
  python scripts/detect_anomalies.py \
      --ckpt-dir results/checkpoints_seasonal \
      --save-dir results/anomalies_seasonal
  python scripts/benchmark_against_labels.py
  ```
- **Validation.**
  - Walk-forward AUC for the seasonal zoo beats the level-only zoo on at
    least two markets by ≥0.01.
  - On gold-tier labels (`build_labels.py` tier == "gold"), the new
    leaderboard's `precision@20` is ≥ the baseline's.
  - If gains are negligible (<0.005 AUC), file an issue and revisit the
    seasonal estimator (Step 2.1) before flipping defaults.
- **Failure modes.**
  - **Retraining without rerunning detect_anomalies.** Old anomaly CSVs
    will reference stale scores. Always run both.
  - **Saving over the level_std checkpoints.** Use a separate
    `--save-dir`. Do not overwrite the baseline.
- **Rollback.** Delete `results/checkpoints_seasonal/`; flip default back.
- **Commit message.** `feat(train): retrain zoo on seasonal residuals`.

---

## Phase 3: Conformal thresholds

The current threshold (`evaluate.calibrated_curve_z` at `evaluate.py:99`)
z-scores raw MSE against per-market training residuals and treats "z > 4" as
a 1-in-tens-of-thousands event. This assumes Gaussian tails. Gas residuals
are heavy-tailed. The empirical FPR is much higher than the nominal one.

Conformal prediction (Vovk; Laxhammar & Falkman; Bates et al. 2023) replaces
the Gaussian assumption with an exchangeability assumption and gives a
distribution-free finite-sample FPR.

### Step 3.1 — Carve a calibration set out of the training period

- **Goal.** Reserve ~10% of clean training-period dates per market as a
  calibration set whose only purpose is to define the conformal threshold.
- **Why.** Inductive conformal (Laxhammar & Falkman, *Ann. Math. AI* 2015,
  DOI 10.1007/s10472-013-9381-7) requires a calibration set disjoint from
  training. For FPR = 1%, you need ≥99 calibration scores per market;
  ~1000 is the comfort zone.
- **Pre-conditions.** Phase 2 complete (or skipped per Phase 1.2 outcome).
- **Files to touch.**
  - `ae_shape/data.py:164-181` — extend `time_aware_split` to return four
    blocks: train / calib / val / test.
  - `ae_shape/train.py:166-188` — after best-epoch reload, additionally
    score the calibration block and persist `calib_scores[market]`.
  - `scripts/detect_anomalies.py:53-80` — `restore()` reads `calib_scores`
    from the checkpoint.
- **Implementation outline.**
  ```python
  # ae_shape/data.py
  def time_aware_split(df, calib_frac=0.10, val_frac=0.10, test_frac=0.10):
      parts = {"train": [], "calib": [], "val": [], "test": []}
      for mkt, g in df.groupby("MARKET"):
          g = g.sort_values("TRADE_DATE").reset_index(drop=True)
          n = len(g)
          n_test  = int(n * test_frac)
          n_val   = int(n * val_frac)
          n_calib = int(n * calib_frac)
          n_train = n - n_calib - n_val - n_test
          parts["train"].append(g.iloc[:n_train])
          parts["calib"].append(g.iloc[n_train:n_train+n_calib])
          parts["val"].append(g.iloc[n_train+n_calib:n_train+n_calib+n_val])
          parts["test"].append(g.iloc[n_train+n_calib+n_val:])
      ...
  ```
  In `train.fit`, after `model.load_state_dict(best["model_state"])`:
  ```python
  Rc = collect_reconstructions(model, loaders["calib_loader"], device)
  mkt_lab = [MARKETS[i] for i in Rc["market_oh"].argmax(axis=1)]
  raw_mse = per_curve_score(Rc["orig"], Rc["recon"])
  calib_scores = {m: raw_mse[np.array(mkt_lab) == m].tolist() for m in MARKETS}
  best["calib_scores"] = calib_scores
  ```
- **Validation.**
  - `len(state["calib_scores"][m]) ≥ 99` for every market `m`.
  - Calibration dates do not overlap training dates (assert in
    `time_aware_split` and in a unit test).
  - The fraction of training-period curves consumed by the calibration set
    is ≈ 10%.
- **Failure modes.**
  - **Calibration leak into training.** Assert disjointness explicitly with
    a unit test on `time_aware_split`.
  - **Calibration straddles a regime break.** The Feb-2022 invasion is a
    regime shift; a calibration window half pre-invasion / half post-
    invasion gives a wrong threshold for both regimes. Mitigation: tie
    calibration refresh to the drift monitor (Phase 5.3 ADWIN).
  - **Calibration set too small on JKM.** JKM is less liquid; check the
    minimum and bail out with a clear error if <99.
- **Rollback.** Drop the `calib` block (re-merge into train). Threshold
  reverts to z-score.
- **Commit message.** `feat(data): inductive-conformal calibration block`.

### Step 3.2 — Replace `calibrated_curve_z` with split-conformal p-values

- **Goal.** Add `conformal_p_value(score, calib_scores) → p` and write a new
  column `score_p_value` to the per-curve CSVs.
- **Why.** Bates et al., *Ann. Stat.* 51(1) 2023, arXiv:2104.08279.
- **Pre-conditions.** Step 3.1 merged; checkpoints contain `calib_scores`.
- **Files to touch.**
  - `ae_shape/evaluate.py` — add `conformal_p_value`.
  - `scripts/detect_anomalies.py:127-143` — compute and write the new
    column. Keep `score_curve_z` for backwards compatibility.
- **Implementation outline.**
  ```python
  # ae_shape/evaluate.py
  def conformal_p_value(test_scores: np.ndarray,
                        calib_scores: np.ndarray) -> np.ndarray:
      """Marginal conformal p-value per Vovk / Bates et al.
      p = (1 + #{s in calib : s ≥ s_test}) / (n + 1)
      """
      calib_sorted = np.sort(calib_scores)
      n = len(calib_sorted)
      # number of calib scores ≥ each test score
      ranks = n - np.searchsorted(calib_sorted, test_scores, side="left")
      return (1 + ranks) / (n + 1)
  ```
  Apply per market in `detect_anomalies.py`:
  ```python
  p_values = np.empty_like(raw_mse, dtype=np.float64)
  for m in MARKETS:
      sel = np.array(market_labels) == m
      p_values[sel] = conformal_p_value(raw_mse[sel],
                                        np.asarray(baseline_calib[m]))
  out["score_p_value"] = p_values
  ```
- **Validation.**
  - On a held-out *clean* block (Step 0.2's test fold under no crisis),
    empirical FPR at α = 0.01 must be within ±0.005 of 0.01. If you get
    0.04, calibration leaked.
  - Empirical FPR at α = 0.05 within ±0.01 of 0.05.
- **Failure modes.**
  - **Ties.** If many calibration scores equal a test score (very low
    precision data), p-values are over-confident on the upper tail. Use the
    smoothed conformal variant (Vovk 2005): add U(0,1) jitter to ranks,
    then take expectation.
  - **Cross-market pooling.** The current `calibrated_curve_z` is already
    per-market; preserve this. Pooling makes the test invalid.
  - **Empty calibration set for a rare market.** Hard-fail with a clear
    error, not a silent fallback to z-score.
- **Rollback.** Stop writing `score_p_value`; consumers fall back to
  `score_curve_z`.
- **Commit message.** `feat(eval): split-conformal p-values per Bates 2023`.

### Step 3.3 — Benjamini–Hochberg FDR on the daily batch

- **Goal.** Convert "alert if `p < α`" to "alert on the BH-selected set at
  FDR `q`."
- **Why.** Bates et al. 2023 prove conformal p-values are PRDS under
  exchangeability; BH gives exact finite-sample FDR control on PRDS
  p-values. Analyst-facing contract is "of today's K alerts, ≤ qK are
  false," which is what they actually want.
- **Pre-conditions.** Step 3.2 merged; `score_p_value` reliable.
- **Files to touch.**
  - `ae_shape/evaluate.py` — add `bh_select`.
  - `scripts/detect_anomalies.py:145-152` — replace per-market top-K
    selection with `bh_select(..., q=0.10)`, but keep the top-K CSV for
    review continuity.
- **Implementation outline.**
  ```python
  def bh_select(p_values: np.ndarray, q: float = 0.10) -> np.ndarray:
      m = len(p_values)
      order = np.argsort(p_values)
      thresh = q * (np.arange(1, m + 1) / m)
      passing = p_values[order] <= thresh
      if not passing.any():
          return np.zeros(m, dtype=bool)
      k = np.where(passing)[0].max()
      out = np.zeros(m, dtype=bool)
      out[order[:k + 1]] = True
      return out
  ```
- **Validation.**
  - Expected daily alert count drops by ~3–5× compared to raw `p < 0.05`.
  - On the known-event windows, BH still selects them at q=0.10.
  - On clean periods, BH selects no rows on most days.
- **Failure modes.**
  - **Pre-filtering that breaks PRDS.** Do not filter "low-volatility days"
    before BH; the dependence structure changes and the FDR guarantee
    weakens. If filtering is required, use BH–Yekutieli (harmonic
    correction).
  - **Per-day vs per-market batches.** Apply BH per (market, day) batch.
    Across-day pooling double-counts and inflates power claims.
- **Rollback.** Keep the top-K legacy output as primary; remove BH column.
- **Commit message.** `feat(eval): Benjamini-Hochberg FDR on daily batch`.

---

## Phase 4: Production plumbing

### Step 4.1 — safetensors + torch ≥ 2.6

- **Goal.** Eliminate the `torch.load(..., weights_only=False)` attack
  surface (CVE-2025-32434).
- **Why.** The CVE allows arbitrary code execution from a malicious `.pt`
  file under torch ≤ 2.5.1, even with `weights_only=True`. safetensors is
  pure-tensor and now a PyTorch Foundation project.
- **Pre-conditions.** Phase 3 merged.
- **Files to touch.**
  - New: `requirements.txt` (pin `torch>=2.6`, `safetensors>=0.4`).
  - `ae_shape/train.py:144-152, 170, 186-188` — switch save to
    `safetensors.torch.save_file` for weights; emit a sibling
    `metadata.json` for `preproc`, `baseline`, `calib_scores`,
    `train_config`, `model_beta`.
  - `scripts/detect_anomalies.py:53-80` — `restore()` reads both.
  - `scripts/build_summary.py:11` — same.
  - New: `ae_shape/checkpoint_io.py` with a thin `save_checkpoint` /
    `load_checkpoint` API that everything else uses.
- **Implementation outline.**
  ```python
  # ae_shape/checkpoint_io.py
  from safetensors.torch import save_file, load_file
  import json

  def save_checkpoint(path_stem, model, metadata: dict):
      save_file(model.state_dict(), f"{path_stem}.safetensors")
      with open(f"{path_stem}.metadata.json", "w") as f:
          json.dump(metadata, f, default=_jsonify)

  def load_checkpoint(path_stem, model_cls, model_kwargs):
      state = load_file(f"{path_stem}.safetensors")
      with open(f"{path_stem}.metadata.json") as f:
          meta = json.load(f)
      model = model_cls(**model_kwargs)
      model.load_state_dict(state, strict=True)
      return model, meta
  ```
  Migration: write a one-off `scripts/migrate_checkpoints.py` that reads each
  old `.pt`, writes the new pair, and asserts numerical equivalence by
  scoring one batch under both.
- **Validation.**
  - For each migrated checkpoint, scores on one shared batch are bitwise
    identical (use `torch.equal`, not `torch.allclose`).
  - `weights_only=False` no longer appears in the codebase
    (`rg "weights_only=False"` returns nothing).
- **Failure modes.**
  - **Lost non-tensor fields.** `model_beta`, `n_train_curves`,
    `train_config` are not tensors; they must go into `metadata.json`.
  - **dtype drift.** safetensors preserves dtype, but verify on a half-
    precision tensor if you ever introduce one.
- **Rollback.** Keep `migrate_checkpoints.py` reversible: it copies, not
  moves. Original `.pt` files stay.
- **Commit message.** `chore(io): safetensors + metadata.json checkpoints`.

### Step 4.2 — Test suite + CI

- **Goal.** A regression suite under 60 seconds that runs on each PR.
- **Why.** No tests exist today. Every refactor risks silent breakage of
  the preproc contract, the VAE-determinism invariant, or the calibration
  baseline disjointness.
- **Pre-conditions.** Phase 4.1 merged so tests can use the new IO API.
- **Files to touch.**
  - New: `tests/` directory.
  - New: `pyproject.toml` (so `pytest` discovers package paths).
  - New: `.github/workflows/ci.yml` if GitHub Actions is wired up.
- **Implementation outline.** Mandatory tests:
  ```
  tests/test_data.py
      - test_time_aware_split_disjoint
      - test_split_per_market_chronology
      - test_seasonal_vector_sums_to_12
      - test_preproc_round_trip
  tests/test_models.py
      - test_each_model_io_shape (parametrised)
      - test_vae_deterministic_in_eval (bitwise; loop 10 times)
      - test_lstm_decoder_seed_shape
  tests/test_evaluate.py
      - test_conformal_fpr_on_uniform (synthetic)
      - test_bh_select_returns_max_index
      - test_range_based_f1_existence_only
  tests/test_checkpoint.py
      - test_save_load_round_trip
      - test_load_strict_rejects_extra_keys
  ```
- **Validation.** `pytest -q tests/` passes on Python 3.11 and 3.12.
- **Failure modes.**
  - **Flaky VAE-determinism test.** If it fails, do not loosen tolerance —
    the bug is real. Bug location: `models.py:165-171`. Fix instead.
  - **CPU vs GPU numerical drift.** Confine numerical equality assertions
    to CPU.
- **Rollback.** Tests are additive; rollback = delete.
- **Commit message.** `test: minimal regression suite + CI`.

### Step 4.3 — MLflow + DVC

- **Goal.** Every checkpoint is traceable to a data SHA and a code SHA.
- **Why.** Without this, an analyst asking "what produced this alert in
  March" cannot be answered.
- **Pre-conditions.** Phases 4.1 and 4.2 merged.
- **Files to touch.**
  - `ae_shape/train.py:88-159` — wrap the training loop in
    `mlflow.start_run`.
  - New: `.dvc/` (from `dvc init`), `.dvcignore`.
  - `ml_wide.csv`, `ml_long.csv` — track via `dvc add`.
- **Implementation outline.**
  ```python
  import mlflow

  with mlflow.start_run(run_name=name):
      mlflow.log_params(cfg.__dict__)
      mlflow.set_tag("data_hash", _read_dvc_md5("ml_wide.csv"))
      mlflow.set_tag("git_sha", _git_sha())
      for ep in range(...):
          ...
          mlflow.log_metrics({"train": tr, "val": va, "val_recon": va_recon},
                             step=ep)
      mlflow.log_artifact(f"{ckpt_path}.safetensors")
      mlflow.log_artifact(f"{ckpt_path}.metadata.json")
  ```
- **Validation.** A reviewer can locate any checkpoint by data hash from
  the MLflow UI, click through to the metrics history, and download the
  artefact pair.
- **Failure modes.**
  - **No DVC remote configured.** Hashes are local-only and reviewers can't
    reproduce. Set up S3 / GCS / internal storage before claiming
    "production." For an internal-only system, a shared NFS path is fine.
  - **MLflow tracking server URL hard-coded.** Read from env var
    `MLFLOW_TRACKING_URI`; default to `./mlruns`.
- **Rollback.** Remove `mlflow` calls; MLflow runs go stale but harmlessly.
  DVC tracking can be torn down by `dvc remove`.
- **Commit message.** `feat(mlops): MLflow run tracking + DVC data hash`.

---

## Phase 5: Drift monitors

### Step 5.1 — PSI per tenor on inputs

- **Goal.** Catch silent input drift cheaply.
- **Why.** Industry standard (Fiddler, Arthur). Threshold bands: <0.10
  stable, 0.10–0.25 moderate (investigate), >0.25 significant (retrain).
- **Pre-conditions.** Phase 4 merged; calibration set exists.
- **Files to touch.**
  - New: `scripts/drift_monitor.py`.
  - New: `ae_shape/drift.py`.
- **Implementation outline.**
  ```python
  def psi(reference: np.ndarray, production: np.ndarray, n_bins=10):
      edges = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
      edges[0], edges[-1] = -np.inf, np.inf
      ref_hist = np.histogram(reference, edges)[0] / len(reference)
      prod_hist = np.histogram(production, edges)[0] / len(production)
      ref_hist = np.clip(ref_hist, 1e-6, None)
      prod_hist = np.clip(prod_hist, 1e-6, None)
      return float(((prod_hist - ref_hist) * np.log(prod_hist / ref_hist)).sum())
  ```
  Run per (market, tenor). Reference = trailing 252-day calibration window;
  production = batch being scored.
- **Validation.** Retro-run on Feb 2022. PSI > 0.5 on multiple TTF tenors
  during invasion week. If not, the bin edges are wrong.
- **Failure modes.**
  - **Bin edges from production window.** Edges must be frozen from
    reference. Recomputing daily silently defeats the test.
  - **Bin starvation.** If a bin has 0 reference observations, the log
    blows up. Clip to 1e-6 (above) or use Laplace smoothing.
- **Rollback.** Trivial; new code.
- **Commit message.** `feat(drift): PSI per tenor on inputs`.

### Step 5.2 — MMD on the joint 36-vector

- **Goal.** Detect correlation/shape drift that per-tenor PSI misses.
- **Why.** MMD with RBF kernel is the standard nonparametric two-sample
  test on vectors. Gretton et al., *JMLR* 13, 2012.
- **Files to touch.** `ae_shape/drift.py` (add `mmd2_rbf`),
  `scripts/drift_monitor.py`.
- **Implementation outline.**
  ```python
  def mmd2_rbf(X, Y, bandwidth):
      Kxx = _rbf(X, X, bandwidth)
      Kyy = _rbf(Y, Y, bandwidth)
      Kxy = _rbf(X, Y, bandwidth)
      n, m = len(X), len(Y)
      return Kxx.sum()/n**2 + Kyy.sum()/m**2 - 2*Kxy.sum()/(n*m)
  ```
  Bandwidth via median heuristic on the reference; permutation null with
  1000 permutations computed weekly (not per-day) for speed.
- **Validation.** MMD must fire on Feb 2022 retro-data **before** PSI on
  any single tenor. If PSI fires first, you've broken the bandwidth.
- **Failure modes.**
  - **Bandwidth drift.** Lock bandwidth to a long-history reference;
    recomputing on a drifting window silently reduces sensitivity.
  - **Quadratic cost.** With 252 reference rows × 21 production rows
    × 36 dims, ~1 ms — fine. If you ever scale up, switch to linear-time
    MMD or random Fourier features.
- **Rollback.** Trivial.
- **Commit message.** `feat(drift): RBF-MMD on joint 36-vector`.

### Step 5.3 — ADWIN on the score stream

- **Goal.** Detect drift in the **score** distribution (model-aware drift)
  and trigger calibration refresh.
- **Why.** Bifet & Gavaldà (2007). Standard for streaming drift detection.
- **Files to touch.** `ae_shape/drift.py`; `scripts/drift_monitor.py`.
- **Implementation outline.** Use `river.drift.ADWIN(delta=1e-3)`. Feed the
  daily mean conformal p-value per market. On detected change, emit a
  notification *and* tag the calibration set for refresh.
- **Validation.** Backfill on the existing score history; ADWIN fires near
  each `KNOWN_EVENT_WINDOWS` boundary and roughly at Feb 2022.
- **Failure modes.**
  - **Auto-retrain loop.** Do not wire ADWIN directly to retraining. Wire
    it to a human-in-the-loop notification. Calibration refresh is cheaper
    than full retrain; do that first.
  - **Multiple-testing across markets.** 3 markets × daily ADWIN tests
    inflates the false-alarm rate ~3×. Use `delta=1e-3 / 3` per market or
    a multi-stream variant.
- **Rollback.** Trivial.
- **Commit message.** `feat(drift): ADWIN on conformal-p-value stream`.

---

## Phase 6: Ensemble (NSS + functional + AE)

This is where the rates/commodities literature (Diebold-Li, Sun & Genton,
Borovkova) recommends going: not a bigger AE, but several cheap, complementary
detectors aggregated by rank.

### Step 6.1 — NSS parametric fit per curve

- **Goal.** Fit Svensson (4-factor or 6-factor) to each daily curve; score
  by (a) residual norm and (b) parameter-jump.
- **Why.** Svensson 1994 (BIS Papers No. 25); MDPI *Energies* 16(12) 4746
  for TTF-specific fits. Catches smooth shape moves the AE struggles with.
- **Files to touch.** New `ae_shape/nss.py`.
- **Implementation outline.**
  ```python
  def svensson_basis(tau, lambda1, lambda2):
      # tau in months (1..36)
      f1 = np.ones_like(tau)
      f2 = (1 - np.exp(-tau / lambda1)) / (tau / lambda1)
      f3 = f2 - np.exp(-tau / lambda1)
      f4 = (1 - np.exp(-tau / lambda2)) / (tau / lambda2) - np.exp(-tau / lambda2)
      return np.stack([f1, f2, f3, f4], axis=-1)

  def fit_nss(curve, lambda1=1.5, lambda2=12.0, restarts=5):
      tau = np.arange(1, 37)
      best = None
      for _ in range(restarts):
          init = np.random.randn(4) * np.std(curve)
          res = scipy.optimize.least_squares(
              lambda p: svensson_basis(tau, lambda1, lambda2) @ p - curve, init)
          if best is None or res.cost < best.cost:
              best = res
      return best.x, np.sqrt(best.cost * 2 / 36)   # params, RMSE
  ```
- **Validation.**
  - Mean residual RMSE < 2% of level on clean training dates.
  - Parameter trajectories visibly continuous outside crisis windows.
- **Failure modes.**
  - **Non-convex optimiser.** Without restarts, ~5% of fits land in bad
    local minima with silent residuals of 10%+. Keep ≥5 random restarts.
  - **Lambda fixed vs estimated.** Estimating λ1, λ2 alongside the four
    coefficients makes the optimiser much less stable. Fix them at
    sensible defaults (1.5 and 12 months) for production; revisit only if
    residual RMSE refuses to come down.
- **Rollback.** Trivial.
- **Commit message.** `feat(nss): Svensson 4-factor fit per curve`.

### Step 6.2 — Functional MS-plot detector

- **Goal.** Compute magnitude outlyingness (MO) and variation outlyingness
  (VO) per curve; flag on `sqrt(MO² + VO²)`.
- **Why.** Sun & Genton, *JCGS* 20(2) 2011; Dai & Genton, arXiv:1703.06419.
  Closest match in the literature to "shape anomaly". Two interpretable
  axes: level shock (MO) vs shape break (VO).
- **Files to touch.** New `ae_shape/functional.py`.
- **Implementation outline.** Implement modified band depth on the
  reference curves; compute the two outlyingness measures per Dai & Genton.
  Or use the R package `fdaoutlier` via `rpy2` if Python implementation
  becomes a time sink.
- **Validation.** On gold-tier crisis curves, MS-plot composite score is in
  the top 5% of its market. If not, band-depth ranking is wrong.
- **Failure modes.**
  - **Sample size.** Functional depth is noisy with <300 reference curves.
    Use the full per-market training history.
  - **Cross-market depth.** Compute depth per market; pooling smears
    shapes.
- **Rollback.** Trivial.
- **Commit message.** `feat(functional): MS-plot outlyingness detector`.

### Step 6.3 — Rank aggregation

- **Goal.** Combine [AE, PCA-Mahalanobis, IForest, NSS-residual, MS-plot]
  into one ensemble score with its own conformal calibration.
- **Why.** Aggarwal & Sathe, *Outlier Ensembles* (Springer 2017); SIGKDD
  Explorations 2015. Rank aggregation beats score averaging by removing
  per-detector scale bias.
- **Files to touch.** New `ae_shape/ensemble.py`; modified
  `scripts/detect_anomalies.py`.
- **Implementation outline.**
  ```python
  def rank_ensemble(score_table: pd.DataFrame,
                    detectors: list[str],
                    method: str = "mean") -> np.ndarray:
      ranks = score_table[detectors].rank(method="average", ascending=False)
      if method == "mean":
          return ranks.mean(axis=1).values
      if method == "max":
          return ranks.max(axis=1).values
      raise ValueError(method)
  ```
  Calibrate the ensemble rank's conformal p-value with its own calibration
  set (re-use the calib block; score with the ensemble, store sorted
  ensemble ranks).
- **Validation.**
  - Walk-forward AP on the ensemble beats every single detector's AP on
    every market.
  - If any single detector dominates by ≥0.02 AP, prune it from the
    ensemble — Rayana & Akoglu (*TKDD* 2016) "less is more."
  - Detector rank correlations: if any pair exceeds 0.85, drop the more
    expensive one.
- **Failure modes.**
  - **Correlated detectors.** AE and PCA both pick up reconstruction-style
    anomalies — verify pairwise rank correlation before keeping both.
  - **Two event-list sources of truth.** Reconcile `KNOWN_EVENT_WINDOWS`
    (`data.py:132`) and `EVENTS` (`build_labels.py:29`) before this step.
    The labels drive validation; mismatched event lists silently bias the
    "ensemble beats single" comparison.
- **Rollback.** Drop the ensemble column; downstream consumers fall back to
  the AE p-value.
- **Commit message.** `feat(ensemble): rank aggregation of 5 detectors`.

---

## Phase 7: Analyst feedback loop

### Step 7.1 — Triage log schema

- **Goal.** Persist every analyst verdict in an append-only log.
- **Why.** Without this, the system cannot improve over time.
- **Files to touch.** New `scripts/triage_log.py` (or a small FastAPI app
  if a UI is later wanted).
- **Implementation outline.** SQLite (a single file under `results/`)
  with one table:
  ```sql
  CREATE TABLE triage_log (
      verdict_id        INTEGER PRIMARY KEY AUTOINCREMENT,
      trade_date        DATE NOT NULL,
      market            TEXT NOT NULL,
      curve_id          TEXT NOT NULL,
      score_p_value     REAL NOT NULL,
      ensemble_rank     REAL,
      analyst_id        TEXT NOT NULL,
      analyst_verdict   TEXT CHECK (analyst_verdict IN
                                    ('true_positive','false_positive','unsure')),
      verdict_ts        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      notes             TEXT
  );
  CREATE INDEX idx_triage_trade_date ON triage_log (trade_date);
  ```
- **Validation.** A dummy CLI tool writes a row and reads it back.
- **Failure modes.**
  - **No UI, empty log.** If logging is awkward, analysts won't do it.
    Build the smallest possible CLI (or one-line `curl` to a local Flask
    app) **before** asking for verdicts.
  - **Mutable verdicts.** Treat the log as append-only; corrections are
    new rows, not edits, otherwise audit is broken.
- **Rollback.** Trivial; SQLite file is local.
- **Commit message.** `feat(triage): SQLite append-only verdict log`.

### Step 7.2 — Weak-supervision re-ranker (deferred)

- **Goal.** Once ≥200 verdicts per class accumulate, fit a gradient-
  boosted re-ranker on
  `[p_value, ensemble_rank, NSS_resid, MO, VO, PSI_max, MMD2]` →
  `verdict`. Use it to reorder daily alerts.
- **Why.** Ratner et al., *VLDB Journal* 2019 (Snorkel); reported 45% lift
  over hand labels. Premature re-ranking with 20 labels is worse than
  nothing.
- **Pre-conditions.** Step 7.1 produces ≥200 verdicts per class.
- **Failure modes.**
  - **Censoring.** Analysts only see top-K. The labelled set is censored.
    Mitigate by routing 5% of "ambiguous" cases (p ∈ [0.05, 0.20]) for
    deliberate review.
- **Status.** Do **not** start this step before the log has the data.

---

## Appendix A: Citations (with arXiv / DOI)

- ADBench — Han, Hu, Zhao et al. *NeurIPS Datasets & Benchmarks*, 2022.
  arXiv:2206.09426.
- Point-adjustment metric critique — Kim et al. *AAAI* 2022.
  arXiv:2109.05257.
- Conformal outlier p-values — Bates, Candès, Lei, Romano, Sesia.
  *Annals of Statistics* 51(1), 2023. arXiv:2104.08279.
- Inductive conformal AD — Laxhammar & Falkman. *Annals of Mathematics
  and AI*, 2015. DOI 10.1007/s10472-013-9381-7.
- Walk-forward TS-AD benchmark — Schmidl, Wenig, Papenbrock. *PVLDB*
  15(9), 2022. DOI 10.14778/3538598.3538602.
- Range-based P/R for time series — Tatbul, Lee, Zdonik, Alam,
  Gottschlich. *NeurIPS* 2018. arXiv:1803.03639.
- Litterman–Scheinkman level/slope/curvature — Litterman & Scheinkman.
  *Journal of Fixed Income* 1(1), 1991.
- Diebold–Li dynamic NS — Diebold & Li. *J. Econometrics* 130, 2006.
- Svensson NSS — Svensson. NBER WP 4871, 1994. BIS Papers No. 25.
- Borovkova–Geman gas seasonality — *Review of Derivatives Research* 9,
  2006. DOI 10.1007/s11147-007-9008-4.
- Functional Boxplot — Sun & Genton. *JCGS* 20(2), 2011. DOI
  10.1198/jcgs.2011.09224.
- MS-plot — Dai & Genton. arXiv:1703.06419.
- Outlier Ensembles — Aggarwal & Sathe. Springer, 2017; SIGKDD
  Explorations 2015.
- Less-is-more pruning — Rayana & Akoglu. *TKDD* 10(4), 2016.
- ADWIN — Bifet & Gavaldà, *SDM* 2007.
- MMD — Gretton et al., *JMLR* 13, 2012.
- Snorkel weak supervision — Ratner et al. *VLDB Journal*, 2019.
  DOI 10.1007/s00778-019-00552-1.
- CVE-2025-32434 — GHSA-53q9-r3pm-6pq6 (PyTorch torch.load RCE).

---

## Appendix B: Pre-flight checklist before each step

Before claiming a step done, verify:

1. Working tree clean on the step's branch.
2. Tests passing locally: `pytest -q tests/`.
3. Walk-forward harness (Phase 0.2) re-run; output written to
   `results/walk_forward/`.
4. Diff against `results/_baseline_2026_05_16/` produced and included in
   the PR description.
5. No `weights_only=False` reintroduced (`rg "weights_only=False"`).
6. No new dependencies added without a corresponding `requirements.txt`
   pin.
7. Step's failure modes section walked through and any relevant ones
   confirmed handled.

---

## Appendix C: Commit and branch conventions

- Branch name: `claude/phase<N>-step<M>-<slug>`, e.g.
  `claude/phase2-step1-seasonal-estimate`.
- Commit prefix: one of `feat`, `fix`, `chore`, `test`, `docs`,
  `refactor`, scoped to the area: `feat(eval): ...`, `chore(io): ...`.
- One logical change per commit. Squash on merge.
- PR description must include:
  - The step name from this document.
  - The walk-forward numbers (before vs after).
  - A "Failure modes considered" checklist with each item ticked or
    explicitly waived with reasoning.

---

## Appendix D: Glossary

- **Anomaly score.** Per-curve scalar where higher = more anomalous.
  Currently raw MSE (`detect_anomalies.py:127`); to become a conformal
  p-value where lower = more anomalous (Phase 3).
- **Baseline (residual).** Per-market mean and std of training-set
  reconstruction MSE; persisted in checkpoints by
  `train.py:166-188`. Used by `calibrated_curve_z` today, to be retired
  in favour of conformal p-values.
- **Calibration set.** Disjoint slice of clean training-period data whose
  only purpose is to define the conformal threshold. Introduced in
  Phase 3.1.
- **Conformal p-value.** Distribution-free finite-sample
  p-value = `(1 + #{calib_s ≥ test_s}) / (n + 1)`. Introduced in 3.2.
- **FDR (false discovery rate).** Expected fraction of alerts that are
  false positives. Controlled at level q by Benjamini–Hochberg, 3.3.
- **PSI (Population Stability Index).** Binned KL-divergence between a
  reference and production window. Industry-standard drift metric.
- **MMD (Maximum Mean Discrepancy).** Kernel two-sample test on
  vectors. Used in 5.2 to catch correlation drift.
- **NSS (Nelson-Siegel-Svensson).** Parametric basis for curve shape;
  4–6 coefficients capture level, slope, curvature, hump.
- **MS-plot.** Magnitude/Variation outlyingness plot from functional
  data analysis (Dai & Genton); two-axis interpretation of "shape
  anomaly."
- **Rank ensemble.** Aggregation of multiple detectors by converting
  each to per-curve ranks, then averaging or maximising. Removes
  per-detector scale bias.
