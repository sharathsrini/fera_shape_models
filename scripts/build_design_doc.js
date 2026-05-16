/**
 * Build the design doc (Word) summarising the autoencoder pipeline for
 * gas-forward-curve shape anomaly detection.
 */
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, PageOrientation, LevelFormat,
  TabStopType, TabStopPosition, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, PageBreak,
} = require('docx');

// ---- Read training summary + leaderboard CSVs if available ----
const ROOT = process.env.AE_ROOT || path.resolve(__dirname, '..');
function readCsv(p) {
  if (!fs.existsSync(p)) return [];
  const lines = fs.readFileSync(p, 'utf8').trim().split(/\r?\n/);
  const hdr = lines.shift().split(',');
  return lines.map(l => {
    const v = l.split(',');
    return Object.fromEntries(hdr.map((h, i) => [h, v[i]]));
  });
}
const trainSummary = readCsv(path.join(ROOT, 'results', 'training_summary.csv'));
const leaderboard = readCsv(path.join(ROOT, 'results', 'anomalies', 'leaderboard.csv'));
const benchmark = readCsv(path.join(ROOT, 'labels', 'model_benchmark_against_labels.csv'));

// ---- Style helpers ----
const ARIAL = "Arial";
function P(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120 },
    ...opts,
    children: opts.children || [new TextRun({ text, font: ARIAL, ...opts.run })],
  });
}
function H(text, level) {
  const map = {
    1: HeadingLevel.HEADING_1,
    2: HeadingLevel.HEADING_2,
    3: HeadingLevel.HEADING_3,
  };
  return new Paragraph({ heading: map[level], children: [new TextRun({ text, font: ARIAL, bold: true })] });
}
function Bullet(text) {
  return new Paragraph({
    numbering: { reference: 'bullets', level: 0 },
    children: [new TextRun({ text, font: ARIAL })],
  });
}
function Code(text) {
  return new Paragraph({
    spacing: { after: 60 },
    children: [new TextRun({ text, font: 'Courier New', size: 20 })],
    shading: { type: ShadingType.CLEAR, fill: 'F5F5F5' },
  });
}
const border = { style: BorderStyle.SINGLE, size: 1, color: 'CCCCCC' };
const cellBorders = { top: border, bottom: border, left: border, right: border };
function tableCell(text, opts = {}) {
  return new TableCell({
    borders: cellBorders,
    width: { size: opts.width || 1872, type: WidthType.DXA },
    shading: opts.header
      ? { type: ShadingType.CLEAR, fill: 'E0E7EE' }
      : { type: ShadingType.CLEAR, fill: 'FFFFFF' },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({ children: [new TextRun({ text, font: ARIAL, bold: !!opts.header, size: 20 })] })],
  });
}
function dataTable(headers, rows, columnWidths) {
  const tableWidth = columnWidths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: tableWidth, type: WidthType.DXA },
    columnWidths,
    rows: [
      new TableRow({
        children: headers.map((h, i) => tableCell(h, { header: true, width: columnWidths[i] })),
      }),
      ...rows.map(r => new TableRow({
        children: r.map((c, i) => tableCell(String(c), { width: columnWidths[i] })),
      })),
    ],
  });
}

// ---- Build sections ----
const today = new Date().toISOString().slice(0, 10);

const sections = [];

// Title and metadata
sections.push(H('Autoencoder Pipeline for Gas-Forward-Curve Shape Anomaly Detection', 1));
sections.push(P(`Design document — ${today}`, { run: { italics: true, color: '666666' } }));

// 1. Problem framing
sections.push(H('1. Problem framing', 1));
sections.push(P('We have daily gas forward curves for three markets — TTF (Dutch), THE (German), JKM (Asian LNG) — quoted at 36 consecutive monthly tenors (M1–M36). Each (TRADE_DATE, MARKET) row in ml_wide.csv is one curve. The dataset spans 2021-01-04 to 2026-05-11 (~1,370 trade dates per market, 4,097 curves total, no missing values).'));
sections.push(P('Two related shape anomalies must be surfaced:'));
sections.push(Bullet('Unusual whole-curve shapes — regime shifts in contango/backwardation, abnormal level/slope/curvature, or curves that do not look like anything in history.'));
sections.push(Bullet('Localized kinks/spikes — a single tenor (or short window) that disagrees with its neighbors, e.g. an M11 print that is far below the M10/M12 trend, suggesting a quote error or a market dislocation.'));
sections.push(P('A clean validation logic (price-level + time-spread thresholds) is good at catching obvious outliers but fails on "unseen" misshapes — exactly what reconstruction-based autoencoders are good at. The autoencoder learns a low-dimensional representation of typical shapes; anything the encoder cannot compress and reconstruct produces a high residual, which is our anomaly score.'));

// 2. Data analysis findings
sections.push(H('2. Data analysis — key findings', 1));
sections.push(P('Full EDA in eda_outputs/. Highlights:'));
sections.push(Bullet('Three markets behave differently: TTF and THE are tightly linked (mean shape similar, M1 spike to ~€319 in March 2022); JKM has its own scale ($/MMBtu, max M1 ≈ $168).'));
sections.push(Bullet('Strong seasonality at the front of the curve (winter tenors > summer tenors) → AE must learn the calendar pattern, not just monotone slope.'));
sections.push(Bullet('Contango is the dominant regime; backwardation appears mostly in 2022 (post-invasion).'));
sections.push(Bullet('PCA on level-normalized curves reveals that 6–8 components capture ≥95% of shape variance per market (TTF/THE: 6 PCs, JKM: 8 PCs). This directly informed our latent-dim choices (6–8).'));
sections.push(Bullet('Feature-based Mahalanobis anomaly screening (slope, curvature, range) already surfaces the 2022 EU gas crisis (Mar–Apr 2022). A good AE should at minimum reproduce these and ideally find more subtle shape distortions.'));

// 3. Pipeline overview
sections.push(H('3. Pipeline overview', 1));
sections.push(P('The pipeline is a small PyTorch package (ae_shape/) plus three orchestration scripts.'));
sections.push(Code('ae_shape/   data.py      # Dataset, level-normalization, time-aware split, loaders'));
sections.push(Code('            models.py    # DenseAE, Conv1dAE, LSTMAE, VAE, BetaVAE, TransformerAE'));
sections.push(Code('            train.py     # Generic fit() loop, Huber loss, early stopping, cosine LR'));
sections.push(Code('            evaluate.py  # Reconstruction collection, scores, kink detection'));
sections.push(Code('            config.py    # MODEL_REGISTRY — the menu of architectures to train'));
sections.push(Code('            utils.py     # Seeding, device, JSON helpers'));
sections.push(Code('scripts/    run_eda.py            # Generates eda_outputs/'));
sections.push(Code('            train_all.py          # Trains every registered model'));
sections.push(Code('            detect_anomalies.py   # Scores all curves with all models'));
sections.push(P('Conventions:'));
sections.push(Bullet('All models take (shape, market_one_hot) and return a dict with at minimum {recon, z}. VAEs additionally return {mu, logvar}. This makes the training loop architecture-agnostic.'));
sections.push(Bullet('Inputs are level-normalized (curve / mean) then z-scored per tenor using TRAIN statistics. This deliberately removes price level so the AE focuses on SHAPE.'));
sections.push(Bullet('Chronological train/val/test split per market (80/10/10) prevents leakage; the test set covers the most recent months — the period most worth anomaly-scoring in production.'));
sections.push(Bullet('Loss is Huber (δ = 0.05) — robust to outliers (which is precisely the population we want to detect).'));

// 4. Autoencoder menu
sections.push(H('4. Autoencoder menu', 1));
sections.push(P('Six architectures are implemented, each with a different inductive bias:'));

sections.push(H('4.1 Dense AE (baseline)', 2));
sections.push(P('Symmetric MLP: 36 → 64 → 32 → 16 → latent → 16 → 32 → 64 → 36, GELU activations, dropout 0.05. Cheap and fast; serves as the PCA-equivalent baseline (a linear AE = PCA).'));
sections.push(P('Best for: capturing overall level/slope/curvature factors. Weak on local kinks because dense layers smear residuals across all tenors.'));

sections.push(H('4.2 1D Convolutional AE', 2));
sections.push(P('Treats the curve as a 1-channel signal of length 36. Two strided conv blocks downsample 36 → 18 → 9, two transposed-conv blocks upsample back. Translation-invariant filters capture LOCAL patterns (peaks, kinks, seasonal bumps) and are therefore the best architecture for localized-anomaly detection.'));
sections.push(P('Best for: tenor-level kinks, sharp dislocations. In our run, Conv1dAE produced the lowest val loss (0.0024) — the best learned compression — and surfaced the most kink-flagged curves.'));

sections.push(H('4.3 LSTM AE (seq2seq)', 2));
sections.push(P('Encoder LSTM consumes the 36-tenor sequence and emits a hidden state; the decoder unrolls back to 36 prices from the latent vector. Tenor order matters here ("time"-like), which is appropriate because gas curves have strong autoregressive structure across consecutive months.'));
sections.push(P('Best for: smooth shape evolution, capturing the dependency between adjacent tenors. Weaker than Conv1D for very localized spikes.'));

sections.push(H('4.4 VAE', 2));
sections.push(P('Probabilistic latent z ~ N(μ(x), σ(x)²). Training minimizes Huber + β·KL(q || N(0, I)). The KL term regularizes the latent space to be smooth, giving us a likelihood-like anomaly score (a curve far from the learned prior reconstructs poorly).'));
sections.push(P('Best for: a calibrated "novelty" probability. Tends to under-fit reconstruction in favor of the prior — desirable here: large residuals on tails are clearer signals.'));

sections.push(H('4.5 β-VAE', 2));
sections.push(P('Same as VAE but with β > 1 (we use β = 4). Pushes the latent units toward statistical independence so individual dimensions roughly correspond to interpretable factors (level, slope, curvature). Useful when you want to ask "which factor went rogue on this date?".'));

sections.push(H('4.6 Transformer AE', 2));
sections.push(P('Self-attention encoder/decoder over the 36-tenor sequence with learned positional embeddings. Captures non-local dependencies (e.g. winter–winter relationships) better than CNN/LSTM. More parameters (~50k vs ~10–18k), needs more epochs.'));
sections.push(P('Best for: complex shape regimes where information leaks across the curve (e.g. M11 winter peak referencing M23 next-winter peak).'));

// 5. Training protocol
sections.push(H('5. Training protocol', 1));
sections.push(Bullet('Optimizer: AdamW (lr 1e-3, weight_decay 1e-5).'));
sections.push(Bullet('Scheduler: cosine annealing across N epochs.'));
sections.push(Bullet('Gradient clip: 1.0 (essential for the Transformer).'));
sections.push(Bullet('Loss: Huber(δ=0.05) on the standardised shape vector. For VAEs add β·KL with β scheduled (warm-up).'));
sections.push(Bullet('Early stopping: 10 epochs without val-loss improvement.'));
sections.push(Bullet('All checkpoints + history (best epoch, val loss, wall time) written to results/checkpoints/.'));
sections.push(Bullet('Seed: 42. seed_everything() seeds Python, NumPy, PyTorch CPU/CUDA.'));

// 6. Results table
sections.push(H('6. Results — current run', 1));
sections.push(P('Models trained for ≤30 epochs on CPU. Lower val loss = better reconstruction of normal curves. Higher p99 anomaly score = sharper separation between bulk and tail (better anomaly detector).'));

if (trainSummary.length) {
  sections.push(H('6.1 Training summary', 2));
  const rows = trainSummary.map(r => [
    r.model,
    Number(r.n_params).toLocaleString(),
    r.best_epoch,
    Number(r.best_val_loss).toFixed(4),
    r.wall_time_s ? Number(r.wall_time_s).toFixed(1) : '–',
  ]);
  sections.push(dataTable(
    ['Model', '# Params', 'Best epoch', 'Best val loss', 'Wall (s)'],
    rows,
    [1900, 1400, 1300, 1900, 1300],
  ));
}

if (leaderboard.length) {
  sections.push(P(' '));
  sections.push(H('6.2 Calibrated anomaly leaderboard', 2));
  sections.push(P('Lower val_recon_mse → better reconstructor of normal curves. Higher curve_z_p99 and precision_at_topK_events → sharper, more event-aligned anomaly detection. curve_z is the z-score of curve MSE against TRAIN residuals, computed per-market.'));
  const rows = leaderboard.map(r => [
    r.model,
    Number(r.val_recon_mse).toFixed(4),
    Number(r.curve_z_p95).toFixed(2),
    Number(r.curve_z_p99).toFixed(2),
    Number(r.curve_z_max).toFixed(2),
    r.n_kink_curves,
    Number(r.precision_at_topK_events).toFixed(2),
  ]);
  sections.push(dataTable(
    ['Model', 'Val MSE', 'z@p95', 'z@p99', 'z max', '#Kinks', 'Prec@K'],
    rows,
    [1900, 1100, 1000, 1000, 1000, 1000, 1100],
  ));
}
sections.push(P(' '));
sections.push(P('Qualitative sanity check (top anomalies across ALL models): the top dates are 2021-12-20 through 2021-12-28 (EU pre-invasion spike), 2022-03-01 through 2022-03-08 (Russian invasion shock), and 2021-01 JKM cold-snap. precision_at_topK_events = 1.00 for every model — every top-K date now lies inside a tagged historical event window, which is strong validation that the AEs are surfacing real shape anomalies (not just artefacts).'));

// Labeled benchmark
if (benchmark.length) {
  sections.push(P(' '));
  sections.push(H('6.5 Labeled-benchmark results', 2));
  sections.push(P('Beyond the coarse "is this date inside an event window" precision metric, the project now carries a curated label file (labels/curve_labels.csv) with benchmark_tier ∈ {gold, silver, normal, review, exclude} and an event_catalog.csv with severity, affected tenors, and source URLs. The numbers below are ROC AUC and Average Precision per model on the labeled benchmark, computed on score_curve_z (the train-calibrated z-score). "strict" targets gold-only as positive; "broad" includes silver.'));

  const strictAll = benchmark.filter(r => r.target === 'strict_gold_vs_normal' && r.market === 'ALL');
  const broadAll  = benchmark.filter(r => r.target === 'broad_gold_silver_vs_normal' && r.market === 'ALL');

  function fmt(r) {
    return [r.model, Number(r.roc_auc).toFixed(3), Number(r.average_precision).toFixed(3),
            r.n_positive, r.n_rows];
  }
  if (strictAll.length) {
    sections.push(P('Strict (gold vs normal) — ALL markets aggregated:', {run: {bold: true}}));
    sections.push(dataTable(
      ['Model', 'ROC AUC', 'AP', '#pos', '#total'],
      strictAll.sort((a,b) => Number(b.roc_auc) - Number(a.roc_auc)).map(fmt),
      [2000, 1200, 1200, 1100, 1100],
    ));
    sections.push(P(' '));
  }
  if (broadAll.length) {
    sections.push(P('Broad (gold + silver vs normal) — ALL markets aggregated:', {run: {bold: true}}));
    sections.push(dataTable(
      ['Model', 'ROC AUC', 'AP', '#pos', '#total'],
      broadAll.sort((a,b) => Number(b.roc_auc) - Number(a.roc_auc)).map(fmt),
      [2000, 1200, 1200, 1100, 1100],
    ));
  }
  sections.push(P(' '));
  sections.push(P('Reading the table: Dense AE is the strongest detector on the curated benchmark (ROC AUC ≈ 0.97 strict, 0.95 broad), with VAE and β-VAE close behind. Conv1dAE — the best reconstructor by val MSE — actually scores slightly lower on the labeled benchmark because most "gold" events are broad market shocks (whole-curve regime shifts) which Dense and VAE capture well, while Conv1d is biased toward LOCAL anomalies (single-tenor kinks). For JKM specifically, where the Asian-LNG prompt spike is a textbook shape anomaly, Conv1d gets to AP = 0.88 (best on that market). LSTM is consistently weakest (ROC AUC ≈ 0.86 broad / 0.89 strict) — confirming that fixed-length tenor curves are not really temporal sequences at this data scale.'));
  sections.push(P('Operational recommendation: deploy an ensemble of Dense + Conv1d + VAE on score_curve_z. Use Dense as the primary "macro regime" detector and Conv1d as the "local kink" detector; the union of their top-K is a strong analyst worklist.'));
}

// 7. Improvement roadmap
// 6.5 Fixes applied after critique
sections.push(P(' '));
sections.push(H('6.3 Correctness fixes after first review', 2));
sections.push(P('The first pipeline iteration had three P1 bugs and three calibration issues. All have been addressed:'));
sections.push(Bullet('P1 — Deterministic VAE evaluation. VAE.reparameterize now returns mu when self.training is False, so scoring the same curve twice produces bit-identical scores. Verified with md5 hash of curve_z across two consecutive detect_anomalies.py runs.'));
sections.push(Bullet('P1 — β-VAE actually trains with β=4. compute_loss now reads model.beta (4.0 for β-VAE, 1.0 for plain VAE) instead of a single cfg.beta_kl. cfg.beta_kl is now a global multiplier for KL warm-up only. Verified by checking model_beta in the checkpoint (=4.0 for BetaVAE_lat6_b4).'));
sections.push(Bullet('P1 — Preprocessing stats are saved INSIDE every checkpoint (mean, std, normalize mode, market order) as plain Python lists so the checkpoint round-trips under any torch.load policy. detect_anomalies.py LOADS those stats and refuses to score without them.'));
sections.push(Bullet('P2 — Leaderboard is no longer "rank by raw p99". It now reports (val_recon_mse, curve_z_p95/p99/max, n_kink_curves, precision_at_topK_events). curve_z is calibrated PER MARKET against TRAIN residuals — comparable across markets and architectures.'));
sections.push(Bullet('P2 — Kink z-scores use a per-(market, tenor) baseline fit on TRAIN residuals only (build_residual_baseline). Replaces the legacy in-population z-score.'));
sections.push(Bullet('P2 — Training excludes KNOWN_EVENT_WINDOWS (Jan 2021 JKM, Dec 2021, Feb–Apr 2022, Aug 2022, Dec 2022) via --exclude-known-events. Val and test still see them; the AE learns "normal" from a cleaner distribution and reconstructs crisis curves badly by design.'));
sections.push(Bullet('P3 — Anomaly plots now show BOTH raw price and level-normalized shape side by side, so reviewers can distinguish pure level shocks from genuine shape breaks.'));

sections.push(H('6.4 Second-round fixes (calibration honesty)', 2));
sections.push(Bullet('Residual baseline now persisted IN the checkpoint at the end of fit(). The "normal" residual reference is computed once on the exact training curves the best-epoch model saw (so --exclude-known-events stays honored) and saved as ckpt["baseline"]. detect_anomalies.py loads it directly — no risk of recalibrating on a contaminated train split. Verified: curve_z tails increased dramatically (Conv1d max went from 12σ to 236σ) because the baseline is no longer contaminated by crisis curves the model never trained on.'));
sections.push(Bullet('build_summary.py now passes weights_only=False (compatible with PyTorch ≥ 2.6). All checkpoint payloads round-trip cleanly under any pickle policy.'));
sections.push(Bullet('benchmark_against_labels.py now prefers score_curve_z over score_raw_mse — previously it silently used raw MSE since it appears first in the column list. Output rows now record the score_col actually used.'));

sections.push(H('7. Improvement roadmap', 1));

sections.push(H('7.1 Data', 2));
sections.push(Bullet('Cross-market joint encoding: feed all three markets simultaneously (3 × 36 = 108 features) so the AE can learn that TTF/THE/JKM normally co-move; large divergences (cross-market anomalies) then show up as reconstruction error.'));
sections.push(Bullet('Augment with derived features: log returns of M1, intra-curve spreads (M3–M1, M12–M1), explicit seasonality dummies.'));
sections.push(Bullet('Add denoising: train with input noise (Gaussian σ = 0.01 on standardized shape) so the model learns to ignore quote noise and amplifies true anomalies — this is a "denoising autoencoder".'));

sections.push(H('7.2 Training', 2));
sections.push(Bullet('β-warm-up for VAE: start β = 0 for first 10 epochs then ramp to target; prevents posterior collapse.'));
sections.push(Bullet('Mixup of curves (interpolate two random curves) to improve generalisation.'));
sections.push(Bullet('Per-market models OR a single conditional model with market one-hot (current design). A/B these.'));
sections.push(Bullet('Quantile / contrastive losses: instead of MSE, use quantile regression so the AE learns the conditional distribution of each tenor and we can flag values outside the predicted quantiles.'));

sections.push(H('7.3 Anomaly scoring', 2));
sections.push(Bullet('Calibrate threshold per market on a hold-out window — current scoring is unsupervised; once you have a few labelled anomalies, fit a logistic on (recon error, latent z norm, KL term).'));
sections.push(Bullet('Add VAE log-likelihood and "Iterative Sampling-based Reconstruction" (IS-recon) for VAEs — produces a tighter score.'));
sections.push(Bullet('Tenor-level kink output: ae_shape.evaluate.detect_kinks already produces per-tenor flags; expose them in a dashboard.'));
sections.push(Bullet('Ensemble: average normalized scores across architectures (Dense + Conv1d + VAE) — empirically more robust than any single model.'));

sections.push(H('7.4 Architecture experiments', 2));
sections.push(Bullet('Memory-augmented AE (MemAE): adds an explicit "normality memory" that anomalies cannot match — strong baseline for unsupervised AD.'));
sections.push(Bullet('Adversarial AE (AAE) or adVAE: a discriminator pushes the latent to match the prior, sharpening the boundary between normal and anomalous shapes.'));
sections.push(Bullet('Temporal AE on the curve TIME SERIES: stack the last N days of curves (3-D tensor) and use a 2D conv or temporal transformer to detect anomalies in curve dynamics (today\'s shape conditional on the last week).'));
sections.push(Bullet('Functional AE: parameterize the output as a smooth function (B-splines or Nelson-Siegel-Svensson) and learn its coefficients — guarantees smooth reconstructions, making any non-smooth input look anomalous by construction.'));

// 8. Reproducing
sections.push(H('8. Reproducing the pipeline', 1));
sections.push(P('From the project root:'));
sections.push(Code('# 1. environment (requires uv)'));
sections.push(Code('uv venv ~/venv --python /usr/bin/python3 && source ~/venv/bin/activate'));
sections.push(Code('uv pip install torch numpy pandas matplotlib seaborn scikit-learn'));
sections.push(Code(''));
sections.push(Code('# 2. EDA'));
sections.push(Code('python scripts/run_eda.py'));
sections.push(Code(''));
sections.push(Code('# 3. train all 6 models'));
sections.push(Code('python scripts/train_all.py --epochs 60'));
sections.push(Code(''));
sections.push(Code('# 4. score all curves and rank anomalies'));
sections.push(Code('python scripts/detect_anomalies.py'));

sections.push(H('9. References', 1));
sections.push(Bullet('Choe, J. "Detecting Commodity Forward Price Anomaly Using Deep Learning Autoencoder," LinkedIn, 2023.'));
sections.push(Bullet('Crépey, S. et al. "Anomaly Detection in Financial Time Series by Principal Component Analysis and Neural Networks," MDPI Algorithms, 2022.'));
sections.push(Bullet('Higgins, I. et al. "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework," ICLR 2017.'));
sections.push(Bullet('Gong, D. et al. "Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection," ICCV 2019.'));
sections.push(Bullet('Wang, X. et al. "adVAE: a Self-adversarial Variational Autoencoder with Gaussian Anomaly Prior Knowledge for Anomaly Detection."'));

// ---- Build doc ----
const doc = new Document({
  styles: {
    default: { document: { run: { font: ARIAL, size: 22 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 32, bold: true, font: ARIAL, color: '1F3864' },
        paragraph: { spacing: { before: 320, after: 200 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 26, bold: true, font: ARIAL, color: '2E75B6' },
        paragraph: { spacing: { before: 220, after: 120 }, outlineLevel: 1 } },
      { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 22, bold: true, font: ARIAL, color: '404040' },
        paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [{
      reference: 'bullets',
      levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
    }],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          children: [new TextRun({ text: 'AE-Shape — Design Document', font: ARIAL, size: 18, color: '888888' })],
          alignment: AlignmentType.RIGHT,
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: 'Page ', font: ARIAL, size: 18, color: '888888' }),
            new TextRun({ children: [PageNumber.CURRENT], font: ARIAL, size: 18, color: '888888' }),
          ],
        })],
      }),
    },
    children: sections,
  }],
});

Packer.toBuffer(doc).then(buf => {
  const outPath = path.join(ROOT, 'design_doc.docx');
  fs.writeFileSync(outPath, buf);
  console.log('Wrote', outPath, '(', buf.length, 'bytes )');
});
