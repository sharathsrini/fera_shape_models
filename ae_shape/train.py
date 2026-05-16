"""Training loop, shared across architectures.

A model `out` dict always contains 'recon'. VAE/BetaVAE additionally have mu/logvar.
The loss is dispatched on the presence of these keys.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import time
import math
import json
import torch
import torch.nn.functional as F
import numpy as np
from .utils import save_json, count_params, get_device

@dataclass
class TrainConfig:
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    beta_kl: float = 0.5             # for VAE
    huber_delta: float = 0.05        # huber loss δ (in normalized units)
    early_stop_patience: int = 10
    log_every: int = 5


def reconstruction_loss(recon, target, kind: str = "huber", delta: float = 0.05):
    if kind == "mse":
        return F.mse_loss(recon, target)
    if kind == "mae":
        return F.l1_loss(recon, target)
    if kind == "huber":
        return F.huber_loss(recon, target, delta=delta)
    raise ValueError(kind)


def compute_loss(out, target, cfg: TrainConfig, model=None):
    """Dispatch on model output shape.

    For VAE-family models, the β weight comes from MODEL.beta (so e.g. β-VAE
    with β=4 actually uses β=4). cfg.beta_kl is an optional global multiplier
    (defaults to 1.0 — set <1.0 for KL warm-up).
    """
    recon = out["recon"]
    if "mu" in out:                              # VAE family
        rec = reconstruction_loss(recon, target, "huber", cfg.huber_delta)
        kl = -0.5 * torch.mean(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())
        beta_model = float(getattr(model, "beta", 1.0)) if model is not None else 1.0
        beta_eff = beta_model * cfg.beta_kl
        loss = rec + beta_eff * kl
        return loss, {"loss": loss.item(), "recon": rec.item(), "kl": kl.item(),
                      "beta_eff": beta_eff}
    rec = reconstruction_loss(recon, target, "huber", cfg.huber_delta)
    return rec, {"loss": rec.item(), "recon": rec.item()}


def train_one_epoch(model, loader, optim, cfg: TrainConfig, device):
    model.train()
    losses = []
    for shape, market, level, raw, idx in loader:
        shape = shape.to(device); market = market.to(device)
        out = model(shape, market)
        loss, _ = compute_loss(out, shape, cfg, model=model)
        optim.zero_grad()
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()
        losses.append(loss.item())
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def evaluate(model, loader, cfg: TrainConfig, device):
    model.eval()
    losses, recons = [], []
    for shape, market, level, raw, idx in loader:
        shape = shape.to(device); market = market.to(device)
        out = model(shape, market)
        loss, log = compute_loss(out, shape, cfg, model=model)
        losses.append(loss.item()); recons.append(log["recon"])
    return sum(losses) / max(len(losses), 1), sum(recons) / max(len(recons), 1)


def fit(
    model,
    loaders,
    cfg: TrainConfig | None = None,
    save_dir: str | Path = "results/checkpoints",
    name: str | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
    preproc: dict | None = None,
):
    """Train one model.

    `preproc` is the preprocessing metadata required to reproduce the exact
    same input transform at inference time. It is pickled into the checkpoint
    so detect_anomalies / serving code never recomputes stats from new data.
    Recommended payload:
        {"normalize": "level_std",
         "mean": np.ndarray[36],     # train-set tenor means (post level-norm)
         "std":  np.ndarray[36],
         "markets": ["TTF","THE","JKM"]}
    """
    cfg = cfg or TrainConfig()
    device = device or get_device()
    model = model.to(device)
    name = name or getattr(model, "name", model.__class__.__name__)
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"{name}.pt"
    log_path = save_dir / f"{name}.history.json"

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    # Build the persistent preproc payload.
    # Convert numpy arrays to plain Python lists so the checkpoint round-trips
    # under any future torch pickle policy.
    if preproc is None:
        ts = loaders.get("train_stats", {}) or {}
        mean = ts.get("mean"); std = ts.get("std")
        preproc = {
            "normalize": "level_std",
            "mean": (mean.tolist() if hasattr(mean, "tolist") else mean),
            "std":  (std.tolist()  if hasattr(std,  "tolist") else std),
            "markets": ["TTF", "THE", "JKM"],
        }

    best_val, best_epoch, hist = math.inf, 0, []
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, loaders["train_loader"], optim, cfg, device)
        va, va_recon = evaluate(model, loaders["val_loader"], cfg, device)
        sched.step()
        hist.append({"epoch": ep, "train": tr, "val": va, "val_recon": va_recon})
        if verbose and (ep == 1 or ep % cfg.log_every == 0 or ep == cfg.epochs):
            print(f"  [{name}] ep {ep:3d} | train {tr:.5f}  val {va:.5f}  recon {va_recon:.5f}")
        if va < best_val:
            best_val, best_epoch = va, ep
            torch.save({
                "model_state": model.state_dict(),
                "name": name, "epoch": ep,
                "val_loss": best_val,
                "n_params": count_params(model),
                "preproc": preproc,
                "model_beta": float(getattr(model, "beta", 1.0)),
                "train_config": cfg.__dict__,
            }, ckpt_path)
        elif ep - best_epoch >= cfg.early_stop_patience:
            if verbose: print(f"  [{name}] early stop at epoch {ep}")
            break

    save_json({"history": hist, "best_val": best_val, "best_epoch": best_epoch,
               "wall_time_s": time.time() - t0, "n_params": count_params(model),
               "config": cfg.__dict__}, log_path)

    # ----------------------------------------------------------------------
    # Compute and persist the residual baseline using the SAME training data
    # the best model was selected on. This makes downstream scoring invariant
    # to the choice of inference-time train loader (in particular, --exclude-
    # known-events stays honored).
    # ----------------------------------------------------------------------
    from .evaluate import collect_reconstructions, build_residual_baseline
    from .data import MARKETS
    # Reload best weights
    best = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    R = collect_reconstructions(model, loaders["train_loader"], device=device)
    market_labels = [MARKETS[i] for i in R["market_oh"].argmax(axis=1)]
    baseline = build_residual_baseline(R["orig"], R["recon"], market_labels)

    # Convert to plain lists/strings for safe pickling
    baseline_safe = {
        "per_market": {str(m): {"mu": float(v["mu"]), "sigma": float(v["sigma"])}
                       for m, v in baseline["per_market"].items()},
        "per_tenor":  {str(m): {"mu": np.asarray(v["mu"]).tolist(),
                                "sigma": np.asarray(v["sigma"]).tolist()}
                       for m, v in baseline["per_tenor"].items()},
        "global":     {"mu": float(baseline["global"]["mu"]),
                       "sigma": float(baseline["global"]["sigma"])},
        "n_train_curves": int(R["orig"].shape[0]),
    }
    best["baseline"] = baseline_safe
    torch.save(best, ckpt_path)

    return {"ckpt": str(ckpt_path), "best_val": best_val, "best_epoch": best_epoch,
            "history": hist, "baseline": baseline_safe}
