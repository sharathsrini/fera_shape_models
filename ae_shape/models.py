"""Autoencoder architectures for forward-curve shape anomaly detection.

All models accept the same input contract:
    shape : Float[B, 36]            level-normalized curve (mean 1 or z-score)
    market: Float[B, 3]             one-hot market id
and return a dict with keys at minimum:
    'recon' : Float[B, 36]          reconstructed curve
    'z'     : Float[B, latent_dim]  bottleneck representation
optionally:
    'mu', 'logvar' : for VAE-style models

This contract makes every model swappable in the training loop.

Architectures
-------------
1. DenseAE        — symmetric MLP, baseline.
2. Conv1dAE       — 1D convolutional encoder/decoder, captures local kinks well.
3. LSTMAE         — sequence AE with reversed-decoder, models tenor order as time-like.
4. VAE            — probabilistic latent, soft generative prior; useful for "likelihood" scoring.
5. BetaVAE        — VAE with β to disentangle factors (level / slope / curvature).
6. TransformerAE  — multi-head self-attention encoder; global shape relationships.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

CURVE_LEN = 36
N_MARKETS = 3


def _mlp(dims, act=nn.GELU, dropout=0.0):
    layers = []
    for a, b in zip(dims[:-1], dims[1:-1] if len(dims) > 2 else []):
        layers += [nn.Linear(a, b), act(), nn.Dropout(dropout)]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


# ----------------------------------------------------------------------
# 1. Dense AE
# ----------------------------------------------------------------------
class DenseAE(nn.Module):
    def __init__(self, latent_dim: int = 6, hidden=(64, 32, 16), dropout: float = 0.05,
                 cond_market: bool = True):
        super().__init__()
        self.cond_market = cond_market
        in_dim = CURVE_LEN + (N_MARKETS if cond_market else 0)
        self.encoder = _mlp([in_dim, *hidden, latent_dim], dropout=dropout)
        dec_in = latent_dim + (N_MARKETS if cond_market else 0)
        self.decoder = _mlp([dec_in, *reversed(hidden), CURVE_LEN], dropout=dropout)
        self.latent_dim = latent_dim
        self.name = "DenseAE"

    def forward(self, x, market):
        xin = torch.cat([x, market], 1) if self.cond_market else x
        z = self.encoder(xin)
        zin = torch.cat([z, market], 1) if self.cond_market else z
        return {"recon": self.decoder(zin), "z": z}


# ----------------------------------------------------------------------
# 2. 1D Conv AE
# ----------------------------------------------------------------------
class Conv1dAE(nn.Module):
    """Treats the curve as a 1D signal of length 36 with 1 channel.

    Good for detecting LOCAL anomalies (kinks at specific tenors).
    """
    def __init__(self, latent_dim: int = 8, base_channels: int = 16,
                 cond_market: bool = True):
        super().__init__()
        self.cond_market = cond_market
        c = base_channels
        # encoder: 36 -> 18 -> 9
        self.enc_conv = nn.Sequential(
            nn.Conv1d(1, c, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(c, c, kernel_size=3, padding=1, stride=2), nn.GELU(),     # 18
            nn.Conv1d(c, 2 * c, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(2 * c, 2 * c, kernel_size=3, padding=1, stride=2), nn.GELU(),  # 9
        )
        self.flat = 2 * c * 9
        self.enc_head = nn.Linear(self.flat + (N_MARKETS if cond_market else 0), latent_dim)
        self.dec_head = nn.Linear(latent_dim + (N_MARKETS if cond_market else 0), self.flat)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose1d(2 * c, 2 * c, kernel_size=4, stride=2, padding=1), nn.GELU(),   # 18
            nn.Conv1d(2 * c, c, kernel_size=3, padding=1), nn.GELU(),
            nn.ConvTranspose1d(c, c, kernel_size=4, stride=2, padding=1), nn.GELU(),           # 36
            nn.Conv1d(c, 1, kernel_size=3, padding=1),
        )
        self.latent_dim = latent_dim
        self.name = "Conv1dAE"

    def forward(self, x, market):
        h = self.enc_conv(x.unsqueeze(1))                                  # B, 2c, 9
        h = h.flatten(1)
        if self.cond_market:
            h = torch.cat([h, market], 1)
        z = self.enc_head(h)
        zin = torch.cat([z, market], 1) if self.cond_market else z
        d = self.dec_head(zin).view(-1, self.dec_conv[0].in_channels, 9)
        out = self.dec_conv(d).squeeze(1)
        return {"recon": out, "z": z}


# ----------------------------------------------------------------------
# 3. LSTM AE
# ----------------------------------------------------------------------
class LSTMAE(nn.Module):
    """Sequence-to-sequence AE treating tenor index as time."""
    def __init__(self, hidden: int = 32, latent_dim: int = 8, cond_market: bool = True):
        super().__init__()
        self.hidden = hidden
        self.cond_market = cond_market
        self.encoder = nn.LSTM(1 + (N_MARKETS if cond_market else 0), hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent_dim)
        self.from_latent = nn.Linear(latent_dim + (N_MARKETS if cond_market else 0), hidden)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)
        self.latent_dim = latent_dim
        self.name = "LSTMAE"

    def forward(self, x, market):
        B, T = x.shape
        xt = x.unsqueeze(-1)                                             # B,T,1
        if self.cond_market:
            mt = market.unsqueeze(1).expand(-1, T, -1)
            xt = torch.cat([xt, mt], -1)
        _, (h, _) = self.encoder(xt)
        z = self.to_latent(h.squeeze(0))                                 # B,latent
        zin = torch.cat([z, market], 1) if self.cond_market else z
        h0 = self.from_latent(zin).unsqueeze(0)
        # decoder input: repeat zero seed length T (teacher-free)
        dec_in = h0.repeat(T, 1, 1).transpose(0, 1)
        out, _ = self.decoder(dec_in, (h0, torch.zeros_like(h0)))
        recon = self.out(out).squeeze(-1)
        return {"recon": recon, "z": z}


# ----------------------------------------------------------------------
# 4. VAE
# ----------------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim: int = 6, hidden=(64, 32), beta: float = 1.0,
                 cond_market: bool = True, dropout: float = 0.05):
        super().__init__()
        self.cond_market = cond_market
        self.beta = beta
        in_dim = CURVE_LEN + (N_MARKETS if cond_market else 0)
        self.enc = _mlp([in_dim, *hidden], dropout=dropout)
        self.fc_mu = nn.Linear(hidden[-1], latent_dim)
        self.fc_lv = nn.Linear(hidden[-1], latent_dim)
        dec_in = latent_dim + (N_MARKETS if cond_market else 0)
        self.dec = _mlp([dec_in, *reversed(hidden), CURVE_LEN], dropout=dropout)
        self.latent_dim = latent_dim
        self.name = "VAE" if beta == 1.0 else f"BetaVAE_b{beta}"

    def encode(self, x, market):
        xin = torch.cat([x, market], 1) if self.cond_market else x
        h = self.enc(xin)
        return self.fc_mu(h), self.fc_lv(h)

    def reparameterize(self, mu, logvar):
        # CRITICAL: deterministic at eval time so anomaly scores are reproducible.
        # Stochastic sampling only during training.
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, market):
        zin = torch.cat([z, market], 1) if self.cond_market else z
        return self.dec(zin)

    def forward(self, x, market):
        mu, logvar = self.encode(x, market)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, market)
        return {"recon": recon, "z": z, "mu": mu, "logvar": logvar}


def vae_loss(out, target, beta: float = 1.0):
    recon = F.mse_loss(out["recon"], target, reduction="mean")
    mu, lv = out["mu"], out["logvar"]
    kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    return recon + beta * kl, {"recon": recon.item(), "kl": kl.item()}


# ----------------------------------------------------------------------
# 5. Transformer AE
# ----------------------------------------------------------------------
class TransformerAE(nn.Module):
    """Self-attention over the 36-tenor sequence.

    Captures long-range dependencies (e.g. winter / summer interactions).
    """
    def __init__(self, d_model: int = 32, n_heads: int = 4, n_layers: int = 2,
                 latent_dim: int = 8, cond_market: bool = True, dropout: float = 0.1):
        super().__init__()
        self.cond_market = cond_market
        self.embed = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.randn(CURVE_LEN, d_model) * 0.02)
        if cond_market:
            self.mkt_embed = nn.Linear(N_MARKETS, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2 * d_model,
                                                   dropout=dropout, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.to_latent = nn.Linear(d_model, latent_dim)
        self.from_latent = nn.Linear(latent_dim + (N_MARKETS if cond_market else 0), d_model * CURVE_LEN)
        dec_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2 * d_model,
                                               dropout=dropout, batch_first=True, activation="gelu")
        self.dec = nn.TransformerEncoder(dec_layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, 1)
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.name = "TransformerAE"

    def forward(self, x, market):
        B, T = x.shape
        h = self.embed(x.unsqueeze(-1)) + self.pos.unsqueeze(0)
        if self.cond_market:
            h = h + self.mkt_embed(market).unsqueeze(1)
        h = self.enc(h)
        z_seq = h.mean(dim=1)                                            # B,d
        z = self.to_latent(z_seq)
        zin = torch.cat([z, market], 1) if self.cond_market else z
        dec_in = self.from_latent(zin).view(B, T, self.d_model) + self.pos.unsqueeze(0)
        dec_out = self.dec(dec_in)
        recon = self.out(dec_out).squeeze(-1)
        return {"recon": recon, "z": z}


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------
def build_model(name: str, **kwargs):
    name = name.lower()
    if name == "dense":
        return DenseAE(**kwargs)
    if name == "conv1d":
        return Conv1dAE(**kwargs)
    if name == "lstm":
        return LSTMAE(**kwargs)
    if name == "vae":
        return VAE(beta=1.0, **kwargs)
    if name in ("beta_vae", "bvae"):
        return VAE(**kwargs)
    if name == "transformer":
        return TransformerAE(**kwargs)
    raise ValueError(f"Unknown model: {name}")
