"""Default model configurations - the menu of architectures we train.

Note on β handling for VAE family:
  - Each VAE model stores its own `beta` (1.0 for plain VAE, 4.0 for β-VAE).
  - `cfg.beta_kl` is a GLOBAL multiplier (use <1.0 for KL warm-up, 1.0 for full β).
  - Effective β = model.beta * cfg.beta_kl  (applied in train.compute_loss).
"""
from .train import TrainConfig

# Each entry: (model_kind, model_kwargs, train_config_override)
MODEL_REGISTRY = {
    "Dense_lat6":          ("dense",       dict(latent_dim=6,  hidden=(64, 32, 16)),                 {}),
    "Conv1d_lat8":         ("conv1d",      dict(latent_dim=8,  base_channels=16),                    {}),
    "LSTM_lat8":           ("lstm",        dict(latent_dim=8,  hidden=32),                           dict(epochs=80)),
    "VAE_lat6":            ("vae",         dict(latent_dim=6,  hidden=(64, 32)),                     dict(beta_kl=1.0)),
    "BetaVAE_lat6_b4":     ("beta_vae",    dict(latent_dim=6,  hidden=(64, 32), beta=4.0),           dict(beta_kl=1.0)),
    "Transformer_lat8":    ("transformer", dict(latent_dim=8,  d_model=32, n_heads=4, n_layers=2),   {}),
}


def get_train_config(overrides: dict | None = None) -> TrainConfig:
    cfg = TrainConfig()
    for k, v in (overrides or {}).items():
        setattr(cfg, k, v)
    return cfg
