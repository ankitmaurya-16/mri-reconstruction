"""
Configuration system for Multi-Contrast DISCO-Diffusion GAN.
Loads configs/default.yaml into typed Python dataclasses.
Supports CLI overrides via dotted-path notation.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import List

import yaml


# ---------------------------------------------------------------------------
# Sub-config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DISCOConfig:
    num_basis: int = 8          # L piecewise-linear basis functions (Paper 2 Supp B.3)
    kernel_size: int = 3        # spatial support of each basis function


@dataclass
class UDNOKConfig:
    """k-space neural operator (NO_k) architecture settings."""
    in_channels: int = 4
    out_channels: int = 4
    base_channels: int = 32
    depth: int = 4


@dataclass
class ModelConfig:
    in_channels: int = 4                        # real_c1, imag_c1, real_c2, imag_c2
    base_channels: int = 64
    channel_mults: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    num_res_blocks: int = 2
    latent_dim: int = 64
    time_emb_dim: int = 128
    disco: DISCOConfig = field(default_factory=DISCOConfig)
    udno_k: UDNOKConfig = field(default_factory=UDNOKConfig)
    num_cascades_image: int = 6                 # NO_i + DC cascades in image space


@dataclass
class DiffusionConfig:
    T: int = 16                                 # fast diffusion steps (FDMR Table 6)
    T_refine: int = 30                          # refinement model steps (FDMR Table 8)
    beta_schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 0.02


@dataclass
class TrainingConfig:
    lr: float = 1e-4                            # Adam lr for G and D (FDMR Section 4.3)
    batch_size: int = 4                         # FDMR Section 4.3
    num_epochs: int = 200
    lambda_tv: float = 1e-3                     # TV weight in DGP (FDMR Eq. 12)
    lambda_mse: float = 1.0                     # MSE weight in generator loss
    lambda_adv: float = 0.01                    # adversarial weight in generator loss
    dgp_steps: int = 50                         # DGP iterations (FDMR Table 7)
    grad_clip: float = 1.0
    save_every: int = 10
    val_every: int = 1
    mixed_precision: bool = True                # use torch.cuda.amp


@dataclass
class InferenceConfig:
    acceleration: int = 5                       # undersampling factor
    mask_type: str = "random"                   # "random" or "radial"
    gamma_t: float = 1.0                        # annealing term (Levac et al. Eq. 13-14)
    sigma_noise: float = 0.01                   # MRI noise std dev (Levac et al. Section III-C)
    langevin_steps: int = 100                   # steps for annealed Langevin dynamics
    eta_langevin: float = 0.01                  # Langevin step size


@dataclass
class DataConfig:
    image_size: int = 256
    num_contrasts: int = 2
    dataset: str = "CC359"                      # "CC359", "IXI", or "SKMTEA"
    train_split: float = 0.9
    normalize_per_volume: bool = True
    center_fraction: float = 0.08              # ACS region fraction for masks


@dataclass
class PathsConfig:
    data_root: str = "/data/mri"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    output_dir: str = "./outputs"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


# ---------------------------------------------------------------------------
# YAML loader with nested dict → dataclass conversion
# ---------------------------------------------------------------------------

def _dict_to_dataclass(cls, d: dict):
    """Recursively convert a nested dict to the target dataclass."""
    if d is None:
        return cls()
    kwargs = {}
    for f in cls.__dataclass_fields__:
        if f not in d:
            continue
        val = d[f]
        field_type = cls.__dataclass_fields__[f].type
        # Handle nested dataclasses
        if hasattr(field_type, "__dataclass_fields__") and isinstance(val, dict):
            kwargs[f] = _dict_to_dataclass(field_type, val)
        else:
            kwargs[f] = val
    return cls(**kwargs)


def _apply_override(config: Config, key_path: str, value: str) -> None:
    """Apply a CLI override like 'model.base_channels=128'."""
    parts = key_path.split(".")
    obj = config
    for part in parts[:-1]:
        obj = getattr(obj, part)
    attr = parts[-1]
    # Infer type from existing attribute
    existing = getattr(obj, attr)
    if isinstance(existing, bool):
        parsed = value.lower() in ("true", "1", "yes")
    elif isinstance(existing, int):
        parsed = int(value)
    elif isinstance(existing, float):
        parsed = float(value)
    elif isinstance(existing, list):
        parsed = [int(x.strip()) for x in value.strip("[]").split(",")]
    else:
        parsed = value
    setattr(obj, attr, parsed)


def load_config(path: str, overrides: List[str] | None = None) -> Config:
    """
    Load config from a YAML file, then apply optional CLI overrides.

    Args:
        path: Path to the YAML config file.
        overrides: List of 'key.path=value' strings, e.g. ['model.base_channels=128'].

    Returns:
        Populated Config instance.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    config = Config()

    if raw:
        for section, cls_map in [
            ("model", ModelConfig),
            ("diffusion", DiffusionConfig),
            ("training", TrainingConfig),
            ("inference", InferenceConfig),
            ("data", DataConfig),
            ("paths", PathsConfig),
        ]:
            if section in raw:
                # Handle nested sub-configs manually for model section
                if section == "model":
                    model_dict = dict(raw["model"])
                    if "disco" in model_dict:
                        model_dict["disco"] = _dict_to_dataclass(DISCOConfig, model_dict["disco"])
                    if "udno_k" in model_dict:
                        model_dict["udno_k"] = _dict_to_dataclass(UDNOKConfig, model_dict["udno_k"])
                    # channel_mults may come in as list
                    obj = ModelConfig(**{k: v for k, v in model_dict.items()
                                        if k in ModelConfig.__dataclass_fields__})
                    setattr(config, section, obj)
                else:
                    setattr(config, section,
                            _dict_to_dataclass(cls_map, raw[section]))

    if overrides:
        for override in overrides:
            if "=" not in override:
                print(f"[Config] Skipping malformed override: {override}", file=sys.stderr)
                continue
            key, val = override.split("=", 1)
            _apply_override(config, key.strip(), val.strip())

    return config
