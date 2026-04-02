"""
Checkpoint save/load utilities.

Saves and restores all model states needed for training resumption and inference:
  - Main generator (G_θ)
  - Refinement generator (G_refine)
  - Discriminator (D_φ)
  - Optimizer states (for training resumption)
  - Current epoch and best metrics
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    generator: nn.Module,
    refinement_gen: nn.Module,
    discriminator: nn.Module,
    opt_G: torch.optim.Optimizer,
    opt_G_refine: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    metrics: Dict[str, float],
    is_best: bool = False,
    filename: Optional[str] = None,
) -> str:
    """
    Save a complete training checkpoint.

    Args:
        checkpoint_dir: Directory to save checkpoints in.
        epoch:          Current epoch number.
        generator:      Main GAN generator.
        refinement_gen: Refinement generator.
        discriminator:  Time-conditioned discriminator.
        opt_G:          Generator optimizer.
        opt_G_refine:   Refinement generator optimizer.
        opt_D:          Discriminator optimizer.
        metrics:        Dict of metric values (e.g. {"psnr_mean": 37.5, ...}).
        is_best:        If True, also saves as "best_model.pth".
        filename:       Custom filename; defaults to "checkpoint_epoch_{epoch:04d}.pth".

    Returns:
        Path to the saved checkpoint file.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"checkpoint_epoch_{epoch:04d}.pth"

    save_path = os.path.join(checkpoint_dir, filename)

    state = {
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "refinement_gen_state_dict": refinement_gen.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "opt_G_state_dict": opt_G.state_dict(),
        "opt_G_refine_state_dict": opt_G_refine.state_dict(),
        "opt_D_state_dict": opt_D.state_dict(),
        "metrics": metrics,
    }

    torch.save(state, save_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_path)
        print(f"[Checkpoint] Saved best model at epoch {epoch} → {best_path}")

    print(f"[Checkpoint] Saved epoch {epoch} → {save_path}")
    return save_path


def load_checkpoint(
    checkpoint_path: str,
    generator: nn.Module,
    refinement_gen: nn.Module,
    discriminator: Optional[nn.Module] = None,
    opt_G: Optional[torch.optim.Optimizer] = None,
    opt_G_refine: Optional[torch.optim.Optimizer] = None,
    opt_D: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a training checkpoint and restore model/optimizer states.

    Supports partial loading (inference-only: pass None for optimizers/discriminator).

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        generator:       Generator module (mutated in place).
        refinement_gen:  Refinement generator module (mutated in place).
        discriminator:   Optional discriminator (mutated in place).
        opt_G:           Optional generator optimizer (mutated in place).
        opt_G_refine:    Optional refinement optimizer (mutated in place).
        opt_D:           Optional discriminator optimizer (mutated in place).
        device:          Device to load tensors to.

    Returns:
        Dict containing: {"epoch", "metrics"}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(checkpoint_path, map_location=device)

    generator.load_state_dict(state["generator_state_dict"])
    refinement_gen.load_state_dict(state["refinement_gen_state_dict"])

    if discriminator is not None and "discriminator_state_dict" in state:
        discriminator.load_state_dict(state["discriminator_state_dict"])

    if opt_G is not None and "opt_G_state_dict" in state:
        opt_G.load_state_dict(state["opt_G_state_dict"])

    if opt_G_refine is not None and "opt_G_refine_state_dict" in state:
        opt_G_refine.load_state_dict(state["opt_G_refine_state_dict"])

    if opt_D is not None and "opt_D_state_dict" in state:
        opt_D.load_state_dict(state["opt_D_state_dict"])

    epoch = state.get("epoch", 0)
    metrics = state.get("metrics", {})

    print(f"[Checkpoint] Loaded epoch {epoch} from {checkpoint_path}")
    if metrics:
        psnr = metrics.get("psnr_mean", 0.0)
        ssim = metrics.get("ssim_mean", 0.0)
        print(f"[Checkpoint] Saved metrics — PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")

    return {"epoch": epoch, "metrics": metrics}


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the most recently modified checkpoint file in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoint .pth files.

    Returns:
        Path to the latest checkpoint, or None if directory is empty/missing.
    """
    cp_dir = Path(checkpoint_dir)
    if not cp_dir.exists():
        return None

    checkpoints = sorted(
        cp_dir.glob("checkpoint_epoch_*.pth"),
        key=lambda p: p.stat().st_mtime,
    )
    return str(checkpoints[-1]) if checkpoints else None
