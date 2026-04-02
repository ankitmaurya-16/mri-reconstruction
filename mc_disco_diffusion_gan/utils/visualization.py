"""
Visualization utilities for MRI reconstruction comparisons.

Generates side-by-side comparison figures showing:
  [GT | ZF | Stage1 | Stage2 | Stage3]  — for each contrast
  [Error maps for each method]

Matches the visual comparison style in FDMR (Zhao et al., Figs. 3–5)
and Paper 2 (Jatyani et al., Fig. 4).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/training environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

from data.transforms import complex_magnitude


def tensor_to_magnitude_numpy(x: torch.Tensor, contrast_idx: int = 0) -> np.ndarray:
    """
    Extract a single contrast magnitude image as a numpy array.

    Args:
        x:            4-channel tensor [4, H, W] or [B, 4, H, W]
        contrast_idx: 0 for contrast 1, 1 for contrast 2

    Returns:
        Magnitude image as float32 numpy array [H, W], normalized to [0,1].
    """
    if x.ndim == 4:
        x = x[0]  # take first item in batch

    # complex_magnitude expects [B, 4, H, W]; add batch dim
    mag = complex_magnitude(x.unsqueeze(0))  # [1, 2, H, W]
    img = mag[0, contrast_idx].detach().cpu().float().numpy()

    # Normalize to [0, 1] for display
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 1e-8:
        img = (img - img_min) / (img_max - img_min)
    return img


def compute_error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Compute absolute error map between predicted and ground truth magnitude.

    Args:
        pred: Predicted magnitude [H, W], values in [0, 1]
        gt:   Ground truth magnitude [H, W], values in [0, 1]

    Returns:
        Error map [H, W], absolute difference.
    """
    return np.abs(gt - pred)


def save_comparison_figure(
    x0_gt: torch.Tensor,
    x_zf: torch.Tensor,
    results: Dict[str, torch.Tensor],
    output_dir: str,
    idx: int,
    contrast_names: Optional[List[str]] = None,
    dpi: int = 150,
    error_scale: float = 5.0,
) -> str:
    """
    Save a multi-panel comparison figure showing all stages and error maps.

    Panel layout (per contrast):
        GT | ZF | Stage1 | Stage2 | Stage3
        -- | Error_ZF | Error_S1 | Error_S2 | Error_S3

    Args:
        x0_gt:      Ground truth, shape [B, 4, H, W] or [4, H, W]
        x_zf:       Zero-fill baseline, shape [B, 4, H, W] or [4, H, W]
        results:    Dict with keys "stage1", "stage2", "stage3", each [B, 4, H, W]
        output_dir: Directory to save the figure.
        idx:        Sample index (used in filename).
        contrast_names: Labels for each contrast (default: ["C1 (T1)", "C2 (T2)"])
        dpi:        Figure DPI.
        error_scale: Multiply error maps by this factor for visibility.

    Returns:
        Path to the saved figure.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if contrast_names is None:
        contrast_names = ["C1 (T1)", "C2 (T2)"]

    n_contrasts = 2
    # Columns: GT, ZF, Stage1, Stage2, Stage3
    col_labels = ["Ground Truth", "Zero-Fill", "Stage 1", "Stage 2 (DGP)", "Stage 3 (Final)"]
    n_cols = len(col_labels)
    n_rows = n_contrasts * 2  # image row + error row per contrast

    fig = plt.figure(figsize=(n_cols * 2.5, n_rows * 2.5))
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.05, wspace=0.05)

    images = {
        "Ground Truth": x0_gt,
        "Zero-Fill":    x_zf,
        "Stage 1":      results.get("stage1", x_zf),
        "Stage 2 (DGP)": results.get("stage2", x_zf),
        "Stage 3 (Final)": results.get("stage3", x_zf),
    }

    for c_idx in range(n_contrasts):
        row_img = c_idx * 2       # image row
        row_err = c_idx * 2 + 1  # error row

        # Extract GT magnitude for error computation
        gt_img = tensor_to_magnitude_numpy(x0_gt, c_idx)

        for col_idx, (label, col_label) in enumerate(zip(images.keys(), col_labels)):
            img = tensor_to_magnitude_numpy(images[label], c_idx)

            # Image panel
            ax_img = fig.add_subplot(gs[row_img, col_idx])
            ax_img.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax_img.axis("off")

            if c_idx == 0:
                ax_img.set_title(col_label, fontsize=8, pad=3)
            if col_idx == 0:
                ax_img.set_ylabel(contrast_names[c_idx], fontsize=8)

            # Error map panel (skip GT column — show blank)
            ax_err = fig.add_subplot(gs[row_err, col_idx])
            if col_idx == 0:
                ax_err.axis("off")
            else:
                err = compute_error_map(img, gt_img) * error_scale
                ax_err.imshow(err, cmap="hot", vmin=0.0, vmax=1.0)
                ax_err.axis("off")

    # Colorbar for error maps
    cbar_ax = fig.add_axes([0.92, 0.05, 0.015, 0.4])
    sm = plt.cm.ScalarMappable(cmap="hot", norm=plt.Normalize(0, 1.0 / error_scale))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Error")

    save_path = os.path.join(output_dir, f"comparison_{idx:05d}.png")
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)

    return save_path


def save_loss_curves(
    train_losses: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    output_dir: str,
    filename: str = "training_curves.png",
) -> str:
    """
    Save training loss curves and validation metric curves.

    Args:
        train_losses: Dict of lists, e.g. {"G_total": [...], "D": [...], "G_adv": [...]}
        val_metrics:  Dict of lists, e.g. {"psnr_mean": [...], "ssim_mean": [...]}
        output_dir:   Directory to save the figure.
        filename:     Output filename.

    Returns:
        Path to the saved figure.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_loss = len(train_losses)
    n_metric = len(val_metrics)
    n_total = n_loss + n_metric

    fig, axes = plt.subplots(1, n_total, figsize=(5 * n_total, 4))
    if n_total == 1:
        axes = [axes]

    ax_idx = 0
    for name, vals in train_losses.items():
        axes[ax_idx].plot(vals, label=name, color="steelblue")
        axes[ax_idx].set_title(f"Train: {name}")
        axes[ax_idx].set_xlabel("Iteration")
        axes[ax_idx].set_ylabel("Loss")
        axes[ax_idx].legend()
        ax_idx += 1

    for name, vals in val_metrics.items():
        axes[ax_idx].plot(vals, label=name, color="coral")
        axes[ax_idx].set_title(f"Val: {name}")
        axes[ax_idx].set_xlabel("Epoch")
        axes[ax_idx].legend()
        ax_idx += 1

    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return save_path
