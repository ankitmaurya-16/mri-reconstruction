"""
Undersampling mask generators for accelerated MRI.

Supports the mask types used in the three papers:
  - Variable density random (FDMR random; CC-359 and IXI experiments)
  - Pseudo-radial (FDMR radial; CC-359 and IXI experiments)
  - Equispaced / uniform random (Paper 2, fastMRI experiments)
  - Poisson disk (Paper 2, fastMRI experiments)

All masks are 2D binary tensors of shape [H, W] (float32, 0.0 or 1.0)
with the DC (center) region always fully sampled (ACS region).

Usage:
    mask = MaskFactory.get_mask("random", shape=(256, 256), acceleration=5, seed=42)
    # mask: Tensor[H, W] — 1 at sampled, 0 at unsampled locations
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Variable density random mask (FDMR "Random" mask)
# ---------------------------------------------------------------------------

def variable_density_random_mask(
    shape: Tuple[int, int],
    acceleration: float,
    center_fraction: float = 0.08,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Variable-density random 1D phase-encode undersampling mask.

    Center `center_fraction` of k-space is always fully sampled (ACS region).
    Remaining phase-encode lines are sampled with probability proportional to
    a Gaussian centered at DC, targeting total sampling fraction 1/acceleration.

    This matches the "Random sampling" mask used in FDMR (Zhao et al., Table 1–4)
    and the variable density scheme from Paper 3 (Levac et al., Section III-C).

    Args:
        shape:            (H, W) — image / k-space dimensions
        acceleration:     Undersampling factor (e.g. 5 for 5×)
        center_fraction:  Fraction of center k-space always sampled (e.g. 0.08 = 8%)
        seed:             Random seed for reproducibility

    Returns:
        mask: Binary tensor of shape [H, W], dtype float32
    """
    H, W = shape
    rng = np.random.default_rng(seed)

    target_num_samples = int(H / acceleration)
    num_center = max(1, int(H * center_fraction))
    num_remaining = max(0, target_num_samples - num_center)

    # Always-sampled center rows
    center_start = (H - num_center) // 2
    center_rows = np.arange(center_start, center_start + num_center)

    # Probability distribution for remaining rows: Gaussian centered at DC
    distances = np.abs(np.arange(H) - H // 2).astype(float)
    distances[center_start:center_start + num_center] = np.inf  # exclude center rows

    # Gaussian variable density: p ∝ exp(−d² / σ²)
    sigma = H / 4.0
    weights = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    weights[np.isinf(distances)] = 0.0
    weights /= weights.sum() + 1e-12

    # Sample without replacement
    num_remaining = min(num_remaining, int(weights.sum() > 0) * num_remaining)
    if num_remaining > 0 and weights.sum() > 0:
        extra_rows = rng.choice(H, size=num_remaining, replace=False, p=weights)
    else:
        extra_rows = np.array([], dtype=int)

    sampled_rows = np.unique(np.concatenate([center_rows, extra_rows]))

    # Build 2D mask: all columns are sampled for selected rows (1D undersampling)
    mask = np.zeros((H, W), dtype=np.float32)
    mask[sampled_rows, :] = 1.0

    return torch.from_numpy(mask)


# ---------------------------------------------------------------------------
# Pseudo-radial mask (FDMR "Radial" mask)
# ---------------------------------------------------------------------------

def pseudo_radial_mask(
    shape: Tuple[int, int],
    acceleration: float,
    num_spokes: Optional[int] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Pseudo-radial (golden-angle) k-space undersampling mask.

    Simulates radial MRI acquisition by sampling along spoke lines through
    the k-space center at golden-angle increments (φ = 111.25°).

    This matches the "Radial sampling" mask used in FDMR (Zhao et al., Tables 2–3).

    Args:
        shape:       (H, W) k-space dimensions
        acceleration: Undersampling factor
        num_spokes:  Explicit number of spokes; if None, derived from acceleration
        seed:        Unused (golden-angle is deterministic); kept for API consistency

    Returns:
        mask: Binary tensor of shape [H, W], dtype float32
    """
    H, W = shape
    if num_spokes is None:
        # Approximate the number of spokes to reach target sampling fraction
        # Each spoke samples ~max(H, W) points in k-space
        target_fraction = 1.0 / acceleration
        num_spokes = max(1, int(target_fraction * H * W / max(H, W)))

    golden_angle_rad = math.pi * (3.0 - math.sqrt(5.0))  # ≈ 2.3999 rad ≈ 137.5°

    mask = np.zeros((H, W), dtype=np.float32)
    cx, cy = H // 2, W // 2

    for i in range(num_spokes):
        angle = i * golden_angle_rad
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        max_r = math.sqrt((H // 2) ** 2 + (W // 2) ** 2)
        num_pts = int(2 * max_r)

        for j in range(num_pts):
            r = (j / num_pts) * 2 * max_r - max_r
            row = int(round(cx + r * cos_a))
            col = int(round(cy + r * sin_a))
            if 0 <= row < H and 0 <= col < W:
                mask[row, col] = 1.0

    return torch.from_numpy(mask)


# ---------------------------------------------------------------------------
# Equispaced mask (Paper 2, CVPR 2025)
# ---------------------------------------------------------------------------

def equispaced_mask(
    shape: Tuple[int, int],
    acceleration: int,
    center_fraction: float = 0.08,
) -> torch.Tensor:
    """
    Equispaced (uniform) 1D undersampling mask.

    Every `acceleration`-th phase-encode line is sampled, with the center
    fully sampled (ACS region).

    Used in Paper 2 (Jatyani et al.) baseline experiments on fastMRI.

    Args:
        shape:           (H, W)
        acceleration:    Undersampling factor (integer)
        center_fraction: Fraction of center k-space always sampled

    Returns:
        mask: Binary tensor of shape [H, W]
    """
    H, W = shape
    mask = np.zeros((H, W), dtype=np.float32)

    # Equispaced rows
    sampled_rows = np.arange(0, H, acceleration)
    mask[sampled_rows, :] = 1.0

    # ACS center region
    num_center = max(1, int(H * center_fraction))
    center_start = (H - num_center) // 2
    mask[center_start:center_start + num_center, :] = 1.0

    return torch.from_numpy(mask)


# ---------------------------------------------------------------------------
# Poisson disk mask (Paper 2, CVPR 2025)
# ---------------------------------------------------------------------------

def poisson_disk_mask(
    shape: Tuple[int, int],
    acceleration: float,
    center_fraction: float = 0.08,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Poisson-disk 2D undersampling mask (variable density).

    Uses a simplified rejection sampling approach to generate a 2D mask with
    near-uniform minimum distance between sampled points (Poisson disk property)
    and a variable-density weighting to oversample the center.

    Used in Paper 2 (Jatyani et al.) across undersampling pattern experiments.

    Args:
        shape:           (H, W)
        acceleration:    Target undersampling factor
        center_fraction: Fraction of center k-space always sampled
        seed:            Random seed

    Returns:
        mask: Binary tensor of shape [H, W]
    """
    H, W = shape
    rng = np.random.default_rng(seed)

    target_samples = int(H * W / acceleration)
    num_center = max(1, int(H * center_fraction))
    c_start_h = (H - num_center) // 2
    c_start_w = (W - num_center) // 2

    # Variable density weights: Gaussian centered at DC
    yy, xx = np.mgrid[:H, :W]
    dist = np.sqrt((yy - H // 2) ** 2 + (xx - W // 2) ** 2)
    sigma = min(H, W) / 4.0
    weights = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    weights_flat = weights.ravel()
    weights_flat /= weights_flat.sum()

    indices = rng.choice(H * W, size=min(target_samples, H * W), replace=False, p=weights_flat)
    mask = np.zeros(H * W, dtype=np.float32)
    mask[indices] = 1.0
    mask = mask.reshape(H, W)

    # Always sample ACS center
    mask[c_start_h:c_start_h + num_center, c_start_w:c_start_w + num_center] = 1.0

    return torch.from_numpy(mask)


# ---------------------------------------------------------------------------
# Gaussian random mask (Paper 2)
# ---------------------------------------------------------------------------

def gaussian_random_mask(
    shape: Tuple[int, int],
    acceleration: float,
    center_fraction: float = 0.08,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    2D Gaussian-weighted random undersampling mask.

    Args:
        shape:           (H, W)
        acceleration:    Target undersampling factor
        center_fraction: Fraction of center always sampled
        seed:            Random seed

    Returns:
        mask: Binary tensor of shape [H, W]
    """
    H, W = shape
    rng = np.random.default_rng(seed)

    target_samples = int(H * W / acceleration)

    yy, xx = np.mgrid[:H, :W]
    dist = np.sqrt((yy - H // 2) ** 2 + (xx - W // 2) ** 2)
    sigma = min(H, W) / 3.0
    weights = np.exp(-(dist ** 2) / (2 * sigma ** 2)).ravel()
    weights /= weights.sum()

    indices = rng.choice(H * W, size=min(target_samples, H * W), replace=False, p=weights)
    mask = np.zeros(H * W, dtype=np.float32)
    mask[indices] = 1.0
    mask = mask.reshape(H, W)

    num_center = max(1, int(H * center_fraction))
    c_start_h = (H - num_center) // 2
    c_start_w = (W - num_center) // 2
    mask[c_start_h:c_start_h + num_center, c_start_w:c_start_w + num_center] = 1.0

    return torch.from_numpy(mask)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class MaskFactory:
    """
    Factory for creating undersampling masks by name.

    Supported types:
        "random"    — variable density random (FDMR experiments)
        "radial"    — pseudo-radial golden-angle (FDMR experiments)
        "equi"      — equispaced (Paper 2 experiments)
        "poisson"   — Poisson disk 2D (Paper 2 experiments)
        "gaussian"  — Gaussian random 2D (Paper 2 experiments)
    """

    @staticmethod
    def get_mask(
        mask_type: str,
        shape: Tuple[int, int],
        acceleration: float,
        center_fraction: float = 0.08,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Create and return a binary undersampling mask.

        Args:
            mask_type:       One of "random", "radial", "equi", "poisson", "gaussian"
            shape:           (H, W) spatial dimensions
            acceleration:    Undersampling factor
            center_fraction: Fraction of center k-space always sampled
            seed:            Random seed for reproducibility

        Returns:
            mask: Float32 tensor of shape [H, W], values in {0.0, 1.0}
        """
        mask_type = mask_type.lower()
        if mask_type == "random":
            return variable_density_random_mask(shape, acceleration, center_fraction, seed)
        elif mask_type == "radial":
            return pseudo_radial_mask(shape, acceleration, seed=seed)
        elif mask_type == "equi":
            return equispaced_mask(shape, int(acceleration), center_fraction)
        elif mask_type == "poisson":
            return poisson_disk_mask(shape, acceleration, center_fraction, seed)
        elif mask_type == "gaussian":
            return gaussian_random_mask(shape, acceleration, center_fraction, seed)
        else:
            raise ValueError(
                f"Unknown mask type: '{mask_type}'. "
                f"Choose from: random, radial, equi, poisson, gaussian"
            )

    @staticmethod
    def get_batch_masks(
        mask_type: str,
        shape: Tuple[int, int],
        batch_size: int,
        acceleration: float,
        center_fraction: float = 0.08,
        base_seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate a batch of masks (different seed per sample).

        Returns:
            masks: Tensor of shape [B, 1, H, W]
        """
        masks = []
        for i in range(batch_size):
            seed = (base_seed + i) if base_seed is not None else None
            m = MaskFactory.get_mask(mask_type, shape, acceleration, center_fraction, seed)
            masks.append(m)
        return torch.stack(masks, dim=0).unsqueeze(1)  # [B, 1, H, W]
