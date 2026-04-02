"""
Total Variation (TV) regularization loss.

Implements the second term of FDMR Eq. 12 for the DGP adaptation stage:
    L_TV = λ · ||G_θ(x_0, z, 1)||_TV

Uses isotropic TV on the magnitude image per contrast:
    ||x||_TV = Σ_{i,j} √((x_{i+1,j} - x_{i,j})^2 + (x_{i,j+1} - x_{i,j})^2 + ε)

Applied to the per-contrast magnitude images (|real + j·imag|) since TV
regularization on magnitudes is the standard in MRI reconstruction.
"""

from __future__ import annotations

import torch

from data.transforms import complex_magnitude


def tv_loss(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Isotropic Total Variation loss on the magnitude image.

    Applies TV to the magnitude of each contrast independently, then sums.

    L_TV = Σ_c Σ_{i,j} √((|x_c|_{i+1,j} - |x_c|_{i,j})² + (|x_c|_{i,j+1} - |x_c|_{i,j})² + ε)

    Args:
        x:   4-channel image [B, 4, H, W] = [real_c1, imag_c1, real_c2, imag_c2]
        eps: Small constant for numerical stability in the square root

    Returns:
        Scalar mean TV loss over the batch.
    """
    # Compute magnitude per contrast: [B, 2, H, W]
    mag = complex_magnitude(x)  # [B, 2, H, W]: [|c1|, |c2|]

    # Horizontal differences (along W)
    diff_h = mag[:, :, :, 1:] - mag[:, :, :, :-1]   # [B, 2, H, W-1]
    # Vertical differences (along H)
    diff_v = mag[:, :, 1:, :] - mag[:, :, :-1, :]   # [B, 2, H-1, W]

    # Isotropic TV: need same spatial size; crop to min
    H_min = min(diff_h.shape[2], diff_v.shape[2])
    W_min = min(diff_h.shape[3], diff_v.shape[3])

    diff_h = diff_h[:, :, :H_min, :W_min]
    diff_v = diff_v[:, :, :H_min, :W_min]

    # Isotropic norm per pixel: √(dh² + dv² + ε)
    tv_map = torch.sqrt(diff_h ** 2 + diff_v ** 2 + eps)

    # Mean over all spatial locations, contrasts, and batch
    return tv_map.mean()


def tv_loss_anisotropic(x: torch.Tensor) -> torch.Tensor:
    """
    Anisotropic (L1) Total Variation loss on the magnitude image.

    L_TV_aniso = Σ_c Σ_{i,j} |dh| + |dv|

    Sometimes preferred for sharper edges at the cost of staircase artifacts.

    Args:
        x:   4-channel image [B, 4, H, W]

    Returns:
        Scalar mean TV loss.
    """
    mag = complex_magnitude(x)  # [B, 2, H, W]
    diff_h = (mag[:, :, :, 1:] - mag[:, :, :, :-1]).abs()
    diff_v = (mag[:, :, 1:, :] - mag[:, :, :-1, :]).abs()
    return diff_h.mean() + diff_v.mean()
