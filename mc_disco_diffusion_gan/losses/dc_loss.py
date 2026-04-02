"""
Data Consistency (DC) loss for the Deep Generative Prior adaptation stage.

Implements the first term of FDMR Eq. 12:
    L_DC = ||M · (F · G_θ(x_0, z, 1) - y)||_2^2

where:
    - F  is the 2D Fourier transform
    - M  is the binary undersampling mask
    - y  is the observed undersampled k-space
    - G_θ(x_0, z, 1) is the generator output (x̂_0)
"""

from __future__ import annotations

import torch

from data.transforms import fft2c_4ch, split_4channel


def compute_dc_loss(
    x0_pred: torch.Tensor,
    y_obs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Data consistency loss: MSE between predicted and observed k-space at sampled locations.

    Implements FDMR Eq. 12 (first term):
        L_DC = Σ_i ||M · (F · x̂_0_i - y_i)||_2^2 / |M|

    Summed over both contrasts, normalized by the number of observed k-space entries.

    Args:
        x0_pred: Predicted clean image, shape [B, 4, H, W]
                 Channels: [real_c1, imag_c1, real_c2, imag_c2]
        y_obs:   Observed undersampled k-space, shape [B, 4, H, W]
                 (zeros at unobserved locations)
        mask:    Binary undersampling mask, shape [1, 1, H, W] or [B, 1, H, W]
                 (broadcast over channel dim)

    Returns:
        Scalar DC loss.
    """
    # Transform predicted image to k-space
    k_pred = fft2c_4ch(x0_pred)  # [B, 4, H, W]

    # Residual at observed locations: M · (F·x̂_0 - y)
    # mask shape [1/B, 1, H, W] broadcasts over all 4 channels
    residual = mask * (k_pred - y_obs)  # [B, 4, H, W]

    # MSE normalized by number of observed entries × channels
    num_observed = mask.sum() * x0_pred.shape[1]  # |M| × C
    loss = (residual.abs() ** 2).sum() / (num_observed + 1e-8)

    return loss


def compute_dc_loss_per_contrast(
    x0_pred: torch.Tensor,
    y_obs: torch.Tensor,
    mask: torch.Tensor,
) -> tuple:
    """
    Compute DC loss separately for each contrast (for logging/debugging).

    Returns:
        (loss_c1, loss_c2): DC losses for contrast 1 and contrast 2
    """
    k_pred = fft2c_4ch(x0_pred)

    # Split into per-contrast tensors
    k_c1 = k_pred[:, :2]
    k_c2 = k_pred[:, 2:]
    y_c1 = y_obs[:, :2]
    y_c2 = y_obs[:, 2:]

    num_obs = mask.sum() * 2  # 2 channels per contrast
    loss_c1 = (mask * (k_c1 - y_c1)).abs().pow(2).sum() / (num_obs + 1e-8)
    loss_c2 = (mask * (k_c2 - y_c2)).abs().pow(2).sum() / (num_obs + 1e-8)

    return loss_c1, loss_c2
