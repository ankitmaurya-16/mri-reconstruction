"""
MRI data transforms: Fourier transforms, complex/real conversions, normalization.

All FFT operations use torch.fft with norm='ortho' for energy conservation
and fftshift/ifftshift for centered k-space convention (DC at center),
matching the MRI scanner acquisition convention.

Convention used throughout:
  - Real representation: Tensor of shape [..., 2, H, W] where dim -3 = [real, imag]
    or [..., 4, H, W] for 2-contrast = [real_c1, imag_c1, real_c2, imag_c2]
  - Complex representation: Tensor of shape [..., H, W] with dtype complex64/complex128
  - k-space: always in centered Fourier convention (fftshift applied)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Complex ↔ real conversions
# ---------------------------------------------------------------------------

def complex_to_real(x: torch.Tensor) -> torch.Tensor:
    """
    Convert complex tensor to real 2-channel representation.

    Args:
        x: Complex tensor, shape [..., H, W] (dtype: complex)

    Returns:
        Real tensor, shape [..., 2, H, W] where dim -3 = (real, imag)
    """
    return torch.stack([x.real, x.imag], dim=-3)


def real_to_complex(x: torch.Tensor) -> torch.Tensor:
    """
    Convert real 2-channel tensor to complex tensor.

    Args:
        x: Real tensor, shape [..., 2, H, W] where dim -3 = (real, imag)

    Returns:
        Complex tensor, shape [..., H, W]
    """
    return torch.complex(x[..., 0, :, :], x[..., 1, :, :])


def pair_to_4channel(
    x1: torch.Tensor, x2: torch.Tensor
) -> torch.Tensor:
    """
    Stack two 2-channel images (one per contrast) into a 4-channel tensor.

    Args:
        x1: Contrast 1 real representation, shape [B, 2, H, W]
        x2: Contrast 2 real representation, shape [B, 2, H, W]

    Returns:
        4-channel tensor [B, 4, H, W] = [real_c1, imag_c1, real_c2, imag_c2]
    """
    return torch.cat([x1, x2], dim=1)


def split_4channel(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split a 4-channel multi-contrast tensor into two 2-channel tensors.

    Args:
        x: 4-channel tensor, shape [B, 4, H, W]

    Returns:
        (x1, x2): each shape [B, 2, H, W]
    """
    return x[:, :2], x[:, 2:]


# ---------------------------------------------------------------------------
# Fourier transforms (centered k-space convention)
# ---------------------------------------------------------------------------

def fft2c(x_real: torch.Tensor) -> torch.Tensor:
    """
    Centered 2D FFT: image domain → k-space.

    Applies ifftshift before FFT and fftshift after to place the DC component
    at the center of k-space (scanner convention).

    Args:
        x_real: Real 2-channel image, shape [B, 2, H, W]

    Returns:
        k-space in real 2-channel format, shape [B, 2, H, W]
    """
    x_cplx = real_to_complex(x_real)                          # [B, H, W] complex
    x_shifted = torch.fft.ifftshift(x_cplx, dim=(-2, -1))
    k_cplx = torch.fft.fft2(x_shifted, norm="ortho")
    k_centered = torch.fft.fftshift(k_cplx, dim=(-2, -1))
    return complex_to_real(k_centered)                         # [B, 2, H, W]


def ifft2c(k_real: torch.Tensor) -> torch.Tensor:
    """
    Centered 2D IFFT: k-space → image domain.

    Args:
        k_real: k-space real 2-channel, shape [B, 2, H, W]

    Returns:
        Image in real 2-channel format, shape [B, 2, H, W]
    """
    k_cplx = real_to_complex(k_real)                          # [B, H, W] complex
    k_shifted = torch.fft.ifftshift(k_cplx, dim=(-2, -1))
    x_cplx = torch.fft.ifft2(k_shifted, norm="ortho")
    x_centered = torch.fft.fftshift(x_cplx, dim=(-2, -1))
    return complex_to_real(x_centered)                         # [B, 2, H, W]


def fft2c_4ch(x: torch.Tensor) -> torch.Tensor:
    """
    Apply centered 2D FFT to a 4-channel (2-contrast) tensor.

    Args:
        x: 4-channel image [B, 4, H, W] = [real_c1, imag_c1, real_c2, imag_c2]

    Returns:
        4-channel k-space [B, 4, H, W]
    """
    x1, x2 = split_4channel(x)
    return pair_to_4channel(fft2c(x1), fft2c(x2))


def ifft2c_4ch(k: torch.Tensor) -> torch.Tensor:
    """
    Apply centered 2D IFFT to a 4-channel (2-contrast) k-space tensor.

    Args:
        k: 4-channel k-space [B, 4, H, W]

    Returns:
        4-channel image [B, 4, H, W]
    """
    k1, k2 = split_4channel(k)
    return pair_to_4channel(ifft2c(k1), ifft2c(k2))


# ---------------------------------------------------------------------------
# Forward MRI measurement model
# ---------------------------------------------------------------------------

def apply_mask_kspace(
    x0_real: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate MRI undersampling: transform to k-space and apply mask.

    This implements the forward measurement model (Paper 3 Eq. 2):
        y_i = F_i · x_i + ε_i
    where F_i = M · F (undersampled Fourier operator).

    Args:
        x0_real: Fully sampled image, shape [B, 4, H, W] (4-channel multi-contrast)
        mask:    Binary undersampling mask, shape [1, 1, H, W] (broadcast over B and C)

    Returns:
        y_obs:    Observed undersampled k-space, shape [B, 4, H, W]
                  Zeros at unobserved locations.
        k_full:   Fully sampled k-space (for DC loss reference), shape [B, 4, H, W]
    """
    k_full = fft2c_4ch(x0_real)                # [B, 4, H, W]
    y_obs = k_full * mask                       # zero at unobserved k-space points
    return y_obs, k_full


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_volume(
    x: torch.Tensor, percentile: float = 99.0
) -> Tuple[torch.Tensor, float]:
    """
    Normalize a volume by its p-th percentile magnitude.

    Paper 3 (Levac et al., Section III-C): data pairs (x1, x2) are normalized
    jointly by the 99th percentile value of the reference magnitude image x1.

    Args:
        x:          Image tensor, shape [B, C, H, W] or [C, H, W]
        percentile: Normalization percentile (default 99)

    Returns:
        x_norm: Normalized image (same shape)
        scale:  The scale factor (for de-normalization)
    """
    # Compute magnitude
    magnitude = torch.sqrt(x[..., 0::2, :, :] ** 2 + x[..., 1::2, :, :] ** 2)
    # Flatten and compute percentile
    flat = magnitude.reshape(-1)
    k = max(1, int(len(flat) * percentile / 100.0))
    scale = float(torch.kthvalue(flat, k).values.item())
    scale = max(scale, 1e-8)  # avoid division by zero

    return x / scale, scale


def complex_magnitude(x_real: torch.Tensor) -> torch.Tensor:
    """
    Compute per-contrast magnitude image from real-valued 4-channel input.

    Args:
        x_real: shape [B, 4, H, W] = [real_c1, imag_c1, real_c2, imag_c2]

    Returns:
        magnitude: shape [B, 2, H, W] = [|c1|, |c2|]
    """
    real_c1, imag_c1 = x_real[:, 0], x_real[:, 1]
    real_c2, imag_c2 = x_real[:, 2], x_real[:, 3]

    mag_c1 = torch.sqrt(real_c1 ** 2 + imag_c1 ** 2 + 1e-8)
    mag_c2 = torch.sqrt(real_c2 ** 2 + imag_c2 ** 2 + 1e-8)

    return torch.stack([mag_c1, mag_c2], dim=1)


def zero_fill(y_obs: torch.Tensor) -> torch.Tensor:
    """
    Zero-fill reconstruction (baseline): IFFT of undersampled k-space.

    Args:
        y_obs: Undersampled k-space, shape [B, 4, H, W] (zeros at unobserved)

    Returns:
        x_zf: Zero-filled image, shape [B, 4, H, W]
    """
    return ifft2c_4ch(y_obs)
