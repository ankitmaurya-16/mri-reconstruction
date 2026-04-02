"""
Time-Conditioned Patch Discriminator for the DISCO-Diffusion GAN.

Implements D_φ(x_{t-1}, x_t, t) from FDMR (Zhao et al., Section 4.1, Eq. 9).

The discriminator:
  - Takes a pair of images (x_{t-1}, x_t) concatenated along channel dim → 8 channels
    (2 × 4-channel multi-contrast images)
  - Conditions on the diffusion timestep t via sinusoidal embedding
  - Uses spectral normalization for stable GAN training
  - Outputs patch-level real/fake scores (PatchGAN style)

Architecture is a 4-level convolutional encoder with:
  - Spectral-normalized DISCO convolutions
  - Time embedding injection at each level
  - Final 1×1 convolution to scalar logit map
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .disco import DISCOConv2d
from .udno import sinusoidal_embedding


class SpectralDISCOConv2d(nn.Module):
    """
    DISCOConv2d with spectral normalization applied to the theta parameter.

    Spectral normalization constrains the Lipschitz constant of the discriminator,
    improving GAN training stability (standard practice in Diffusion-GAN, Xiao et al. 2022).

    The spectral norm is applied via PyTorch's nn.utils.spectral_norm on a thin
    wrapper module that exposes theta as its 'weight' attribute.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_basis: int = 8,
        stride: int = 1,
        padding: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.disco = DISCOConv2d(
            in_channels, out_channels, kernel_size, num_basis,
            stride=stride, padding=padding
        )
        # Apply spectral norm to the learnable theta parameter
        nn.utils.parametrize.register_parametrization(
            self.disco, "theta", _SpectralNormTheta(self.disco.theta.shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.disco(x)


class _SpectralNormTheta(nn.Module):
    """Spectral normalization parametrization for the theta parameter of DISCOConv2d."""

    def __init__(self, shape: tuple) -> None:
        super().__init__()
        # u and v vectors for power iteration (shape of the flattened matrix)
        out_ch, in_ch, L = shape
        self.register_buffer("u", F.normalize(torch.randn(out_ch), dim=0))
        self.register_buffer("v", F.normalize(torch.randn(in_ch * L), dim=0))

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        out_ch, in_ch, L = theta.shape
        W = theta.reshape(out_ch, in_ch * L)  # flatten for SVD approximation
        # One step of power iteration
        u = F.normalize(W @ self.v, dim=0)
        v = F.normalize(W.T @ u, dim=0)
        sigma = u @ W @ v
        self.u.copy_(u.detach())
        self.v.copy_(v.detach())
        return theta / sigma


class DiscriminatorBlock(nn.Module):
    """
    Single level of the patch discriminator.

    Structure:
        [SpectralDISCOConv2d → InstanceNorm → LeakyReLU] + time_emb injection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_basis: int,
        time_emb_dim: int,
        stride: int = 2,
    ) -> None:
        super().__init__()

        self.conv = SpectralDISCOConv2d(
            in_channels, out_channels, kernel_size=4,
            num_basis=num_basis, stride=stride, padding=1
        )
        # InstanceNorm works well for discriminators (no batch-size dependency)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)

        # Time embedding injection
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor
    ) -> torch.Tensor:
        h = F.leaky_relu(self.norm(self.conv(x)), negative_slope=0.2)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        return h


class TimeConditionedDiscriminator(nn.Module):
    """
    Time-Conditioned PatchGAN Discriminator D_φ(x_{t-1}, x_t, t).

    Input:
        x_prev: x_{t-1} image, shape [B, 4, H, W]  (multi-contrast)
        x_curr: x_t image,   shape [B, 4, H, W]
        t:      Timestep (1-indexed), shape [B]

    Output:
        Logit map, shape [B, 1, H', W']  (real/fake scores per patch)

    Implements FDMR Eq. 9:
        D_φ(x_{t-1}, x_t, t) → discriminate whether x_{t-1} is a real
        denoised sample from q(x_{t-1}|x_t) or a GAN-generated fake.

    Args:
        in_channels:    Channels per image (4 for multi-contrast). The
                        discriminator concatenates x_{t-1} and x_t → 8 total.
        base_channels:  Width of the first discriminator level.
        num_basis:      DISCO basis functions L.
        time_emb_dim:   Sinusoidal time embedding dimension.
        num_levels:     Number of downsampling levels (4 recommended).
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        num_basis: int = 8,
        time_emb_dim: int = 128,
        num_levels: int = 4,
    ) -> None:
        super().__init__()

        self.time_emb_dim = time_emb_dim

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Initial projection from 2*in_channels (concatenated pair) to base_channels
        # No norm or time emb at the first layer (standard PatchGAN practice)
        self.input_conv = SpectralDISCOConv2d(
            2 * in_channels, base_channels,
            kernel_size=4, num_basis=num_basis, stride=2, padding=1
        )

        # Downsampling blocks with time conditioning
        blocks = []
        ch = base_channels
        for i in range(1, num_levels):
            out_ch = min(base_channels * (2 ** i), 512)
            stride = 2 if i < num_levels - 1 else 1  # last level: stride 1
            blocks.append(
                DiscriminatorBlock(ch, out_ch, num_basis, time_emb_dim, stride=stride)
            )
            ch = out_ch
        self.blocks = nn.ModuleList(blocks)

        # Final 1×1 convolution to scalar logit map
        self.output_conv = SpectralDISCOConv2d(
            ch, 1, kernel_size=4, num_basis=num_basis, stride=1, padding=1
        )

    def forward(
        self,
        x_prev: torch.Tensor,
        x_curr: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_prev: Previous timestep image (real or fake), shape [B, 4, H, W]
            x_curr: Current noisy image x_t, shape [B, 4, H, W]
            t:      Diffusion timestep (1-indexed), shape [B]

        Returns:
            logits: Patch real/fake scores, shape [B, 1, H', W']
        """
        # Build time embedding
        t_emb_raw = sinusoidal_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb_raw)  # [B, time_emb_dim]

        # Concatenate pair along channel dimension
        x = torch.cat([x_prev, x_curr], dim=1)  # [B, 8, H, W]

        # Input conv (no time emb, no norm — standard PatchGAN first layer)
        h = F.leaky_relu(self.input_conv(x), negative_slope=0.2)

        # Downsampling blocks with time conditioning
        for block in self.blocks:
            h = block(h, t_emb)

        # Final logit map
        return self.output_conv(h)
