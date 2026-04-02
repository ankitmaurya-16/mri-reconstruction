"""
Multi-Contrast DISCO-Diffusion GAN Generator.

Implements G_θ(x_t, z, t) → x̂_0 from FDMR (Zhao et al., Eq. 7–8),
extended with:
  - DISCO/UDNO backbone (Paper 2) replacing standard convolutions
  - 4-channel multi-contrast input [real_c1, imag_c1, real_c2, imag_c2] (Paper 3)
  - Latent vector z injected at the U-Net bottleneck

The generator learns the joint score function s_θ(x_1, x_2) over paired
contrast images (Paper 3, Section III), realized through the 4-channel
architecture trained on paired multi-contrast data.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .disco import DISCOConv2d
from .udno import UDNO, sinusoidal_embedding, ResidualDISCOBlock
from utils.config import Config


class MCDISCOGenerator(nn.Module):
    """
    Multi-Contrast DISCO-Diffusion GAN Generator.

    Architecture:
        1. UDNO backbone (encoder + bottleneck + decoder) with DISCO convolutions
        2. Latent z injected at the bottleneck via a learnable MLP projection
        3. Sinusoidal time embedding t fed to every ResidualDISCOBlock
        4. 4-channel input/output for multi-contrast complex MRI

    Args:
        config: Global Config instance. Reads from config.model and config.diffusion.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()

        mc = config.model
        dc = mc.disco

        self.in_channels = mc.in_channels      # 4
        self.out_channels = mc.in_channels     # 4 (same — predict x̂_0)
        self.latent_dim = mc.latent_dim        # 64
        self.time_emb_dim = mc.time_emb_dim   # 128

        # Bottleneck channel size: base_channels * max(channel_mults)
        self.bottleneck_ch = mc.base_channels * max(mc.channel_mults)

        # ----------------------------------------------------------------
        # UDNO backbone (NO_i — image-space operator with time conditioning)
        # ----------------------------------------------------------------
        self.backbone = UDNO(
            in_channels=mc.in_channels,
            out_channels=mc.in_channels,
            base_channels=mc.base_channels,
            channel_mults=mc.channel_mults,
            num_res_blocks=mc.num_res_blocks,
            num_basis=dc.num_basis,
            kernel_size=dc.kernel_size,
            time_emb_dim=mc.time_emb_dim,
        )

        # ----------------------------------------------------------------
        # Latent z → bottleneck projection (FDMR Section 4.1)
        # Produces a feature offset added to the bottleneck activations.
        # Using an MLP projection keeps the input dimension fixed,
        # which is critical for DGP adaptation (z is optimized separately).
        # ----------------------------------------------------------------
        self.latent_proj = nn.Sequential(
            nn.Linear(mc.latent_dim, mc.latent_dim * 2),
            nn.SiLU(),
            nn.Linear(mc.latent_dim * 2, self.bottleneck_ch),
        )

        # ----------------------------------------------------------------
        # Bottleneck injection hook:
        # We register a forward hook on backbone.bottleneck to add the
        # projected z. Alternatively, we pass z through a custom forward.
        # We use the clean approach: subclass the backbone's bottleneck pass.
        # ----------------------------------------------------------------
        # The latent z is added at the bottleneck inside forward().

    def forward(
        self,
        x_t: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the clean image x̂_0 given noisy x_t, latent z, and timestep t.

        Implements FDMR Eq. 7:
            p_θ(x_{t-1} | x_t) = ∫ q(x_{t-1} | x_t, x̂_0) · p(z) dz,
            where x̂_0 = G_θ(x_t, z, t)

        For multi-contrast:
            x̂_0 ∈ R^{B×4×H×W} where ch 0,1 = contrast 1 (real,imag)
                                              ch 2,3 = contrast 2 (real,imag)

        Args:
            x_t: Noisy image, shape [B, 4, H, W]
            z:   Latent vector, shape [B, latent_dim]
            t:   Timestep (1-indexed), shape [B]

        Returns:
            x̂_0: Predicted clean image, shape [B, 4, H, W]
        """
        B = x_t.shape[0]

        # Build time embedding (shared with backbone)
        t_emb_raw = sinusoidal_embedding(t, self.time_emb_dim)
        t_emb = self.backbone.time_mlp(t_emb_raw)  # [B, time_emb_dim]

        # Project latent z to bottleneck channel space
        z_proj = self.latent_proj(z)  # [B, bottleneck_ch]

        # --- Encoder ---
        bottleneck, skips = self.backbone.encoder(x_t, t_emb)

        # --- Bottleneck with latent injection ---
        h = bottleneck
        for i, block in enumerate(self.backbone.bottleneck):
            h = block(h, t_emb)
            # Inject z after the first bottleneck block
            if i == 0:
                h = h + z_proj[:, :, None, None]

        # --- Decoder with skip connections ---
        h = self.backbone.decoder(h, skips, t_emb)

        # --- Output projection ---
        x0_pred = self.backbone.output_proj(h)

        return x0_pred


class KSpaceNeuralOperator(nn.Module):
    """
    k-space neural operator (NO_k) from Paper 2 (Section 3.1, Eq. 4).

    Operates on undersampled k-space measurements k̃ to learn a prior in
    frequency space before inverse Fourier transform. This is used as an
    optional preprocessing step in the training pipeline.

    NO_k is a UDNO without time conditioning applied directly to k-space data.
    Input: undersampled k-space (4 channels: real_c1, imag_c1, real_c2, imag_c2)
    Output: refined k-space (same 4 channels)
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        mc = config.model
        uck = mc.udno_k

        self.udno_k = UDNO(
            in_channels=uck.in_channels,
            out_channels=uck.out_channels,
            base_channels=uck.base_channels,
            channel_mults=[1, 2, 4, 4],      # lighter architecture for k-space
            num_res_blocks=2,
            num_basis=mc.disco.num_basis,
            kernel_size=mc.disco.kernel_size,
            time_emb_dim=0,                   # no time conditioning for NO_k
        )

    def forward(self, k_undersampled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            k_undersampled: Undersampled k-space, shape [B, 4, H, W]
                            Channels: [real_c1, imag_c1, real_c2, imag_c2]

        Returns:
            k_refined: Refined k-space, shape [B, 4, H, W]
        """
        return self.udno_k(k_undersampled, t=None)
