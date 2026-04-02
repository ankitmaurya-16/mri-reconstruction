"""
U-Shaped DISCO Neural Operator (UDNO).

Implements the backbone architecture from:
  Jatyani et al., "A Unified Model for Compressed Sensing MRI Across
  Undersampling Patterns," CVPR 2025, Section 3.2 (UDNO paragraph).

UDNO replaces every standard nn.Conv2d in a U-Net encoder/decoder with
DISCOConv2d, making the architecture resolution-agnostic. It is used as:
  - NO_k : k-space neural operator (no timestep conditioning)
  - NO_i : image-space neural operator inside the GAN generator (with timestep conditioning)

The U-shaped design captures multi-scale features via skip connections,
following the motivation in Paper 2 Section 3.2 (UDNO paragraph) and
its Appendix A.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .disco import DISCOConv2d


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding (shared with generator)
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standard sinusoidal positional embedding for diffusion timesteps.

    Args:
        t: Timestep integers, shape [B] (1-indexed, values in 1..T)
        dim: Embedding dimension (must be even)

    Returns:
        Embedding tensor, shape [B, dim]
    """
    assert dim % 2 == 0, "Embedding dim must be even"
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


import math


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualDISCOBlock(nn.Module):
    """
    Residual block with DISCO convolutions and optional time embedding injection.

    Structure:
        GroupNorm → SiLU → DISCOConv2d → (+ time_emb) → GroupNorm → SiLU → DISCOConv2d
        + skip (1×1 DISCOConv2d if in_ch ≠ out_ch)

    This is the core building block of the UDNO encoder and decoder.

    Args:
        in_channels:  Input channel count.
        out_channels: Output channel count.
        num_basis:    Number of DISCO basis functions L.
        kernel_size:  Spatial support of DISCO kernels.
        time_emb_dim: Dimension of time embedding; 0 means no time conditioning.
        num_groups:   Number of groups for GroupNorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_basis: int = 8,
        kernel_size: int = 3,
        time_emb_dim: int = 0,
        num_groups: int = 8,
    ) -> None:
        super().__init__()

        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
        self.conv1 = DISCOConv2d(in_channels, out_channels, kernel_size, num_basis)

        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.conv2 = DISCOConv2d(out_channels, out_channels, kernel_size, num_basis)

        # Time embedding projection (injected between the two conv layers)
        if time_emb_dim > 0:
            self.time_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels),
            )
        else:
            self.time_proj = None

        # Skip connection: project if channel count changes
        if in_channels != out_channels:
            self.skip = DISCOConv2d(in_channels, out_channels, kernel_size=1, num_basis=num_basis, padding=0)
        else:
            self.skip = nn.Identity()

    def forward(
        self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x:     Input feature map, shape [B, in_channels, H, W]
            t_emb: Time embedding, shape [B, time_emb_dim] or None

        Returns:
            Output feature map, shape [B, out_channels, H, W]
        """
        h = self.conv1(F.silu(self.norm1(x)))

        # Inject time embedding
        if self.time_proj is not None and t_emb is not None:
            h = h + self.time_proj(t_emb)[:, :, None, None]

        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class DownsampleDISCO(nn.Module):
    """
    Spatial downsampling via strided DISCO convolution (stride=2).
    Preserves channel count.
    """

    def __init__(self, channels: int, num_basis: int = 8) -> None:
        super().__init__()
        self.conv = DISCOConv2d(channels, channels, kernel_size=3, num_basis=num_basis, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsampleDISCO(nn.Module):
    """
    Spatial upsampling: bilinear interpolation followed by a DISCO 1×1 projection
    to avoid checkerboard artifacts from transposed convolutions.
    """

    def __init__(self, channels: int, num_basis: int = 8) -> None:
        super().__init__()
        self.conv = DISCOConv2d(channels, channels, kernel_size=3, num_basis=num_basis, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


# ---------------------------------------------------------------------------
# UDNO encoder and decoder
# ---------------------------------------------------------------------------

class UDNOEncoder(nn.Module):
    """
    U-Net encoder built from ResidualDISCOBlocks with downsampling.

    Args:
        in_channels:   Input channels (4 for multi-contrast complex MRI).
        base_channels: Width at the first encoder level.
        channel_mults: Multipliers for each encoder level.
        num_res_blocks: Number of residual blocks per level.
        num_basis:     DISCO basis count L.
        kernel_size:   DISCO kernel spatial support.
        time_emb_dim:  Time embedding size (0 = no time conditioning).
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        channel_mults: List[int],
        num_res_blocks: int,
        num_basis: int,
        kernel_size: int,
        time_emb_dim: int = 0,
    ) -> None:
        super().__init__()

        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks

        # Input projection
        self.input_proj = DISCOConv2d(in_channels, base_channels, kernel_size, num_basis)

        # Build encoder levels
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        ch = base_channels
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(
                    ResidualDISCOBlock(ch, out_ch, num_basis, kernel_size, time_emb_dim)
                )
                ch = out_ch
            self.down_blocks.append(level_blocks)

            # Downsample after every level except the last
            if level < len(channel_mults) - 1:
                self.downsamplers.append(DownsampleDISCO(ch, num_basis))
            else:
                self.downsamplers.append(None)

        self.out_channels = ch

    def forward(
        self,
        x: torch.Tensor,
        t_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            bottleneck: Feature map after the deepest encoder level.
            skips:      List of feature maps from each level (for decoder skip connections).
        """
        h = self.input_proj(x)
        skips = []

        for level, (level_blocks, downsampler) in enumerate(
            zip(self.down_blocks, self.downsamplers)
        ):
            for block in level_blocks:
                h = block(h, t_emb)
            skips.append(h)  # save before downsampling

            if downsampler is not None:
                h = downsampler(h)

        return h, skips


class UDNODecoder(nn.Module):
    """
    U-Net decoder built from ResidualDISCOBlocks with upsampling and skip connections.

    Args:
        base_channels: Width at the first encoder level.
        channel_mults: Same multipliers used by the encoder (reversed here).
        num_res_blocks: Number of residual blocks per level.
        num_basis:     DISCO basis count L.
        kernel_size:   DISCO kernel spatial support.
        time_emb_dim:  Time embedding size.
    """

    def __init__(
        self,
        base_channels: int,
        channel_mults: List[int],
        num_res_blocks: int,
        num_basis: int,
        kernel_size: int,
        time_emb_dim: int = 0,
    ) -> None:
        super().__init__()

        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        # Build decoder in reverse order
        reversed_mults = list(reversed(channel_mults))
        for level, mult in enumerate(reversed_mults):
            in_ch = base_channels * mult
            # After concatenation with skip, input doubles
            if level == 0:
                block_in_ch = in_ch  # bottleneck — no skip at the very bottom
            else:
                prev_mult = reversed_mults[level - 1]
                block_in_ch = in_ch + base_channels * prev_mult

            out_ch = base_channels * mult if level == len(reversed_mults) - 1 else base_channels * reversed_mults[level + 1]
            # Simpler: match the encoder's skip output channels
            skip_ch = base_channels * mult
            concat_ch = in_ch + skip_ch if level > 0 else in_ch

            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                in_c = (concat_ch if i == 0 else skip_ch)
                level_blocks.append(
                    ResidualDISCOBlock(in_c, skip_ch, num_basis, kernel_size, time_emb_dim)
                )
            self.up_blocks.append(level_blocks)

            if level < len(reversed_mults) - 1:
                self.upsamplers.append(UpsampleDISCO(skip_ch, num_basis))
            else:
                self.upsamplers.append(None)

        self.out_channels = base_channels * reversed_mults[-1]

    def forward(
        self,
        h: torch.Tensor,
        skips: List[torch.Tensor],
        t_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h:     Bottleneck feature map.
            skips: List of encoder skip features (deepest first after reversing).
            t_emb: Optional time embedding.

        Returns:
            Decoded feature map.
        """
        skips_rev = list(reversed(skips))

        for level, (level_blocks, upsampler) in enumerate(
            zip(self.up_blocks, self.upsamplers)
        ):
            if level > 0 and level <= len(skips_rev):
                # Concatenate with the corresponding encoder skip
                skip = skips_rev[level]
                # Crop skip if spatial sizes differ (can happen with odd sizes)
                if h.shape != skip.shape:
                    skip = skip[:, :, : h.shape[2], : h.shape[3]]
                h = torch.cat([h, skip], dim=1)

            for block in level_blocks:
                h = block(h, t_emb)

            if upsampler is not None:
                h = upsampler(h)

        return h


# ---------------------------------------------------------------------------
# Full UDNO
# ---------------------------------------------------------------------------

class UDNO(nn.Module):
    """
    U-Shaped DISCO Neural Operator.

    This is the unified backbone used for both:
      - NO_k (k-space operator): time_emb_dim=0
      - NO_i (image-space operator in the generator): time_emb_dim > 0

    Args:
        in_channels:   Input channels (4 for 2-contrast complex MRI).
        out_channels:  Output channels.
        base_channels: Feature map width at the coarsest encoder level.
        channel_mults: Width multipliers per encoder level.
        num_res_blocks: Residual blocks per level.
        num_basis:     L, DISCO basis functions.
        kernel_size:   DISCO kernel spatial support.
        time_emb_dim:  Sinusoidal time embedding size (0 → no time conditioning).
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 64,
        channel_mults: List[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        num_basis: int = 8,
        kernel_size: int = 3,
        time_emb_dim: int = 0,
    ) -> None:
        super().__init__()

        self.time_emb_dim = time_emb_dim

        # Optional time embedding MLP: sinusoidal → linear projection
        if time_emb_dim > 0:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, time_emb_dim * 4),
                nn.SiLU(),
                nn.Linear(time_emb_dim * 4, time_emb_dim),
            )
        else:
            self.time_mlp = None

        # Encoder
        self.encoder = UDNOEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=list(channel_mults),
            num_res_blocks=num_res_blocks,
            num_basis=num_basis,
            kernel_size=kernel_size,
            time_emb_dim=time_emb_dim,
        )

        # Bottleneck residual blocks (at the deepest level)
        bottleneck_ch = self.encoder.out_channels
        self.bottleneck = nn.ModuleList([
            ResidualDISCOBlock(bottleneck_ch, bottleneck_ch, num_basis, kernel_size, time_emb_dim),
            ResidualDISCOBlock(bottleneck_ch, bottleneck_ch, num_basis, kernel_size, time_emb_dim),
        ])

        # Decoder
        self.decoder = UDNODecoder(
            base_channels=base_channels,
            channel_mults=list(channel_mults),
            num_res_blocks=num_res_blocks,
            num_basis=num_basis,
            kernel_size=kernel_size,
            time_emb_dim=time_emb_dim,
        )

        # Output projection to the desired channel count
        self.output_proj = nn.Sequential(
            nn.GroupNorm(min(8, self.decoder.out_channels), self.decoder.out_channels),
            nn.SiLU(),
            DISCOConv2d(self.decoder.out_channels, out_channels, kernel_size=1,
                        num_basis=num_basis, padding=0),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape [B, in_channels, H, W]
            t: Optional timestep (1-indexed integers), shape [B].
               If None, time conditioning is skipped.

        Returns:
            Output tensor, shape [B, out_channels, H, W]
        """
        # Build time embedding
        t_emb = None
        if self.time_mlp is not None and t is not None:
            t_emb_raw = sinusoidal_embedding(t, self.time_emb_dim)
            t_emb = self.time_mlp(t_emb_raw)

        # Encoder
        bottleneck, skips = self.encoder(x, t_emb)

        # Bottleneck
        h = bottleneck
        for block in self.bottleneck:
            h = block(h, t_emb)

        # Decoder with skip connections
        h = self.decoder(h, skips, t_emb)

        # Output projection
        return self.output_proj(h)
