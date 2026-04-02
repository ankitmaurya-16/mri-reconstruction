"""
DISCO (Discrete-Continuous) Convolution layer.

Implements the resolution-agnostic local integration operator from:
  Jatyani et al., "A Unified Model for Compressed Sensing MRI Across
  Undersampling Patterns," CVPR 2025, Eq. 7.

The kernel is parameterized as a linear combination of fixed piecewise-linear
basis functions:
    κ = Σ_ℓ θ^ℓ · κ^ℓ

On an equidistant grid this reduces to a standard convolution whose kernel
is constrained to the span of the L basis functions. As the resolution
increases, the DISCO kernel converges to a local integral (not a point
operator), making the architecture resolution-agnostic.

No custom CUDA kernels required — the materialized kernel is passed to
torch.nn.functional.conv2d, enjoying full GPU acceleration.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DISCOConv2d(nn.Module):
    """
    Resolution-agnostic 2-D convolution using DISCO (Discrete-Continuous) kernels.

    The effective convolutional kernel is computed as:
        w[c_out, c_in, i, j] = Σ_{ℓ=1}^{L} θ^ℓ[c_out, c_in] · κ^ℓ[i, j]

    where:
        - θ^ℓ  are learnable scalar weights, shape [out_channels, in_channels, L]
        - κ^ℓ  are fixed piecewise-linear basis functions, shape [L, kH, kW]

    The basis functions are defined in the continuous domain [0,1]^2 and
    evaluated on the kH × kW grid once at initialization.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels.
        kernel_size:  Spatial support of each basis function (e.g. 3 → 3×3 grid).
        num_basis:    L, the number of piecewise-linear basis functions.
        stride:       Convolution stride.
        padding:      Explicit padding. If None, uses kernel_size//2 (same conv).
        bias:         If True, adds a learnable bias.
        groups:       Number of blocked connections (same semantics as nn.Conv2d).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_basis: int = 8,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = True,
        groups: int = 1,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_basis = num_basis
        self.stride = stride
        self.padding = kernel_size // 2 if padding is None else padding
        self.groups = groups

        # ----------------------------------------------------------------
        # θ^ℓ: learnable coefficients, shape [out_channels, in_channels/groups, L]
        # ----------------------------------------------------------------
        self.theta = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, num_basis)
        )
        nn.init.kaiming_uniform_(self.theta, a=math.sqrt(5))

        # ----------------------------------------------------------------
        # κ^ℓ: fixed piecewise-linear basis functions, shape [L, kH, kW]
        # Registered as a non-trainable buffer so it moves with .to(device).
        # ----------------------------------------------------------------
        basis = self._build_piecewise_linear_basis(num_basis, kernel_size)
        self.register_buffer("basis", basis)

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    # ------------------------------------------------------------------
    # Basis construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_piecewise_linear_basis(
        num_basis: int, kernel_size: int
    ) -> torch.Tensor:
        """
        Build L piecewise-linear (hat) basis functions sampled on a kH×kW grid.

        Each basis function κ^ℓ is a 2D tent function centered at the ℓ-th
        uniformly spaced knot in [0,1]^2. The tent decays linearly to 0 at
        the neighboring knots.

        For a 1-D tent function centered at knot c with spacing h:
            φ(x) = max(0, 1 - |x - c| / h)

        The 2-D basis is formed as the outer product of 1-D tents:
            κ^ℓ(i, j) = φ_row^{ℓ_row}(i) · φ_col^{ℓ_col}(j)

        This matches the "linear piecewise basis" described in Paper 2's
        Supplement B.3 as the empirically optimal choice.

        Args:
            num_basis: L, total number of basis functions.
            kernel_size: Grid resolution (kH = kW = kernel_size).

        Returns:
            Tensor of shape [L, kernel_size, kernel_size] (float32).
        """
        # Distribute L basis functions on a grid of knots.
        # Choose the smallest square grid that fits L knots.
        n_1d = math.ceil(math.sqrt(num_basis))

        # Knot positions in [0, 1] (including endpoints)
        if n_1d > 1:
            knots = torch.linspace(0.0, 1.0, n_1d)
            h = knots[1] - knots[0]  # knot spacing
        else:
            knots = torch.tensor([0.5])
            h = torch.tensor(1.0)

        # Grid sampling positions for the kernel: [0, 1] mapped over kernel_size pts
        if kernel_size > 1:
            grid = torch.linspace(0.0, 1.0, kernel_size)
        else:
            grid = torch.tensor([0.5])

        # Build 1-D tent functions for each knot: shape [n_1d, kernel_size]
        # φ^k(x) = max(0, 1 - |x - knot_k| / h)
        knots_ = knots.unsqueeze(1)   # [n_1d, 1]
        grid_  = grid.unsqueeze(0)    # [1, kernel_size]
        phi_1d = (1.0 - torch.abs(grid_ - knots_) / h).clamp(min=0.0)
        # phi_1d shape: [n_1d, kernel_size]

        # 2-D basis via outer product of 1-D tents
        basis_list = []
        for row_idx in range(n_1d):
            for col_idx in range(n_1d):
                # [kernel_size, 1] x [1, kernel_size] → [kernel_size, kernel_size]
                b2d = phi_1d[row_idx].unsqueeze(1) * phi_1d[col_idx].unsqueeze(0)
                basis_list.append(b2d)
                if len(basis_list) == num_basis:
                    break
            if len(basis_list) == num_basis:
                break

        basis = torch.stack(basis_list, dim=0)  # [L, kH, kW]

        # Normalize each basis function to unit max (for stable initialization)
        max_vals = basis.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        basis = basis / max_vals

        return basis.float()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply DISCO convolution to input x.

        Step 1: Materialize the effective kernel from θ and basis:
            w[c_out, c_in, i, j] = Σ_ℓ θ[c_out, c_in, ℓ] · basis[ℓ, i, j]

        Step 2: Standard F.conv2d with the materialized kernel.

        Args:
            x: Input tensor, shape [B, in_channels, H, W]

        Returns:
            Output tensor, shape [B, out_channels, H', W']
        """
        # Materialize kernel: einsum over the basis dimension ℓ
        # theta: [out_ch, in_ch/groups, L]
        # basis: [L, kH, kW]
        # → weight: [out_ch, in_ch/groups, kH, kW]
        weight = torch.einsum("o i l, l h w -> o i h w", self.theta, self.basis)

        return F.conv2d(
            x,
            weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )

    def extra_repr(self) -> str:
        return (
            f"in={self.in_channels}, out={self.out_channels}, "
            f"k={self.kernel_size}, L={self.num_basis}, "
            f"stride={self.stride}, pad={self.padding}"
        )
