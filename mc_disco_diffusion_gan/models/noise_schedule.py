"""
Diffusion noise schedule for the Multi-Contrast DISCO-Diffusion GAN.

Implements the forward/reverse diffusion quantities from:
  FDMR (Zhao et al.) Equations 3–4.

Two schedule instances are used during training:
  1. Main schedule: T=16 steps (fast diffusion, FDMR Table 6)
  2. Refinement schedule: T=30 steps (diffusion refinement, FDMR Table 8)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple


class DiffusionNoiseSchedule(nn.Module):
    """
    Manages all diffusion schedule tensors for forward and reverse processes.

    All quantities are precomputed and stored as non-trainable buffers so that
    they are automatically moved to the correct device with .to(device).

    Attributes:
        T (int): Total number of diffusion steps.
        betas (Tensor): β_t for t = 1,...,T  shape [T]
        alphas (Tensor): α_t = 1 - β_t       shape [T]
        alpha_bars (Tensor): ᾱ_t = Π_{s=1}^{t} α_s  shape [T]
        sqrt_alpha_bars (Tensor): √(ᾱ_t)     shape [T]
        sqrt_one_minus_alpha_bars (Tensor): √(1 - ᾱ_t)  shape [T]
        posterior_variance (Tensor): σ_t² = β_t(1-ᾱ_{t-1})/(1-ᾱ_t)  shape [T]
    """

    def __init__(
        self,
        T: int,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.T = T

        # --- Build β schedule (FDMR Eq. 3; FDMR footnote 1: monotonically decreasing) ---
        if schedule == "linear":
            # Linear from beta_start to beta_end; FDMR uses a decreasing schedule,
            # which corresponds to larger noise at early (high-t) steps and smaller
            # at late (low-t) steps. We match the DDPM convention: β increases with t.
            betas = torch.linspace(beta_start, beta_end, T)
        elif schedule == "cosine":
            # Cosine schedule (improved DDPM) — not used by FDMR but exposed for ablation
            steps = T + 1
            s = 0.008
            x = torch.linspace(0, T, steps)
            alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas_raw = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = betas_raw.clamp(0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        # ᾱ_t = Π_{s=1}^{t} α_s  (FDMR Eq. 3)
        alpha_bars = torch.cumprod(alphas, dim=0)

        # Prepend ᾱ_0 = 1 for t=0 indexing in posterior calculations
        alpha_bars_prev = torch.cat([torch.tensor([1.0]), alpha_bars[:-1]])

        # Posterior variance σ_t² = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)  (DDPM Eq. 7)
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        # Clamp to avoid log(0) at t=1 where ᾱ_{t-1}=1
        posterior_variance = posterior_variance.clamp(min=1e-20)

        # Log of posterior variance (numerically stable)
        log_posterior_variance = torch.log(posterior_variance)

        # Posterior mean coefficients
        # μ_t(x_t, x_0) = (√ᾱ_{t-1}·β_t)/(1-ᾱ_t) · x_0 + (√α_t·(1-ᾱ_{t-1}))/(1-ᾱ_t) · x_t
        post_mean_coef1 = torch.sqrt(alpha_bars_prev) * betas / (1.0 - alpha_bars)
        post_mean_coef2 = torch.sqrt(alphas) * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

        # Register as buffers (moved with .to(device))
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("log_posterior_variance", log_posterior_variance)
        self.register_buffer("post_mean_coef1", post_mean_coef1)
        self.register_buffer("post_mean_coef2", post_mean_coef2)

    # ------------------------------------------------------------------
    # Helpers for batch-dimension broadcasting
    # ------------------------------------------------------------------

    def _gather(self, buffer: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Index into a schedule buffer using a batch of timestep indices.

        Args:
            buffer: 1-D tensor of shape [T]
            t: integer tensor of shape [B], values in {1,...,T} (1-indexed)

        Returns:
            Tensor of shape [B, 1, 1, 1] for broadcasting over [B, C, H, W].
        """
        # Convert 1-indexed t to 0-indexed
        idx = (t - 1).long().clamp(0, self.T - 1)
        return buffer[idx].reshape(-1, 1, 1, 1)

    # ------------------------------------------------------------------
    # Forward diffusion process
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample x_t from the forward process q(x_t | x_0).

        Implements FDMR Eq. 4:
            x_t = √(ᾱ_t) · x_0 + √(1 − ᾱ_t) · ε,   ε ~ N(0, I)

        Args:
            x0: Clean image, shape [B, C, H, W]
            t: Timestep (1-indexed), shape [B]
            noise: Optional pre-sampled noise; if None, samples fresh N(0,I)

        Returns:
            x_t: Noisy image at timestep t, shape [B, C, H, W]
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab = self._gather(self.sqrt_alpha_bars, t)
        sqrt_one_minus_ab = self._gather(self.sqrt_one_minus_alpha_bars, t)

        return sqrt_ab * x0 + sqrt_one_minus_ab * noise

    # ------------------------------------------------------------------
    # Reverse diffusion quantities
    # ------------------------------------------------------------------

    def q_posterior_mean_variance(
        self,
        x0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute posterior mean and variance q(x_{t-1} | x_t, x_0).

        μ_t = post_mean_coef1(t) · x_0 + post_mean_coef2(t) · x_t
        σ_t² = posterior_variance(t)

        Args:
            x0: Predicted clean image, shape [B, C, H, W]
            x_t: Noisy image at timestep t, shape [B, C, H, W]
            t: Timestep (1-indexed), shape [B]

        Returns:
            (posterior_mean, posterior_variance): both shape [B, C, H, W]
        """
        coef1 = self._gather(self.post_mean_coef1, t)
        coef2 = self._gather(self.post_mean_coef2, t)
        mean = coef1 * x0 + coef2 * x_t

        var = self._gather(self.posterior_variance, t).expand_as(x_t)
        return mean, var

    def sample_from_posterior(
        self,
        x0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample x_{t-1} from the posterior q(x_{t-1} | x_t, x_0).
        Used as the SP (Sample Prior) step in FDMR Stage 1.

        Args:
            x0: Predicted clean image (possibly DC-corrected), shape [B, C, H, W]
            x_t: Current noisy image, shape [B, C, H, W]
            t: Current timestep (1-indexed), shape [B]

        Returns:
            x_{t-1}: Sample from the posterior, shape [B, C, H, W]
        """
        mean, var = self.q_posterior_mean_variance(x0, x_t, t)
        noise = torch.randn_like(x_t)

        # Mask noise at t=1 to avoid adding noise at the final denoising step
        t_is_one = (t == 1).reshape(-1, 1, 1, 1).float()
        return mean + (1.0 - t_is_one) * torch.sqrt(var) * noise

    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Return ᾱ_t for the given batch of timesteps."""
        return self._gather(self.alpha_bars, t)

    def get_alpha_bar_prev(self, t: torch.Tensor) -> torch.Tensor:
        """Return ᾱ_{t-1} for the given batch of timesteps."""
        return self._gather(self.alpha_bars_prev, t)

    def get_beta(self, t: torch.Tensor) -> torch.Tensor:
        """Return β_t for the given batch of timesteps."""
        return self._gather(self.betas, t)

    def get_sqrt_one_minus_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Return √(1 − ᾱ_t) for the given batch of timesteps."""
        return self._gather(self.sqrt_one_minus_alpha_bars, t)
