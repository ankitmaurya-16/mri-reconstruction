"""
Data consistency layers for FDMR inference.

Implements three k-space data correction operations from FDMR (Zhao et al.):
  1. DC  (Data Consistency Layer)        — FDMR Eq. 10, Fig. 2a
  2. NDC (Noise-Mixed Data Consistency)  — FDMR Eq. 11, Fig. 2b
  3. JointLikelihoodScore                — Levac et al. Eq. 13-14 (multi-contrast)

All operations handle the 4-channel [real_c1, imag_c1, real_c2, imag_c2] format
and apply k-space corrections per contrast independently using centered 2D FFT.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from data.transforms import (
    fft2c,
    ifft2c,
    split_4channel,
    pair_to_4channel,
    real_to_complex,
    complex_to_real,
)


# ---------------------------------------------------------------------------
# DC Layer (FDMR Eq. 10)
# ---------------------------------------------------------------------------

class DataConsistencyLayer(nn.Module):
    """
    Data Consistency Layer (DC).

    Corrects the predicted clean image x̂_0 by replacing the k-space values
    at observed (sampled) locations with the true measurements y.

    Implements FDMR Eq. 10 (per contrast i):
        x̃_0_i = F⁻¹((1-M) · F · x̂_0_i + M · y_i)

    where:
        F   = centered 2D FFT
        F⁻¹ = centered 2D IFFT
        M   = binary undersampling mask (1 at sampled, 0 at unsampled)
        y   = observed undersampled k-space

    The DC layer is a no-parameter operation applied at every step during
    Stage 1 (fast diffusion generation) and once after Stage 3 (refinement).
    """

    def forward(
        self,
        x0_pred: torch.Tensor,
        y_obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply data consistency correction to predicted clean image.

        Args:
            x0_pred: Predicted clean image, shape [B, 4, H, W]
            y_obs:   Observed undersampled k-space, shape [B, 4, H, W]
                     (zeros at unobserved k-space locations)
            mask:    Binary undersampling mask, shape [B, 1, H, W] or [1, 1, H, W]
                     (broadcast over channel dimension)

        Returns:
            x̃_0: DC-corrected clean image, shape [B, 4, H, W]
        """
        x1, x2 = split_4channel(x0_pred)   # each [B, 2, H, W]
        y1, y2 = split_4channel(y_obs)

        # Per-contrast DC correction
        x1_dc = self._dc_per_contrast(x1, y1, mask)
        x2_dc = self._dc_per_contrast(x2, y2, mask)

        return pair_to_4channel(x1_dc, x2_dc)

    @staticmethod
    def _dc_per_contrast(
        x_pred: torch.Tensor,
        y_obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        DC correction for a single contrast.

        x̃ = F⁻¹((1-M)·F·x̂ + M·y)

        Args:
            x_pred: Shape [B, 2, H, W]  (real + imag)
            y_obs:  Shape [B, 2, H, W]  (observed k-space, real + imag)
            mask:   Shape [B/1, 1, H, W] (broadcastable)

        Returns:
            DC-corrected image, shape [B, 2, H, W]
        """
        # Transform prediction to k-space
        k_pred = fft2c(x_pred)  # [B, 2, H, W]

        # Replace observed k-space locations with true measurements
        # k_corrected = (1-M)·k_pred + M·y_obs
        k_corrected = (1.0 - mask) * k_pred + mask * y_obs

        # Transform back to image domain
        return ifft2c(k_corrected)  # [B, 2, H, W]


# ---------------------------------------------------------------------------
# NDC Layer (FDMR Eq. 11)
# ---------------------------------------------------------------------------

class NoiseMixedDataConsistencyLayer(nn.Module):
    """
    Noise-Mixed Data Consistency Layer (NDC).

    Enhances data consistency during noisy intermediate reverse diffusion steps
    by blending known sampling data with Gaussian noise at the current noise level.

    Implements FDMR Eq. 11 (per contrast i):
        x_{t-1} = F⁻¹((1-M)·F·x̃_{t-1} + √(ᾱ_{t-1})·y + M·F(√(1-ᾱ_t)·ε))

    where:
        x̃_{t-1} = posterior mean sample from q(x_{t-1}|x_t, x̃_0)  [from SP step]
        y        = observed k-space measurements (unscaled, already in k-space)
        ε        = i.i.d. Gaussian noise N(0,I)
        ᾱ_{t-1}  = cumulative noise retention at timestep t-1
        ᾱ_t      = cumulative noise retention at timestep t

    The NDC layer blends known k-space data with Gaussian noise to guide the
    reverse diffusion process, making generated samples closely mirror the target.
    (FDMR Section 4.2.1, last paragraph)
    """

    def forward(
        self,
        x_prev: torch.Tensor,
        y_obs: torch.Tensor,
        mask: torch.Tensor,
        alpha_bar_t: torch.Tensor,
        alpha_bar_t_prev: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply NDC correction to an intermediate noisy sample x_{t-1}.

        Args:
            x_prev:         Posterior sample x_{t-1} from SP step, shape [B, 4, H, W]
            y_obs:          Observed k-space (zeros at unobserved), shape [B, 4, H, W]
            mask:           Binary mask, shape [B/1, 1, H, W]
            alpha_bar_t:    ᾱ_t, shape [B, 1, 1, 1]
            alpha_bar_t_prev: ᾱ_{t-1}, shape [B, 1, 1, 1]
            eps:            Optional pre-sampled noise [B, 4, H, W]; sampled if None

        Returns:
            NDC-corrected x_{t-1}, shape [B, 4, H, W]
        """
        if eps is None:
            eps = torch.randn_like(x_prev)

        x1, x2 = split_4channel(x_prev)
        y1, y2 = split_4channel(y_obs)
        eps1, eps2 = split_4channel(eps)

        x1_ndc = self._ndc_per_contrast(x1, y1, mask, alpha_bar_t, alpha_bar_t_prev, eps1)
        x2_ndc = self._ndc_per_contrast(x2, y2, mask, alpha_bar_t, alpha_bar_t_prev, eps2)

        return pair_to_4channel(x1_ndc, x2_ndc)

    @staticmethod
    def _ndc_per_contrast(
        x_prev: torch.Tensor,
        y_obs: torch.Tensor,
        mask: torch.Tensor,
        alpha_bar_t: torch.Tensor,
        alpha_bar_t_prev: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """
        NDC correction for a single contrast (FDMR Eq. 11).

        x_{t-1} = F⁻¹((1-M)·F·x̃_{t-1} + √(ᾱ_{t-1})·y + M·F(√(1-ᾱ_t)·ε))

        Args:
            x_prev:   [B, 2, H, W] — current x_{t-1} from SP step
            y_obs:    [B, 2, H, W] — observed k-space (zeros at unobserved)
            mask:     [B/1, 1, H, W]
            alpha_bar_t:      [B, 1, 1, 1]
            alpha_bar_t_prev: [B, 1, 1, 1]
            eps:      [B, 2, H, W] — Gaussian noise

        Returns:
            NDC-corrected x_{t-1}, shape [B, 2, H, W]
        """
        # F · x̃_{t-1}
        k_prev = fft2c(x_prev)

        # F(√(1-ᾱ_t) · ε) — noise term in k-space
        sqrt_one_minus_ab_t = torch.sqrt(1.0 - alpha_bar_t)
        k_noise = fft2c(sqrt_one_minus_ab_t * eps)

        # NDC equation (Eq. 11):
        # k_out = (1-M)·k_prev + √(ᾱ_{t-1})·y_obs + M·k_noise
        sqrt_ab_prev = torch.sqrt(alpha_bar_t_prev)
        k_out = (1.0 - mask) * k_prev + sqrt_ab_prev * y_obs + mask * k_noise

        return ifft2c(k_out)  # [B, 2, H, W]


# ---------------------------------------------------------------------------
# Joint Likelihood Score (Levac et al. Eq. 13-14)
# ---------------------------------------------------------------------------

class JointLikelihoodScore(nn.Module):
    """
    Likelihood gradient term for joint multi-contrast reconstruction.

    Implements the measurement likelihood score from Paper 3 (Levac et al.):
    For contrast i, the likelihood gradient is (Eq. 13-14):
        ∇_{x_i} log p(y_i | x_i) ≈ -F_i^H(F_i·x_i - y_i) / (γ_t² + σ²)

    where:
        F_i     = undersampled Fourier operator for contrast i (M_i · F)
        F_i^H   = adjoint (conjugate transpose of F_i = F^H · M_i)
        y_i     = observed undersampled k-space for contrast i
        γ_t     = annealing parameter at timestep t
        σ       = MRI acquisition noise standard deviation

    This is added to the score function gradient during Langevin dynamics
    to enforce measurement consistency (Paper 3 Eq. 9-10, 13-14).

    In the GAN diffusion framework, this is used to compute the data-fidelity
    gradient for the multi-contrast Langevin update, complementing the
    generator's joint score prediction.
    """

    def __init__(self, sigma: float = 0.01) -> None:
        """
        Args:
            sigma: MRI acquisition noise standard deviation σ
                   (Paper 3 Section III-C: σ = 0.01 · max(x_GT))
        """
        super().__init__()
        self.sigma = sigma

    def forward(
        self,
        x: torch.Tensor,
        y_obs: torch.Tensor,
        mask: torch.Tensor,
        gamma_t: float,
    ) -> torch.Tensor:
        """
        Compute the likelihood gradient for joint reconstruction.

        L_score_i = -F_i^H(F_i·x_i - y_i) / (γ_t² + σ²)

        Args:
            x:       Current image estimate, shape [B, 4, H, W]
            y_obs:   Observed undersampled k-space, shape [B, 4, H, W]
            mask:    Binary undersampling mask, shape [B/1, 1, H, W]
            gamma_t: Annealing term γ_t at current noise level

        Returns:
            Likelihood gradient, shape [B, 4, H, W]
        """
        x1, x2 = split_4channel(x)
        y1, y2 = split_4channel(y_obs)

        score1 = self._likelihood_score_per_contrast(x1, y1, mask, gamma_t)
        score2 = self._likelihood_score_per_contrast(x2, y2, mask, gamma_t)

        return pair_to_4channel(score1, score2)

    def _likelihood_score_per_contrast(
        self,
        x_i: torch.Tensor,
        y_i: torch.Tensor,
        mask: torch.Tensor,
        gamma_t: float,
    ) -> torch.Tensor:
        """
        Likelihood score for one contrast (Paper 3 Eq. 13-14 numerator/denominator).

        score_i = -F^H(M · (F·x_i - y_i/√(ᾱ))) / (γ_t² + σ²)

        Note: y_obs here already encodes M·y (zeros at unobserved), so:
            M · (F·x - y_obs/√ᾱ) reduces to F·x - y_obs/√ᾱ at observed locations
            and zero elsewhere (via the mask).

        Args:
            x_i:    Single contrast image [B, 2, H, W]
            y_i:    Observed k-space for this contrast [B, 2, H, W]
            mask:   Undersampling mask [B/1, 1, H, W]
            gamma_t: Annealing term (scalar)

        Returns:
            Likelihood gradient image [B, 2, H, W]
        """
        # Forward: F · x_i
        k_xi = fft2c(x_i)  # [B, 2, H, W]

        # Residual in k-space at observed locations: M · (F·x_i - y_i)
        k_residual = mask * (k_xi - y_i)  # [B, 2, H, W]

        # Adjoint (inverse FFT of masked residual): F^H · (M · (F·x_i - y_i))
        img_residual = ifft2c(k_residual)  # [B, 2, H, W]

        # Divide by (γ_t² + σ²) and negate (gradient ascent on log-likelihood)
        denom = gamma_t ** 2 + self.sigma ** 2 + 1e-12
        return -img_residual / denom
