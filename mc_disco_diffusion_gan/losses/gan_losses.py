"""
GAN adversarial losses for the DISCO-Diffusion GAN.

Implements the generator and discriminator losses from:
  FDMR (Zhao et al.) Equations 8 and 9.

Generator loss (Eq. 8):
    L_G = Σ_{t≥1} E[-log D_φ(x_{t-1}^fake, x_t, t)] + λ_mse · ||x̂_0 - x_0||_2^2

Discriminator loss (Eq. 9):
    L_D = Σ_{t≥1} E[-log D_φ(x_{t-1}^real, x_t, t)
                    - log(1 - D_φ(x_{t-1}^fake, x_t, t))]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def adversarial_generator_loss(
    discriminator: nn.Module,
    x_prev_fake: torch.Tensor,
    x_curr: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Generator adversarial loss: fool the discriminator into predicting real.

    L_adv_G = -log D_φ(x_{t-1}^fake, x_t, t)  (non-saturating GAN loss)

    Args:
        discriminator: D_φ model
        x_prev_fake:   Generator's x_{t-1} sample, shape [B, 4, H, W]
        x_curr:        Current noisy x_t, shape [B, 4, H, W]
        t:             Timestep (1-indexed), shape [B]

    Returns:
        Scalar adversarial loss for the generator.
    """
    logits_fake = discriminator(x_prev_fake, x_curr, t)
    # Non-saturating loss: -log(sigmoid(logits)) = softplus(-logits)
    loss = F.softplus(-logits_fake).mean()
    return loss


def mse_reconstruction_loss(
    x0_pred: torch.Tensor,
    x0_real: torch.Tensor,
) -> torch.Tensor:
    """
    MSE loss between predicted and ground truth clean images.

    Second term in FDMR Eq. 8: λ_mse · ||x̂_0 - x_0||_2^2

    Applied to all 4 channels (both contrasts together), which couples the
    joint reconstruction and implicitly enforces the joint Bayesian prior
    from Paper 3 (Levac et al.).

    Args:
        x0_pred: Predicted clean image, shape [B, 4, H, W]
        x0_real: Ground truth clean image, shape [B, 4, H, W]

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(x0_pred, x0_real)


def generator_loss(
    discriminator: nn.Module,
    x_prev_fake: torch.Tensor,
    x_curr: torch.Tensor,
    t: torch.Tensor,
    x0_pred: torch.Tensor,
    x0_real: torch.Tensor,
    lambda_adv: float = 0.01,
    lambda_mse: float = 1.0,
) -> torch.Tensor:
    """
    Combined generator loss (FDMR Eq. 8).

    L_G = λ_adv · L_adv_G + λ_mse · MSE(x̂_0, x_0)

    Args:
        discriminator: D_φ model
        x_prev_fake:   Sampled x_{t-1} from the fake distribution, shape [B, 4, H, W]
        x_curr:        Current noisy x_t, shape [B, 4, H, W]
        t:             Diffusion timestep, shape [B]
        x0_pred:       Generator's predicted x̂_0, shape [B, 4, H, W]
        x0_real:       Ground truth x_0, shape [B, 4, H, W]
        lambda_adv:    Weight for adversarial loss (default 0.01)
        lambda_mse:    Weight for MSE loss (default 1.0)

    Returns:
        Scalar total generator loss.
    """
    l_adv = adversarial_generator_loss(discriminator, x_prev_fake, x_curr, t)
    l_mse = mse_reconstruction_loss(x0_pred, x0_real)
    return lambda_adv * l_adv + lambda_mse * l_mse


def discriminator_loss(
    discriminator: nn.Module,
    x_prev_real: torch.Tensor,
    x_prev_fake: torch.Tensor,
    x_curr: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Discriminator loss (FDMR Eq. 9).

    L_D = -log D_φ(x_{t-1}^real, x_t, t) - log(1 - D_φ(x_{t-1}^fake, x_t, t))
        = softplus(-logits_real) + softplus(logits_fake)  [non-saturating form]

    Args:
        discriminator: D_φ model
        x_prev_real:   True denoised sample x_{t-1} from q(x_{t-1}|x_t, x_0_real),
                       shape [B, 4, H, W]
        x_prev_fake:   GAN-generated sample (detached from generator graph),
                       shape [B, 4, H, W]
        x_curr:        Current noisy x_t, shape [B, 4, H, W]
        t:             Diffusion timestep, shape [B]

    Returns:
        Scalar total discriminator loss.
    """
    logits_real = discriminator(x_prev_real, x_curr, t)
    logits_fake = discriminator(x_prev_fake.detach(), x_curr, t)

    loss_real = F.softplus(-logits_real).mean()   # -log D(real)
    loss_fake = F.softplus(logits_fake).mean()    # -log(1 - D(fake)) = softplus(logits_fake)

    return loss_real + loss_fake
