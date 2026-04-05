"""
Three-Stage Inference Pipeline for Multi-Contrast DISCO-Diffusion GAN.

Implements the full FDMR inference framework (Zhao et al., Section 4.2) extended to
multi-contrast reconstruction (Levac et al., Paper 3), with DISCO backbones.

Three stages:
  Stage 1: Fast Diffusion Generation     (FDMR Section 4.2.1, Fig. 1a)
  Stage 2: Early-stopped DGP Adaptation  (FDMR Section 4.2.2, Fig. 1b)
  Stage 3: Diffusion Refinement          (FDMR Section 4.2.3, Fig. 1c)

Each stage is implemented as a distinct method so that ablation studies
(Table 5 of FDMR) can be reproduced by calling individual stages.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Adam

from models.generator import MCDISCOGenerator
from models.noise_schedule import DiffusionNoiseSchedule
from inference.data_consistency import (
    DataConsistencyLayer,
    NoiseMixedDataConsistencyLayer,
    JointLikelihoodScore,
)
from losses.dc_loss import compute_dc_loss
from losses.tv_loss import tv_loss
from utils.config import Config


class ThreeStageInference:
    """
    Three-stage FDMR inference pipeline for multi-contrast MRI reconstruction.

    Attributes:
        generator:          Main trained GAN generator G_θ
        refinement_gen:     Separate refinement generator G_refine
        schedule:           Noise schedule for the main T=16 model
        refine_schedule:    Noise schedule for the refinement T=30 model
        dc_layer:           Data consistency layer (DC, Eq. 10)
        ndc_layer:          Noise-mixed data consistency layer (NDC, Eq. 11)
        config:             Global Config instance
        device:             Torch device
    """

    def __init__(
        self,
        generator: MCDISCOGenerator,
        refinement_gen: MCDISCOGenerator,
        schedule: DiffusionNoiseSchedule,
        refine_schedule: DiffusionNoiseSchedule,
        dc_layer: DataConsistencyLayer,
        ndc_layer: NoiseMixedDataConsistencyLayer,
        config: Config,
        device: Optional[torch.device] = None,
    ) -> None:
        self.generator = generator
        self.refinement_gen = refinement_gen
        self.schedule = schedule
        self.refine_schedule = refine_schedule
        self.dc_layer = dc_layer
        self.ndc_layer = ndc_layer
        self.config = config
        self.device = device or next(generator.parameters()).device

        self._likelihood_score = JointLikelihoodScore(
            sigma=config.inference.sigma_noise
        ).to(self.device)

    # ------------------------------------------------------------------
    # Stage 1: Fast Diffusion Generation (FDMR Section 4.2.1)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def stage1_fast_diffusion(
        self,
        y_obs: torch.Tensor,
        mask: torch.Tensor,
        mode: str = "joint",
    ) -> torch.Tensor:
        """
        Stage 1: Reverse diffusion with DC → SP → NDC at each step.

        Implements FDMR Fig. 1a (Fast Diffusion Generation) extended to
        multi-contrast via the 4-channel generator (Paper 3 joint reconstruction).

        Algorithm (for t = T, T-1, ..., 1):
          1. z ~ N(0, I)
          2. x̂_0 = G_θ(x_t, z, t)                 [generator prediction]
          3. x̃_0 = DC(x̂_0, y_obs, mask)            [Eq. 10: replace observed k-space]
          4. x_{t-1} = SP(x_t, x̃_0, t)              [posterior sample]
          5. if t > 1: x_{t-1} = NDC(x_{t-1}, y_obs, mask, ᾱ_{t-1}, ᾱ_t)  [Eq. 11]

        Args:
            y_obs:  Undersampled k-space, shape [B, 4, H, W]
            mask:   Binary undersampling mask, shape [B/1, 1, H, W]
            mode:   "joint" (both contrasts undersampled) or "conditional"
                    (contrast 1 fully sampled, only contrast 2 undersampled)

        Returns:
            x0_s1: Stage 1 reconstruction, shape [B, 4, H, W]
        """
        B, C, H, W = y_obs.shape
        T = self.schedule.T
        latent_dim = self.config.model.latent_dim

        self.generator.eval()

        # Start from pure Gaussian noise (DDPM reverse diffusion start)
        x_t = torch.randn(B, C, H, W, device=self.device)

        for t_int in reversed(range(1, T + 1)):
            t = torch.full((B,), t_int, device=self.device, dtype=torch.long)

            # Step 1–2: Generator prediction x̂_0 = G_θ(x_t, z, t)
            z = torch.randn(B, latent_dim, device=self.device)
            x0_pred = self.generator(x_t, z, t)

            # Step 3: DC correction on x̂_0 (Eq. 10)
            x0_tilde = self.dc_layer(x0_pred, y_obs, mask)

            # In conditional mode (x1 fully sampled): override contrast-1 channel
            if mode == "conditional":
                # x1 = F^{-1}(y_obs_c1) — pseudo-inverse reconstruction for c1
                from data.transforms import ifft2c_4ch
                x_c1_recon = ifft2c_4ch(y_obs)[:, :2]  # fully sampled → exact
                x0_tilde = torch.cat([x_c1_recon, x0_tilde[:, 2:]], dim=1)

            # Step 4: SP — sample x_{t-1} from posterior q(x_{t-1}|x_t, x̃_0)
            x_prev = self.schedule.sample_from_posterior(x0_tilde, x_t, t)

            # Step 5: NDC — noise-mixed data consistency (only for t > 1)
            if t_int > 1:
                t_prev = torch.full((B,), t_int - 1, device=self.device, dtype=torch.long)
                alpha_bar_t = self.schedule.get_alpha_bar(t)        # [B, 1, 1, 1]
                alpha_bar_t_prev = self.schedule.get_alpha_bar(t_prev)

                eps = torch.randn_like(x_prev)
                x_prev = self.ndc_layer(
                    x_prev, y_obs, mask,
                    alpha_bar_t=alpha_bar_t,
                    alpha_bar_t_prev=alpha_bar_t_prev,
                    eps=eps,
                )

            x_t = x_prev

        # Final DC correction on output
        x0_s1 = self.dc_layer(x_t, y_obs, mask)
        return x0_s1

    # ------------------------------------------------------------------
    # Stage 2: Early-stopped Deep Generative Prior Adaptation (FDMR 4.2.2)
    # ------------------------------------------------------------------

    def stage2_dgp_adaptation(
        self,
        x0_stage1: torch.Tensor,
        y_obs: torch.Tensor,
        mask: torch.Tensor,
        num_steps: Optional[int] = None,
        lambda_tv: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Stage 2: Fine-tune a copy of the generator using data consistency loss.

        Implements FDMR Eq. 12 (DGP optimization):
            θ*, z* = argmin_{θ,z} ||MF(G_θ(x0, z, 1)) - y||_2^2 + λ·||G_θ(x0, z, 1)||_TV

        Key design: Uses a DEEP COPY of the generator — original weights unchanged.
        The DGP acts as a deep image prior (DIP) by optimizing both θ and z.

        Early stopping (fixed iterations = dgp_steps) prevents overfitting, per
        FDMR Section 4.2.2 and ablation in Table 7 (50 steps optimal).

        Args:
            x0_stage1: Stage 1 reconstruction (input context), shape [B, 4, H, W]
            y_obs:     Observed k-space, shape [B, 4, H, W]
            mask:      Undersampling mask, shape [B/1, 1, H, W]
            num_steps: Override for dgp_steps from config
            lambda_tv: Override for lambda_tv from config

        Returns:
            x0_dgp: DGP-adapted reconstruction, shape [B, 4, H, W]
        """
        num_steps = num_steps or self.config.training.dgp_steps
        lambda_tv_val = lambda_tv or self.config.training.lambda_tv
        B = x0_stage1.shape[0]
        latent_dim = self.config.model.latent_dim

        # Deep copy of generator to avoid modifying original weights
        gen_copy = copy.deepcopy(self.generator)
        gen_copy.train()

        # Learnable latent z* initialized to zeros (stable starting point)
        z_star = nn.Parameter(torch.zeros(B, latent_dim, device=self.device))

        # Optimize both generator copy weights and z*
        optimizer = Adam(
            list(gen_copy.parameters()) + [z_star],
            lr=self.config.training.lr,
        )

        # Use t=1 for DGP (generator evaluated at the last denoising step)
        t_one = torch.ones(B, device=self.device, dtype=torch.long)

        # DGP requires gradients even when called from a no_grad context (validation)
        with torch.enable_grad():
            for step in range(num_steps):
                optimizer.zero_grad()

                x0_pred = gen_copy(x0_stage1, z_star, t_one)

                # Data consistency loss: L_DC (FDMR Eq. 12, term 1)
                l_dc = compute_dc_loss(x0_pred, y_obs, mask)

                # Total variation loss: L_TV (FDMR Eq. 12, term 2)
                l_tv = tv_loss(x0_pred)

                loss = l_dc + lambda_tv_val * l_tv
                loss.backward()
                optimizer.step()

        # Evaluate adapted generator with optimized z*
        with torch.no_grad():
            gen_copy.eval()
            x0_dgp = gen_copy(x0_stage1, z_star.detach(), t_one)
            # Final DC correction
            x0_dgp = self.dc_layer(x0_dgp, y_obs, mask)

        # Discard gen_copy and z_star — original generator unchanged
        del gen_copy, z_star

        return x0_dgp

    # ------------------------------------------------------------------
    # Stage 3: Diffusion Refinement (FDMR Section 4.2.3)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def stage3_diffusion_refinement(
        self,
        x0_dgp: torch.Tensor,
        y_obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Stage 3: Small-step diffusion refinement for sharpening.

        Uses only the LAST step of the refinement model's reverse diffusion
        (t=1 of a T_refine=30 model) to enrich fine image details and sharpen
        edges without re-running the full diffusion chain.

        Implements FDMR Section 4.2.3:
          1. Add noise to x*_0: x_T_refine ~ q(x_{T_refine} | x*_0)
          2. Run ONE step of the refinement model: x0_final = G_refine(x_T_refine, z, t=1)
          3. Apply DC correction: x0_final = DC(x0_final, y_obs, mask)

        The refinement model has more steps (T=30, smaller β) and its last step
        has demonstrated strong detail enrichment and edge sharpening capability
        (FDMR Section 4.2.3, Table 8).

        Args:
            x0_dgp: DGP-adapted reconstruction, shape [B, 4, H, W]
            y_obs:  Observed k-space, shape [B, 4, H, W]
            mask:   Undersampling mask, shape [B/1, 1, H, W]

        Returns:
            x0_final: Final refined reconstruction, shape [B, 4, H, W]
        """
        B = x0_dgp.shape[0]
        latent_dim = self.config.model.latent_dim
        T_refine = self.refine_schedule.T

        self.refinement_gen.eval()

        # Step 1: Add noise to reach x_{T_refine} level
        t_refine = torch.full((B,), T_refine, device=self.device, dtype=torch.long)
        eps = torch.randn_like(x0_dgp)
        x_T_refine = self.refine_schedule.q_sample(x0_dgp, t_refine, noise=eps)

        # Step 2: ONE reverse diffusion step at t=1 (last step only, FDMR 4.2.3)
        z = torch.randn(B, latent_dim, device=self.device)
        t_one = torch.ones(B, device=self.device, dtype=torch.long)
        x0_refined_pred = self.refinement_gen(x_T_refine, z, t_one)

        # Step 3: DC correction (no NDC in Stage 3, per FDMR Section 4.2.3)
        x0_final = self.dc_layer(x0_refined_pred, y_obs, mask)

        return x0_final

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        y_obs: torch.Tensor,
        mask: torch.Tensor,
        mode: str = "joint",
        run_dgp: bool = True,
        run_refinement: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Run the full 3-stage inference pipeline.

        Returns intermediate results at each stage boundary, enabling
        reproduction of FDMR Table 5 ablation studies (ZF+①, ZF+②, etc.).

        Args:
            y_obs:          Undersampled k-space, shape [B, 4, H, W]
            mask:           Binary undersampling mask, shape [B/1, 1, H, W]
            mode:           "joint" or "conditional" multi-contrast mode
            run_dgp:        Whether to run Stage 2 (set False to skip)
            run_refinement: Whether to run Stage 3 (set False to skip)

        Returns:
            Dict with keys:
              "stage1": Stage 1 output  [B, 4, H, W]
              "stage2": Stage 2 output  [B, 4, H, W] (or same as stage1 if skipped)
              "stage3": Stage 3 output  [B, 4, H, W] (or same as stage2 if skipped)
              "final":  Same as "stage3"
        """
        # Move data to device
        y_obs = y_obs.to(self.device)
        mask = mask.to(self.device)

        # Stage 1: Fast diffusion generation
        x0_s1 = self.stage1_fast_diffusion(y_obs, mask, mode=mode)

        # Stage 2: DGP adaptation
        if run_dgp:
            x0_s2 = self.stage2_dgp_adaptation(x0_s1, y_obs, mask)
        else:
            x0_s2 = x0_s1

        # Stage 3: Diffusion refinement
        if run_refinement:
            x0_s3 = self.stage3_diffusion_refinement(x0_s2, y_obs, mask)
        else:
            x0_s3 = x0_s2

        return {
            "stage1": x0_s1,
            "stage2": x0_s2,
            "stage3": x0_s3,
            "final": x0_s3,
        }
