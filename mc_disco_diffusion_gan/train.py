"""
Training entry point for Multi-Contrast DISCO-Diffusion GAN.

Trains two generators (main + refinement) and one discriminator simultaneously
using the FDMR adversarial training framework (Zhao et al., Section 4.1, Eq. 8-9),
extended to multi-contrast data.

Features:
  - Dual generator training (main G_θ + refinement G_refine)
  - Time-conditioned discriminator (D_φ)
  - Mixed precision training (torch.amp)
  - Gradient clipping
  - Per-epoch validation with PSNR/SSIM/NMSE metrics
  - Best-model checkpoint saving
  - Resume training from checkpoint

Usage:
    python train.py --config configs/default.yaml [--overrides key=value ...]
    python train.py --config configs/default.yaml --overrides model.base_channels=128
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
try:
    from torch.amp import GradScaler
    from torch.amp import autocast as _autocast
    def amp_autocast(device_type, enabled=True):
        return _autocast(device_type=device_type, enabled=enabled)
except ImportError:
    from torch.cuda.amp import GradScaler
    from torch.cuda.amp import autocast as _autocast_cuda
    def amp_autocast(device_type, enabled=True):
        return _autocast_cuda(enabled=enabled)

# Project imports
from utils.config import load_config, Config
from utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from utils.visualization import save_loss_curves

from models.generator import MCDISCOGenerator
from models.discriminator import TimeConditionedDiscriminator
from models.noise_schedule import DiffusionNoiseSchedule

from data.datasets import MultiContrastMRIDataset
from data.transforms import apply_mask_kspace, zero_fill

from losses.gan_losses import generator_loss, discriminator_loss
from losses.dc_loss import compute_dc_loss
from losses.tv_loss import tv_loss

from inference.data_consistency import DataConsistencyLayer, NoiseMixedDataConsistencyLayer
from inference.pipeline import ThreeStageInference

from metrics.reconstruction import MetricsTracker


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_epoch(
    generator: MCDISCOGenerator,
    refinement_gen: MCDISCOGenerator,
    discriminator: TimeConditionedDiscriminator,
    schedule: DiffusionNoiseSchedule,
    refine_schedule: DiffusionNoiseSchedule,
    dataloader: DataLoader,
    opt_G: Adam,
    opt_G_refine: Adam,
    opt_D: Adam,
    scaler_G: GradScaler,
    scaler_G_refine: GradScaler,
    scaler_D: GradScaler,
    config: Config,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Run one full training epoch.

    Training strategy (FDMR Section 4.1):
      For each batch:
        1. Sample random diffusion timestep t ~ U{1,...,T}
        2. Corrupt ground truth: x_t = q_sample(x_0, t)  [Eq. 4]
        3. Generator predicts x̂_0 = G_θ(x_t, z, t)
        4. Sample x_{t-1}^real from q(x_{t-1}|x_t, x_0)   [real transition]
        5. Sample x_{t-1}^fake from q(x_{t-1}|x_t, x̂_0)  [generated transition]
        6. Update D: minimize L_D(real, fake)               [Eq. 9]
        7. Update G: minimize L_G(fake) + λ_mse·MSE        [Eq. 8]
        8. Same procedure for refinement generator G_refine

    Returns:
        Dict of mean losses for this epoch.
    """
    generator.train()
    refinement_gen.train()
    discriminator.train()

    T = config.diffusion.T
    T_refine = config.diffusion.T_refine
    latent_dim = config.model.latent_dim

    loss_history = defaultdict(list)

    for batch_idx, batch in enumerate(dataloader):
        x0 = batch["x0"].to(device)     # [B, 4, H, W] ground truth
        y_obs = batch["y_obs"].to(device) # [B, 4, H, W] undersampled k-space
        mask = batch["mask"].to(device)   # [B, 1, H, W] binary mask

        B = x0.shape[0]

        # ----------------------------------------------------------------
        # Main generator training (T=16 diffusion steps)
        # ----------------------------------------------------------------
        t = torch.randint(1, T + 1, (B,), device=device)
        eps = torch.randn_like(x0)
        x_t = schedule.q_sample(x0, t, noise=eps)

        z = torch.randn(B, latent_dim, device=device)

        with amp_autocast(device.type, enabled=config.training.mixed_precision):
            x0_pred = generator(x_t, z, t)

            # Real and fake x_{t-1} samples (for discriminator)
            with torch.no_grad():
                x_prev_real, _ = schedule.q_posterior_mean_variance(x0, x_t, t)
                x_prev_fake, _ = schedule.q_posterior_mean_variance(x0_pred.detach(), x_t, t)

            # Discriminator update
            opt_D.zero_grad(set_to_none=True)
            d_loss = discriminator_loss(discriminator, x_prev_real, x_prev_fake, x_t, t)

        scaler_D.scale(d_loss).backward()
        scaler_D.unscale_(opt_D)
        nn.utils.clip_grad_norm_(discriminator.parameters(), config.training.grad_clip)
        scaler_D.step(opt_D)
        scaler_D.update()

        # Generator update
        with amp_autocast(device.type, enabled=config.training.mixed_precision):
            # Re-generate for generator gradient computation
            x0_pred_g = generator(x_t, z, t)
            x_prev_fake_g, _ = schedule.q_posterior_mean_variance(x0_pred_g, x_t, t)

            g_loss = generator_loss(
                discriminator=discriminator,
                x_prev_fake=x_prev_fake_g,
                x_curr=x_t,
                t=t,
                x0_pred=x0_pred_g,
                x0_real=x0,
                lambda_adv=config.training.lambda_adv,
                lambda_mse=config.training.lambda_mse,
            )

        opt_G.zero_grad(set_to_none=True)
        scaler_G.scale(g_loss).backward()
        scaler_G.unscale_(opt_G)
        nn.utils.clip_grad_norm_(generator.parameters(), config.training.grad_clip)
        scaler_G.step(opt_G)
        scaler_G.update()

        # ----------------------------------------------------------------
        # Refinement generator training (T_refine=30, smaller β)
        # ----------------------------------------------------------------
        t_r = torch.randint(1, T_refine + 1, (B,), device=device)
        eps_r = torch.randn_like(x0)
        x_t_r = refine_schedule.q_sample(x0, t_r, noise=eps_r)

        z_r = torch.randn(B, latent_dim, device=device)

        with amp_autocast(device.type, enabled=config.training.mixed_precision):
            x0_pred_r = refinement_gen(x_t_r, z_r, t_r)

            with torch.no_grad():
                x_prev_real_r, _ = refine_schedule.q_posterior_mean_variance(x0, x_t_r, t_r)
                x_prev_fake_r, _ = refine_schedule.q_posterior_mean_variance(
                    x0_pred_r.detach(), x_t_r, t_r
                )

            opt_D.zero_grad(set_to_none=True)
            d_loss_r = discriminator_loss(discriminator, x_prev_real_r, x_prev_fake_r, x_t_r, t_r)

        scaler_D.scale(d_loss_r).backward()
        scaler_D.unscale_(opt_D)
        nn.utils.clip_grad_norm_(discriminator.parameters(), config.training.grad_clip)
        scaler_D.step(opt_D)
        scaler_D.update()

        with amp_autocast(device.type, enabled=config.training.mixed_precision):
            x0_pred_rg = refinement_gen(x_t_r, z_r, t_r)
            x_prev_fake_rg, _ = refine_schedule.q_posterior_mean_variance(x0_pred_rg, x_t_r, t_r)

            g_loss_r = generator_loss(
                discriminator=discriminator,
                x_prev_fake=x_prev_fake_rg,
                x_curr=x_t_r,
                t=t_r,
                x0_pred=x0_pred_rg,
                x0_real=x0,
                lambda_adv=config.training.lambda_adv,
                lambda_mse=config.training.lambda_mse,
            )

        opt_G_refine.zero_grad(set_to_none=True)
        scaler_G_refine.scale(g_loss_r).backward()
        scaler_G_refine.unscale_(opt_G_refine)
        nn.utils.clip_grad_norm_(refinement_gen.parameters(), config.training.grad_clip)
        scaler_G_refine.step(opt_G_refine)
        scaler_G_refine.update()

        # Record losses
        loss_history["G_total"].append(g_loss.item())
        loss_history["G_refine"].append(g_loss_r.item())
        loss_history["D"].append((d_loss.item() + d_loss_r.item()) / 2)

        if batch_idx % 50 == 0:
            print(
                f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"G: {g_loss.item():.4f} | D: {d_loss.item():.4f} | "
                f"G_refine: {g_loss_r.item():.4f}"
            )

    return {k: sum(v) / len(v) for k, v in loss_history.items()}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    inference_pipeline: ThreeStageInference,
    val_loader: DataLoader,
    metrics_tracker: MetricsTracker,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run validation using the full 3-stage inference pipeline.

    Args:
        inference_pipeline: Configured ThreeStageInference instance.
        val_loader:         Validation DataLoader.
        metrics_tracker:    Metrics accumulator (reset before calling).
        device:             Compute device.

    Returns:
        Dict of metrics from MetricsTracker.compute().
    """
    metrics_tracker.reset()

    for batch in val_loader:
        x0 = batch["x0"].to(device)
        y_obs = batch["y_obs"].to(device)
        mask = batch["mask"].to(device)

        # Run Stage 1 only during training validation (DGP is too slow)
        # Full 3-stage pipeline is used at test time only
        results = inference_pipeline.run(y_obs, mask, run_dgp=False, run_refinement=False)
        x0_final = results["final"]

        metrics_tracker.update(x0_final, x0)

    return metrics_tracker.compute()


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    # Load configuration
    overrides = args.overrides if args.overrides else []
    config = load_config(args.config, overrides=overrides)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # Create output directories
    Path(config.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.output_dir).mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Build models
    # ----------------------------------------------------------------
    print("[Train] Building models...")
    generator = MCDISCOGenerator(config).to(device)
    refinement_gen = MCDISCOGenerator(config).to(device)
    discriminator = TimeConditionedDiscriminator(
        in_channels=config.model.in_channels,
        base_channels=config.model.base_channels,
        num_basis=config.model.disco.num_basis,
        time_emb_dim=config.model.time_emb_dim,
    ).to(device)

    n_params_G = sum(p.numel() for p in generator.parameters())
    n_params_D = sum(p.numel() for p in discriminator.parameters())
    print(f"[Train] Generator parameters: {n_params_G:,}")
    print(f"[Train] Discriminator parameters: {n_params_D:,}")

    # ----------------------------------------------------------------
    # Build noise schedules
    # ----------------------------------------------------------------
    schedule = DiffusionNoiseSchedule(
        T=config.diffusion.T,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        schedule=config.diffusion.beta_schedule,
    ).to(device)
    refine_schedule = DiffusionNoiseSchedule(
        T=config.diffusion.T_refine,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end / 4,  # finer noise for refinement model
        schedule=config.diffusion.beta_schedule,
    ).to(device)

    # ----------------------------------------------------------------
    # Build optimizers & Scalers
    # ----------------------------------------------------------------
    opt_G = Adam(generator.parameters(), lr=config.training.lr, betas=(0.5, 0.999))
    opt_G_refine = Adam(refinement_gen.parameters(), lr=config.training.lr, betas=(0.5, 0.999))
    opt_D = Adam(discriminator.parameters(), lr=config.training.lr, betas=(0.5, 0.999))
    
    # Use individual scalers for separate gradient graphs
    use_amp = config.training.mixed_precision and device.type == "cuda"
    scaler_G = GradScaler(enabled=use_amp)
    scaler_G_refine = GradScaler(enabled=use_amp)
    scaler_D = GradScaler(enabled=use_amp)

    # ----------------------------------------------------------------
    # Resume from checkpoint if available
    # ----------------------------------------------------------------
    start_epoch = 0
    best_psnr = 0.0

    if args.resume:
        cp_path = args.resume
    elif args.auto_resume:
        cp_path = find_latest_checkpoint(config.paths.checkpoint_dir)
    else:
        cp_path = None

    if cp_path and os.path.exists(cp_path):
        info = load_checkpoint(
            cp_path, generator, refinement_gen, discriminator,
            opt_G, opt_G_refine, opt_D, device=device
        )
        start_epoch = info["epoch"] + 1
        best_psnr = info["metrics"].get("psnr_mean", 0.0)
        print(f"[Train] Resuming from epoch {start_epoch}")

    # ----------------------------------------------------------------
    # Build data loaders
    # ----------------------------------------------------------------
    print(f"[Train] Loading {config.data.dataset} dataset...")
    train_dataset = MultiContrastMRIDataset(config, split="train")
    val_dataset   = MultiContrastMRIDataset(config, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    # Optionally cap validation to a random subset for speed
    max_val = config.training.max_val_samples
    if max_val > 0 and len(val_dataset) > max_val:
        import random
        val_indices = random.sample(range(len(val_dataset)), max_val)
        val_subset = torch.utils.data.Subset(val_dataset, val_indices)
        print(f"[Train] Val capped to {max_val}/{len(val_dataset)} slices")
    else:
        val_subset = val_dataset

    val_loader = DataLoader(
        val_subset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"[Train] Train samples: {len(train_dataset)} | Val samples: {len(val_subset)}")

    # ----------------------------------------------------------------
    # Build inference pipeline (for validation)
    # ----------------------------------------------------------------
    dc_layer  = DataConsistencyLayer().to(device)
    ndc_layer = NoiseMixedDataConsistencyLayer().to(device)
    inference_pipeline = ThreeStageInference(
        generator, refinement_gen,
        schedule, refine_schedule,
        dc_layer, ndc_layer, config, device=device,
    )

    metrics_tracker = MetricsTracker()

    # History for plotting
    train_loss_history: Dict[str, List[float]] = defaultdict(list)
    val_metric_history: Dict[str, List[float]] = defaultdict(list)

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    print(f"[Train] Starting training for {config.training.num_epochs} epochs...")

    for epoch in range(start_epoch, config.training.num_epochs):
        t0 = time.time()

        # Train one epoch
        epoch_losses = train_epoch(
            generator, refinement_gen, discriminator,
            schedule, refine_schedule,
            train_loader, opt_G, opt_G_refine, opt_D,
            scaler_G, scaler_G_refine, scaler_D, 
            config, device, epoch,
        )

        for k, v in epoch_losses.items():
            train_loss_history[k].append(v)

        elapsed = time.time() - t0
        print(
            f"[Epoch {epoch:03d}] "
            + " | ".join(f"{k}: {v:.4f}" for k, v in epoch_losses.items())
            + f" | time: {elapsed:.1f}s"
        )

        # Validation
        if epoch % config.training.val_every == 0:
            print(f"[Epoch {epoch:03d}] Running validation...")
            val_metrics = validate(inference_pipeline, val_loader, metrics_tracker, device)

            for k, v in val_metrics.items():
                val_metric_history[k].append(v)

            psnr = val_metrics.get("psnr_mean", 0.0)
            ssim = val_metrics.get("ssim_mean", 0.0)
            nmse = val_metrics.get("nmse_mean", 0.0)
            print(
                f"[Epoch {epoch:03d}] Val — "
                f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | NMSE: {nmse:.4f}"
            )

            # Save best model
            is_best = psnr > best_psnr
            if is_best:
                best_psnr = psnr
                print(f"[Epoch {epoch:03d}] New best PSNR: {best_psnr:.2f} dB")
        else:
            val_metrics = {}
            is_best = False

        # Save checkpoint
        if epoch % config.training.save_every == 0 or is_best:
            save_checkpoint(
                config.paths.checkpoint_dir, epoch,
                generator, refinement_gen, discriminator,
                opt_G, opt_G_refine, opt_D,
                metrics=val_metrics,
                is_best=is_best,
            )

    # Save training curves
    save_loss_curves(
        dict(train_loss_history),
        {k: v for k, v in val_metric_history.items() if "mean" in k},
        output_dir=config.paths.log_dir,
    )

    print(f"\n[Train] Done. Best PSNR: {best_psnr:.2f} dB")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Multi-Contrast DISCO-Diffusion GAN")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--overrides", nargs="*", default=[],
        help="Config overrides in key=value format, e.g. model.base_channels=128"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a specific checkpoint to resume from"
    )
    parser.add_argument(
        "--auto_resume", action="store_true",
        help="Automatically resume from the latest checkpoint in checkpoint_dir"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)