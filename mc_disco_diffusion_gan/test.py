"""
Test/Inference entry point for Multi-Contrast DISCO-Diffusion GAN.

Runs the full 3-stage inference pipeline on the test set and:
  1. Evaluates PSNR/SSIM/NMSE at each stage (mirrors FDMR Table 5 ablation)
  2. Saves visual comparison figures (GT | ZF | S1 | S2 | S3 + error maps)
  3. Prints a quantitative summary table

Usage:
    python test.py --config configs/default.yaml --checkpoint checkpoints/best_model.pth
    python test.py --config configs/default.yaml --checkpoint checkpoints/best_model.pth \
                   --mask_type radial --acceleration 10 --no_dgp
"""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from utils.config import load_config
from utils.checkpoint import load_checkpoint
from utils.visualization import save_comparison_figure, save_loss_curves

from models.generator import MCDISCOGenerator
from models.discriminator import TimeConditionedDiscriminator
from models.noise_schedule import DiffusionNoiseSchedule

from data.datasets import MultiContrastMRIDataset

from inference.data_consistency import DataConsistencyLayer, NoiseMixedDataConsistencyLayer
from inference.pipeline import ThreeStageInference

from metrics.reconstruction import MetricsTracker, compute_all_metrics


# ---------------------------------------------------------------------------
# Quantitative summary table
# ---------------------------------------------------------------------------

def print_summary_table(
    all_metrics: Dict[str, List[Dict[str, float]]],
    stage_names: List[str],
) -> None:
    """
    Print a formatted metrics table (matching FDMR Table 5 style).

    Args:
        all_metrics: Dict mapping stage name → list of per-sample metric dicts.
        stage_names: Ordered list of stage names to print.
    """
    import numpy as np

    header = f"{'Method':<20} {'PSNR (dB)':<15} {'SSIM':<12} {'NMSE':<12}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for stage in stage_names:
        samples = all_metrics.get(stage, [])
        if not samples:
            continue

        psnr_vals = [m["psnr"] for m in samples]
        ssim_vals = [m["ssim"] for m in samples]
        nmse_vals = [m["nmse"] for m in samples]

        psnr_m, psnr_s = float(np.mean(psnr_vals)), float(np.std(psnr_vals))
        ssim_m, ssim_s = float(np.mean(ssim_vals)), float(np.std(ssim_vals))
        nmse_m, nmse_s = float(np.mean(nmse_vals)), float(np.std(nmse_vals))

        print(
            f"{stage:<20} "
            f"{psnr_m:.2f}±{psnr_s:.2f}       "
            f"{ssim_m:.4f}±{ssim_s:.4f}  "
            f"{nmse_m:.4f}±{nmse_s:.4f}"
        )

    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# Main test function
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    # Load config with any CLI overrides
    overrides = list(args.overrides) if args.overrides else []
    if args.mask_type:
        overrides.append(f"inference.mask_type={args.mask_type}")
    if args.acceleration:
        overrides.append(f"inference.acceleration={args.acceleration}")

    config = load_config(args.config, overrides=overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Test] Using device: {device}")

    # Output directory for this test run
    out_dir = os.path.join(
        config.paths.output_dir,
        f"{config.data.dataset}_acc{config.inference.acceleration}_{config.inference.mask_type}",
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Build models
    # ----------------------------------------------------------------
    print("[Test] Building models...")
    generator = MCDISCOGenerator(config).to(device)
    refinement_gen = MCDISCOGenerator(config).to(device)

    # Load checkpoint
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    load_checkpoint(
        args.checkpoint, generator, refinement_gen, device=device
    )
    generator.eval()
    refinement_gen.eval()

    # ----------------------------------------------------------------
    # Build inference pipeline
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
        beta_end=config.diffusion.beta_end / 4,
        schedule=config.diffusion.beta_schedule,
    ).to(device)

    dc_layer  = DataConsistencyLayer().to(device)
    ndc_layer = NoiseMixedDataConsistencyLayer().to(device)

    inference_pipeline = ThreeStageInference(
        generator, refinement_gen,
        schedule, refine_schedule,
        dc_layer, ndc_layer, config, device=device,
    )

    # ----------------------------------------------------------------
    # Build test data loader
    # ----------------------------------------------------------------
    test_split = getattr(args, 'split', 'val') or 'val'
    print(f"[Test] Loading {config.data.dataset} {test_split} set...")
    test_dataset = MultiContrastMRIDataset(config, split=test_split)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"[Test] Test samples: {len(test_dataset)}")

    # ----------------------------------------------------------------
    # Metrics accumulators
    # ----------------------------------------------------------------
    stage_names = ["Zero-Fill", "Stage1", "Stage2 (DGP)", "Stage3 (Final)"]
    all_metrics: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    # Maximum number of samples for which to save visual figures
    max_figures = min(args.max_figures, len(test_dataset))

    inference_times = []

    # ----------------------------------------------------------------
    # Test loop
    # ----------------------------------------------------------------
    print(f"\n[Test] Running inference on {len(test_dataset)} samples...")

    with torch.no_grad():
        for sample_idx, batch in enumerate(test_loader):
            x0    = batch["x0"].to(device)      # [1, 4, H, W]
            y_obs = batch["y_obs"].to(device)    # [1, 4, H, W]
            mask  = batch["mask"].to(device)     # [1, 1, H, W]
            x_zf  = batch["x_zf"].to(device)    # [1, 4, H, W]

            t_start = time.perf_counter()

            # Run full 3-stage pipeline
            results = inference_pipeline.run(
                y_obs, mask,
                run_dgp=not args.no_dgp,
                run_refinement=not args.no_refinement,
            )

            t_elapsed = time.perf_counter() - t_start
            inference_times.append(t_elapsed)

            # Metrics per stage (mirroring FDMR Table 5 ablation)
            m_zf = compute_all_metrics(x_zf, x0)
            m_s1 = compute_all_metrics(results["stage1"], x0)
            m_s2 = compute_all_metrics(results["stage2"], x0)
            m_s3 = compute_all_metrics(results["stage3"], x0)

            all_metrics["Zero-Fill"].append(m_zf)
            all_metrics["Stage1"].append(m_s1)
            all_metrics["Stage2 (DGP)"].append(m_s2)
            all_metrics["Stage3 (Final)"].append(m_s3)

            # Progress
            if (sample_idx + 1) % 10 == 0 or sample_idx == 0:
                print(
                    f"  [{sample_idx+1}/{len(test_dataset)}] "
                    f"PSNR: ZF={m_zf['psnr']:.2f} "
                    f"S1={m_s1['psnr']:.2f} "
                    f"S2={m_s2['psnr']:.2f} "
                    f"S3={m_s3['psnr']:.2f} | "
                    f"time: {t_elapsed:.2f}s"
                )

            # Save comparison figures for the first max_figures samples
            if sample_idx < max_figures:
                save_comparison_figure(
                    x0_gt=x0,
                    x_zf=x_zf,
                    results={
                        "stage1": results["stage1"],
                        "stage2": results["stage2"],
                        "stage3": results["stage3"],
                    },
                    output_dir=out_dir,
                    idx=sample_idx,
                )

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    import numpy as np
    mean_time = float(np.mean(inference_times))
    print(f"\n[Test] Average inference time: {mean_time:.2f}s per sample")

    print_summary_table(all_metrics, stage_names)

    # Save metrics to text file
    summary_path = os.path.join(out_dir, "metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Dataset: {config.data.dataset}\n")
        f.write(f"Acceleration: {config.inference.acceleration}x\n")
        f.write(f"Mask type: {config.inference.mask_type}\n")
        f.write(f"Avg inference time: {mean_time:.2f}s\n\n")

        for stage in stage_names:
            samples = all_metrics[stage]
            if not samples:
                continue
            psnr_m = float(np.mean([m["psnr"] for m in samples]))
            psnr_s = float(np.std([m["psnr"] for m in samples]))
            ssim_m = float(np.mean([m["ssim"] for m in samples]))
            ssim_s = float(np.std([m["ssim"] for m in samples]))
            nmse_m = float(np.mean([m["nmse"] for m in samples]))
            nmse_s = float(np.std([m["nmse"] for m in samples]))
            f.write(
                f"{stage}: PSNR={psnr_m:.2f}±{psnr_s:.2f} "
                f"SSIM={ssim_m:.4f}±{ssim_s:.4f} "
                f"NMSE={nmse_m:.4f}±{nmse_s:.4f}\n"
            )

    print(f"[Test] Metrics summary saved to: {summary_path}")
    print(f"[Test] Visual comparisons saved to: {out_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Multi-Contrast DISCO-Diffusion GAN")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the trained model checkpoint (.pth)"
    )
    parser.add_argument(
        "--overrides", nargs="*", default=[],
        help="Config overrides in key=value format"
    )
    parser.add_argument(
        "--mask_type", type=str, default=None,
        choices=["random", "radial", "equi", "poisson", "gaussian"],
        help="Override mask type from config"
    )
    parser.add_argument(
        "--acceleration", type=int, default=None,
        help="Override acceleration factor (e.g. 5 or 10)"
    )
    parser.add_argument(
        "--no_dgp", action="store_true",
        help="Skip Stage 2 DGP adaptation (ablation)"
    )
    parser.add_argument(
        "--no_refinement", action="store_true",
        help="Skip Stage 3 diffusion refinement (ablation)"
    )
    parser.add_argument(
        "--max_figures", type=int, default=20,
        help="Maximum number of comparison figures to save"
    )
    parser.add_argument(
        "--split", type=str, default="val",
        choices=["val", "test"],
        help="Which data split to evaluate on (default: val)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
