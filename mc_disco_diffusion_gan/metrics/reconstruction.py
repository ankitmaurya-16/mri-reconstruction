"""
MRI reconstruction quality metrics.

Implements the three standard metrics used across all three papers:
  - PSNR  (Peak Signal-to-Noise Ratio) — higher is better
  - SSIM  (Structural Similarity Index) — higher is better
  - NMSE  (Normalized Mean Squared Error) — lower is better

All metrics are computed on the magnitude image (|real + j·imag|) per contrast,
then averaged across contrasts, consistent with FDMR Table 1–4 and Paper 2 Table 1.

Also provides NMSEᵣ = √NMSE (NRMSE) as used in Paper 3 (Levac et al., Eq. 21).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch

from data.transforms import complex_magnitude


# ---------------------------------------------------------------------------
# Core metric functions (operate on magnitude images)
# ---------------------------------------------------------------------------

def compute_psnr(
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    data_range: Optional[float] = None,
) -> float:
    """
    Peak Signal-to-Noise Ratio on magnitude images.

    PSNR = 20 · log10(data_range / RMSE)

    Applied per contrast, then averaged.

    Args:
        x_pred:     Predicted image [B, 4, H, W] (real 4-channel multi-contrast)
        x_gt:       Ground truth image [B, 4, H, W]
        data_range: Value range for normalization. If None, uses max of x_gt magnitude.

    Returns:
        Mean PSNR over batch and contrasts (dB).
    """
    mag_pred = complex_magnitude(x_pred)  # [B, 2, H, W]
    mag_gt   = complex_magnitude(x_gt)   # [B, 2, H, W]

    psnr_vals = []
    for b in range(mag_pred.shape[0]):
        for c in range(mag_pred.shape[1]):
            pred_c = mag_pred[b, c].float()
            gt_c   = mag_gt[b, c].float()

            dr = data_range if data_range is not None else gt_c.max().item()
            if dr < 1e-8:
                continue

            mse_val = ((pred_c - gt_c) ** 2).mean().item()
            if mse_val < 1e-12:
                psnr_vals.append(100.0)
            else:
                psnr_vals.append(20.0 * math.log10(dr / math.sqrt(mse_val)))

    return float(np.mean(psnr_vals)) if psnr_vals else 0.0


def compute_ssim(
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    data_range: Optional[float] = None,
) -> float:
    """
    Structural Similarity Index (SSIM) on magnitude images.

    Uses the standard SSIM formula (Wang et al., 2004) with
    k1=0.01, k2=0.03, 11×11 Gaussian window.

    Applied per contrast, then averaged.

    Args:
        x_pred:     Predicted image [B, 4, H, W]
        x_gt:       Ground truth image [B, 4, H, W]
        data_range: Value range. If None, uses max of x_gt magnitude.

    Returns:
        Mean SSIM over batch and contrasts.
    """
    try:
        from skimage.metrics import structural_similarity as sk_ssim
    except ImportError:
        raise ImportError("scikit-image required for SSIM: pip install scikit-image")

    mag_pred = complex_magnitude(x_pred).detach().cpu().numpy()
    mag_gt   = complex_magnitude(x_gt).detach().cpu().numpy()

    ssim_vals = []
    for b in range(mag_pred.shape[0]):
        for c in range(mag_pred.shape[1]):
            pred_c = mag_pred[b, c]
            gt_c   = mag_gt[b, c]

            dr = data_range if data_range is not None else gt_c.max()
            if dr < 1e-8:
                continue

            val = sk_ssim(pred_c, gt_c, data_range=float(dr))
            ssim_vals.append(val)

    return float(np.mean(ssim_vals)) if ssim_vals else 0.0


def compute_nmse(
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
) -> float:
    """
    Normalized Mean Squared Error on magnitude images.

    NMSE = ||x_gt - x_pred||_2^2 / ||x_gt||_2^2

    Applied per contrast, then averaged.

    Note: Paper 3 (Levac et al., Eq. 21) uses NRMSE = √NMSE.
    Both are returned by compute_all_metrics().

    Args:
        x_pred: Predicted image [B, 4, H, W]
        x_gt:   Ground truth image [B, 4, H, W]

    Returns:
        Mean NMSE over batch and contrasts.
    """
    mag_pred = complex_magnitude(x_pred)  # [B, 2, H, W]
    mag_gt   = complex_magnitude(x_gt)

    nmse_vals = []
    for b in range(mag_pred.shape[0]):
        for c in range(mag_pred.shape[1]):
            pred_c = mag_pred[b, c].float()
            gt_c   = mag_gt[b, c].float()

            norm_gt = (gt_c ** 2).sum().item()
            if norm_gt < 1e-12:
                continue

            nmse_val = ((gt_c - pred_c) ** 2).sum().item() / norm_gt
            nmse_vals.append(nmse_val)

    return float(np.mean(nmse_vals)) if nmse_vals else 0.0


def compute_all_metrics(
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    data_range: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute all reconstruction metrics at once.

    Returns:
        dict with keys: "psnr", "ssim", "nmse", "nrmse"
    """
    psnr = compute_psnr(x_pred, x_gt, data_range)
    ssim = compute_ssim(x_pred, x_gt, data_range)
    nmse = compute_nmse(x_pred, x_gt)
    nrmse = math.sqrt(max(nmse, 0.0))  # Paper 3 Eq. 21

    return {"psnr": psnr, "ssim": ssim, "nmse": nmse, "nrmse": nrmse}


# ---------------------------------------------------------------------------
# Per-contrast metrics (for detailed analysis)
# ---------------------------------------------------------------------------

def compute_metrics_per_contrast(
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    data_range: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics separately for each contrast.

    Returns:
        dict with keys "c1" and "c2", each containing {"psnr", "ssim", "nmse"}
    """
    results = {}
    for c_idx, c_name in enumerate(["c1", "c2"]):
        # Extract single-contrast 4-ch tensor by zeroing out the other contrast
        pred_c = x_pred.clone()
        gt_c = x_gt.clone()

        if c_idx == 0:
            pred_c = pred_c[:, :2]   # [B, 2, H, W]
            gt_c   = gt_c[:, :2]
        else:
            pred_c = pred_c[:, 2:]
            gt_c   = gt_c[:, 2:]

        # Wrap as 4ch with zeros for the other contrast to reuse compute functions
        def _to_4ch(x2: torch.Tensor) -> torch.Tensor:
            zeros = torch.zeros_like(x2)
            return torch.cat([x2, zeros] if c_idx == 0 else [zeros, x2], dim=1)

        results[c_name] = compute_all_metrics(_to_4ch(pred_c), _to_4ch(gt_c), data_range)

    return results


# ---------------------------------------------------------------------------
# Running metrics tracker
# ---------------------------------------------------------------------------

class MetricsTracker:
    """
    Running-average metrics tracker for one epoch of validation/test.

    Usage:
        tracker = MetricsTracker()
        for batch in loader:
            x_pred = model(batch)
            tracker.update(x_pred, batch['x0'])
        results = tracker.compute()
        # {'psnr': 37.5, 'ssim': 0.982, 'nmse': 0.003, 'nrmse': 0.055}
        tracker.reset()
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators."""
        self._psnr: List[float] = []
        self._ssim: List[float] = []
        self._nmse: List[float] = []
        self._nrmse: List[float] = []

    def update(
        self,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        data_range: Optional[float] = None,
    ) -> None:
        """
        Update running averages with a batch.

        Args:
            x_pred: Predicted images [B, 4, H, W]
            x_gt:   Ground truth images [B, 4, H, W]
        """
        with torch.no_grad():
            metrics = compute_all_metrics(x_pred, x_gt, data_range)
        self._psnr.append(metrics["psnr"])
        self._ssim.append(metrics["ssim"])
        self._nmse.append(metrics["nmse"])
        self._nrmse.append(metrics["nrmse"])

    def compute(self) -> Dict[str, float]:
        """
        Compute mean ± std for all metrics.

        Returns:
            dict with "psnr_mean", "psnr_std", "ssim_mean", "ssim_std",
                      "nmse_mean", "nmse_std", "nrmse_mean", "nrmse_std"
        """
        def _stats(vals: List[float]) -> tuple:
            arr = np.array(vals)
            return float(arr.mean()), float(arr.std())

        p_m, p_s = _stats(self._psnr)
        s_m, s_s = _stats(self._ssim)
        n_m, n_s = _stats(self._nmse)
        nr_m, nr_s = _stats(self._nrmse)

        return {
            "psnr_mean": p_m, "psnr_std": p_s,
            "ssim_mean": s_m, "ssim_std": s_s,
            "nmse_mean": n_m, "nmse_std": n_s,
            "nrmse_mean": nr_m, "nrmse_std": nr_s,
        }

    def summary_str(self) -> str:
        """Return a human-readable summary string."""
        m = self.compute()
        return (
            f"PSNR: {m['psnr_mean']:.2f}±{m['psnr_std']:.2f} dB | "
            f"SSIM: {m['ssim_mean']:.4f}±{m['ssim_std']:.4f} | "
            f"NMSE: {m['nmse_mean']:.4f}±{m['nmse_std']:.4f}"
        )
