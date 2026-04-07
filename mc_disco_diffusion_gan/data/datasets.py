"""
Dataset classes for multi-contrast MRI reconstruction.

Supports three datasets from the papers:
  - CC-359 (Calgary-Campinas-359): T1 brain, single contrast (FDMR experiments)
  - IXI:   T1/T2/PD brain, multi-modal (FDMR and Paper 3 multi-contrast)
  - SKM-TEA: DESS gradient echo knee, two echoes as paired contrasts (Paper 3)

All datasets return complex MRI data in real 4-channel format:
    [real_c1, imag_c1, real_c2, imag_c2]

For single-contrast datasets (CC-359), the same contrast is duplicated into
both channels for compatibility with the 4-channel architecture.

The MultiContrastMRIDataset wrapper applies on-the-fly undersampling and
returns ready-to-use training batches.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .transforms import (
        complex_to_real,
        normalize_volume,
        pair_to_4channel,
        apply_mask_kspace,
        zero_fill,
    )
    from .masks import MaskFactory
except ImportError:
    from data.transforms import (
        complex_to_real,
        normalize_volume,
        pair_to_4channel,
        apply_mask_kspace,
        zero_fill,
    )
    from data.masks import MaskFactory
from utils.config import Config


# ---------------------------------------------------------------------------
# Base helpers
# ---------------------------------------------------------------------------

def _load_h5_slice(filepath: str, key: str = "reconstruction_rss") -> np.ndarray:
    """Load a 2D slice from an HDF5 file (fastMRI format)."""
    with h5py.File(filepath, "r") as f:
        return f[key][:]


def _to_complex_slice(arr: np.ndarray) -> torch.Tensor:
    """
    Convert a numpy array to a complex torch tensor.
    Handles both real (treated as real-only) and complex numpy arrays.
    """
    if np.isrealobj(arr):
        return torch.from_numpy(arr.astype(np.float32)).to(torch.complex64)
    else:
        return torch.from_numpy(arr.astype(np.complex64))


# ---------------------------------------------------------------------------
# CC-359 Dataset (single contrast: T1 brain)
# ---------------------------------------------------------------------------

class CC359Dataset(Dataset):
    """
    Calgary-Campinas-359 (CC-359) T1 brain MRI dataset.

    Structure:
        data_root/
          subject_001.h5   (key: "reconstruction_rss", shape [num_slices, H, W])
          subject_002.h5
          ...

    FDMR experiments: 24 volumes for training, 10 for inference (Section 5.1.1).
    Image size: 256 × 256.

    Since CC-359 is single contrast, both contrast channels are set to the same
    T1 image to maintain 4-channel compatibility with the multi-contrast architecture.
    For ablation to single-contrast FDMR, set config.data.num_contrasts=1.

    Args:
        data_root:  Path to the directory containing .h5 files.
        split:      "train" or "val" (or "test").
        train_frac: Fraction of volumes used for training.
        image_size: Spatial crop/resize target.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        train_frac: float = 0.9,
        image_size: int = 256,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.split = split

        # Collect all .h5 files
        root = Path(data_root)
        all_files = sorted(root.glob("*.h5"))

        n_train = int(len(all_files) * train_frac)
        if split == "train":
            self.files = all_files[:n_train]
        else:
            self.files = all_files[n_train:]

        # Build a flat index: (file_idx, slice_idx)
        self.index: List[Tuple[int, int]] = []
        self._slice_counts: List[int] = []
        for f_idx, fpath in enumerate(self.files):
            with h5py.File(str(fpath), "r") as hf:
                # Try different common keys
                for key in ["reconstruction_rss", "kspace", "reconstruction_esc"]:
                    if key in hf:
                        n_slices = hf[key].shape[0]
                        break
                else:
                    n_slices = 0
            self._slice_counts.append(n_slices)
            self.index.extend([(f_idx, s) for s in range(n_slices)])

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        f_idx, s_idx = self.index[idx]
        fpath = self.files[f_idx]

        with h5py.File(str(fpath), "r") as hf:
            for key in ["reconstruction_rss", "kspace", "reconstruction_esc"]:
                if key in hf:
                    sl = hf[key][s_idx]  # [H, W] real or [coils, H, W] complex
                    break

        # If multi-coil k-space, do RSS coil combination
        if sl.ndim == 3:
            sl_cplx = np.fft.ifft2(sl, axes=(-2, -1))
            sl = np.sqrt((np.abs(sl_cplx) ** 2).sum(axis=0))

        # To complex tensor [H, W]
        x_cplx = _to_complex_slice(sl.squeeze())

        # Crop / pad to image_size × image_size
        x_cplx = self._center_crop(x_cplx, self.image_size)

        # Real 2-channel representation [2, H, W]
        x_real = complex_to_real(x_cplx.unsqueeze(0)).squeeze(0)

        # Normalize by 99th percentile magnitude
        x_4ch = torch.cat([x_real, x_real], dim=0)  # duplicate: [4, H, W]
        x_4ch, _ = normalize_volume(x_4ch.unsqueeze(0))
        x_4ch = x_4ch.squeeze(0)

        return {"x0": x_4ch, "file": str(fpath), "slice": s_idx}

    @staticmethod
    def _center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
        """Center crop or pad a 2D tensor [H, W] to [size, size]."""
        H, W = x.shape[-2], x.shape[-1]
        # Crop
        h_start = max(0, (H - size) // 2)
        w_start = max(0, (W - size) // 2)
        x = x[..., h_start:h_start + min(H, size), w_start:w_start + min(W, size)]
        # Pad if smaller than size
        H2, W2 = x.shape[-2], x.shape[-1]
        pad_h = max(0, size - H2)
        pad_w = max(0, size - W2)
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
        return x


# ---------------------------------------------------------------------------
# IXI Dataset (multi-modal: T1, T2, PD)
# ---------------------------------------------------------------------------

class IXIDataset(Dataset):
    """
    IXI multi-modal brain dataset (T1, T2, PD-weighted images).

    Expected directory structure:
        data_root/
          T1/  subject_001.h5  subject_002.h5  ...
          T2/  subject_001.h5  ...
          PD/  subject_001.h5  ...

    FDMR experiments: 578 subjects, 256×256, 90/10 split (Section 5.1.1).
    Paper 3: Uses T1 and T2 as the two contrasts for multi-contrast reconstruction.

    __getitem__ returns a paired (c1=T1, c2=T2) slice as a 4-channel tensor.
    """

    def __init__(
        self,
        data_root: str,
        contrast_1: str = "T1",
        contrast_2: str = "T2",
        split: str = "train",
        train_frac: float = 0.9,
        image_size: int = 256,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.c1 = contrast_1
        self.c2 = contrast_2

        root = Path(data_root)
        dir_c1 = root / contrast_1
        dir_c2 = root / contrast_2

        files_c1 = sorted(dir_c1.glob("*.h5")) if dir_c1.exists() else []
        files_c2 = sorted(dir_c2.glob("*.h5")) if dir_c2.exists() else []

        # Match paired subjects by filename stem
        stems_c1 = {f.stem: f for f in files_c1}
        stems_c2 = {f.stem: f for f in files_c2}
        common = sorted(set(stems_c1) & set(stems_c2))

        n_train = int(len(common) * train_frac)
        subjects = common[:n_train] if split == "train" else common[n_train:]

        self.pairs: List[Tuple[Path, Path]] = [
            (stems_c1[s], stems_c2[s]) for s in subjects
        ]

        # Build flat slice index
        self.index: List[Tuple[int, int]] = []
        for subj_idx, (fc1, fc2) in enumerate(self.pairs):
            with h5py.File(str(fc1), "r") as hf:
                for key in ["reconstruction_rss", "kspace"]:
                    if key in hf:
                        n = hf[key].shape[0]
                        break
                else:
                    n = 0
            self.index.extend([(subj_idx, s) for s in range(n)])

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        subj_idx, s_idx = self.index[idx]
        fc1, fc2 = self.pairs[subj_idx]

        def _load(fpath: Path) -> torch.Tensor:
            with h5py.File(str(fpath), "r") as hf:
                for key in ["reconstruction_rss", "kspace"]:
                    if key in hf:
                        sl = hf[key][s_idx]
                        break
            if sl.ndim == 3:  # multi-coil
                sl_cplx = np.fft.ifft2(sl, axes=(-2, -1))
                sl = np.sqrt((np.abs(sl_cplx) ** 2).sum(axis=0))
            x_cplx = _to_complex_slice(sl.squeeze())
            x_cplx = CC359Dataset._center_crop(x_cplx, self.image_size)
            return complex_to_real(x_cplx.unsqueeze(0)).squeeze(0)  # [2, H, W]

        x_c1 = _load(fc1)  # [2, H, W]
        x_c2 = _load(fc2)  # [2, H, W]

        # Joint 99th-percentile normalization (Paper 3 Section III-C)
        x_4ch = pair_to_4channel(x_c1.unsqueeze(0), x_c2.unsqueeze(0)).squeeze(0)
        x_4ch, _ = normalize_volume(x_4ch.unsqueeze(0))
        x_4ch = x_4ch.squeeze(0)

        return {"x0": x_4ch, "file_c1": str(fc1), "file_c2": str(fc2), "slice": s_idx}


# ---------------------------------------------------------------------------
# SKM-TEA Dataset (DESS knee: two echoes as paired contrasts)
# ---------------------------------------------------------------------------

class SKMTEADataset(Dataset):
    """
    SKM-TEA dataset (Paper 3, Levac et al., Section III-C).

    Contains paired DESS (Double Echo Steady State) gradient echo knee images.
    The two echoes serve as the two contrasts (x_1, x_2) for multi-contrast
    reconstruction. x_2 has weaker signal due to T2 relaxation between echoes.

    Expected structure:
        data_root/
          volume_001.h5  (keys: "echo1" and "echo2", shape [slices, H, W] complex)
          volume_002.h5
          ...

    Paper 3 experiments: 150 training volumes, 100 test slices from 3 held-out volumes.

    Args:
        data_root:  Path to data directory.
        split:      "train", "val", or "test".
        train_frac: Training fraction (default 0.85 to leave ~3 volumes for test).
        image_size: Spatial size.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        train_frac: float = 0.85,
        image_size: int = 256,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        root = Path(data_root)
        all_files = sorted(root.glob("*.h5"))

        n_train = int(len(all_files) * train_frac)
        if split == "train":
            self.files = all_files[:n_train]
        elif split in ("val", "test"):
            self.files = all_files[n_train:]
        else:
            self.files = all_files

        self.index: List[Tuple[int, int]] = []
        for f_idx, fpath in enumerate(self.files):
            with h5py.File(str(fpath), "r") as hf:
                key = "echo1" if "echo1" in hf else list(hf.keys())[0]
                n = hf[key].shape[0]
            self.index.extend([(f_idx, s) for s in range(n)])

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        f_idx, s_idx = self.index[idx]
        fpath = self.files[f_idx]

        with h5py.File(str(fpath), "r") as hf:
            key1 = "echo1" if "echo1" in hf else list(hf.keys())[0]
            key2 = "echo2" if "echo2" in hf else list(hf.keys())[1]
            sl1 = hf[key1][s_idx]
            sl2 = hf[key2][s_idx]

        def _process(sl: np.ndarray) -> torch.Tensor:
            if sl.ndim == 3:  # coil dimension
                sl = np.sqrt((np.abs(sl) ** 2).sum(axis=0))
            x = _to_complex_slice(sl.squeeze())
            x = CC359Dataset._center_crop(x, self.image_size)
            return complex_to_real(x.unsqueeze(0)).squeeze(0)  # [2, H, W]

        x_c1 = _process(sl1)
        x_c2 = _process(sl2)

        # Joint normalization by 99th percentile of |x_c1| (Paper 3 Section III-C)
        x_4ch = pair_to_4channel(x_c1.unsqueeze(0), x_c2.unsqueeze(0)).squeeze(0)
        x_4ch, _ = normalize_volume(x_4ch.unsqueeze(0))
        x_4ch = x_4ch.squeeze(0)

        return {"x0": x_4ch, "file": str(fpath), "slice": s_idx}


# ---------------------------------------------------------------------------
# fastMRI Dataset (single contrast: knee/brain)
# ---------------------------------------------------------------------------

class FastMRIDataset(Dataset):
    """
    fastMRI dataset (single contrast knee or brain MRI).

    Supports TWO directory layouts:

    Layout A — Pre-split directories (standard fastMRI download):
        data_root/
          knee_singlecoil_train/   (or singlecoil_train/, multicoil_train/, etc.)
            file1000000.h5
            file1000001.h5  ...
          knee_singlecoil_val/
            ...
          knee_singlecoil_test/
            ...

    Layout B — Flat directory (all .h5 files together):
        data_root/
          file1.h5
          file2.h5  ...

    Each .h5 file uses the standard fastMRI format with keys:
      - 'kspace': multi-coil k-space data [slices, coils, H, W] complex64
      - 'reconstruction_rss': RSS magnitude images [slices, H, W] float32
      - 'reconstruction_esc': ESC magnitude images [slices, H, W] float32

    Since fastMRI is single contrast, both contrast channels are set to the same
    image to maintain 4-channel compatibility with the multi-contrast architecture.

    Args:
        data_root:  Path to the root data directory.
        split:      "train", "val", or "test".
        train_frac: Fraction of data used for training (only for flat layout).
        image_size: Spatial crop/resize target.
        max_files:  Max number of .h5 files to use (None = all). Useful for
                    Colab where you can't fit the full 72 GB train set.
    """

    # Patterns to search for pre-split subdirectories (tried in order)
    _SPLIT_DIR_PATTERNS = {
        "train": ["knee_singlecoil_train", "singlecoil_train", "multicoil_train", "train"],
        "val":   ["knee_singlecoil_val", "singlecoil_val", "multicoil_val", "val"],
        "test":  ["knee_singlecoil_test", "singlecoil_test", "multicoil_test", "test"],
    }

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        train_frac: float = 0.9,
        image_size: int = 320,
        max_files: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.split = split

        root = Path(data_root)

        # --- Resolve file list ---
        split_dir = self._find_split_dir(root, split)

        if split_dir is not None:
            # Layout A: pre-split directories
            all_files = sorted(split_dir.glob("*.h5"))
            if len(all_files) == 0:
                raise FileNotFoundError(
                    f"No .h5 files found in {split_dir}. "
                    f"Check that you placed your fastMRI data correctly."
                )
            self.files = all_files
            self._single_file_mode = False
            print(f"[FastMRI] Using pre-split dir: {split_dir}  ({len(all_files)} files)")
        else:
            # Layout B: flat directory — split by fraction
            all_files = sorted(root.glob("*.h5"))
            if len(all_files) == 0:
                raise FileNotFoundError(
                    f"No .h5 files found in {data_root} or its subdirectories.\n"
                    f"Expected either:\n"
                    f"  {data_root}/knee_singlecoil_train/*.h5  (pre-split layout)\n"
                    f"  {data_root}/*.h5                        (flat layout)"
                )

            if len(all_files) == 1:
                self.files = all_files
                self._single_file_mode = True
            else:
                n_train = max(1, int(len(all_files) * train_frac))
                if split == "train":
                    self.files = all_files[:n_train]
                else:
                    self.files = all_files[n_train:] if n_train < len(all_files) else all_files[-1:]
                self._single_file_mode = False

        # Optionally cap number of files (for Colab / quick experiments)
        if max_files is not None and len(self.files) > max_files:
            print(f"[FastMRI] Limiting to {max_files}/{len(self.files)} files (max_files={max_files})")
            self.files = self.files[:max_files]

        # --- Build flat slice index: (file_idx, slice_idx) ---
        self.index: List[Tuple[int, int]] = []
        self._slice_counts: List[int] = []
        for f_idx, fpath in enumerate(self.files):
            with h5py.File(str(fpath), "r") as hf:
                for key in ["reconstruction_rss", "kspace", "reconstruction_esc"]:
                    if key in hf:
                        n_slices = hf[key].shape[0]
                        break
                else:
                    n_slices = 0
            self._slice_counts.append(n_slices)
            self.index.extend([(f_idx, s) for s in range(n_slices)])

        # If single file, split slices between train and val
        if getattr(self, '_single_file_mode', False) and len(self.index) > 1:
            n_train_slices = max(1, int(len(self.index) * train_frac))
            if split == "train":
                self.index = self.index[:n_train_slices]
            else:
                self.index = self.index[n_train_slices:]
                if len(self.index) == 0:
                    self.index = [(0, self._slice_counts[0] - 1)]

    @classmethod
    def _find_split_dir(cls, root: Path, split: str) -> Optional[Path]:
        """Search for a pre-split subdirectory matching the requested split."""
        patterns = cls._SPLIT_DIR_PATTERNS.get(split, cls._SPLIT_DIR_PATTERNS.get("val", []))
        for name in patterns:
            candidate = root / name
            if candidate.is_dir():
                return candidate
        return None

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        f_idx, s_idx = self.index[idx]
        fpath = self.files[f_idx]

        with h5py.File(str(fpath), "r") as hf:
            # Prefer reconstruction_rss (magnitude), fall back to kspace
            if "reconstruction_rss" in hf:
                sl = hf["reconstruction_rss"][s_idx]  # [H, W] float32
            elif "kspace" in hf:
                ksp = hf["kspace"][s_idx]  # [coils, H, W] complex64
                sl_cplx = np.fft.ifft2(ksp, axes=(-2, -1))
                sl = np.sqrt((np.abs(sl_cplx) ** 2).sum(axis=0))  # RSS
            elif "reconstruction_esc" in hf:
                sl = hf["reconstruction_esc"][s_idx]
            else:
                raise KeyError(f"No recognized data key in {fpath}")

        # Ensure 2D
        if sl.ndim == 3:
            sl_cplx = np.fft.ifft2(sl, axes=(-2, -1))
            sl = np.sqrt((np.abs(sl_cplx) ** 2).sum(axis=0))

        # To complex tensor [H, W]
        x_cplx = _to_complex_slice(sl.squeeze())

        # Center crop/pad to image_size × image_size
        x_cplx = CC359Dataset._center_crop(x_cplx, self.image_size)

        # Real 2-channel representation [2, H, W]
        x_real = complex_to_real(x_cplx.unsqueeze(0)).squeeze(0)

        # Duplicate for 4-channel (single contrast → both channels same)
        x_4ch = torch.cat([x_real, x_real], dim=0)  # [4, H, W]
        x_4ch, _ = normalize_volume(x_4ch.unsqueeze(0))
        x_4ch = x_4ch.squeeze(0)

        return {"x0": x_4ch, "file": str(fpath), "slice": s_idx}


# ---------------------------------------------------------------------------
# Multi-Contrast MRI Dataset Wrapper
# ---------------------------------------------------------------------------

class MultiContrastMRIDataset(Dataset):
    """
    Wrapper dataset that applies on-the-fly undersampling to any of the base datasets.

    Returns batches with:
        x0:      Fully sampled 4-channel ground truth [4, H, W]
        y_obs:   Undersampled k-space [4, H, W] (zeros at unobserved)
        mask:    Binary undersampling mask [1, H, W]
        x_zf:    Zero-fill reconstruction (baseline) [4, H, W]

    The mask is regenerated per sample using a random seed derived from the
    sample index, ensuring deterministic but diverse masks across the dataset.

    Args:
        config:  Global Config instance.
        split:   "train", "val", or "test".
    """

    def __init__(self, config: Config, split: str = "train") -> None:
        super().__init__()

        self.config = config
        self.split = split
        dc = config.data
        pc = config.paths

        # Select base dataset based on config
        if dc.dataset.upper() == "CC359":
            self.base_dataset = CC359Dataset(
                data_root=pc.data_root,
                split=split,
                train_frac=dc.train_split,
                image_size=dc.image_size,
            )
        elif dc.dataset.upper() == "IXI":
            self.base_dataset = IXIDataset(
                data_root=pc.data_root,
                split=split,
                train_frac=dc.train_split,
                image_size=dc.image_size,
            )
        elif dc.dataset.upper() == "SKMTEA":
            self.base_dataset = SKMTEADataset(
                data_root=pc.data_root,
                split=split,
                train_frac=dc.train_split,
                image_size=dc.image_size,
            )
        elif dc.dataset.upper() == "FASTMRI":
            self.base_dataset = FastMRIDataset(
                data_root=pc.data_root,
                split=split,
                train_frac=dc.train_split,
                image_size=dc.image_size,
                max_files=getattr(dc, 'max_files', None) or None,
            )
        else:
            raise ValueError(f"Unknown dataset: {dc.dataset}")

        self.mask_type = config.inference.mask_type
        self.acceleration = config.inference.acceleration
        self.center_fraction = dc.center_fraction
        self.image_size = dc.image_size

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get ground truth
        sample = self.base_dataset[idx]
        x0 = sample["x0"]  # [4, H, W]

        # Generate a deterministic mask per sample (train: varied; test: fixed)
        seed = idx if self.split != "train" else None
        mask = MaskFactory.get_mask(
            self.mask_type,
            shape=(self.image_size, self.image_size),
            acceleration=self.acceleration,
            center_fraction=self.center_fraction,
            seed=seed,
        )  # [H, W]
        mask = mask.unsqueeze(0)  # [1, H, W] — broadcast over channel dim

        # Simulate undersampled measurement
        y_obs, _ = apply_mask_kspace(x0.unsqueeze(0), mask.unsqueeze(0))
        y_obs = y_obs.squeeze(0)  # [4, H, W]

        # Zero-fill baseline reconstruction
        x_zf = zero_fill(y_obs.unsqueeze(0)).squeeze(0)  # [4, H, W]

        result = {
            "x0": x0,          # [4, H, W] ground truth
            "y_obs": y_obs,    # [4, H, W] undersampled k-space
            "mask": mask,      # [1, H, W] binary mask
            "x_zf": x_zf,     # [4, H, W] zero-fill reconstruction
        }

        # Pass through any metadata from base dataset
        for k, v in sample.items():
            if k != "x0":
                result[k] = v

        return result
