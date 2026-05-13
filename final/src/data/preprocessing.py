"""
RGB fusion and slice extraction from DCE-MRI NIfTI volumes.

Implements the paper's approach (Section 2.5):
  - Pre-contrast      -> Red channel
  - Early post-contrast -> Green channel
  - Late post-contrast  -> Blue channel
Each channel is MinMax-normalized per-slice to [0, 255] uint8.
"""

import os
import re
import functools
import numpy as np
import nibabel as nib


NIFTI_SUFFIX = {
    "spy2": "_spy2_vis1_dce_aqc_",
    "spy1": "_spy1_vis1_acq",
    "duke": "_duke_aqc_",
}
MASK_SUFFIX = {
    "spy2": "_spy2_vis1_mask",
    "spy1": "_spy1_vis1_mask",
    "duke": "_duke_mask",
}

_nifti_paths: dict[str, str] = {}
_mask_paths: dict[str, str] = {}


def setup_paths(nifti_dirs: dict[str, str], mask_dirs: dict[str, str]):
    global _nifti_paths, _mask_paths
    _nifti_paths = nifti_dirs
    _mask_paths = mask_dirs
    # Cohort dirs may have changed; cached arrays would point at the wrong cohort.
    load_acquisitions.cache_clear()
    load_mask.cache_clear()


def _cohort_from_pid(pid: str) -> str:
    if "ISPY1" in pid or "SPY1" in pid:
        return "spy1"
    if "ISPY2" in pid or "SPY2" in pid or "ACRIN-6698" in pid:
        return "spy2"
    if "Breast_MRI" in pid or "MRI" in pid:
        return "duke"
    raise ValueError(f"Cannot infer cohort for pid={pid!r}")


def _last_int(path):
    m = re.search(r"(\d+)\.nii\.gz$", path)
    return int(m.group(1)) if m else -1


# Per-worker LRU caches: __getitem__ is called n_slices times per patient,
# so without caching we'd decode each NIfTI 8x per epoch. dataobj is a lazy
# proxy; np.asarray materialises in float32 to halve RAM vs get_fdata's float64.
@functools.lru_cache(maxsize=64)
def load_acquisitions(pid: str) -> list[np.ndarray] | None:
    cohort = _cohort_from_pid(pid)
    fpath = _nifti_paths.get(cohort)
    if fpath is None or not os.path.isdir(fpath):
        return None
    files = sorted(
        [f for f in os.listdir(fpath) if pid in f],
        key=_last_int,
    )
    if not files:
        return None
    return [
        np.asarray(nib.load(os.path.join(fpath, f)).dataobj, dtype=np.float32)
        for f in files
    ]


@functools.lru_cache(maxsize=256)
def load_mask(pid: str) -> np.ndarray | None:
    cohort = _cohort_from_pid(pid)
    fpath = _mask_paths.get(cohort)
    if fpath is None or not os.path.isdir(fpath):
        return None
    files = [f for f in os.listdir(fpath) if pid in f]
    if not files:
        return None
    return np.asarray(nib.load(os.path.join(fpath, files[0])).dataobj, dtype=np.uint8)


def find_tumor_z_range(mask: np.ndarray) -> tuple[int, int] | None:
    active = np.any(mask, axis=(0, 1))
    idx = np.where(active)[0]
    if idx.size == 0:
        return None
    return int(idx[0]), int(idx[-1])


def _minmax_uint8(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def _percentile_uint8_volume(
    volumes: list[np.ndarray],
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
) -> list[np.ndarray]:
    """Per-volume, channel-SHARED percentile normalization for one patient's
    DCE timepoints. Computes (lo, hi) from nonzero voxels pooled across all
    volumes, then maps each volume to uint8 with the SHARED range so that the
    relative intensity between timepoints — the actual DCE signal — is
    preserved across the resulting RGB channels."""
    stacked = np.concatenate([v[v > 0].ravel() for v in volumes]) \
        if any(v.size > 0 for v in volumes) else np.array([], dtype=np.float32)
    if stacked.size == 0:
        return [np.zeros_like(v, dtype=np.uint8) for v in volumes]
    lo = float(np.percentile(stacked, lo_pct))
    hi = float(np.percentile(stacked, hi_pct))
    if hi - lo < 1e-8:
        return [np.zeros_like(v, dtype=np.uint8) for v in volumes]
    out = []
    for v in volumes:
        scaled = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
        out.append((scaled * 255).astype(np.uint8))
    return out


def fuse_rgb_slice(
    pre: np.ndarray,
    early_post: np.ndarray,
    late_post: np.ndarray,
) -> np.ndarray:
    """
    DEPRECATED. Per-slice MinMax fusion. Each channel is independently
    rescaled to [0, 255] so the relative intensity *between* DCE timepoints —
    the entire signal of DCE-MRI — is normalized away. Kept only for the
    `normalization: minmax_per_slice` ablation path in BreastDCEDataset. New
    code should use `_percentile_uint8_volume` and stack slices manually.
    """
    r = _minmax_uint8(pre.astype(np.float32))
    g = _minmax_uint8(early_post.astype(np.float32))
    b = _minmax_uint8(late_post.astype(np.float32))
    return np.stack([r, g, b], axis=-1)


def select_timepoints(
    acqs: list[np.ndarray],
    cohort: str,
    idx_pre: int | None = None,
    idx_early: int | None = None,
    idx_late: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select the 3 acquisitions for RGB fusion.

    When explicit indices are provided (from the MinCrop metadata columns
    `pre`, `post_early`, `post_late`), use them directly — each patient has
    per-scan clinically-curated timepoints. Otherwise fall back to the
    paper's cohort-level defaults:
      I-SPY: 0, 2, min(last_index, 6)
      Duke:  0, 1, final acquisition
    """
    n = len(acqs)

    def _pick(i):
        return acqs[max(0, min(int(i), n - 1))]

    if idx_pre is not None and idx_early is not None and idx_late is not None:
        return _pick(idx_pre), _pick(idx_early), _pick(idx_late)

    if cohort in ("spy1", "spy2"):
        pre = acqs[0]
        early = acqs[min(2, n - 1)]
        late = acqs[min(n - 1, 6)]
    else:
        pre = acqs[0]
        early = acqs[min(1, n - 1)]
        late = acqs[-1]
    return pre, early, late
