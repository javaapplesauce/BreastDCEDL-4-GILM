"""
DCE-MRI dataset for pCR prediction.

Each patient contributes n_slices 2D RGB images centered on the tumor mid-plane.
RGB channels encode pre-contrast, early post-contrast, and late post-contrast
DCE phases via MinMax normalization (Section 2.5 of the paper).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from src.data.preprocessing import (
    load_acquisitions,
    load_mask,
    find_tumor_z_range,
    fuse_rgb_slice,
    select_timepoints,
    _cohort_from_pid,
)


def build_transforms(cfg_aug: dict, is_train: bool) -> transforms.Compose:
    if not is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    ops = []
    if cfg_aug.get("horizontal_flip"):
        ops.append(transforms.RandomHorizontalFlip())
    if cfg_aug.get("vertical_flip"):
        ops.append(transforms.RandomVerticalFlip())
    rot = cfg_aug.get("rotation", 0)
    if rot:
        ops.append(transforms.RandomRotation(rot))
    tr = cfg_aug.get("translate")
    sc = cfg_aug.get("scale")
    if tr or sc:
        ops.append(transforms.RandomAffine(
            degrees=0,
            translate=tuple(tr) if tr else None,
            scale=tuple(sc) if sc else None,
        ))
    ops.append(transforms.Resize((224, 224)))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]))
    er = cfg_aug.get("random_erasing", 0)
    if er:
        ops.append(transforms.RandomErasing(p=er, scale=(0.02, 0.15)))
    return transforms.Compose(ops)


def _roi_centre(row: pd.Series, vol_shape: tuple) -> tuple[int, int]:
    h, w = vol_shape[0], vol_shape[1]
    if "scol" in row.index and pd.notna(row.get("scol")):
        sc, ec = int(row["scol"]), int(row.get("ecol", row["scol"]))
        sr, er = int(row["sraw"]), int(row.get("eraw", row["sraw"]))
        return (sc + ec) // 2, (sr + er) // 2
    if "voi_start_x" in row.index and pd.notna(row.get("voi_start_x")):
        cx = int((row["voi_start_x"] + row["voi_end_x"]) / 2)
        cy = int((row["voi_start_y"] + row["voi_end_y"]) / 2)
        return cx, cy
    return w // 2, h // 2


def _z_range(row: pd.Series, vol_depth: int) -> tuple[int, int]:
    if "mask_start" in row.index and pd.notna(row.get("mask_start")):
        return max(int(row["mask_start"]), 0), min(int(row["mask_end"]), vol_depth - 1)
    if "voi_start_z" in row.index and pd.notna(row.get("voi_start_z")):
        return max(int(row["voi_start_z"]), 0), min(int(row["voi_end_z"]), vol_depth - 1)
    return 0, vol_depth - 1


class BreastDCEDataset(Dataset):
    """
    Yields (image, label) pairs. Each patient produces n_slices samples
    centered around the tumor mid-plane. Slices are cropped around the
    tumor ROI and fused into RGB via the paper's MinMax approach.

    Optionally returns clinical features as a third element.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label_col: str = "pCR",
        crop_size: int = 224,
        n_slices: int = 8,
        transform=None,
        clinical_cols: list[str] | None = None,
    ):
        self.df = df.dropna(subset=[label_col]).reset_index(drop=True)
        self.label_col = label_col
        self.crop_size = crop_size
        self.n_slices = n_slices
        self.transform = transform
        self.clinical_cols = clinical_cols

        # Precompute clinical feature normalization stats
        if clinical_cols:
            available = [c for c in clinical_cols if c in self.df.columns]
            self.clinical_cols = available
            self._clin_mean = self.df[available].mean().values.astype(np.float32)
            self._clin_std = self.df[available].std().values.astype(np.float32) + 1e-6
        else:
            self.clinical_cols = None

        self._index = [(i, s) for i in range(len(self.df)) for s in range(n_slices)]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        row_idx, slice_offset = self._index[idx]
        row = self.df.iloc[row_idx]
        pid = row["pid"]
        label = int(row[self.label_col])

        try:
            acqs = load_acquisitions(pid)
            if acqs is None or len(acqs) < 2:
                return self._blank(label, row_idx)

            cohort = _cohort_from_pid(pid)
            pre, early, late = select_timepoints(acqs, cohort)
            vol_depth = pre.shape[2]

            f, l = _z_range(row, vol_depth)
            if f == 0 and l == vol_depth - 1:
                mask = load_mask(pid)
                if mask is not None:
                    zr = find_tumor_z_range(mask)
                    if zr is not None:
                        f, l = zr

            mid = (f + l) // 2
            k = max(f, min(l, mid - self.n_slices // 2 + slice_offset))

            rgb = fuse_rgb_slice(pre[:, :, k], early[:, :, k], late[:, :, k])
            img = Image.fromarray(rgb, mode="RGB")

            cx, cy = _roi_centre(row, pre.shape)
            half = self.crop_size // 2
            w, h = img.size
            left = max(0, cx - half)
            top = max(0, cy - half)
            right = min(w, left + self.crop_size)
            bottom = min(h, top + self.crop_size)
            if right - left < self.crop_size:
                left = max(0, right - self.crop_size)
            if bottom - top < self.crop_size:
                top = max(0, bottom - self.crop_size)
            img = img.crop((left, top, right, bottom))
            if img.size != (self.crop_size, self.crop_size):
                img = img.resize((self.crop_size, self.crop_size), Image.BILINEAR)

        except Exception:
            return self._blank(label, row_idx)

        if self.transform:
            img = self.transform(img)

        if self.clinical_cols:
            clin = self.df.iloc[row_idx][self.clinical_cols].values.astype(np.float32)
            clin = (clin - self._clin_mean) / self._clin_std
            return img, label, torch.from_numpy(clin)

        return img, label

    def _blank(self, label: int, row_idx: int = 0):
        blank = torch.zeros(3, self.crop_size, self.crop_size)
        if self.clinical_cols:
            return blank, label, torch.zeros(len(self.clinical_cols))
        return blank, label
