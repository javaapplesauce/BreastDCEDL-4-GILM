"""
finetune.py – Fine-tune the BreastDCEDL ViT (pCR classifier) on a custom split.

Hardware 
    GPU 0 : TITAN V  (12 GB VRAM)
    GPU 1 : RTX 2080 Ti (11 GB VRAM)

Run with:
    CUDA_VISIBLE_DEVICES=0,1 python finetune.py
    
    * Automatic Mixed Precision (torch.cuda.amp)
    * Gradient Accumulation
    * Weights loaded to CPU first
"""

import os
import sys
import warnings
import argparse
import configparser
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from transformers import ViTForImageClassification

sys.path.append(os.path.abspath("utils"))
import data_utils as ds

warnings.filterwarnings("ignore")


_CFG_FILE = os.path.join(os.path.dirname(__file__), "finetune.cfg")

def load_cfg(cfg_path: str = _CFG_FILE) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    cfg.read(cfg_path)
    return cfg


TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _to_rgb(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Stack three 2-D slices into a uint8 RGB image using ds.minmax."""
    rgb = ds.minmax(np.stack([a, b, c], axis=2))
    return (rgb * 255).astype(np.uint8)


def _roi_centre_from_row(row: pd.Series, vol_shape: tuple) -> tuple[int, int]:
    """
    Extract the (cx, cy) tumour ROI centre from a metadata row.

    Handles two column conventions:
      - Duke / ISPY2-style  : sraw, eraw, scol, ecol
      - ISPY1 VOI-style     : voi_start_x, voi_end_x, voi_start_y, voi_end_y
    Falls back to the volume centre if no columns are present.
    """
    h, w = vol_shape[0], vol_shape[1]

    if "scol" in row.index and not pd.isna(row.get("scol")):
        sc = int(row["scol"]); ec = int(row.get("ecol", sc))
        sr = int(row["sraw"]); er = int(row.get("eraw", sr))
        return (sc + ec) // 2, (sr + er) // 2

    if "voi_start_x" in row.index and not pd.isna(row.get("voi_start_x")):
        cx = int((row["voi_start_x"] + row["voi_end_x"]) / 2)
        cy = int((row["voi_start_y"] + row["voi_end_y"]) / 2)
        return cx, cy

    # Fallback: image centre
    return w // 2, h // 2


def _z_range_from_row(row: pd.Series, vol_depth: int) -> tuple[int, int]:
    """
    Return the (first, last) tumour Z-slice indices from a metadata row.

    Handles:
      - Duke / ISPY2-style  : mask_start, mask_end
      - ISPY1 VOI-style     : voi_start_z, voi_end_z
    Falls back to ds.find_first_last_planes when neither is present.
    """
    if "mask_start" in row.index and not pd.isna(row.get("mask_start")):
        f = max(int(row["mask_start"]), 0)
        l = min(int(row.get("mask_end", vol_depth - 1)), vol_depth - 1)
        return f, l

    if "voi_start_z" in row.index and not pd.isna(row.get("voi_start_z")):
        f = max(int(row["voi_start_z"]), 0)
        l = min(int(row["voi_end_z"]), vol_depth - 1)
        return f, l

    return 0, vol_depth - 1


class BreastDCEDataset(Dataset):
    """
    Yields individual 2-D RGB slices (crop around tumour ROI) from NIfTI DCE
    volumes.  Each patient contributes `n_slices` samples centred on the tumour
    mid-plane, all sharing the same label.

    Data I/O is entirely delegated to data_utils:
      ds.get_all_nifti_acquisitions  – loads NIfTI volumes
      ds.minmax                      – normalises slices (via _to_rgb)
      ds.find_first_last_planes      – fallback Z-range when metadata is absent
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label_col: str = "pCR",
        crop_size: int = 224,
        n_slices: int = 4,
        transform=None,
    ):
        self.df        = df.dropna(subset=[label_col]).reset_index(drop=True)
        self.label_col = label_col
        self.crop_size = crop_size
        self.n_slices  = n_slices
        self.transform = transform

        # Flat index: (patient_row, slice_offset)
        self._index: list[tuple[int, int]] = [
            (i, s) for i in range(len(self.df)) for s in range(n_slices)
        ]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        row_idx, slice_offset = self._index[idx]
        row   = self.df.iloc[row_idx]
        pid   = row["pid"]
        label = int(row[self.label_col])

        try:
            # ── Load NIfTI acquisitions via data_utils ────────────────────────
            acqs = ds.get_all_nifti_acquisitions(pid)
            if acqs is None or len(acqs) < 3:
                return self._blank(label)

            vol_depth = acqs[0].shape[2]

            # ── Z-range: prefer metadata columns; fall back to mask scan ──────
            f, l = _z_range_from_row(row, vol_depth)
            if f == 0 and l == vol_depth - 1:
                # Try to find active planes from the mask volume
                mask = ds.get_nifti_mask(pid)
                if mask is not None:
                    f_m, l_m = ds.find_first_last_planes(mask)
                    if f_m is not None:
                        f, l = f_m, l_m

            mid = (f + l) // 2
            k   = max(f, min(l, mid - self.n_slices // 2 + slice_offset))

            # ROI centre via data_utils helper
            cx, cy = _roi_centre_from_row(row, acqs[0].shape)

            # Build RGB image and crop
            rgb = _to_rgb(acqs[0][:, :, k], acqs[1][:, :, k], acqs[2][:, :, k])
            img = Image.fromarray(rgb, mode="RGB")

            # Crop centred on tumour ROI (same logic as predict notebook's
            # safe_crop_around_roi, but uses PIL directly – no extra copy needed)
            w, h   = img.size
            half   = self.crop_size // 2
            left   = max(0, cx - half);  right  = left + self.crop_size
            top    = max(0, cy - half);  bottom = top  + self.crop_size
            if right  > w: left   = max(0, w - self.crop_size); right  = w
            if bottom > h: top    = max(0, h - self.crop_size); bottom = h
            img = img.crop((left, top, right, bottom))
            if img.size != (self.crop_size, self.crop_size):
                img = img.resize((self.crop_size, self.crop_size), Image.BILINEAR)

        except Exception:
            return self._blank(label)

        if self.transform:
            img = self.transform(img)

        return img, label

    def _blank(self, label: int):
        return torch.zeros(3, self.crop_size, self.crop_size), label


# Model helpers
def build_model(hf_checkpoint: str, weights_path: str, num_classes: int,
                freeze_backbone: bool) -> nn.Module:
    """
    Instantiate ViTForImageClassification (identical to predict notebook),
    load weights onto CPU first (no VRAM spike), then optionally freeze the
    encoder so only the classifier head trains.
    """
    print(f"[model] Loading architecture from '{hf_checkpoint}' …")
    model = ViTForImageClassification.from_pretrained(
        hf_checkpoint,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    print(f"[model] Loading weights from '{weights_path}' …")
    state_dict = torch.load(weights_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  ⚠  Missing keys ({len(missing)}) – classifier head randomly initialised")
    if unexpected:
        print(f"  ⚠  Unexpected keys ({len(unexpected)}) – ignored")

    if freeze_backbone:
        print("[model] Freezing ViT encoder – only classifier head will be trained.")
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[model] Trainable params: {trainable:,} / {total:,}")
    return model


# Training / validation loop
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    accum_steps: int,
    epoch: int,
    num_epochs: int,
    is_train: bool = True,
) -> float:
    model.train() if is_train else model.eval()
    phase = "Train" if is_train else "  Val"

    running_loss  = 0.0
    running_corr  = 0
    total_samples = 0
    optimizer.zero_grad()

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for step, (images, labels) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(images).logits
                loss    = criterion(outputs, labels)
                scaled_loss = loss / accum_steps   # average over accum steps

            if is_train:
                scaler.scale(scaled_loss).backward()
                if step % accum_steps == 0 or step == len(loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            preds = outputs.argmax(dim=1)
            running_corr  += (preds == labels).sum().item()
            running_loss  += loss.item() * images.size(0)
            total_samples += images.size(0)

            if step % 10 == 0 or step == len(loader):
                print(
                    f"  [{phase}] Epoch {epoch:02d}/{num_epochs} "
                    f"step {step:04d}/{len(loader):04d} | "
                    f"loss={running_loss/total_samples:.4f}  "
                    f"acc={running_corr/total_samples:.3f}",
                    end="\r",
                )

    print()
    return running_loss / total_samples, running_corr / total_samples


# CLI
def parse_args(cfg: configparser.ConfigParser) -> argparse.Namespace:
    p = cfg["paths"]; t = cfg["training"]
    ap = argparse.ArgumentParser(description="Fine-tune BreastDCEDL ViT")
    ap.add_argument("--config",          default=_CFG_FILE,          help="Path to .cfg file")
    ap.add_argument("--weights",         default=p["weights_path"],  help="Pre-trained .pth")
    ap.add_argument("--checkpoint-dir",  default=p["checkpoint_dir"])
    ap.add_argument("--epochs",          default=t.getint("num_epochs"),          type=int)
    ap.add_argument("--batch-size",      default=t.getint("physical_batch_size"), type=int)
    ap.add_argument("--accum",           default=t.getint("accum_steps"),         type=int)
    ap.add_argument("--lr",              default=t.getfloat("lr"),                type=float)
    ap.add_argument("--freeze-backbone", action="store_true",
                    default=t.getboolean("freeze_backbone"),
                    help="Freeze ViT encoder; train classifier head only")
    return ap.parse_args()


def main():
    cfg  = load_cfg()
    args = parse_args(cfg)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"[hardware] Using device: {device}")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  "
              f"({torch.cuda.get_device_properties(i).total_memory/1024**3:.1f} GB)")

    d = cfg["data"]
    nifti_paths = {
        "spy1": d["nifti_spy1"],
        "spy2": d["nifti_spy2"],
        "duke": d["nifti_duke"],
    }
    mask_paths = {
        "spy1": d["mask_spy1"],
        "spy2": d["mask_spy2"],
        "duke": d["mask_duke"],
    }
    ds.setup_paths(".", nifti_paths, mask_paths)

    # Load and merge per-dataset metadata CSVs
    dfs = []
    for key in ("spy1_metadata_csv", "duke_metadata_csv"):
        csv_path = d[key]
        if os.path.isfile(csv_path):
            dfs.append(pd.read_csv(csv_path))
            print(f"[data] Loaded {csv_path}  ({len(dfs[-1])} rows)")
        else:
            print(f"[data] ⚠  {csv_path} not found – skipping")
    if not dfs:
        raise RuntimeError("No metadata CSVs found. Check paths in finetune.cfg.")
    df = pd.concat(dfs, ignore_index=True)
    print(f"[data] Combined dataset: {len(df)} patients")

    label_col  = d["label_col"]
    crop_size  = d.getint("crop_size")
    n_slices   = d.getint("n_slices_per_patient")
    num_workers = cfg["training"].getint("num_workers")

    # Train / val split
    if "test" in df.columns:
        train_df = df[df["test"] == False].reset_index(drop=True)
        val_df   = df[df["test"] == True ].reset_index(drop=True)
        print(f"[data] Train: {len(train_df)}  |  Val: {len(val_df)}")
    else:
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=42,
            stratify=df[label_col].fillna(-1),
        )
        train_df = train_df.reset_index(drop=True)
        val_df   = val_df.reset_index(drop=True)
        print(f"[data] (80/20 split) Train: {len(train_df)}  Val: {len(val_df)}")

    train_ds = BreastDCEDataset(train_df, label_col=label_col, crop_size=crop_size,
                                n_slices=n_slices, transform=TRAIN_TRANSFORMS)
    val_ds   = BreastDCEDataset(val_df,   label_col=label_col, crop_size=crop_size,
                                n_slices=n_slices, transform=VAL_TRANSFORMS)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"[data] Physical batch/GPU={args.batch_size}{max(n_gpus,1)} GPU(s)"
          f"{args.accum} accum  →  effective batch = "
          f"{args.batch_size * max(n_gpus, 1) * args.accum}")

    # Model 
    m = cfg["model"]
    model = build_model(
        hf_checkpoint   = m["hf_checkpoint"],
        weights_path    = args.weights,
        num_classes     = m.getint("num_classes"),
        freeze_backbone = args.freeze_backbone,
    )
    if n_gpus > 1:
        print(f"[hardware] Wrapping in nn.DataParallel across {n_gpus} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Loss – weighted to handle pCR class imbalance
    label_counts  = train_df[label_col].dropna().value_counts().sort_index()
    num_classes   = m.getint("num_classes")
    class_weights = torch.tensor(
        [1.0 / label_counts.get(i, 1) for i in range(num_classes)],
        dtype=torch.float32,
    ).to(device)
    class_weights /= class_weights.sum()
    # No label_smoothing: with only ~45 val patients and class imbalance,
    # smoothing pushes the model toward the majority class and hurts val acc.
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=cfg["training"].getfloat("weight_decay"),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler    = GradScaler()



    best_val_loss = float("inf")
    best_val_acc  = 0.0
    print("\n" + "=" * 60)
    print("Starting fine-tuning")
    print("=" * 60 + "\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler,
                                          device, args.accum, epoch, args.epochs, is_train=True)
        val_loss, val_acc     = run_epoch(model, val_loader,   criterion, optimizer, scaler,
                                          device, args.accum, epoch, args.epochs, is_train=False)
        scheduler.step()

        print(f"  ✔  Epoch {epoch:02d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Save per-epoch checkpoint
        inner = model.module if isinstance(model, nn.DataParallel) else model
        ckpt  = os.path.join(args.checkpoint_dir, f"breastdcedl_vit_epoch{epoch:02d}.pth")
        torch.save(inner.state_dict(), ckpt)
        print(f"  💾 Checkpoint → {ckpt}")

        # Save best checkpoint by val ACCURACY (more stable than loss on small val sets)
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            best = os.path.join(args.checkpoint_dir, "breastdcedl_vit_best.pth")
            torch.save(inner.state_dict(), best)
            print(f"  ⭐ New best val acc ({val_acc:.3f}, loss={val_loss:.4f}) → {best}")

        print()

    print("Fine-tuning complete.")
    print(f"Best val acc  : {best_val_acc:.3f}")
    print(f"Best val loss : {best_val_loss:.4f}  (at best-acc epoch)")
    print(f"Checkpoints saved in : {args.checkpoint_dir}/")


if __name__ == "__main__":
    main()

