"""
finetune.py – Fine-tune the BreastDCEDL ViT (pCR classifier) on a custom split.

Hardware
    GPU 0 : TITAN V  (12 GB VRAM)
    GPU 1 : RTX 2080 Ti (11 GB VRAM)

Run with:
    CUDA_VISIBLE_DEVICES=0 python finetune.py

    * Two-phase training: freeze backbone → unfreeze with LLRD
    * Automatic Mixed Precision (torch.cuda.amp)
    * Gradient Accumulation
    * Patient-level validation pooling
    * Early stopping
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
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from transformers import ViTForImageClassification, Dinov2ForImageClassification
from sklearn.metrics import roc_auc_score

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


# Channels are Z-scored then encoded to uint8 via: uint8 = (clip(z,-3,3) + 3) / 6 * 255
# ToTensor() divides by 255 → tensor ∈ [0,1].  Normalize(0.5, 1/6) recovers Z-score range.
_ZSCORE_MEAN = [0.5, 0.5, 0.5]
_ZSCORE_STD  = [1/6, 1/6, 1/6]

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_ZSCORE_MEAN, std=_ZSCORE_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_ZSCORE_MEAN, std=_ZSCORE_STD),
])


def _to_float_channels(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """
    Build a (H, W, 3) uint8 image from clinically-meaningful DCE channels:
      ch0 = pre-contrast
      ch1 = first post-contrast
      ch2 = subtraction (post - pre)  — primary enhancement signal

    Each channel is Z-score normalised per-slice, clipped to [-3, 3], then
    encoded to uint8 so standard PIL spatial transforms can be applied.
    The paired Normalize(mean=0.5, std=1/6) transform recovers the Z-score range.
    """
    diff = post.astype(np.float32) - pre.astype(np.float32)
    channels = np.stack([pre.astype(np.float32), post.astype(np.float32), diff], axis=2)
    for c in range(3):
        mu  = channels[:, :, c].mean()
        sig = channels[:, :, c].std() + 1e-6
        channels[:, :, c] = (channels[:, :, c] - mu) / sig
    channels = np.clip(channels, -3.0, 3.0)
    return ((channels + 3.0) / 6.0 * 255).astype(np.uint8)


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
                mask = ds.get_nifti_mask(pid)
                if mask is not None:
                    f_m, l_m = ds.find_first_last_planes(mask)
                    if f_m is not None:
                        f, l = f_m, l_m

            mid = (f + l) // 2
            k   = max(f, min(l, mid - self.n_slices // 2 + slice_offset))

            # ROI centre via data_utils helper
            cx, cy = _roi_centre_from_row(row, acqs[0].shape)

            # Build clinical DCE channels (pre, post, subtraction) with Z-score encoding
            rgb = _to_float_channels(acqs[0][:, :, k], acqs[1][:, :, k])
            img = Image.fromarray(rgb, mode="RGB")

            # Crop centred on tumour ROI
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


# ── Loss ──────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Focuses training on hard, misclassified examples; down-weights easy ones.
    Replaces weighted cross-entropy for better handling of pCR class imbalance.
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce  = F.cross_entropy(logits, labels, weight=self.weight, reduction="none")
        p_t = torch.exp(-ce)
        return (((1 - p_t) ** self.gamma) * ce).mean()


# ── Model helpers ─────────────────────────────────────────────────────────────

def build_model(hf_checkpoint: str, weights_path: str, num_classes: int,
                freeze_backbone: bool, dropout: float = 0.3) -> nn.Module:
    """
    Instantiate ViTForImageClassification, load weights onto CPU first,
    inject dropout before the classifier head, and optionally freeze encoder.
    """
    print(f"[model] Loading architecture from '{hf_checkpoint}' …")
    ModelClass = Dinov2ForImageClassification if "dinov2" in hf_checkpoint.lower() \
                 else ViTForImageClassification
    model = ModelClass.from_pretrained(
        hf_checkpoint,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    if os.path.isfile(weights_path):
        print(f"[model] Loading weights from '{weights_path}' …")
        state_dict = torch.load(weights_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  ⚠  Missing keys ({len(missing)}) – classifier head randomly initialised")
        if unexpected:
            print(f"  ⚠  Unexpected keys ({len(unexpected)}) – ignored")
    else:
        print(f"[model] No weights file at '{weights_path}' – using HF pretrained only")

    # Inject dropout before classifier for regularization
    hidden_size = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(hidden_size, num_classes),
    )
    print(f"[model] Injected Dropout(p={dropout}) before classifier head")

    if freeze_backbone:
        print("[model] Freezing ViT encoder – only classifier head will be trained.")
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[model] Trainable params: {trainable:,} / {total:,}")
    return model


def unfreeze_with_llrd(model: nn.Module, backbone_lr: float, head_lr: float,
                       weight_decay: float, llrd: float = 0.85) -> list[dict]:
    """
    Unfreeze all parameters and return AdamW param groups with layer-wise LR
    decay (LLRD). Classifier gets head_lr; each ViT encoder layer below gets
    backbone_lr * llrd^(distance_from_top); embeddings get the lowest LR.
    """
    for param in model.parameters():
        param.requires_grad = True

    no_decay = {"bias", "LayerNorm.weight"}
    seen     = set()
    groups   = []

    def _add(params, lr):
        d_params  = [p for n, p in params if n not in seen and not any(nd in n for nd in no_decay)]
        nd_params = [p for n, p in params if n not in seen and     any(nd in n for nd in no_decay)]
        seen.update(n for n, _ in params)
        if d_params:
            groups.append({"params": d_params,  "lr": lr, "weight_decay": weight_decay})
        if nd_params:
            groups.append({"params": nd_params, "lr": lr, "weight_decay": 0.0})

    # Classifier head (highest LR)
    _add([(n, p) for n, p in model.named_parameters() if "classifier" in n], head_lr)

    # Encoder layers — LLRD from top (layer 11) to bottom (layer 0)
    num_layers = 12
    for i in range(num_layers - 1, -1, -1):
        depth = num_layers - 1 - i          # 0 for layer 11, 11 for layer 0
        layer_lr = backbone_lr * (llrd ** depth)
        layer_named = [(n, p) for n, p in model.named_parameters()
                       if f"encoder.layer.{i}." in n and n not in seen]
        _add(layer_named, layer_lr)

    # Embeddings and top-level layernorm (lowest LR)
    rest = [(n, p) for n, p in model.named_parameters() if n not in seen]
    _add(rest, backbone_lr * (llrd ** num_layers))

    trainable = sum(p.numel() for g in groups for p in g["params"] if p.requires_grad)
    print(f"[model] Unfrozen all params. Trainable: {trainable:,}  "
          f"(backbone_lr={backbone_lr:.1e}, head_lr={head_lr:.1e}, llrd={llrd})")
    return groups


# ── Training / validation loop ────────────────────────────────────────────────

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
) -> tuple[float, float]:
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
                scaled_loss = loss / accum_steps

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


def eval_patient_level(
    model: nn.Module,
    val_df: pd.DataFrame,
    label_col: str,
    crop_size: int,
    n_slices: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> float:
    """
    Compute patient-level validation accuracy by mean-pooling logits across
    all n_slices slices for each patient.  More stable than slice-level acc
    on small validation sets.
    """
    val_ds = BreastDCEDataset(val_df, label_col=label_col, crop_size=crop_size,
                               n_slices=n_slices, transform=VAL_TRANSFORMS)
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with autocast():
                logits = model(images).logits
            all_logits.append(logits.cpu().float())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)   # (N_patients * n_slices, 2)
    all_labels = torch.cat(all_labels)   # (N_patients * n_slices,)

    n_patients = len(val_ds.df)
    # Reshape: (n_patients, n_slices, 2) — loader is unshuffled, slices are contiguous
    logits_3d  = all_logits.view(n_patients, n_slices, -1)
    labels_2d  = all_labels.view(n_patients, n_slices)

    pooled_logits   = logits_3d.mean(dim=1)      # (n_patients, 2)
    patient_labels  = labels_2d[:, 0]             # same label for all slices

    preds     = pooled_logits.argmax(dim=1)
    acc       = (preds == patient_labels).float().mean().item()
    probs_pos = torch.softmax(pooled_logits, dim=1)[:, 1].numpy()
    try:
        auc = float(roc_auc_score(patient_labels.numpy(), probs_pos))
    except ValueError:
        auc = 0.5
    return acc, auc


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(cfg: configparser.ConfigParser) -> argparse.Namespace:
    p = cfg["paths"]; t = cfg["training"]
    ap = argparse.ArgumentParser(description="Fine-tune BreastDCEDL ViT")
    ap.add_argument("--config",          default=_CFG_FILE)
    ap.add_argument("--weights",         default=p.get("weights_path", ""),  help="Pre-trained .pth")
    ap.add_argument("--checkpoint-dir",  default=p["checkpoint_dir"])
    ap.add_argument("--epochs",          default=t.getint("num_epochs"),          type=int)
    ap.add_argument("--freeze-epochs",   default=t.getint("freeze_epochs"),       type=int,
                    help="Epochs to train head-only before unfreezing backbone")
    ap.add_argument("--batch-size",      default=t.getint("physical_batch_size"), type=int)
    ap.add_argument("--accum",           default=t.getint("accum_steps"),         type=int)
    ap.add_argument("--lr",              default=t.getfloat("lr"),                type=float,
                    help="Backbone LR for phase 2 (LLRD base)")
    ap.add_argument("--head-lr",         default=t.getfloat("head_lr"),           type=float,
                    help="Classifier head LR (phase 1 and phase 2)")
    ap.add_argument("--llrd",            default=t.getfloat("llrd"),              type=float)
    ap.add_argument("--patience",        default=t.getint("patience"),            type=int)
    ap.add_argument("--resume",          default=0,   type=int,
                    help="Resume from this epoch number (loads checkpoints/breastdcedl_vit_epochNN.pth)")
    ap.add_argument("--resume-best-acc", default=0.0, type=float,
                    help="Best patient-level val acc achieved before interruption")
    ap.add_argument("--resume-patience", default=0,   type=int,
                    help="Patience counter at the point of interruption")
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
    nifti_paths = {"spy1": d["nifti_spy1"], "spy2": d["nifti_spy2"], "duke": d["nifti_duke"]}
    mask_paths  = {"spy1": d["mask_spy1"],  "spy2": d["mask_spy2"],  "duke": d["mask_duke"]}
    ds.setup_paths(".", nifti_paths, mask_paths)

    # Load and merge metadata CSVs
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

    label_col   = d["label_col"]
    crop_size   = d.getint("crop_size")
    n_slices    = d.getint("n_slices_per_patient")
    num_workers = cfg["training"].getint("num_workers")
    weight_decay = cfg["training"].getfloat("weight_decay")
    dropout      = cfg["training"].getfloat("dropout")

    # Train / val split
    if "test" in df.columns:
        train_df = df[df["test"] == 0].reset_index(drop=True)
        val_df   = df[df["test"] != 0].reset_index(drop=True)
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

    eff_batch = args.batch_size * max(n_gpus, 1) * args.accum
    print(f"[data] Physical batch/GPU={args.batch_size}  ×  {max(n_gpus,1)} GPU(s)  "
          f"×  {args.accum} accum  →  effective batch = {eff_batch}")
    print(f"[data] Train samples: {len(train_ds)}  |  Val samples (slice-level): {len(val_ds)}")

    # Model
    m = cfg["model"]
    model = build_model(
        hf_checkpoint = m["hf_checkpoint"],
        weights_path  = args.weights,
        num_classes   = m.getint("num_classes"),
        freeze_backbone = True,   # always start frozen; phase 2 will unfreeze
        dropout       = dropout,
    )
    model = model.to(device)

    # Class-weighted loss
    label_counts  = train_df[label_col].dropna().value_counts().sort_index()
    num_classes   = m.getint("num_classes")
    class_weights = torch.tensor(
        [1.0 / label_counts.get(i, 1) for i in range(num_classes)],
        dtype=torch.float32,
    ).to(device)
    class_weights /= class_weights.sum()
    criterion = FocalLoss(gamma=2.0, weight=class_weights)

    scaler = GradScaler()

    best_val_auc  = args.resume_best_acc   # arg reused: now stores best AUC
    best_val_loss = float("inf")
    epochs_no_imp = args.resume_patience
    phase2_start  = args.freeze_epochs + 1

    if args.resume > 0:
        ckpt_path = os.path.join(args.checkpoint_dir, f"breastdcedl_vit_epoch{args.resume:02d}.pth")
        print(f"[resume] Loading checkpoint from '{ckpt_path}' (epoch {args.resume}) …")
        inner = model.module if isinstance(model, nn.DataParallel) else model
        inner.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[resume] Restored best_val_auc={best_val_auc:.3f}, patience={epochs_no_imp}/{args.patience}")

    print("\n" + "=" * 60)
    print(f"Starting fine-tuning  ({args.freeze_epochs} freeze + "
          f"{args.epochs - args.freeze_epochs} unfreeze epochs, patience={args.patience})")
    print("=" * 60 + "\n")

    # ── Phase 1: head-only ────────────────────────────────────────────────────
    if args.freeze_epochs > 0 and args.resume < args.freeze_epochs:
        print(f"── Phase 1: head-only, lr={args.head_lr:.1e} ──")
        optimizer_p1 = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.head_lr, weight_decay=weight_decay,
        )
        scheduler_p1 = CosineAnnealingLR(optimizer_p1, T_max=args.freeze_epochs, eta_min=1e-6)

        for epoch in range(1, args.freeze_epochs + 1):
            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, optimizer_p1, scaler,
                device, args.accum, epoch, args.epochs, is_train=True)
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, optimizer_p1, scaler,
                device, args.accum, epoch, args.epochs, is_train=False)
            scheduler_p1.step()

            # Patient-level val accuracy + AUC
            pt_acc, pt_auc = eval_patient_level(model, val_df, label_col, crop_size, n_slices,
                                                args.batch_size, num_workers, device)

            print(f"  ✔  Epoch {epoch:02d}/{args.epochs}  [Phase 1]  "
                  f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
                  f"val_acc(slice)={val_acc:.3f}  val_acc(patient)={pt_acc:.3f}  "
                  f"val_auc={pt_auc:.3f}  lr={scheduler_p1.get_last_lr()[0]:.2e}")

            if pt_auc > best_val_auc:
                best_val_auc  = pt_auc
                best_val_loss = val_loss
                epochs_no_imp = 0
                best = os.path.join(args.checkpoint_dir, "breastdcedl_vit_best.pth")
                inner = model.module if isinstance(model, nn.DataParallel) else model
                torch.save(inner.state_dict(), best)
                print(f"  ⭐ New best val AUC: {pt_auc:.3f} → {best}")
            else:
                epochs_no_imp += 1

            ckpt = os.path.join(args.checkpoint_dir, f"breastdcedl_vit_epoch{epoch:02d}.pth")
            inner = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(inner.state_dict(), ckpt)
            print(f"  💾 Checkpoint → {ckpt}\n")

    # ── Phase 2: full fine-tune with LLRD ────────────────────────────────────
    # Reset patience so phase-1 non-improvements don't bleed into phase 2
    epochs_no_imp = 0

    remaining_epochs = args.epochs - args.freeze_epochs
    if remaining_epochs > 0:
        print(f"\n── Phase 2: full fine-tune with LLRD  "
              f"(backbone_lr={args.lr:.1e}, head_lr={args.head_lr:.1e}, llrd={args.llrd}) ──")
        param_groups = unfreeze_with_llrd(
            model, backbone_lr=args.lr, head_lr=args.head_lr,
            weight_decay=weight_decay, llrd=args.llrd,
        )
        optimizer_p2 = optim.AdamW(param_groups)

        # 1-epoch linear warmup then cosine
        warmup = LinearLR(optimizer_p2, start_factor=0.1, end_factor=1.0, total_iters=1)
        cosine = CosineAnnealingLR(optimizer_p2, T_max=max(remaining_epochs - 1, 1), eta_min=1e-7)
        scheduler_p2 = SequentialLR(optimizer_p2, schedulers=[warmup, cosine], milestones=[1])

        # Fast-forward scheduler to match already-completed phase-2 epochs
        phase2_done = max(0, args.resume - args.freeze_epochs)
        for _ in range(phase2_done):
            scheduler_p2.step()

        for ep_offset in range(phase2_done + 1, remaining_epochs + 1):
            epoch = args.freeze_epochs + ep_offset

            if epochs_no_imp >= args.patience:
                print(f"  Early stopping: no patient-level improvement for {args.patience} epochs.")
                break

            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, optimizer_p2, scaler,
                device, args.accum, epoch, args.epochs, is_train=True)
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, optimizer_p2, scaler,
                device, args.accum, epoch, args.epochs, is_train=False)
            scheduler_p2.step()

            # Patient-level val accuracy + AUC
            pt_acc, pt_auc = eval_patient_level(model, val_df, label_col, crop_size, n_slices,
                                                args.batch_size, num_workers, device)

            # Report the LR of the head param group (highest LR)
            current_lr = optimizer_p2.param_groups[0]["lr"]
            print(f"  ✔  Epoch {epoch:02d}/{args.epochs}  [Phase 2]  "
                  f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
                  f"val_acc(slice)={val_acc:.3f}  val_acc(patient)={pt_acc:.3f}  "
                  f"val_auc={pt_auc:.3f}  lr={current_lr:.2e}")

            if pt_auc >= best_val_auc:
                best_val_auc  = pt_auc
                best_val_loss = val_loss
                epochs_no_imp = 0
                best = os.path.join(args.checkpoint_dir, "breastdcedl_vit_best.pth")
                inner = model.module if isinstance(model, nn.DataParallel) else model
                torch.save(inner.state_dict(), best)
                print(f"  ⭐ New best val AUC: {pt_auc:.3f} → {best}")
            else:
                epochs_no_imp += 1
                print(f"  (no improvement, patience {epochs_no_imp}/{args.patience})")

            ckpt = os.path.join(args.checkpoint_dir, f"breastdcedl_vit_epoch{epoch:02d}.pth")
            inner = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(inner.state_dict(), ckpt)
            print(f"  💾 Checkpoint → {ckpt}\n")

    print("Fine-tuning complete.")
    print(f"Best patient-level val AUC : {best_val_auc:.3f}")
    print(f"Best val loss              : {best_val_loss:.4f}  (at best-AUC epoch)")
    print(f"Checkpoints saved in       : {args.checkpoint_dir}/")


if __name__ == "__main__":
    main()
