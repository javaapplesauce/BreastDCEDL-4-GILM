"""
Training entry point for BreastDCEDL ViT pCR prediction.

Usage:
    python scripts/train.py --config configs/default.yaml

    # Override specific parameters:
    python scripts/train.py --config configs/default.yaml \
        --backbone facebook/dinov2-base \
        --epochs 20 --batch-size 8

    # Resume from checkpoint:
    python scripts/train.py --config configs/default.yaml --resume checkpoints/epoch_10.pth
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import setup_paths
from src.data.dataset import BreastDCEDataset, build_transforms
from src.data.splits import load_and_split
from src.models.vit import BreastDCEViT
from src.models.losses import FocalLoss, build_class_weights
from src.training.trainer import Trainer


def train_from_config(cfg: dict) -> float:
    seed = cfg["training"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"[device] {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        print("[device] CPU")

    # Data paths
    dcfg = cfg["data"]
    setup_paths(
        nifti_dirs={k: dcfg["nifti"][k] for k in dcfg["nifti"]},
        mask_dirs={k: dcfg["masks"][k] for k in dcfg["masks"]},
    )

    # Load metadata and split
    metadata = dcfg.get("combined_metadata")
    if metadata and os.path.isfile(metadata):
        train_df, val_df, test_df = load_and_split(
            metadata, label_col=dcfg["label_col"], seed=seed,
        )
    else:
        # Merge individual cohort CSVs
        dfs = []
        for key in dcfg.get("metadata", {}):
            csv = dcfg["metadata"][key]
            if os.path.isfile(csv):
                dfs.append(pd.read_csv(csv))
                print(f"[data] Loaded {csv} ({len(dfs[-1])} rows)")
        if not dfs:
            raise RuntimeError("No metadata CSVs found.")
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.dropna(subset=[dcfg["label_col"]]).reset_index(drop=True)
        from src.data.splits import load_and_split as _  # noqa
        from sklearn.model_selection import train_test_split
        train_df, temp = train_test_split(
            combined, test_size=0.2, random_state=seed,
            stratify=combined[dcfg["label_col"]],
        )
        val_df, test_df = train_test_split(
            temp, test_size=0.5, random_state=seed,
            stratify=temp[dcfg["label_col"]],
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    print(f"[data] Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    pcr_rate = train_df[dcfg["label_col"]].mean()
    print(f"[data] Train pCR rate: {pcr_rate:.1%}")

    # Augmentation
    aug_mode = cfg["training"].get("augmentation_mode", "full")
    if aug_mode == "none":
        aug_cfg = {}
    elif aug_mode == "minimal":
        aug_cfg = {"horizontal_flip": True, "vertical_flip": True}
    else:
        aug_cfg = cfg["training"].get("augmentation", {})

    train_transform = build_transforms(aug_cfg, is_train=True)
    val_transform = build_transforms({}, is_train=False)

    # Clinical features
    mcfg = cfg["model"]
    clinical_cols = mcfg.get("clinical_features") if mcfg.get("use_clinical") else None

    # Datasets
    crop = dcfg.get("crop_size", 224)
    n_slices = dcfg.get("n_slices", 8)
    train_ds = BreastDCEDataset(
        train_df, label_col=dcfg["label_col"], crop_size=crop,
        n_slices=n_slices, transform=train_transform,
        clinical_cols=clinical_cols,
    )
    val_ds = BreastDCEDataset(
        val_df, label_col=dcfg["label_col"], crop_size=crop,
        n_slices=n_slices, transform=val_transform,
        clinical_cols=clinical_cols,
    )

    tcfg = cfg["training"]
    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True,
        num_workers=tcfg["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg["batch_size"], shuffle=False,
        num_workers=tcfg["num_workers"], pin_memory=True,
    )

    eff_batch = tcfg["batch_size"] * tcfg["accum_steps"]
    print(f"[data] Effective batch size: {eff_batch}")
    print(f"[data] Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")

    # Model
    clinical_dim = mcfg.get("clinical_dim", 0) if mcfg.get("use_clinical") else 0
    model = BreastDCEViT(
        backbone=mcfg["backbone"],
        num_classes=mcfg["num_classes"],
        dropout=mcfg["dropout"],
        use_clinical=mcfg.get("use_clinical", False),
        clinical_dim=clinical_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {mcfg['backbone']}  params: {total_params:,}")

    # Loss
    loss_type = tcfg.get("loss", "focal")
    labels_list = train_df[dcfg["label_col"]].tolist()
    weights = build_class_weights(labels_list, mcfg["num_classes"]).to(device)
    if loss_type == "ce":
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = FocalLoss(gamma=2.0, weight=weights)
    print(f"[loss] {loss_type} with class weights {weights.cpu().tolist()}")

    # Resume
    resume_path = cfg.get("resume")
    if resume_path and os.path.isfile(resume_path):
        model.load_state_dict(torch.load(resume_path, map_location=device))
        print(f"[resume] Loaded weights from {resume_path}")

    # Train
    trainer = Trainer(model, train_loader, val_loader, val_df, cfg, device)
    best_auc = trainer.train(criterion)
    return best_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.backbone:
        cfg["model"]["backbone"] = args.backbone
    if args.epochs:
        cfg["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["backbone_lr"] = args.lr
    cfg["resume"] = args.resume
    cfg["checkpoint_dir"] = args.checkpoint_dir

    train_from_config(cfg)


if __name__ == "__main__":
    main()
