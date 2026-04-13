"""
Two-phase training loop for BreastDCEDL ViT.

Phase 1: Frozen backbone, train classifier head only (warmup).
Phase 2: Unfreeze all layers with layer-wise learning rate decay (LLRD).

Uses AMP, gradient accumulation, cosine annealing, and early stopping
on patient-level AUC.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.models.vit import BreastDCEViT, build_llrd_param_groups
from src.evaluation.metrics import patient_level_eval

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def run_epoch(
    model: BreastDCEViT,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    accum_steps: int,
    is_train: bool = True,
    use_clinical: bool = False,
):
    model.train() if is_train else model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0
    all_logits = []
    all_labels = []

    if is_train:
        optimizer.zero_grad()

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for step, batch in enumerate(loader, 1):
            if use_clinical:
                images, labels, clinical = batch
                clinical = clinical.to(device, non_blocking=True)
            else:
                images, labels = batch
                clinical = None

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                logits = model(images, clinical)
                loss = criterion(logits, labels)

            if is_train:
                scaled = loss / accum_steps
                scaler.scale(scaled).backward()
                if step % accum_steps == 0 or step == len(loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            running_correct += (logits.argmax(1) == labels).sum().item()
            running_loss += loss.item() * images.size(0)
            total += images.size(0)
            all_logits.append(logits.detach().cpu().float())
            all_labels.append(labels.cpu())

    avg_loss = running_loss / max(total, 1)
    avg_acc = running_correct / max(total, 1)
    return avg_loss, avg_acc, torch.cat(all_logits), torch.cat(all_labels)


class Trainer:

    def __init__(self, model, train_loader, val_loader, val_df, cfg, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_df = val_df
        self.cfg = cfg
        self.device = device

        self.tcfg = cfg["training"]
        self.use_clinical = cfg["model"].get("use_clinical", False)
        self.checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.scaler = GradScaler()
        self.history = []

        # W&B
        wcfg = cfg.get("wandb", {})
        self.use_wandb = (wcfg.get("enabled", False)
                          and _WANDB_AVAILABLE
                          and wandb.run is not None)

    def train(self, criterion):
        num_epochs = self.tcfg["num_epochs"]
        freeze_epochs = self.tcfg["freeze_epochs"]
        accum = self.tcfg["accum_steps"]
        patience = self.tcfg["patience"]
        n_slices = self.cfg["data"]["n_slices"]

        best_auc = 0.0
        epochs_no_imp = 0

        # Phase 1: frozen backbone
        if freeze_epochs > 0:
            self.model.freeze_backbone()
            opt_p1 = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.tcfg["head_lr"],
                weight_decay=self.tcfg["weight_decay"],
            )
            sched_p1 = CosineAnnealingLR(opt_p1, T_max=freeze_epochs, eta_min=1e-6)

            print(f"\n--- Phase 1: head-only ({freeze_epochs} epochs) ---")
            for epoch in range(1, freeze_epochs + 1):
                tr_loss, tr_acc, _, _ = run_epoch(
                    self.model, self.train_loader, criterion, opt_p1,
                    self.scaler, self.device, accum, is_train=True,
                    use_clinical=self.use_clinical,
                )
                vl_loss, vl_acc, vl_logits, vl_labels = run_epoch(
                    self.model, self.val_loader, criterion, opt_p1,
                    self.scaler, self.device, accum, is_train=False,
                    use_clinical=self.use_clinical,
                )
                sched_p1.step()

                metrics = patient_level_eval(vl_logits, vl_labels, n_slices)
                self._log(epoch, "P1", tr_loss, tr_acc, vl_loss, vl_acc, metrics)
                best_auc, epochs_no_imp = self._checkpoint(
                    epoch, metrics, best_auc, epochs_no_imp,
                )

        # Phase 2: full fine-tune with LLRD
        self.model.unfreeze_backbone()
        groups = build_llrd_param_groups(
            self.model,
            backbone_lr=self.tcfg["backbone_lr"],
            head_lr=self.tcfg["head_lr"],
            weight_decay=self.tcfg["weight_decay"],
            llrd=self.tcfg["llrd"],
        )
        opt_p2 = optim.AdamW(groups)
        remaining = num_epochs - freeze_epochs
        warmup = LinearLR(opt_p2, start_factor=0.1, end_factor=1.0, total_iters=1)
        cosine = CosineAnnealingLR(opt_p2, T_max=max(remaining - 1, 1), eta_min=1e-7)
        sched_p2 = SequentialLR(opt_p2, schedulers=[warmup, cosine], milestones=[1])

        epochs_no_imp = 0  # reset for phase 2

        print(f"\n--- Phase 2: LLRD fine-tune ({remaining} epochs) ---")
        for ep_offset in range(1, remaining + 1):
            epoch = freeze_epochs + ep_offset

            if epochs_no_imp >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

            tr_loss, tr_acc, _, _ = run_epoch(
                self.model, self.train_loader, criterion, opt_p2,
                self.scaler, self.device, accum, is_train=True,
                use_clinical=self.use_clinical,
            )
            vl_loss, vl_acc, vl_logits, vl_labels = run_epoch(
                self.model, self.val_loader, criterion, opt_p2,
                self.scaler, self.device, accum, is_train=False,
                use_clinical=self.use_clinical,
            )
            sched_p2.step()

            metrics = patient_level_eval(vl_logits, vl_labels, n_slices)
            self._log(epoch, "P2", tr_loss, tr_acc, vl_loss, vl_acc, metrics)
            best_auc, epochs_no_imp = self._checkpoint(
                epoch, metrics, best_auc, epochs_no_imp,
            )

        self._save_history()
        if self.use_wandb:
            best_path = os.path.join(self.checkpoint_dir, "best.pth")
            if os.path.isfile(best_path):
                artifact = wandb.Artifact("best-model", type="model",
                                          metadata={"best_auc": best_auc})
                artifact.add_file(best_path)
                wandb.log_artifact(artifact)
        print(f"\nTraining complete. Best val AUC: {best_auc:.4f}")
        return best_auc

    def _log(self, epoch, phase, tr_loss, tr_acc, vl_loss, vl_acc, metrics):
        entry = {
            "epoch": epoch, "phase": phase,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": vl_loss, "val_acc_slice": vl_acc,
            **metrics,
        }
        self.history.append(entry)
        print(
            f"  Epoch {epoch:02d} [{phase}]  "
            f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f}  "
            f"vl_acc={vl_acc:.3f}  "
            f"pt_acc={metrics['accuracy']:.3f}  "
            f"auc={metrics['auc']:.3f}  "
            f"sens={metrics['sensitivity']:.3f}  "
            f"spec={metrics['specificity']:.3f}"
        )

        if self.use_wandb:
            wandb.log({
                "epoch": epoch,
                "phase": phase,
                "train/loss": tr_loss,
                "train/acc": tr_acc,
                "val/loss": vl_loss,
                "val/acc_slice": vl_acc,
                "val/acc_patient": metrics["accuracy"],
                "val/auc": metrics["auc"],
                "val/sensitivity": metrics["sensitivity"],
                "val/specificity": metrics["specificity"],
                "val/precision": metrics["precision"],
                "val/npv": metrics["npv"],
            }, step=epoch)

    def _checkpoint(self, epoch, metrics, best_auc, epochs_no_imp):
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            epochs_no_imp = 0
            path = os.path.join(self.checkpoint_dir, "best.pth")
            torch.save(self.model.state_dict(), path)
            print(f"  -> New best AUC: {best_auc:.4f} saved to {path}")
            if self.use_wandb:
                wandb.run.summary["best_auc"] = best_auc
                wandb.run.summary["best_epoch"] = epoch
        else:
            epochs_no_imp += 1

        path = os.path.join(self.checkpoint_dir, f"epoch_{epoch:02d}.pth")
        torch.save(self.model.state_dict(), path)
        return best_auc, epochs_no_imp

    def _save_history(self):
        path = os.path.join(self.checkpoint_dir, "history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
