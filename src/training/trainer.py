"""
Two-phase training loop for BreastDCEDL ViT.

Phase 1: Frozen backbone, train classifier head only (warmup).
Phase 2: Unfreeze all layers with layer-wise learning rate decay (LLRD).

Uses AMP, gradient accumulation, cosine annealing, and early stopping
on patient-level AUC.

Checkpoints save full training state (model + optimizer + scheduler +
epoch + phase + best_auc + history) so a Colab disconnect mid-run can
resume without re-warmup. Old bare-state-dict checkpoints are still
recognised for backward compat — epoch is inferred from filename.
"""

import os
import re
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


def _resolve_amp_dtype(requested: str) -> torch.dtype:
    """bf16 on A100/H100 is more numerically stable for ViT than fp16.
    Fall back to fp16 if bf16 isn't supported (V100/older)."""
    if requested == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

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
    scaler: GradScaler | None,
    device: torch.device,
    accum_steps: int,
    is_train: bool = True,
    use_clinical: bool = False,
    amp_dtype: torch.dtype = torch.float16,
):
    model.train() if is_train else model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0
    all_logits = []
    all_labels = []

    if is_train:
        optimizer.zero_grad()

    use_scaler = scaler is not None and amp_dtype == torch.float16

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

            with autocast(dtype=amp_dtype):
                logits = model(images, clinical)
                loss = criterion(logits, labels)

            if is_train:
                scaled = loss / accum_steps
                if use_scaler:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                if step % accum_steps == 0 or step == len(loader):
                    if use_scaler:
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if use_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
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

        self.amp_dtype = _resolve_amp_dtype(self.tcfg.get("amp_dtype", "bf16"))
        # GradScaler is fp16-only; bf16 has the f32 exponent range and doesn't need it.
        self.scaler = GradScaler() if self.amp_dtype == torch.float16 else None
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

        resume = self._maybe_load_resume()
        if resume and resume["mode"] == "full":
            start_epoch = resume["epoch"] + 1
            best_auc = resume["best_auc"]
        elif resume and resume["mode"] == "weights_only":
            # Old-format ckpt: weights only, no opt/sched. Pick up after the
            # epoch parsed from the filename; opt/sched will be fresh.
            start_epoch = resume["epoch"] + 1
            best_auc = 0.0
        else:
            start_epoch = 1
            best_auc = 0.0
        epochs_no_imp = 0

        # Phase 1: frozen backbone
        if freeze_epochs > 0 and start_epoch <= freeze_epochs:
            self.model.freeze_backbone()
            opt_p1 = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.tcfg["head_lr"],
                weight_decay=self.tcfg["weight_decay"],
            )
            sched_p1 = CosineAnnealingLR(opt_p1, T_max=freeze_epochs, eta_min=1e-6)

            if resume and resume["mode"] == "full" and resume["phase"] == "P1":
                opt_p1.load_state_dict(resume["optimizer"])
                sched_p1.load_state_dict(resume["scheduler"])
                epochs_no_imp = resume["epochs_no_imp"]
                print(f"  [resume] P1 optimizer/scheduler restored")

            print(f"\n--- Phase 1: head-only (epochs {start_epoch}..{freeze_epochs}) ---")
            for epoch in range(start_epoch, freeze_epochs + 1):
                tr_loss, tr_acc, _, _ = run_epoch(
                    self.model, self.train_loader, criterion, opt_p1,
                    self.scaler, self.device, accum, is_train=True,
                    use_clinical=self.use_clinical, amp_dtype=self.amp_dtype,
                )
                vl_loss, vl_acc, vl_logits, vl_labels = run_epoch(
                    self.model, self.val_loader, criterion, opt_p1,
                    self.scaler, self.device, accum, is_train=False,
                    use_clinical=self.use_clinical, amp_dtype=self.amp_dtype,
                )
                sched_p1.step()

                metrics = patient_level_eval(vl_logits, vl_labels, n_slices)
                self._log(epoch, "P1", tr_loss, tr_acc, vl_loss, vl_acc, metrics)
                best_auc, epochs_no_imp = self._checkpoint(
                    epoch, "P1", opt_p1, sched_p1, metrics, best_auc, epochs_no_imp,
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

        if resume and resume["mode"] == "full" and resume["phase"] == "P2":
            opt_p2.load_state_dict(resume["optimizer"])
            sched_p2.load_state_dict(resume["scheduler"])
            epochs_no_imp = resume["epochs_no_imp"]
            print(f"  [resume] P2 optimizer/scheduler restored")
        else:
            epochs_no_imp = 0  # reset for phase 2

        p2_start = max(start_epoch, freeze_epochs + 1)
        print(f"\n--- Phase 2: LLRD fine-tune (epochs {p2_start}..{num_epochs}) ---")
        for epoch in range(p2_start, num_epochs + 1):
            if epochs_no_imp >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

            tr_loss, tr_acc, _, _ = run_epoch(
                self.model, self.train_loader, criterion, opt_p2,
                self.scaler, self.device, accum, is_train=True,
                use_clinical=self.use_clinical, amp_dtype=self.amp_dtype,
            )
            vl_loss, vl_acc, vl_logits, vl_labels = run_epoch(
                self.model, self.val_loader, criterion, opt_p2,
                self.scaler, self.device, accum, is_train=False,
                use_clinical=self.use_clinical, amp_dtype=self.amp_dtype,
            )
            sched_p2.step()

            metrics = patient_level_eval(vl_logits, vl_labels, n_slices)
            self._log(epoch, "P2", tr_loss, tr_acc, vl_loss, vl_acc, metrics)
            best_auc, epochs_no_imp = self._checkpoint(
                epoch, "P2", opt_p2, sched_p2, metrics, best_auc, epochs_no_imp,
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

    def _checkpoint(self, epoch, phase, optimizer, scheduler, metrics, best_auc, epochs_no_imp):
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            epochs_no_imp = 0
            path = os.path.join(self.checkpoint_dir, "best.pth")
            self._save_full(path, epoch, phase, optimizer, scheduler, best_auc, epochs_no_imp)
            print(f"  -> New best AUC: {best_auc:.4f} saved to {path}")
            if self.use_wandb:
                wandb.run.summary["best_auc"] = best_auc
                wandb.run.summary["best_epoch"] = epoch
        else:
            epochs_no_imp += 1

        path = os.path.join(self.checkpoint_dir, f"epoch_{epoch:02d}.pth")
        self._save_full(path, epoch, phase, optimizer, scheduler, best_auc, epochs_no_imp)
        return best_auc, epochs_no_imp

    def _save_full(self, path, epoch, phase, optimizer, scheduler, best_auc, epochs_no_imp):
        """Save model + training state so a disconnect mid-run can fully resume."""
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "phase": phase,
            "best_auc": best_auc,
            "epochs_no_imp": epochs_no_imp,
            "history": self.history,
            "amp_dtype": str(self.amp_dtype),
            "backbone": self.cfg["model"]["backbone"],
        }, path)

    def _maybe_load_resume(self):
        """Detect and load a resume checkpoint pointed at by cfg['resume'].
        Returns None if no resume; dict with mode='full'|'weights_only' otherwise.
        Aborts cleanly on backbone or shape mismatch — caller proceeds from scratch."""
        path = self.cfg.get("resume")
        if not path or not os.path.isfile(path):
            return None

        print(f"\n[resume] loading from {path}")
        blob = torch.load(path, map_location=self.device, weights_only=False)
        cfg_backbone = self.cfg["model"]["backbone"]

        if isinstance(blob, dict) and "model" in blob and "epoch" in blob:
            ckpt_backbone = blob.get("backbone")
            if ckpt_backbone and ckpt_backbone != cfg_backbone:
                print(f"[resume] BACKBONE MISMATCH  ckpt={ckpt_backbone}  cfg={cfg_backbone}")
                print(f"[resume] training from scratch — move/delete the stale checkpoints if intentional")
                return None
            try:
                self.model.load_state_dict(blob["model"])
            except RuntimeError as e:
                print(f"[resume] state_dict load failed: {str(e)[:200]}")
                print(f"[resume] training from scratch")
                return None
            if blob.get("history"):
                self.history = list(blob["history"])
            print(f"[resume] full state — epoch={blob['epoch']} phase={blob['phase']} "
                  f"best_auc={blob.get('best_auc', 0.0):.4f}")
            return {
                "mode": "full",
                "epoch": int(blob["epoch"]),
                "phase": blob["phase"],
                "best_auc": float(blob.get("best_auc", 0.0)),
                "epochs_no_imp": int(blob.get("epochs_no_imp", 0)),
                "optimizer": blob.get("optimizer"),
                "scheduler": blob.get("scheduler"),
            }

        # Old-format bare state_dict. Infer epoch from filename; phase-1 warmup
        # is already in the loaded weights so we skip it.
        try:
            self.model.load_state_dict(blob)
        except RuntimeError as e:
            print(f"[resume] state_dict load failed (old format, likely architecture mismatch):")
            print(f"         {str(e)[:200]}")
            print(f"[resume] training from scratch")
            return None
        m = re.search(r"epoch_(\d+)", os.path.basename(path))
        inferred = int(m.group(1)) if m else self.tcfg["freeze_epochs"]
        print(f"[resume] weights-only (old format)  inferred epoch={inferred}")
        return {"mode": "weights_only", "epoch": inferred}

    def _save_history(self):
        path = os.path.join(self.checkpoint_dir, "history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
