"""
Patient-level evaluation metrics matching the paper's Table 2:
  AUC, Accuracy, Sensitivity, Specificity, Precision, NPV

Also provides subtype-stratified evaluation and visualization.
"""

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score,
    confusion_matrix, classification_report,
)


def patient_level_eval(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_slices: int,
) -> dict[str, float]:
    """
    Mean-pool logits across n_slices per patient, then compute metrics.
    Assumes logits/labels are ordered contiguously by patient (no shuffle).
    """
    n_total = logits.shape[0]
    n_patients = n_total // n_slices

    if n_patients == 0 or n_total % n_slices != 0:
        return _empty_metrics()

    logits_3d = logits[:n_patients * n_slices].view(n_patients, n_slices, -1)
    labels_2d = labels[:n_patients * n_slices].view(n_patients, n_slices)

    pooled = logits_3d.mean(dim=1)
    patient_labels = labels_2d[:, 0].numpy()
    probs = torch.softmax(pooled, dim=1)[:, 1].numpy()
    preds = pooled.argmax(dim=1).numpy()

    return compute_metrics(patient_labels, probs, preds)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.5

    acc = float(accuracy_score(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "auc": auc,
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
    }


def _empty_metrics() -> dict[str, float]:
    return {k: 0.0 for k in ["auc", "accuracy", "sensitivity", "specificity", "precision", "npv"]}


def evaluate_by_subtype(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    label_col: str = "pCR",
    subtypes: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """
    Compute metrics for overall and each receptor subtype.
    Returns a DataFrame matching the paper's Table 2 format.
    """
    y_true = df[label_col].values
    results = [{"Group": "Overall", "N": len(y_true),
                "pCR_rate": y_true.mean(),
                **compute_metrics(y_true, y_prob, y_pred)}]

    if subtypes:
        for name, filters in subtypes.items():
            mask = pd.Series(True, index=df.index)
            for col, val in filters.items():
                if col in df.columns:
                    mask &= df[col] == val
            idx = mask.values
            if idx.sum() < 5:
                continue
            results.append({
                "Group": name, "N": int(idx.sum()),
                "pCR_rate": y_true[idx].mean(),
                **compute_metrics(y_true[idx], y_prob[idx], y_pred[idx]),
            })

    return pd.DataFrame(results)


def predict_with_tta(
    model,
    loader,
    device,
    n_slices: int,
    use_clinical: bool = False,
    tta: bool = True,
):
    """
    Run inference with optional test-time augmentation (horizontal + vertical flips),
    average logits across augmentations and across n_slices per patient.
    Returns (patient_labels, probs, preds) numpy arrays.
    """
    import torch
    from torch.cuda.amp import autocast

    model.eval()
    views = [lambda x: x]
    if tta:
        views.extend([
            lambda x: torch.flip(x, dims=[-1]),
            lambda x: torch.flip(x, dims=[-2]),
            lambda x: torch.flip(x, dims=[-1, -2]),
        ])

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            if use_clinical:
                images, labels, clinical = batch
                clinical = clinical.to(device, non_blocking=True)
            else:
                images, labels = batch
                clinical = None
            images = images.to(device, non_blocking=True)
            logits_sum = None
            for v in views:
                with autocast():
                    out = model(v(images), clinical) if use_clinical \
                          else model(v(images))
                out = out.float()
                logits_sum = out if logits_sum is None else logits_sum + out
            all_logits.append((logits_sum / len(views)).cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    n_total = all_logits.shape[0]
    n_patients = n_total // n_slices
    logits_3d = all_logits[:n_patients * n_slices].view(n_patients, n_slices, -1)
    labels_2d = all_labels[:n_patients * n_slices].view(n_patients, n_slices)
    pooled = logits_3d.mean(dim=1)
    patient_labels = labels_2d[:, 0].numpy()
    probs = torch.softmax(pooled, dim=1)[:, 1].numpy()
    preds = pooled.argmax(dim=1).numpy()
    return patient_labels, probs, preds


def plot_roc(y_true, y_prob, title="ROC Curve", save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_confusion_matrix(y_true, y_pred, labels=("Non-pCR", "pCR"),
                          title="Confusion Matrix", save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
