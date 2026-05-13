"""
Loss functions for imbalanced pCR classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights easy examples to focus on hard, misclassified cases.
    Addresses the ~70/30 pCR class imbalance in BreastDCEDL.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, labels, weight=self.weight, reduction="none")
        p_t = torch.exp(-ce)
        return (((1 - p_t) ** self.gamma) * ce).mean()


def build_class_weights(labels: list[int], num_classes: int = 2) -> torch.Tensor:
    """
    Inverse-frequency class weights:  w_c = N / (num_classes * count_c).
    For a 71/29 split this returns ~[0.704, 1.724]: the majority class is
    down-weighted and the minority up-weighted around 1.0, so gradient
    magnitudes are preserved.

    The previous form returned `weights / weights.sum()`, which kept the
    same RATIO but shrank gradient magnitudes by ~3x. Combined with focal
    gamma=2 that left phase 1 head-warmup with almost no learning signal
    (run9.log showed val_acc 0.301, sens 1.0, spec 0.0 across epochs 1-6).
    """
    counts = torch.zeros(num_classes)
    for lab in labels:
        counts[int(lab)] += 1
    N = counts.sum()
    return N / (num_classes * counts.clamp(min=1))
