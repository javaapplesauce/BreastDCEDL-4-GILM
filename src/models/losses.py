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
    counts = torch.zeros(num_classes)
    for lab in labels:
        counts[int(lab)] += 1
    weights = 1.0 / counts.clamp(min=1)
    return weights / weights.sum()
