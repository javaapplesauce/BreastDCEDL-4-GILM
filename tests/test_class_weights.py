"""Inverse-frequency class weights: w_c = N / (num_classes * count_c).
For a 71/29 split this must come out near [0.704, 1.724]."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.losses import build_class_weights


def test_weights_for_71_29_split():
    n = 1000
    labels = [0] * 710 + [1] * 290
    w = build_class_weights(labels, 2)
    expected = torch.tensor([0.704, 1.724])
    assert torch.allclose(w, expected, rtol=0.01), (
        f"weights for 71/29 split={w.tolist()} expected~{expected.tolist()}"
    )


def test_weights_for_balanced_split():
    labels = [0] * 500 + [1] * 500
    w = build_class_weights(labels, 2)
    assert torch.allclose(w, torch.tensor([1.0, 1.0]), atol=1e-4)


def test_weights_handle_empty_class():
    labels = [0] * 100
    w = build_class_weights(labels, 2)
    assert w[0].item() == 0.5
    assert torch.isfinite(w[1])


def test_weights_are_not_rescaled_to_sum_one():
    """The previous implementation returned weights / weights.sum(),
    shrinking gradient magnitudes by ~3x. The new form preserves them."""
    labels = [0] * 710 + [1] * 290
    w = build_class_weights(labels, 2)
    assert w.sum().item() > 1.5, (
        f"weights sum={w.sum().item():.3f} — regressed to normalized form?"
    )
