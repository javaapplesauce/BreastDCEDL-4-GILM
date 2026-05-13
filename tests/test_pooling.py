"""Patient-id-based pooling agrees with the old reshape-based pooling on
the happy path (slices arrive in strict patient-contiguous order, no
patient dropped). The new code remains correct under reorder/drop; the
old code did not."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import patient_level_eval, compute_metrics


def _build(n_patients=10, n_slices=8, num_classes=2, seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(n_patients * n_slices, num_classes)
    labels = torch.tensor(
        [i % 2 for i in range(n_patients) for _ in range(n_slices)]
    )
    pids = [f"P{i:03d}" for i in range(n_patients) for _ in range(n_slices)]
    return logits, labels, pids


def _legacy_reshape_pool(logits, labels, n_slices):
    n_total = logits.shape[0]
    n_patients = n_total // n_slices
    logits_3d = logits[: n_patients * n_slices].view(n_patients, n_slices, -1)
    labels_2d = labels[: n_patients * n_slices].view(n_patients, n_slices)
    pooled = logits_3d.mean(dim=1)
    plabs = labels_2d[:, 0].numpy()
    probs = torch.softmax(pooled, dim=1)[:, 1].numpy()
    preds = pooled.argmax(dim=1).numpy()
    return compute_metrics(plabs, probs, preds)


def test_new_pool_matches_legacy_on_happy_path():
    n_slices = 8
    logits, labels, pids = _build(n_patients=10, n_slices=n_slices)
    new = patient_level_eval(logits, labels, pids)
    old = _legacy_reshape_pool(logits, labels, n_slices)
    for k in new:
        assert abs(new[k] - old[k]) < 1e-6, (
            f"{k}: new={new[k]} old={old[k]}"
        )


def test_new_pool_invariant_under_reorder():
    """Old reshape code would silently corrupt labels under any reorder.
    The new code returns identical metrics because pooling is keyed on pid."""
    n_slices = 8
    logits, labels, pids = _build(n_patients=10, n_slices=n_slices)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(pids))
    logits_p = logits[perm]
    labels_p = labels[perm]
    pids_p = [pids[i] for i in perm]
    a = patient_level_eval(logits, labels, pids)
    b = patient_level_eval(logits_p, labels_p, pids_p)
    for k in a:
        assert abs(a[k] - b[k]) < 1e-6, f"{k}: original={a[k]} permuted={b[k]}"


def test_new_pool_handles_unequal_slice_counts():
    """Pre-resolve drops some patients, so loader may produce different
    counts per patient. Dict-based pool handles this; reshape-based did not."""
    torch.manual_seed(0)
    logits = torch.randn(20, 2)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0])
    pids = (
        ["P0"] * 3 + ["P1"] * 5 + ["P2"] * 4 + ["P3"] * 2 + ["P4"] * 6
    )
    m = patient_level_eval(logits, labels, pids)
    assert 0.0 <= m["auc"] <= 1.0
    assert 0.0 <= m["accuracy"] <= 1.0


def test_new_pool_raises_on_length_mismatch():
    logits, labels, pids = _build(n_patients=10, n_slices=8)
    import pytest
    with pytest.raises(ValueError):
        patient_level_eval(logits, labels, pids[:-1])
