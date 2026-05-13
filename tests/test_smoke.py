"""Smoke tests: one forward+backward on random data + fresh-python
import surface. Run before opening any PR."""
import importlib
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_imports_from_fresh_python():
    """`from src.data.dataset import BreastDCEDataset; from
    src.evaluation.metrics import predict_with_tta` must work from a
    cold Python with the repo on sys.path."""
    repo = Path(__file__).resolve().parents[1]
    code = (
        f"import sys; sys.path.insert(0, {str(repo)!r}); "
        "from src.data.dataset import BreastDCEDataset; "
        "from src.evaluation.metrics import predict_with_tta; "
        "print('OK')"
    )
    out = subprocess.check_output([sys.executable, "-c", code], stderr=subprocess.STDOUT, text=True)
    assert "OK" in out, out


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="transformers not installed locally",
)
def test_forward_backward_random_input():
    """Build BreastDCEViT (no pretrained), push a random 3x224x224 batch
    through it, take a backward step. Asserts loss is finite and at least
    one parameter received a gradient."""
    from src.models.vit import BreastDCEViT
    from src.models.losses import FocalLoss, build_class_weights

    torch.manual_seed(0)
    model = BreastDCEViT(
        backbone="google/vit-base-patch16-224-in21k",
        num_classes=2,
        dropout=0.1,
        pretrained_weights=None,
        pos_class_prior=0.3,
    )
    model.train()

    x = torch.randn(2, 3, 224, 224)
    y = torch.tensor([0, 1])
    weights = build_class_weights([0, 1] * 10, 2)
    criterion = FocalLoss(gamma=2.0, weight=weights)

    logits = model(x)
    loss = criterion(logits, y)
    assert torch.isfinite(loss), f"loss not finite: {loss}"

    loss.backward()
    grad_norms = [
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    ]
    assert grad_norms, "no parameter received a gradient"
    assert all(g >= 0 for g in grad_norms)
    assert max(grad_norms) > 0, "all gradients zero"
