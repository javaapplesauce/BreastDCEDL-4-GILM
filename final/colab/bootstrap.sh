#!/usr/bin/env bash
# Colab A100 bootstrap for BreastDCEDL-4-GILM.
# Installs Python deps and sanity-checks the GPU.

set -euo pipefail

REPO_DIR="${1:-/content/breastdcedl}"
cd "$REPO_DIR"

echo "=== pip install ==="
python -m pip install -q -U pip
python -m pip install -q \
    "torch>=2.0" "torchvision>=0.15" \
    "transformers>=4.30" \
    "nibabel>=5.0" "numpy>=1.24" "pandas>=2.0" \
    "scikit-learn>=1.3" "Pillow>=10.0" \
    "pyyaml>=6.0" "matplotlib>=3.7" "seaborn>=0.12" \
    "tqdm>=4.66" "requests>=2.31" "wandb>=0.16"

echo ""
echo "=== GPU check ==="
python - <<'PY'
import torch
print(f"torch  : {torch.__version__}")
print(f"cuda   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("no CUDA device — switch runtime to GPU / A100")
PY

echo ""
echo "=== repo check ==="
python - <<'PY'
import sys, os
sys.path.insert(0, os.getcwd())
from src.data.zenodo import download_zenodo, FILES
from src.data.preprocessing import select_timepoints
from src.data.dataset import BreastDCEDataset
from src.models.vit import BreastDCEViT
from src.training.trainer import Trainer
from src.evaluation.metrics import patient_level_eval, predict_with_tta
print("OK: src/ imports")
print(f"zenodo files available: {list(FILES.keys())}")
PY

echo ""
echo "bootstrap complete."
