#!/usr/bin/env bash
#
# Download and prepare BreastDCEDL data from Zenodo.
#
# Prerequisites:
#   pip install -r requirements.txt
#   pip install zenodo_get
#
# The dataset is hosted at https://doi.org/10.5281/zenodo.15627233
# This script downloads the I-SPY2 subset by default (largest cohort,
# 982 patients, best documented results in the paper).
#
# For full reproduction, download all three cohorts.

set -euo pipefail

DATA_DIR="${1:-.}"
ZENODO_DOI="10.5281/zenodo.15627233"

echo "=== BreastDCEDL Data Setup ==="
echo "Target directory: $DATA_DIR"

# Check for required tools
command -v python3 >/dev/null 2>&1 || { echo "python3 required"; exit 1; }

# Install Python dependencies
echo "Installing dependencies..."
pip install -r requirements.txt -q

# Download from Zenodo (if not already present)
if [ ! -d "$DATA_DIR/ISPY2" ]; then
    echo "Downloading I-SPY2 data from Zenodo..."
    echo "DOI: $ZENODO_DOI"
    echo ""
    echo "NOTE: The full dataset is large (~50GB+). For initial testing,"
    echo "download a subset manually from the Zenodo page."
    echo ""
    echo "Alternative: download directly from TCIA:"
    echo "  - I-SPY2: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230072"
    echo "  - Duke:   https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903"
    echo "  - I-SPY1: https://wiki.cancerimagingarchive.net/display/Public/Breast-MRI-NACT-Pilot"
    echo ""
    echo "After downloading, place NIfTI files in the paths specified in configs/default.yaml"
else
    echo "I-SPY2 data directory already exists, skipping download."
fi

# Verify metadata
for csv in BreastDCEDL_metadata.csv BreastDCEDL_metadata_min_crop.csv; do
    if [ -f "$DATA_DIR/$csv" ]; then
        echo "Found metadata: $csv"
        python3 -c "
import pandas as pd
df = pd.read_csv('$DATA_DIR/$csv')
print(f'  Rows: {len(df)}')
if 'pCR' in df.columns:
    pcr = df['pCR'].dropna()
    print(f'  pCR labels: {len(pcr)} ({pcr.mean():.1%} positive)')
if 'pid' in df.columns:
    for cohort, pat in [('Duke','Breast_MRI'),('I-SPY1','SPY1'),('I-SPY2','SPY2')]:
        n = df['pid'].str.contains(pat, na=False).sum()
        if n: print(f'  {cohort}: {n} patients')
"
    fi
done

echo ""
echo "Setup complete. Run training with:"
echo "  python scripts/train.py --config configs/default.yaml"
