#!/bin/bash
source /home/richard/miniconda3/etc/profile.d/conda.sh
conda activate brestdcedl
cd /home/richard/Desktop/workspace/BreastDCEDL-4-GILM
exec python finetune.py
