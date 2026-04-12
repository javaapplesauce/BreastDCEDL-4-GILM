"""
Ablation study runner.

Systematically varies one hyperparameter at a time from the default config,
trains each variant, and collects results into a summary table.

Usage:
    python scripts/ablation.py --config configs/default.yaml --output results/ablation

Ablation axes (each run modifies ONE axis from the default):
    1. Backbone:     vit-base-in21k vs dinov2-base
    2. Loss:         focal (gamma=2) vs cross-entropy
    3. Augmentation: full vs minimal (flip-only) vs none
    4. Slices:       4, 8, 12
    5. Clinical:     with vs without clinical features
    6. LLRD:         0.75, 0.85, 0.95
    7. Dropout:      0.1, 0.3, 0.5
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import yaml
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train import train_from_config


ABLATION_AXES = {
    "backbone": [
        {"model.backbone": "google/vit-base-patch16-224-in21k"},
        {"model.backbone": "facebook/dinov2-base"},
    ],
    "loss": [
        {"training.loss": "focal"},
        {"training.loss": "ce"},
    ],
    "augmentation": [
        {"training.augmentation_mode": "full"},
        {"training.augmentation_mode": "minimal"},
        {"training.augmentation_mode": "none"},
    ],
    "n_slices": [
        {"data.n_slices": 4},
        {"data.n_slices": 8},
        {"data.n_slices": 12},
    ],
    "clinical": [
        {"model.use_clinical": False},
        {"model.use_clinical": True},
    ],
    "llrd": [
        {"training.llrd": 0.75},
        {"training.llrd": 0.85},
        {"training.llrd": 0.95},
    ],
    "dropout": [
        {"model.dropout": 0.1},
        {"model.dropout": 0.3},
        {"model.dropout": 0.5},
    ],
}


def _set_nested(cfg: dict, dotted_key: str, value):
    keys = dotted_key.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def run_ablation(base_cfg: dict, axis: str, output_dir: str):
    variants = ABLATION_AXES[axis]
    results = []

    for i, overrides in enumerate(variants):
        cfg = copy.deepcopy(base_cfg)
        name_parts = []
        for key, val in overrides.items():
            _set_nested(cfg, key, val)
            name_parts.append(f"{key.split('.')[-1]}={val}")
        run_name = f"{axis}__{'-'.join(name_parts)}"
        cfg["checkpoint_dir"] = os.path.join(output_dir, run_name)

        print(f"\n{'='*60}")
        print(f"Ablation: {run_name}")
        print(f"{'='*60}")

        best_auc = train_from_config(cfg)
        history_path = os.path.join(cfg["checkpoint_dir"], "history.json")
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)
            best_entry = max(history, key=lambda e: e.get("auc", 0))
        else:
            best_entry = {}

        results.append({
            "axis": axis,
            "run": run_name,
            "best_auc": best_auc,
            **{k: best_entry.get(k, None) for k in
               ["accuracy", "sensitivity", "specificity", "precision", "npv"]},
        })

    df = pd.DataFrame(results)
    summary_path = os.path.join(output_dir, f"ablation_{axis}.csv")
    df.to_csv(summary_path, index=False)
    print(f"\nAblation results saved to {summary_path}")
    print(df.to_string(index=False))
    return df


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="results/ablation")
    parser.add_argument("--axis", default=None,
                        help=f"Ablation axis to run. Options: {list(ABLATION_AXES.keys())}. "
                             "If omitted, runs all axes.")
    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    os.makedirs(args.output, exist_ok=True)
    axes = [args.axis] if args.axis else list(ABLATION_AXES.keys())

    all_results = []
    for axis in axes:
        df = run_ablation(base_cfg, axis, args.output)
        all_results.append(df)

    if len(all_results) > 1:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(os.path.join(args.output, "ablation_all.csv"), index=False)
        print(f"\nCombined ablation results saved.")


if __name__ == "__main__":
    main()
