"""
Train/validation/test splitting.

Supports:
  1. Paper's pre-defined splits via 'split' or 'test' column in metadata
  2. Stratified random split as fallback
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_split(
    metadata_path: str,
    label_col: str = "pCR",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load metadata and return (train_df, val_df, test_df).

    If the CSV has a 'split' column with values in {train, val, test},
    use that directly (paper's benchmark partitions). Otherwise, if 'test'
    column exists (0/1), use it for val+test and sub-split. As a last
    resort, do an 80/10/10 stratified random split.
    """
    df = pd.read_csv(metadata_path)
    df = df.dropna(subset=[label_col]).reset_index(drop=True)

    if "split" in df.columns:
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        val_df = df[df["split"] == "val"].reset_index(drop=True)
        test_df = df[df["split"] == "test"].reset_index(drop=True)
        return train_df, val_df, test_df

    if "test" in df.columns:
        # MinCrop metadata encodes: 0=train, 1=val, 2=test
        train_df = df[df["test"] == 0].reset_index(drop=True)
        val_df = df[df["test"] == 1].reset_index(drop=True)
        test_df = df[df["test"] == 2].reset_index(drop=True)

        # Fall back to sub-splitting if the CSV only uses 0/1 (old convention)
        if len(test_df) == 0:
            holdout = df[df["test"] == 1].reset_index(drop=True)
            if len(holdout) > 20:
                val_df, test_df = train_test_split(
                    holdout, test_size=0.5, random_state=seed,
                    stratify=holdout[label_col],
                )
            else:
                val_df = holdout
                test_df = holdout.copy()

        return (
            train_df,
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    # Fallback: stratified 80/10/10
    train_df, temp = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df[label_col],
    )
    val_df, test_df = train_test_split(
        temp, test_size=0.5, random_state=seed, stratify=temp[label_col],
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def filter_by_subtype(df: pd.DataFrame, subtype: dict) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, val in subtype.items():
        if col in df.columns:
            mask &= df[col] == val
    return df[mask].reset_index(drop=True)
