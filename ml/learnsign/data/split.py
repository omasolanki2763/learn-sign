"""Signer-independent train/validation split logic."""

from pathlib import Path

import numpy as np
import pandas as pd


def signer_split(
    train_csv: str | Path | pd.DataFrame,
    n_val_signers: int = 4,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split GISLR dataset by participant_id to enforce signer-independent validation.

    Ensures that no signer appears in both training and validation sets, preventing
    signer leakage and guaranteeing true out-of-domain evaluation.

    Args:
        train_csv: Path to CSV or existing DataFrame containing 'participant_id'.
        n_val_signers: Number of distinct signers reserved for the validation set.
        seed: Random seed for reproducible signer sampling.

    Returns:
        Tuple of (train_df, val_df).

    Raises:
        ValueError: If n_val_signers is not strictly between 1 and total_signers - 1.
    """
    if isinstance(train_csv, (str, Path)):
        df = pd.read_csv(train_csv)
    else:
        df = train_csv.copy()

    if "participant_id" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'participant_id' column for signer-independent splitting."
        )

    unique_signers = np.array(sorted(df["participant_id"].unique()))
    total_signers = len(unique_signers)

    if n_val_signers < 1 or n_val_signers >= total_signers:
        raise ValueError(
            f"n_val_signers ({n_val_signers}) must be between 1 and "
            f"total signers - 1 ({total_signers - 1})."
        )

    rng = np.random.default_rng(seed)
    val_signers = set(rng.choice(unique_signers, size=n_val_signers, replace=False))

    is_val = df["participant_id"].isin(val_signers)
    val_df = df[is_val].copy().reset_index(drop=True)
    train_df = df[~is_val].copy().reset_index(drop=True)

    return train_df, val_df
