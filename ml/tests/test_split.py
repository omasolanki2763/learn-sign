"""Unit tests for signer-stratified dataset splitting."""

import pandas as pd
import pytest

from learnsign.data.split import signer_split


def test_signer_split_no_signer_leakage() -> None:
    """Review Focus 5: Assert no participant_id appears in both train and validation splits."""
    data = []
    for participant_id in range(10):
        for sample_id in range(5):
            data.append(
                {
                    "path": f"train_landmarks/{participant_id}_{sample_id}.parquet",
                    "participant_id": participant_id,
                    "sequence_id": sample_id,
                    "sign": "water" if sample_id % 2 == 0 else "book",
                }
            )
    df = pd.DataFrame(data)

    n_val_signers = 3
    train_df, val_df = signer_split(df, n_val_signers=n_val_signers, seed=42)

    train_signers = set(train_df["participant_id"].unique())
    val_signers = set(val_df["participant_id"].unique())

    # Critical Review Focus 5 check: zero signer leakage
    intersection = train_signers.intersection(val_signers)
    assert len(intersection) == 0, f"Signer leakage detected across splits: {intersection}"

    # Validation participant count matches specification
    assert len(val_signers) == n_val_signers
    assert len(train_signers) == 10 - n_val_signers
    assert len(train_df) + len(val_df) == len(df)


def test_signer_split_invalid_signer_count() -> None:
    """Assert signer_split raises ValueError when validation signer count is out of bounds."""
    df = pd.DataFrame({"participant_id": [1, 1, 2, 2], "sign": ["a", "b", "a", "b"]})
    with pytest.raises(ValueError, match="n_val_signers"):
        signer_split(df, n_val_signers=5)
