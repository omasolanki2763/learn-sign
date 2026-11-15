"""Unit tests for the quick baseline 1D-CNN, resampling, and overfit-one-batch check."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from learnsign.data.gislr import load_sequence
from learnsign.features.schema import FEATURES_PER_FRAME
from learnsign.train.quick_baseline import (
    DEFAULT_TARGET_FRAMES,
    MAX_MODEL_PARAMS,
    QuickBaseline1DCNN,
    overfit_one_batch,
    resample_sequence,
)


def test_quick_baseline_parameter_count() -> None:
    """Step 3 test: Assert quick baseline 1D-CNN parameter count is strictly <= 500k."""
    model = QuickBaseline1DCNN(in_features=FEATURES_PER_FRAME, num_classes=250)
    param_count = model.count_parameters()

    assert param_count <= MAX_MODEL_PARAMS, (
        f"Model params {param_count} exceeded {MAX_MODEL_PARAMS}"
    )
    assert param_count > 100_000, f"Model params {param_count} unexpectedly low"


def test_resample_sequence() -> None:
    """Assert resampling variable-length sequences produces exact target frame count."""
    # Test short sequence (10 frames) -> 64 frames
    seq_10 = (
        np.linspace(0.0, 1.0, 10 * FEATURES_PER_FRAME)
        .reshape(10, FEATURES_PER_FRAME)
        .astype(np.float32)
    )
    res_64 = resample_sequence(seq_10, target_frames=DEFAULT_TARGET_FRAMES)
    assert res_64.shape == (DEFAULT_TARGET_FRAMES, FEATURES_PER_FRAME)

    # Test long sequence (120 frames) -> 64 frames
    seq_120 = np.random.randn(120, FEATURES_PER_FRAME).astype(np.float32)
    res_from_120 = resample_sequence(seq_120, target_frames=DEFAULT_TARGET_FRAMES)
    assert res_from_120.shape == (DEFAULT_TARGET_FRAMES, FEATURES_PER_FRAME)

    # Test single frame repeated
    seq_1 = np.ones((1, FEATURES_PER_FRAME), dtype=np.float32)
    res_1 = resample_sequence(seq_1, target_frames=DEFAULT_TARGET_FRAMES)
    assert res_1.shape == (DEFAULT_TARGET_FRAMES, FEATURES_PER_FRAME)
    np.testing.assert_allclose(res_1, 1.0)


def test_overfit_one_batch_passes() -> None:
    """Step 3 test: Assert overfit_one_batch drives loss below 0.1 on a small batch."""
    torch.manual_seed(42)
    model = QuickBaseline1DCNN(in_features=FEATURES_PER_FRAME, num_classes=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    batch_x = torch.randn(4, FEATURES_PER_FRAME, DEFAULT_TARGET_FRAMES)
    batch_y = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    final_loss = overfit_one_batch(
        model,
        batch_x,
        batch_y,
        criterion,
        optimizer,
        threshold=0.1,
        max_steps=100,
    )

    assert final_loss < 0.1


def test_load_sequence_from_parquet(tmp_path: Path) -> None:
    """Assert load_sequence properly reads and normalizes a parquet file."""
    rows = []
    # 2 frames of synthetic landmarks
    for f in range(2):
        for i in range(468):
            rows.append(
                {"frame": f, "type": "face", "landmark_index": i, "x": 0.5, "y": 0.5, "z": 0.0}
            )
        for i in range(21):
            rows.append(
                {"frame": f, "type": "left_hand", "landmark_index": i, "x": 0.6, "y": 0.7, "z": 0.1}
            )
        for i in range(33):
            # Left shoulder (11): (0.6, 0.4), Right shoulder (12): (0.2, 0.4) => width = 0.4
            rows.append(
                {
                    "frame": f,
                    "type": "pose",
                    "landmark_index": i,
                    "x": 0.6 if i == 11 else 0.2,
                    "y": 0.4,
                    "z": 0.0,
                }
            )
        for i in range(21):
            rows.append(
                {
                    "frame": f,
                    "type": "right_hand",
                    "landmark_index": i,
                    "x": 0.2,
                    "y": 0.7,
                    "z": -0.1,
                }
            )

    df = pa.Table.from_pandas(pytest.importorskip("pandas").DataFrame(rows))
    parquet_path = tmp_path / "test_clip.parquet"
    pq.write_table(df, str(parquet_path))

    seq = load_sequence(parquet_path)
    assert seq.shape == (2, FEATURES_PER_FRAME)
    assert not np.isnan(seq[0, 0])
