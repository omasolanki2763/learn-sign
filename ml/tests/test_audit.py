"""Unit tests for dataset audit and sanity check flags."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from learnsign.data.audit import audit


def _create_synthetic_parquet(file_path: Path, n_frames: int = 5, is_all_nan: bool = False) -> None:
    """Helper to generate a synthetic GISLR parquet file with 543 landmarks per frame."""
    rows = []
    for f in range(n_frames):
        # 468 face
        for i in range(468):
            rows.append(
                {
                    "frame": f,
                    "type": "face",
                    "landmark_index": i,
                    "x": np.nan if is_all_nan else 0.5,
                    "y": np.nan if is_all_nan else 0.5,
                    "z": np.nan if is_all_nan else 0.0,
                }
            )
        # 21 left hand
        for i in range(21):
            rows.append(
                {
                    "frame": f,
                    "type": "left_hand",
                    "landmark_index": i,
                    "x": np.nan if is_all_nan else 0.6,
                    "y": np.nan if is_all_nan else 0.7,
                    "z": np.nan if is_all_nan else 0.1,
                }
            )
        # 33 pose
        for i in range(33):
            rows.append(
                {
                    "frame": f,
                    "type": "pose",
                    "landmark_index": i,
                    "x": np.nan if is_all_nan else 0.5,
                    "y": np.nan if is_all_nan else 0.4,
                    "z": np.nan if is_all_nan else 0.0,
                }
            )
        # 21 right hand
        for i in range(21):
            rows.append(
                {
                    "frame": f,
                    "type": "right_hand",
                    "landmark_index": i,
                    "x": np.nan,
                    "y": np.nan,
                    "z": np.nan,
                }
            )  # Right hand missing

    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(file_path))


def test_audit_flags_nan_and_duplicates(tmp_path: Path) -> None:
    """Step 1 test: Assert audit flags an injected NaN file and duplicate path."""
    data_dir = tmp_path / "data"
    p_valid = data_dir / "valid.parquet"
    p_nan = data_dir / "all_nan.parquet"

    _create_synthetic_parquet(p_valid, n_frames=10, is_all_nan=False)
    _create_synthetic_parquet(p_nan, n_frames=5, is_all_nan=True)

    # Injected duplicate path in train CSV
    csv_data = [
        {"path": "valid.parquet", "participant_id": 1, "sign": "water"},
        {"path": "valid.parquet", "participant_id": 1, "sign": "water"},  # Duplicate
        {"path": "all_nan.parquet", "participant_id": 2, "sign": "book"},  # Injected NaN
    ]
    train_csv = tmp_path / "train.csv"
    pd.DataFrame(csv_data).to_csv(train_csv, index=False)

    report_path = tmp_path / "reports" / "data_audit.md"
    results = audit(train_csv, data_dir, output_report=report_path)

    # Assert audit summary dictionary structure
    assert "class_counts" in results
    assert "sequence_length_stats" in results
    assert "missing_hand_frame_ratio" in results
    assert "nan_count" in results
    assert "duplicate_paths" in results
    assert "flagged_files" in results

    # Verify duplicate detection
    assert len(results["duplicate_paths"]) == 1
    assert "valid.parquet" in results["duplicate_paths"][0]

    # Verify all-NaN file detection
    flagged_nan_paths = [f["path"] for f in results["flagged_files"] if f["reason"] == "all_nan"]
    assert any("all_nan.parquet" in p for p in flagged_nan_paths)

    # Verify report was written
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "Data Audit Report" in content
