"""Pre-run dataset audit, sanity checks, and data quality report generation."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from learnsign.features.schema import HAND_POINTS


def audit(
    train_csv: str | Path | pd.DataFrame,
    data_dir: str | Path,
    output_report: str | Path | None = None,
) -> dict[str, Any]:
    """Audit landmark dataset files for integrity, missing data, and statistical distributions.

    Scans the training index and parquet files to flag:
    - Injected or corrupted all-NaN files
    - Duplicate landmark file paths in the metadata index
    - Class distribution imbalances
    - Sequence length percentiles and missing-hand frame ratios

    Args:
        train_csv: Path to metadata CSV or pre-loaded DataFrame.
        data_dir: Base directory containing parquet landmark files.
        output_report: Optional file path where Markdown report is written.

    Returns:
        Dictionary containing audit metrics and flagged anomalies.
    """
    if isinstance(train_csv, (str, Path)):
        df = pd.read_csv(train_csv)
    else:
        df = train_csv.copy()

    base_dir = Path(data_dir)

    # Detect duplicate paths in metadata index
    path_counts = df["path"].value_counts()
    duplicate_paths = path_counts[path_counts > 1].index.tolist()

    # Class distribution counts
    class_counts = df["sign"].value_counts().to_dict() if "sign" in df.columns else {}

    sequence_lengths: list[int] = []
    total_frames = 0
    missing_hand_frames = 0
    total_nan_coords = 0
    total_coords = 0
    flagged_files: list[dict[str, str]] = []

    # Process unique file paths to evaluate data integrity
    unique_paths = df["path"].unique()
    for rel_path in unique_paths:
        full_path = base_dir / rel_path
        if not full_path.exists():
            flagged_files.append({"path": str(rel_path), "reason": "missing_file"})
            continue

        try:
            table = pq.read_table(str(full_path), columns=["frame", "type", "x", "y", "z"])
            pdf = table.to_pandas()
        except Exception as exc:
            flagged_files.append({"path": str(rel_path), "reason": f"read_error: {exc}"})
            continue

        if pdf.empty:
            flagged_files.append({"path": str(rel_path), "reason": "empty_table"})
            continue

        # Check for all-NaN values across spatial coordinates
        x_col = pdf["x"].values
        y_col = pdf["y"].values
        z_col = pdf["z"].values
        nan_mask = np.isnan(x_col) & np.isnan(y_col) & np.isnan(z_col)
        n_nans = int(np.sum(nan_mask))
        n_points = len(pdf)

        total_nan_coords += n_nans * 3
        total_coords += n_points * 3

        if n_nans == n_points:
            flagged_files.append({"path": str(rel_path), "reason": "all_nan"})

        # Frame statistics and missing hand calculation
        frames = pdf["frame"].unique()
        n_frames = len(frames)
        sequence_lengths.append(n_frames)
        total_frames += n_frames

        # Count frames missing both hands
        hand_df = pdf[pdf["type"].isin(["left_hand", "right_hand"])]
        if not hand_df.empty:
            for _f, f_data in hand_df.groupby("frame"):
                f_nans = np.isnan(f_data["x"].values)
                # If all hand landmarks in this frame are NaN or missing
                if len(f_data) < (HAND_POINTS * 2) or np.all(f_nans):
                    missing_hand_frames += 1
        else:
            missing_hand_frames += n_frames

    seq_arr = np.array(sequence_lengths) if sequence_lengths else np.array([0])
    sequence_length_stats = {
        "min": int(np.min(seq_arr)),
        "max": int(np.max(seq_arr)),
        "median": float(np.median(seq_arr)),
        "mean": round(float(np.mean(seq_arr)), 2),
        "p95": round(float(np.percentile(seq_arr, 95)), 2),
    }

    missing_hand_frame_ratio = (
        round(missing_hand_frames / total_frames, 4) if total_frames > 0 else 0.0
    )

    results = {
        "total_clips_indexed": len(df),
        "unique_clips_scanned": len(unique_paths),
        "duplicate_paths": duplicate_paths,
        "class_counts": class_counts,
        "sequence_length_stats": sequence_length_stats,
        "missing_hand_frame_ratio": missing_hand_frame_ratio,
        "nan_count": total_nan_coords,
        "flagged_files": flagged_files,
    }

    if output_report is not None:
        rep_path = Path(output_report)
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        report_md = _generate_markdown_report(results)
        rep_path.write_text(report_md, encoding="utf-8")

    return results


def _generate_markdown_report(res: dict[str, Any]) -> str:
    """Format audit results as a structured GitHub-Flavored Markdown report."""
    seq = res["sequence_length_stats"]

    flagged_table_lines = [
        "| Path | Reason |",
        "| :--- | :--- |",
    ]
    if res["flagged_files"]:
        for f in res["flagged_files"]:
            flagged_table_lines.append(f"| {f['path']} | {f['reason']} |")
    else:
        flagged_table_lines.append("| None | No anomalies detected |")

    flagged_table = "\n".join(flagged_table_lines)

    return f"""# Data Audit Report

- **Total Clips Indexed:** {res["total_clips_indexed"]}
- **Unique Clips Scanned:** {res["unique_clips_scanned"]}
- **Duplicate Paths Detected:** {len(res["duplicate_paths"])}
- **Flagged Anomaly Files:** {len(res["flagged_files"])}
- **Missing Hand Frame Ratio:** {res["missing_hand_frame_ratio"] * 100:.2f}%
- **Total NaN Coordinates:** {res["nan_count"]}

## Sequence Length Distribution (Frames)
| Metric | Value |
| :--- | :--- |
| Min | {seq["min"]} |
| Max | {seq["max"]} |
| Median | {seq["median"]} |
| Mean | {seq["mean"]} |
| p95 | {seq["p95"]} |

## Flagged Anomalies
{flagged_table}
"""
