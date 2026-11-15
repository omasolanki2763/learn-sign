"""GISLR competition parquet loader and landmark sequence extractor."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from learnsign.features.normalize import from_gislr_frame, normalize_frame
from learnsign.features.schema import FEATURES_PER_FRAME


def load_sequence(parquet_path: str | Path, aspect_ratio: float = 1.0) -> np.ndarray:
    """Load a GISLR parquet clip and convert it to a normalized (T, 144) landmark array.

    Reads 543-row landmark frames, extracts hands and upper-body pose landmarks,
    and applies canonical 2D torso-anchored normalization per frame.

    Args:
        parquet_path: File system path to the GISLR parquet file.
        aspect_ratio: Optional aspect ratio correction (width / height) applied to x coordinates.

    Returns:
        float32 NumPy array of shape (T, 144), where T is the number of unique frames.
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet landmark file not found: {path}")

    # Read parquet table directly
    table = pq.read_table(str(path), columns=["frame", "type", "landmark_index", "x", "y", "z"])
    df = table.to_pandas()

    if df.empty:
        return np.empty((0, FEATURES_PER_FRAME), dtype=np.float32)

    frames = sorted(df["frame"].unique())
    num_frames = len(frames)
    seq = np.empty((num_frames, FEATURES_PER_FRAME), dtype=np.float32)

    # Group landmarks by frame
    grouped = df.groupby("frame", sort=True)
    for idx, (_frame_num, frame_df) in enumerate(grouped):
        lh, rh, pose = from_gislr_frame(frame_df)
        seq[idx] = normalize_frame(lh, rh, pose, aspect_ratio=aspect_ratio)

    return seq
