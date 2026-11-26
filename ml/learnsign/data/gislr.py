"""GISLR competition parquet loader, vectorized normalization, and .npy cache."""

from pathlib import Path
from typing import Final

import numpy as np
import pyarrow.parquet as pq

from learnsign.features.normalize import (
    canonicalize_handedness,
    estimate_dominant_hand,
    from_gislr_frame,
    normalize_frame,
    normalize_sequence,
)
from learnsign.features.schema import (
    COORDINATES_PER_POINT,
    FEATURES_PER_FRAME,
    HAND_POINTS,
)

# GISLR 543-row fixed per-frame layout: face (468) + left_hand (21) + pose (33) + right_hand (21)
GISLR_FACE_POINTS: Final[int] = 468
GISLR_POSE_POINTS: Final[int] = 33
GISLR_ROWS_PER_FRAME: Final[int] = (
    GISLR_FACE_POINTS + HAND_POINTS + GISLR_POSE_POINTS + HAND_POINTS
)

_LEFT_HAND_OFFSET: Final[int] = GISLR_FACE_POINTS
_POSE_OFFSET: Final[int] = _LEFT_HAND_OFFSET + HAND_POINTS
_RIGHT_HAND_OFFSET: Final[int] = _POSE_OFFSET + GISLR_POSE_POINTS


def _resolve_cache_path(
    parquet_path: Path,
    aspect_ratio: float,
    cache_dir: str | Path | None,
) -> Path:
    """Compute cache path keyed by clip name and aspect_ratio (pre-canonicalization)."""
    aspect_tag = "" if np.isclose(aspect_ratio, 1.0) else f".ar{aspect_ratio:.6g}"
    filename = f"{parquet_path.stem}{aspect_tag}.npy"
    if cache_dir is not None:
        parent_tag = parquet_path.parent.name
        scoped = f"{parent_tag}__{filename}" if parent_tag else filename
        return Path(cache_dir) / scoped
    return parquet_path.with_name(filename)


def load_sequence_reference(parquet_path: str | Path, aspect_ratio: float = 1.0) -> np.ndarray:
    """Legacy per-frame pandas reference loader retained for equivalence testing."""
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet landmark file not found: {path}")

    table = pq.read_table(str(path), columns=["frame", "type", "landmark_index", "x", "y", "z"])
    df = table.to_pandas()

    if df.empty:
        return np.empty((0, FEATURES_PER_FRAME), dtype=np.float32)

    frames = sorted(df["frame"].unique())
    seq = np.empty((len(frames), FEATURES_PER_FRAME), dtype=np.float32)

    grouped = df.groupby("frame", sort=True)
    for idx, (_frame_num, frame_df) in enumerate(grouped):
        lh, rh, pose = from_gislr_frame(frame_df)
        seq[idx] = normalize_frame(lh, rh, pose, aspect_ratio=aspect_ratio)

    return seq


def load_sequence(
    parquet_path: str | Path,
    aspect_ratio: float = 1.0,
    canonicalize: bool = False,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> np.ndarray:
    """Load a GISLR parquet clip and convert it to a normalized landmark sequence.

    Reshapes the fixed 543-row per-frame layout in one array operation `(T, 543, 3)`
    with optional `.npy` caching and optional handedness canonicalization.

    Handedness in training: GISLR training/validation loaders pass `canonicalize=True`
    to apply `estimate_dominant_hand` + `canonicalize_handedness` here (mirroring
    left-dominant clips to right-dominant orientation, matching browser inference).
    The `.npy` cache stores uncanonicalized coordinates so toggling `canonicalize`
    never serves stale orientation.

    Args:
        parquet_path: File system path to the GISLR parquet file.
        aspect_ratio: Optional aspect ratio correction (width / height) applied to x coordinates.
        canonicalize: If True, apply `estimate_dominant_hand` + `canonicalize_handedness`.
        cache_dir: Optional writable directory for `.npy` cache files (useful when the input
            dataset directory is read-only, e.g. `/kaggle/input`). Defaults to next to the parquet.
        use_cache: Whether to read/write the `.npy` cache (default True).

    Returns:
        float32 NumPy array of shape (T, FEATURES_PER_FRAME).
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet landmark file not found: {path}")

    cache_path = _resolve_cache_path(path, aspect_ratio=aspect_ratio, cache_dir=cache_dir)

    seq: np.ndarray | None = None
    if (
        use_cache
        and cache_path.exists()
        and cache_path.stat().st_mtime >= path.stat().st_mtime
    ):
        seq = np.load(str(cache_path), allow_pickle=False).astype(np.float32, copy=False)

    if seq is None:
        table = pq.read_table(str(path), columns=["x", "y", "z"])
        num_rows = len(table)

        if num_rows == 0:
            seq = np.empty((0, FEATURES_PER_FRAME), dtype=np.float32)
        elif num_rows % GISLR_ROWS_PER_FRAME != 0:
            seq = load_sequence_reference(path, aspect_ratio=aspect_ratio)
        else:
            coords = np.column_stack(
                [
                    table["x"].to_numpy(zero_copy_only=False),
                    table["y"].to_numpy(zero_copy_only=False),
                    table["z"].to_numpy(zero_copy_only=False),
                ]
            ).astype(np.float32, copy=False)
            num_frames = num_rows // GISLR_ROWS_PER_FRAME
            frames = coords.reshape((num_frames, GISLR_ROWS_PER_FRAME, COORDINATES_PER_POINT))

            left_hands = frames[:, _LEFT_HAND_OFFSET:_POSE_OFFSET, :]
            poses = frames[:, _POSE_OFFSET:_RIGHT_HAND_OFFSET, :]
            right_hands = frames[:, _RIGHT_HAND_OFFSET : _RIGHT_HAND_OFFSET + HAND_POINTS, :]
            seq = normalize_sequence(left_hands, right_hands, poses, aspect_ratio=aspect_ratio)

        if use_cache:
            if cache_dir is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(cache_path), seq)
            else:
                try:
                    np.save(str(cache_path), seq)
                except OSError:
                    # Input directories (such as Kaggle's /kaggle/input) can be read-only
                    pass

    if canonicalize and seq.shape[0] > 0:
        is_left = estimate_dominant_hand(seq) == "left"
        seq = canonicalize_handedness(seq, is_left_dominant=is_left)

    return seq
