"""Unit tests for vectorized GISLR parquet sequence loading, caching, and canonicalization."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from learnsign.data.gislr import load_sequence, load_sequence_reference
from learnsign.features.normalize import canonicalize_handedness, estimate_dominant_hand
from learnsign.features.schema import (
    FEATURES_PER_FRAME,
    HAND_POINTS,
    LEFT_HAND_END,
    LEFT_HAND_START,
    RIGHT_HAND_END,
    RIGHT_HAND_START,
)


def _write_gislr_parquet(
    file_path: Path,
    n_frames: int = 4,
    left_active: bool = False,
    missing_right_on_frame_1: bool = False,
    invalid_torso_last_frame: bool = False,
) -> None:
    """Create a synthetic GISLR parquet file matching the 543-row fixed per-frame layout."""
    rows = []
    for f in range(n_frames):
        for i in range(468):
            rows.append(
                {"frame": f, "type": "face", "landmark_index": i, "x": 0.5, "y": 0.5, "z": 0.0}
            )
        for i in range(HAND_POINTS):
            lx = 0.6 + (0.05 * f if left_active else 0.0) + i * 0.001
            rows.append(
                {
                    "frame": f,
                    "type": "left_hand",
                    "landmark_index": i,
                    "x": lx,
                    "y": 0.7 - (0.04 * f if left_active else 0.0),
                    "z": 0.05 * i,
                }
            )
        for i in range(33):
            if invalid_torso_last_frame and f == n_frames - 1 and i in (11, 12):
                px, py, pz = np.nan, np.nan, np.nan
            elif i == 11:
                px, py, pz = 0.65, 0.4, 0.02
            elif i == 12:
                px, py, pz = 0.25, 0.4, -0.02
            else:
                px, py, pz = 0.45 + 0.002 * i, 0.5 + 0.001 * i, 0.01
            rows.append(
                {"frame": f, "type": "pose", "landmark_index": i, "x": px, "y": py, "z": pz}
            )
        for i in range(HAND_POINTS):
            if missing_right_on_frame_1 and f == 1:
                rx, ry, rz = np.nan, np.nan, np.nan
            else:
                rx = 0.2 + (0.0 if left_active else 0.04 * f) - i * 0.001
                ry = 0.7
                rz = -0.03 * i
            rows.append(
                {"frame": f, "type": "right_hand", "landmark_index": i, "x": rx, "y": ry, "z": rz}
            )

    table = pa.Table.from_pandas(pd.DataFrame(rows))
    pq.write_table(table, str(file_path))


@pytest.mark.parametrize("aspect_ratio", [1.0, 16.0 / 9.0])
def test_vectorized_load_sequence_matches_reference(tmp_path: Path, aspect_ratio: float) -> None:
    """Assert vectorized load_sequence produces exact match to legacy per-frame loader."""
    parquet_path = tmp_path / "equiv_clip.parquet"
    _write_gislr_parquet(
        parquet_path,
        n_frames=5,
        missing_right_on_frame_1=True,
        invalid_torso_last_frame=True,
    )

    ref = load_sequence_reference(parquet_path, aspect_ratio=aspect_ratio)
    vec = load_sequence(parquet_path, aspect_ratio=aspect_ratio, use_cache=False)

    assert vec.shape == ref.shape == (5, FEATURES_PER_FRAME)
    assert vec.dtype == np.float32
    assert np.array_equal(np.isnan(vec), np.isnan(ref))

    valid = ~np.isnan(ref)
    np.testing.assert_allclose(vec[valid], ref[valid], atol=1e-5)


def test_load_sequence_npy_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert load_sequence writes and reuses .npy cache on subsequent loads."""
    parquet_path = tmp_path / "cached_clip.parquet"
    _write_gislr_parquet(parquet_path, n_frames=4)

    first = load_sequence(parquet_path)
    expected_cache = parquet_path.with_suffix(".npy")
    assert expected_cache.exists()

    # Ensure second call does not read parquet at all
    def _fail_read(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Expected .npy cache hit without reading parquet")

    monkeypatch.setattr(pq, "read_table", _fail_read)
    second = load_sequence(parquet_path)
    np.testing.assert_array_equal(first, second, strict=True)


def test_load_sequence_custom_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert cache_dir stores .npy cache outside read-only input directory."""
    input_dir = tmp_path / "readonly_input"
    input_dir.mkdir()
    cache_dir = tmp_path / "cache"
    parquet_path = input_dir / "clip_123.parquet"
    _write_gislr_parquet(parquet_path, n_frames=3)

    first = load_sequence(parquet_path, cache_dir=cache_dir)
    assert not parquet_path.with_suffix(".npy").exists()
    assert len(list(cache_dir.glob("*.npy"))) == 1

    monkeypatch.setattr(pq, "read_table", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError()))
    second = load_sequence(parquet_path, cache_dir=cache_dir)
    np.testing.assert_array_equal(first, second, strict=True)


def test_load_sequence_canonicalize_flag(tmp_path: Path) -> None:
    """Assert canonicalize=True mirrors left-dominant sequences to right-dominant orientation."""
    parquet_path = tmp_path / "left_dom.parquet"
    _write_gislr_parquet(parquet_path, n_frames=6, left_active=True)

    raw_seq = load_sequence(parquet_path, canonicalize=False)
    assert estimate_dominant_hand(raw_seq) == "left"

    canon_seq = load_sequence(parquet_path, canonicalize=True)
    expected_canon = canonicalize_handedness(raw_seq, is_left_dominant=True)
    np.testing.assert_allclose(canon_seq, expected_canon, atol=1e-5)

    # Confirm right-hand slot now carries the canonicalized left-hand landmarks (mirrored x)
    np.testing.assert_allclose(
        canon_seq[:, RIGHT_HAND_START:RIGHT_HAND_END : 3],
        -raw_seq[:, LEFT_HAND_START:LEFT_HAND_END : 3],
        atol=1e-5,
    )

    # Cache must remain uncanonicalized when querying canonicalize=False afterwards
    raw_again = load_sequence(parquet_path, canonicalize=False)
    np.testing.assert_allclose(raw_again, raw_seq, atol=1e-5)


def test_load_sequence_rejects_bad_first_frame_row_order(tmp_path: Path) -> None:
    """Assert first-frame row-order guard raises ValueError when GISLR layout differs."""
    parquet_path = tmp_path / "bad_order.parquet"
    _write_gislr_parquet(parquet_path, n_frames=2)

    df = pq.read_table(str(parquet_path)).to_pandas()
    # Swap row 0 (face) with row 542 (right_hand) in the first frame
    first_row = df.iloc[0].copy()
    last_first_frame_row = df.iloc[542].copy()
    df.iloc[0] = last_first_frame_row
    df.iloc[542] = first_row
    pq.write_table(pa.Table.from_pandas(df), str(parquet_path))

    with pytest.raises(ValueError, match="first-frame row order"):
        load_sequence(parquet_path, use_cache=False)
