"""Unit tests for feature normalization, schema, and handedness canonicalization."""

import json
from pathlib import Path

import numpy as np
import pytest

from learnsign.features.normalize import (
    canonicalize_handedness,
    estimate_dominant_hand,
    from_gislr_frame,
    normalize_frame,
)
from learnsign.features.schema import (
    FEATURES_PER_FRAME,
    HAND_POINTS,
    POSE_IDS,
)


@pytest.fixture
def golden_cases() -> dict:
    """Load the golden normalization test cases shared across Python and TypeScript."""
    golden_path = Path(__file__).resolve().parents[2] / "tests" / "golden" / "normalize_cases.json"
    with open(golden_path, encoding="utf-8") as f:
        return json.load(f)


def _to_numpy_array(data: list | None) -> np.ndarray | None:
    """Helper to convert JSON lists with None to float32 NumPy arrays with NaN."""
    if data is None:
        return None
    arr = np.array(
        [np.nan if x is None else float(x) for x in np.asarray(data).flatten()],
        dtype=np.float32,
    )
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        # Preserve 2D shape if original was nested list
        n_rows = len(data)
        n_cols = len(data[0])
        return arr.reshape((n_rows, n_cols))
    return arr


def test_schema_constants() -> None:
    """Verify core landmark dimensions and feature count contract."""
    assert HAND_POINTS == 21
    assert POSE_IDS == [11, 12, 13, 14, 15, 16]
    assert FEATURES_PER_FRAME == 144


@pytest.mark.parametrize(
    "case_key",
    [
        "both_hands",
        "left_only",
        "right_only",
        "no_hands",
        "no_pose",
        "shoulders_swapped",
        "zero_shoulder_width",
    ],
)
def test_normalize_frame_against_golden(golden_cases: dict, case_key: str) -> None:
    """Assert normalize_frame produces exact expected output matching the golden file."""
    case = golden_cases[case_key]
    left_hand = _to_numpy_array(case["left_hand"])
    right_hand = _to_numpy_array(case["right_hand"])
    pose = _to_numpy_array(case["pose"])
    expected = _to_numpy_array(case["expected_frame"])

    actual = normalize_frame(left_hand, right_hand, pose)

    assert isinstance(actual, np.ndarray)
    assert actual.shape == (FEATURES_PER_FRAME,)
    assert actual.dtype == np.float32

    # Verify identical NaN mask
    actual_nan = np.isnan(actual)
    expected_nan = np.isnan(expected)
    assert np.array_equal(actual_nan, expected_nan), f"NaN mismatch in {case_key}"

    # Verify non-NaN values match within atol=1e-5
    valid_mask = ~actual_nan
    if np.any(valid_mask):
        np.testing.assert_allclose(actual[valid_mask], expected[valid_mask], atol=1e-5)


def test_left_dominant_sequence_and_canonicalization(golden_cases: dict) -> None:
    """Assert left-dominant signer detection and handedness mirroring contract."""
    case = golden_cases["left_dominant"]
    seq_raw = case["sequence"]
    seq = np.array(
        [[np.nan if x is None else float(x) for x in frame] for frame in seq_raw],
        dtype=np.float32,
    )
    expected_dominant = case["expected_dominant"]
    expected_canon_raw = case["expected_canonicalized"]
    expected_canon = np.array(
        [[np.nan if x is None else float(x) for x in frame] for frame in expected_canon_raw],
        dtype=np.float32,
    )

    detected_dom = estimate_dominant_hand(seq)
    assert detected_dom == expected_dominant == "left"

    actual_canon = canonicalize_handedness(seq, is_left_dominant=True)
    assert actual_canon.shape == seq.shape
    assert np.array_equal(np.isnan(actual_canon), np.isnan(expected_canon))

    valid_mask = ~np.isnan(actual_canon)
    np.testing.assert_allclose(actual_canon[valid_mask], expected_canon[valid_mask], atol=1e-5)


def test_right_dominant_sequence_unchanged() -> None:
    """Assert right-dominant input passes through unchanged by canonicalize_handedness."""
    # Create sequence where right hand moves more
    seq = np.zeros((3, FEATURES_PER_FRAME), dtype=np.float32)
    # Right hand moves (indices 63..125)
    seq[1, 63:126] = 0.5
    seq[2, 63:126] = 1.0

    detected_dom = estimate_dominant_hand(seq)
    assert detected_dom == "right"

    # Canonicalizing right-dominant with is_left_dominant=False must return unchanged array
    result = canonicalize_handedness(seq, is_left_dominant=False)
    np.testing.assert_array_equal(result, seq)


def test_from_gislr_frame() -> None:
    """Assert GISLR 543-row layout adapter extracts correct hand and pose arrays."""
    # Build synthetic 543-row dataset frame
    # GISLR standard row layout: face 0..467, left_hand 0..20, pose 0..32, right_hand 0..20
    rows = []
    # 468 face rows
    for i in range(468):
        rows.append({"type": "face", "landmark_index": i, "x": 0.1, "y": 0.2, "z": 0.3})
    # 21 left_hand rows
    for i in range(21):
        rows.append(
            {"type": "left_hand", "landmark_index": i, "x": 0.6 + i * 0.01, "y": 0.8, "z": 0.2}
        )
    # 33 pose rows
    for i in range(33):
        rows.append({"type": "pose", "landmark_index": i, "x": 0.5, "y": 0.4, "z": 0.1})
    # 21 right_hand rows
    for i in range(21):
        rows.append(
            {"type": "right_hand", "landmark_index": i, "x": 0.2 - i * 0.01, "y": 0.8, "z": -0.2}
        )

    lh, rh, pose = from_gislr_frame(rows)

    assert lh is not None and lh.shape == (21, 3)
    assert rh is not None and rh.shape == (21, 3)
    assert pose is not None and pose.shape == (33, 3)
    np.testing.assert_allclose(lh[0], [0.6, 0.8, 0.2], atol=1e-5)
    np.testing.assert_allclose(rh[0], [0.2, 0.8, -0.2], atol=1e-5)
    np.testing.assert_allclose(pose[11], [0.5, 0.4, 0.1], atol=1e-5)
