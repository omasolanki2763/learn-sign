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
        "aspect_ratio_scaling",
    ],
)
def test_normalize_frame_against_golden(golden_cases: dict, case_key: str) -> None:
    """Assert normalize_frame produces exact expected output matching the golden file."""
    case = golden_cases[case_key]
    left_hand = _to_numpy_array(case["left_hand"])
    right_hand = _to_numpy_array(case["right_hand"])
    pose = _to_numpy_array(case["pose"])
    aspect_ratio = float(case.get("aspect_ratio", 1.0))
    expected = _to_numpy_array(case["expected_frame"])

    actual = normalize_frame(left_hand, right_hand, pose, aspect_ratio=aspect_ratio)

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


def test_mirroring_swaps_bilateral_pose_pairs() -> None:
    """Blocker 1 test: Assert canonicalize_handedness swaps pose pairs (11,12) (13,14) (15,16)."""
    # Create single frame with distinct values in pose slots
    frame = np.zeros((1, FEATURES_PER_FRAME), dtype=np.float32)
    # Pose: slot 0 (left shoulder: 126..128) = [1.0, 2.0, 3.0]
    #       slot 1 (right shoulder: 129..131) = [4.0, 5.0, 6.0]
    #       slot 2 (left elbow: 132..134) = [7.0, 8.0, 9.0]
    #       slot 3 (right elbow: 135..137) = [10.0, 11.0, 12.0]
    #       slot 4 (left wrist: 138..140) = [13.0, 14.0, 15.0]
    #       slot 5 (right wrist: 141..143) = [16.0, 17.0, 18.0]
    frame[0, 126:129] = [1.0, 2.0, 3.0]
    frame[0, 129:132] = [4.0, 5.0, 6.0]
    frame[0, 132:135] = [7.0, 8.0, 9.0]
    frame[0, 135:138] = [10.0, 11.0, 12.0]
    frame[0, 138:141] = [13.0, 14.0, 15.0]
    frame[0, 141:144] = [16.0, 17.0, 18.0]

    canon = canonicalize_handedness(frame, is_left_dominant=True)

    # In canonicalized frame:
    # Slot 0 should have mirrored previous slot 1: [-4.0, 5.0, 6.0]
    # Slot 1 should have mirrored previous slot 0: [-1.0, 2.0, 3.0]
    # Slot 2 should have mirrored previous slot 3: [-10.0, 11.0, 12.0]
    # Slot 3 should have mirrored previous slot 2: [-7.0, 8.0, 9.0]
    # Slot 4 should have mirrored previous slot 5: [-16.0, 17.0, 18.0]
    # Slot 5 should have mirrored previous slot 4: [-13.0, 14.0, 15.0]
    np.testing.assert_allclose(canon[0, 126:129], [-4.0, 5.0, 6.0], atol=1e-5)
    np.testing.assert_allclose(canon[0, 129:132], [-1.0, 2.0, 3.0], atol=1e-5)
    np.testing.assert_allclose(canon[0, 132:135], [-10.0, 11.0, 12.0], atol=1e-5)
    np.testing.assert_allclose(canon[0, 135:138], [-7.0, 8.0, 9.0], atol=1e-5)
    np.testing.assert_allclose(canon[0, 138:141], [-16.0, 17.0, 18.0], atol=1e-5)
    np.testing.assert_allclose(canon[0, 141:144], [-13.0, 14.0, 15.0], atol=1e-5)


def test_z_divided_by_scale_not_shifted() -> None:
    """Blocker 2 test: Assert z coordinates are divided by 2D shoulder width but NOT shifted."""
    pose = np.full((33, 3), np.nan, dtype=np.float32)
    # 2D shoulder distance: ls_x=0.6, rs_x=0.2, ls_y=0.4, rs_y=0.4 => 2D width = 0.4
    # Set shoulders with non-zero z values (e.g. ls_z = 0.8, rs_z = 0.4 => midpoint would be 0.6)
    pose[11] = [0.6, 0.4, 0.8]
    pose[12] = [0.2, 0.4, 0.4]

    # Hand point with z = 0.2
    lh = np.zeros((21, 3), dtype=np.float32)
    lh[0] = [0.6, 0.4, 0.2]

    out = normalize_frame(lh, None, pose)
    # Expected z for hand point 0: 0.2 / 0.4 = 0.5 (NOT (0.2 - 0.6) / 0.4 = -1.0)
    assert np.isclose(out[2], 0.2 / 0.4, atol=1e-5)
    # Expected z for left shoulder: 0.8 / 0.4 = 2.0 (NOT (0.8 - 0.6) / 0.4 = 0.5)
    assert np.isclose(out[126 + 2], 0.8 / 0.4, atol=1e-5)


def test_right_dominant_sequence_unchanged() -> None:
    """Assert right-dominant input passes through unchanged by canonicalize_handedness."""
    seq = np.zeros((3, FEATURES_PER_FRAME), dtype=np.float32)
    seq[1, 63:126] = 0.5
    seq[2, 63:126] = 1.0

    detected_dom = estimate_dominant_hand(seq)
    assert detected_dom == "right"

    result = canonicalize_handedness(seq, is_left_dominant=False)
    np.testing.assert_array_equal(result, seq)


def test_from_gislr_frame() -> None:
    """Assert GISLR 543-row layout adapter extracts correct hand and pose arrays."""
    rows = []
    for i in range(468):
        rows.append({"type": "face", "landmark_index": i, "x": 0.1, "y": 0.2, "z": 0.3})
    for i in range(21):
        rows.append(
            {"type": "left_hand", "landmark_index": i, "x": 0.6 + i * 0.01, "y": 0.8, "z": 0.2}
        )
    for i in range(33):
        rows.append({"type": "pose", "landmark_index": i, "x": 0.5, "y": 0.4, "z": 0.1})
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
