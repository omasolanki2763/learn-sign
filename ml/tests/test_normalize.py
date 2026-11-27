"""Unit tests for v2 feature normalization, schema, and handedness canonicalization."""

import json
from pathlib import Path

import numpy as np
import pytest

from learnsign.features.normalize import (
    canonicalize_handedness,
    estimate_dominant_hand,
    from_gislr_frame,
    normalize_frame,
    normalize_frame_with_anchor,
    normalize_sequence_with_anchors,
)
from learnsign.features.schema import (
    COORDINATES_PER_POINT,
    EYE_TO_SHOULDER_RATIO,
    EYE_TO_SHOULDER_Y_OFFSET,
    FEATURE_CONTRACT_VERSION,
    FEATURES_PER_FRAME,
    HAND_POINTS,
    LEFT_HAND_END,
    LEFT_HAND_START,
    POSE_BILATERAL_PAIRS,
    POSE_IDS,
    POSE_POINTS,
    POSE_SLOT_LEFT_EYE,
    POSE_SLOT_LEFT_SHOULDER,
    POSE_SLOT_LEFT_WRIST,
    POSE_SLOT_NOSE,
    POSE_SLOT_RIGHT_EYE,
    POSE_SLOT_RIGHT_SHOULDER,
    POSE_SLOT_RIGHT_WRIST,
    POSE_START,
    RIGHT_HAND_END,
    RIGHT_HAND_START,
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
    """Verify frozen contract v2 landmark dimensions and feature count contract."""
    assert FEATURE_CONTRACT_VERSION == 2
    assert HAND_POINTS == 21
    assert POSE_IDS == list(range(17))
    assert POSE_POINTS == len(POSE_IDS) == 17
    assert FEATURES_PER_FRAME == (HAND_POINTS * 2 + POSE_POINTS) * COORDINATES_PER_POINT
    assert EYE_TO_SHOULDER_RATIO > 1.0
    assert EYE_TO_SHOULDER_Y_OFFSET > 0.0


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
        "head_anchors",
        "missing_shoulders_eye_fallback",
        "same_frame_with_shoulder",
        "same_frame_without_shoulder",
    ],
)
def test_normalize_frame_against_golden(golden_cases: dict, case_key: str) -> None:
    """Assert normalize_frame and normalize_frame_with_anchor match the golden file."""
    case = golden_cases[case_key]
    left_hand = _to_numpy_array(case["left_hand"])
    right_hand = _to_numpy_array(case["right_hand"])
    pose = _to_numpy_array(case["pose"])
    aspect_ratio = float(case.get("aspect_ratio", 1.0))
    expected = _to_numpy_array(case["expected_frame"])
    expected_source = case.get("expected_anchor_source")

    actual = normalize_frame(left_hand, right_hand, pose, aspect_ratio=aspect_ratio)
    actual_with_src, anchor_source = normalize_frame_with_anchor(
        left_hand, right_hand, pose, aspect_ratio=aspect_ratio
    )

    assert anchor_source == expected_source
    assert isinstance(actual, np.ndarray)
    assert actual.shape == (FEATURES_PER_FRAME,)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(actual_with_src))
    valid_both = ~np.isnan(actual)
    np.testing.assert_allclose(actual[valid_both], actual_with_src[valid_both], atol=1e-7)

    actual_nan = np.isnan(actual)
    expected_nan = np.isnan(expected)
    assert np.array_equal(actual_nan, expected_nan), f"NaN mismatch in {case_key}"

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


def test_mirroring_swaps_bilateral_head_and_torso_pairs() -> None:
    """Assert canonicalize_handedness swaps all bilateral head/torso pairs and keeps nose center."""
    frame = np.zeros((1, FEATURES_PER_FRAME), dtype=np.float32)

    nose_offset = POSE_START + POSE_SLOT_NOSE * COORDINATES_PER_POINT
    frame[0, nose_offset : nose_offset + 3] = [0.5, -0.5, 0.1]

    for left_slot, right_slot in POSE_BILATERAL_PAIRS:
        l_off = POSE_START + left_slot * COORDINATES_PER_POINT
        r_off = POSE_START + right_slot * COORDINATES_PER_POINT
        frame[0, l_off : l_off + 3] = [float(left_slot + 1), 0.1, 0.2]
        frame[0, r_off : r_off + 3] = [float(right_slot + 1), -0.1, -0.2]

    canon = canonicalize_handedness(frame, is_left_dominant=True)

    # Nose stays in midline slot with mirrored x
    np.testing.assert_allclose(
        canon[0, nose_offset : nose_offset + 3], [-0.5, -0.5, 0.1], atol=1e-5
    )

    # Bilateral pairs swap and mirror x -> -x
    for left_slot, right_slot in POSE_BILATERAL_PAIRS:
        l_off = POSE_START + left_slot * COORDINATES_PER_POINT
        r_off = POSE_START + right_slot * COORDINATES_PER_POINT
        np.testing.assert_allclose(
            canon[0, l_off : l_off + 3], [-float(right_slot + 1), -0.1, -0.2], atol=1e-5
        )
        np.testing.assert_allclose(
            canon[0, r_off : r_off + 3], [-float(left_slot + 1), 0.1, 0.2], atol=1e-5
        )


def test_z_divided_by_scale_not_shifted() -> None:
    """Assert z coordinates are divided by 2D scale without shifting."""
    pose = np.full((33, 3), np.nan, dtype=np.float32)
    pose[POSE_SLOT_LEFT_SHOULDER] = [0.6, 0.4, 0.8]
    pose[POSE_SLOT_RIGHT_SHOULDER] = [0.2, 0.4, 0.4]

    lh = np.zeros((HAND_POINTS, COORDINATES_PER_POINT), dtype=np.float32)
    lh[0] = [0.6, 0.4, 0.2]

    out = normalize_frame(lh, None, pose)
    assert np.isclose(out[2], 0.2 / 0.4, atol=1e-5)
    ls_z_offset = POSE_START + POSE_SLOT_LEFT_SHOULDER * COORDINATES_PER_POINT + 2
    assert np.isclose(out[ls_z_offset], 0.8 / 0.4, atol=1e-5)


def test_eye_fallback_when_shoulders_missing() -> None:
    """Assert missing shoulder shifts eye midpoint to shoulder midpoint and keeps scale."""
    pose = np.full((33, 3), np.nan, dtype=np.float32)
    pose[POSE_SLOT_LEFT_EYE] = [0.55, 0.25, 0.0]
    pose[POSE_SLOT_RIGHT_EYE] = [0.45, 0.25, 0.0]  # eye dist = 0.1, midpoint = (0.5, 0.25)
    pose[POSE_SLOT_RIGHT_WRIST] = [0.5, 0.55, 0.0]

    rh = np.zeros((HAND_POINTS, COORDINATES_PER_POINT), dtype=np.float32)
    rh[0] = [0.5, 0.55, 0.0]

    out, src = normalize_frame_with_anchor(None, rh, pose)
    expected_scale = 0.1 * EYE_TO_SHOULDER_RATIO
    expected_mid_y = 0.25 + expected_scale * EYE_TO_SHOULDER_Y_OFFSET
    assert src == "eyes"
    assert np.isclose(out[RIGHT_HAND_START], 0.0, atol=1e-5)
    assert np.isclose(
        out[RIGHT_HAND_START + 1], (0.55 - expected_mid_y) / expected_scale, atol=1e-5
    )
    rw_offset = POSE_START + POSE_SLOT_RIGHT_WRIST * COORDINATES_PER_POINT
    assert np.isclose(out[rw_offset + 1], (0.55 - expected_mid_y) / expected_scale, atol=1e-5)
    lw_offset = POSE_START + POSE_SLOT_LEFT_WRIST * COORDINATES_PER_POINT
    assert np.isnan(out[lw_offset])


def test_same_frame_with_vs_without_shoulder_consistency(golden_cases: dict) -> None:
    """Assert dropping one shoulder yields identical normalized hands and switches anchor_source."""
    with_sh = golden_cases["same_frame_with_shoulder"]
    without_sh = golden_cases["same_frame_without_shoulder"]

    feat_with, src_with = normalize_frame_with_anchor(
        _to_numpy_array(with_sh["left_hand"]),
        _to_numpy_array(with_sh["right_hand"]),
        _to_numpy_array(with_sh["pose"]),
    )
    feat_without, src_without = normalize_frame_with_anchor(
        _to_numpy_array(without_sh["left_hand"]),
        _to_numpy_array(without_sh["right_hand"]),
        _to_numpy_array(without_sh["pose"]),
    )

    assert src_with == "shoulders"
    assert src_without == "eyes"
    np.testing.assert_allclose(
        feat_with[LEFT_HAND_START:LEFT_HAND_END],
        feat_without[LEFT_HAND_START:LEFT_HAND_END],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        feat_with[RIGHT_HAND_START:RIGHT_HAND_END],
        feat_without[RIGHT_HAND_START:RIGHT_HAND_END],
        atol=1e-5,
    )


def test_normalize_sequence_with_anchors(golden_cases: dict) -> None:
    """Assert normalize_sequence_with_anchors returns per-frame features and anchor sources."""
    c1 = golden_cases["same_frame_with_shoulder"]
    c2 = golden_cases["same_frame_without_shoulder"]
    lh = np.stack([_to_numpy_array(c1["left_hand"]), _to_numpy_array(c2["left_hand"])], axis=0)
    rh = np.stack([_to_numpy_array(c1["right_hand"]), _to_numpy_array(c2["right_hand"])], axis=0)
    poses = np.stack([_to_numpy_array(c1["pose"]), _to_numpy_array(c2["pose"])], axis=0)

    seq, sources = normalize_sequence_with_anchors(lh, rh, poses)
    assert seq.shape == (2, FEATURES_PER_FRAME)
    assert sources == ["shoulders", "eyes"]
    np.testing.assert_allclose(
        seq[0, LEFT_HAND_START:RIGHT_HAND_END],
        seq[1, LEFT_HAND_START:RIGHT_HAND_END],
        atol=1e-5,
    )


def test_right_dominant_sequence_unchanged() -> None:
    """Assert right-dominant input passes through unchanged by canonicalize_handedness."""
    seq = np.zeros((3, FEATURES_PER_FRAME), dtype=np.float32)
    seq[1, RIGHT_HAND_START:RIGHT_HAND_END] = 0.5
    seq[2, RIGHT_HAND_START:RIGHT_HAND_END] = 1.0

    detected_dom = estimate_dominant_hand(seq)
    assert detected_dom == "right"

    result = canonicalize_handedness(seq, is_left_dominant=False)
    np.testing.assert_array_equal(result, seq)


def test_from_gislr_frame() -> None:
    """Assert GISLR 543-row layout adapter extracts correct hand and pose arrays."""
    rows = []
    for i in range(468):
        rows.append({"type": "face", "landmark_index": i, "x": 0.1, "y": 0.2, "z": 0.3})
    for i in range(HAND_POINTS):
        rows.append(
            {"type": "left_hand", "landmark_index": i, "x": 0.6 + i * 0.01, "y": 0.8, "z": 0.2}
        )
    for i in range(33):
        rows.append({"type": "pose", "landmark_index": i, "x": 0.5, "y": 0.4, "z": 0.1})
    for i in range(HAND_POINTS):
        rows.append(
            {"type": "right_hand", "landmark_index": i, "x": 0.2 - i * 0.01, "y": 0.8, "z": -0.2}
        )

    lh, rh, pose = from_gislr_frame(rows)

    assert lh is not None and lh.shape == (HAND_POINTS, COORDINATES_PER_POINT)
    assert rh is not None and rh.shape == (HAND_POINTS, COORDINATES_PER_POINT)
    assert pose is not None and pose.shape == (33, COORDINATES_PER_POINT)
    np.testing.assert_allclose(lh[0], [0.6, 0.8, 0.2], atol=1e-5)
    np.testing.assert_allclose(rh[0], [0.2, 0.8, -0.2], atol=1e-5)
    np.testing.assert_allclose(pose[POSE_SLOT_LEFT_SHOULDER], [0.5, 0.4, 0.1], atol=1e-5)
