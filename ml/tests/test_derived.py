"""Unit tests for v2 derived geometric features shared across Python and TypeScript."""

import json
from pathlib import Path

import numpy as np
import pytest

from learnsign.features.derived import (
    DERIVED_BEND_START,
    DERIVED_FEATURES_PER_FRAME,
    DERIVED_HAND_DIST_START,
    DERIVED_PRIMARY_COUNT,
    DERIVED_SPREAD_START,
    DERIVED_TIP_DIST_START,
    DERIVED_VELOCITY_START,
    DERIVED_WRIST_ANCHOR_START,
    DERIVED_Z_COUNT,
    DERIVED_Z_START,
    FINGER_BEND_JOINTS_PER_HAND,
    derived_features,
    derived_sequence,
)
from learnsign.features.schema import FEATURES_PER_FRAME


@pytest.fixture
def derived_golden() -> dict:
    """Load golden derived feature cases shared between Python and TypeScript."""
    path = Path(__file__).resolve().parents[2] / "tests" / "golden" / "derived_cases.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _to_float32_nan(values: list[float | None]) -> np.ndarray:
    """Convert JSON numeric list containing None into float32 NumPy array with NaN."""
    return np.array(
        [np.nan if v is None else float(v) for v in values],
        dtype=np.float32,
    )


def test_derived_schema_counts_consistent() -> None:
    """Verify derived feature partition constants sum cleanly without gaps."""
    assert FINGER_BEND_JOINTS_PER_HAND == 15
    assert DERIVED_BEND_START == 0
    assert DERIVED_Z_START == DERIVED_PRIMARY_COUNT
    assert DERIVED_Z_START + DERIVED_Z_COUNT == DERIVED_FEATURES_PER_FRAME
    assert DERIVED_TIP_DIST_START > DERIVED_BEND_START
    assert DERIVED_SPREAD_START > DERIVED_TIP_DIST_START
    assert DERIVED_WRIST_ANCHOR_START > DERIVED_SPREAD_START
    assert DERIVED_VELOCITY_START > DERIVED_WRIST_ANCHOR_START
    assert DERIVED_HAND_DIST_START > DERIVED_VELOCITY_START


@pytest.mark.parametrize(
    "case_key",
    ["straight_hand_at_nose", "fist_with_velocity", "missing_hands"],
)
def test_derived_features_against_golden(derived_golden: dict, case_key: str) -> None:
    """Assert derived_features matches shared golden outputs within 1e-5."""
    case = derived_golden[case_key]
    frame = _to_float32_nan(case["frame"])
    prev = _to_float32_nan(case["prev_frame"]) if case["prev_frame"] is not None else None
    expected = _to_float32_nan(case["expected"])

    actual = derived_features(frame, prev_frame=prev)

    assert actual.shape == (DERIVED_FEATURES_PER_FRAME,)
    assert actual.dtype == np.float32
    assert np.array_equal(np.isnan(actual), np.isnan(expected))

    valid = ~np.isnan(expected)
    if np.any(valid):
        np.testing.assert_allclose(actual[valid], expected[valid], atol=1e-5)


def test_derived_semantic_properties(derived_golden: dict) -> None:
    """Verify key Task 7 semantic invariants on derived features."""
    straight_case = derived_golden["straight_hand_at_nose"]
    straight_feat = derived_features(_to_float32_nan(straight_case["frame"]))

    # Straight left hand has zero bend across all 15 finger joints
    left_bends = straight_feat[
        DERIVED_BEND_START : DERIVED_BEND_START + FINGER_BEND_JOINTS_PER_HAND
    ]
    np.testing.assert_allclose(left_bends, 0.0, atol=1e-5)

    # Curled right fist has large bends (> 1.0 rad) across all 15 joints
    right_bends = straight_feat[
        DERIVED_BEND_START + FINGER_BEND_JOINTS_PER_HAND : DERIVED_TIP_DIST_START
    ]
    assert np.all(right_bends > 1.0)

    # Left wrist placed at nose -> wrist-to-nose distance == 0
    assert np.isclose(straight_feat[DERIVED_WRIST_ANCHOR_START], 0.0, atol=1e-5)


def test_derived_sequence_shape_and_first_frame_velocity(derived_golden: dict) -> None:
    """Assert derived_sequence handles frame 0 prev=None and frame 1 velocity correctly."""
    vel_case = derived_golden["fist_with_velocity"]
    seq = np.stack(
        [
            _to_float32_nan(vel_case["prev_frame"]),
            _to_float32_nan(vel_case["frame"]),
        ],
        axis=0,
    )
    assert seq.shape == (2, FEATURES_PER_FRAME)

    out = derived_sequence(seq)
    assert out.shape == (2, DERIVED_FEATURES_PER_FRAME)
    assert out.dtype == np.float32
    assert np.isnan(out[0, DERIVED_VELOCITY_START : DERIVED_VELOCITY_START + 6]).all()
    np.testing.assert_allclose(
        out[1, DERIVED_VELOCITY_START + 3 : DERIVED_VELOCITY_START + 6],
        [0.5, 0.6, -0.8],
        atol=1e-5,
    )
