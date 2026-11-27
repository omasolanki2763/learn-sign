"""Interpretable v2 geometric derived features shared with browser inference."""

import math
from collections.abc import Sequence
from typing import Final

import numpy as np

from learnsign.features.schema import (
    COORDINATES_PER_POINT,
    HAND_POINTS,
    LEFT_HAND_END,
    LEFT_HAND_START,
    POSE_END,
    POSE_POINTS,
    POSE_SLOT_LEFT_MOUTH,
    POSE_SLOT_LEFT_SHOULDER,
    POSE_SLOT_NOSE,
    POSE_SLOT_RIGHT_MOUTH,
    POSE_SLOT_RIGHT_SHOULDER,
    POSE_START,
    RIGHT_HAND_END,
    RIGHT_HAND_START,
)

# Minimum 2D bone/segment magnitude guard before computing normalized directions or angles.
MIN_SEGMENT_NORM: Final[float] = 1e-6

# 15 interior finger joint triplets (proximal_a, vertex_b, distal_c) per hand -> angle in [0, pi].
FINGER_BEND_TRIPLETS: Final[tuple[tuple[int, int, int], ...]] = (
    (0, 1, 2),
    (1, 2, 3),
    (2, 3, 4),
    (0, 5, 6),
    (5, 6, 7),
    (6, 7, 8),
    (0, 9, 10),
    (9, 10, 11),
    (10, 11, 12),
    (0, 13, 14),
    (13, 14, 15),
    (14, 15, 16),
    (0, 17, 18),
    (17, 18, 19),
    (18, 19, 20),
)
FINGER_BEND_JOINTS_PER_HAND: Final[int] = len(FINGER_BEND_TRIPLETS)

# All 10 unique fingertip pairs per hand (thumb=4, index=8, middle=12, ring=16, pinky=20).
FINGERTIP_IDS: Final[tuple[int, ...]] = (4, 8, 12, 16, 20)
FINGERTIP_PAIRS: Final[tuple[tuple[int, int], ...]] = tuple(
    (FINGERTIP_IDS[i], FINGERTIP_IDS[j])
    for i in range(len(FINGERTIP_IDS))
    for j in range(i + 1, len(FINGERTIP_IDS))
)
FINGERTIP_DISTS_PER_HAND: Final[int] = len(FINGERTIP_PAIRS)

# 4 adjacent finger ray pairs per hand measuring angular spread between adjacent fingers.
FINGER_SPREAD_RAY_PAIRS: Final[tuple[tuple[tuple[int, int], tuple[int, int]], ...]] = (
    ((0, 4), (5, 8)),
    ((5, 8), (9, 12)),
    ((9, 12), (13, 16)),
    ((13, 16), (17, 20)),
)
FINGER_SPREADS_PER_HAND: Final[int] = len(FINGER_SPREAD_RAY_PAIRS)

WRIST_ANCHORS_PER_HAND: Final[int] = 3  # nose, mouth_centre, chest_centre
VELOCITY_CHANNELS_PER_HAND: Final[int] = 3  # speed, dir_x, dir_y
TWO_HAND_DIST_COUNT: Final[int] = 1
DERIVED_Z_COUNT: Final[int] = 3  # left_wrist_dz, right_wrist_dz, two_hand_dz

# Frozen derived layout slice offsets (primary 2D geometry first, untrusted z last).
DERIVED_BEND_START: Final[int] = 0
DERIVED_TIP_DIST_START: Final[int] = DERIVED_BEND_START + FINGER_BEND_JOINTS_PER_HAND * 2
DERIVED_SPREAD_START: Final[int] = DERIVED_TIP_DIST_START + FINGERTIP_DISTS_PER_HAND * 2
DERIVED_WRIST_ANCHOR_START: Final[int] = DERIVED_SPREAD_START + FINGER_SPREADS_PER_HAND * 2
DERIVED_VELOCITY_START: Final[int] = DERIVED_WRIST_ANCHOR_START + WRIST_ANCHORS_PER_HAND * 2
DERIVED_HAND_DIST_START: Final[int] = DERIVED_VELOCITY_START + VELOCITY_CHANNELS_PER_HAND * 2
DERIVED_PRIMARY_COUNT: Final[int] = DERIVED_HAND_DIST_START + TWO_HAND_DIST_COUNT
DERIVED_Z_START: Final[int] = DERIVED_PRIMARY_COUNT
DERIVED_FEATURES_PER_FRAME: Final[int] = DERIVED_Z_START + DERIVED_Z_COUNT


def _valid_2d(pts: np.ndarray, idx: int) -> bool:
    return not (np.isnan(pts[idx, 0]) or np.isnan(pts[idx, 1]))


def _valid_3d(pts: np.ndarray, idx: int) -> bool:
    return not np.any(np.isnan(pts[idx]))


def _angle_2d(ux: float, uy: float, vx: float, vy: float) -> float:
    nu = math.hypot(ux, uy)
    nv = math.hypot(vx, vy)
    if nu <= MIN_SEGMENT_NORM or nv <= MIN_SEGMENT_NORM:
        return math.nan
    cos_val = max(-1.0, min(1.0, (ux * vx + uy * vy) / (nu * nv)))
    return math.acos(cos_val)


def _midpoint_2d(pts: np.ndarray, idx_a: int, idx_b: int) -> tuple[float, float] | None:
    if not (_valid_2d(pts, idx_a) and _valid_2d(pts, idx_b)):
        return None
    return (
        float(pts[idx_a, 0] + pts[idx_b, 0]) / 2.0,
        float(pts[idx_a, 1] + pts[idx_b, 1]) / 2.0,
    )


def derived_features(
    frame: np.ndarray,
    prev_frame: np.ndarray | None = None,
    dt_s: float | None = None,
) -> np.ndarray:
    """Compute interpretable 2D-primary derived features from a normalized v2 frame.

    Layout (`DERIVED_FEATURES_PER_FRAME`):
    - [DERIVED_BEND_START .. ): 15 finger bend angles (rad) per hand (left, right)
    - [DERIVED_TIP_DIST_START .. ): 10 fingertip-to-fingertip 2D distances per hand
    - [DERIVED_SPREAD_START .. ): 4 adjacent finger ray spread angles (rad) per hand
    - [DERIVED_WRIST_ANCHOR_START .. ): wrist 2D distance to nose / mouth / chest per hand
    - [DERIVED_VELOCITY_START .. ): wrist 2D (speed / s, dir_x, dir_y) from `prev_frame` per hand
    - [DERIVED_HAND_DIST_START]: 2D wrist-to-wrist distance
    - [DERIVED_Z_START .. ]: separate untrusted z channels (left_dz / s, right_dz / s, two_hand_dz)

    Args:
        frame: Normalized frame array of shape (FEATURES_PER_FRAME,).
        prev_frame: Optional previous normalized frame for velocity/direction channels.
        dt_s: Elapsed seconds since `prev_frame`. Velocity and temporal wrist `dz`
            channels remain NaN unless `dt_s` is finite and strictly positive.

    Returns:
        float32 array of shape (DERIVED_FEATURES_PER_FRAME,).
    """
    arr = np.asarray(frame, dtype=np.float32)
    lh = arr[LEFT_HAND_START:LEFT_HAND_END].reshape((HAND_POINTS, COORDINATES_PER_POINT))
    rh = arr[RIGHT_HAND_START:RIGHT_HAND_END].reshape((HAND_POINTS, COORDINATES_PER_POINT))
    pose = arr[POSE_START:POSE_END].reshape((POSE_POINTS, COORDINATES_PER_POINT))

    out = np.full(DERIVED_FEATURES_PER_FRAME, np.nan, dtype=np.float32)
    idx = 0

    for hand in (lh, rh):
        for a, b, c in FINGER_BEND_TRIPLETS:
            if _valid_2d(hand, a) and _valid_2d(hand, b) and _valid_2d(hand, c):
                out[idx] = _angle_2d(
                    float(hand[b, 0] - hand[a, 0]),
                    float(hand[b, 1] - hand[a, 1]),
                    float(hand[c, 0] - hand[b, 0]),
                    float(hand[c, 1] - hand[b, 1]),
                )
            idx += 1

    for hand in (lh, rh):
        for p_a, p_b in FINGERTIP_PAIRS:
            if _valid_2d(hand, p_a) and _valid_2d(hand, p_b):
                out[idx] = math.hypot(
                    float(hand[p_a, 0] - hand[p_b, 0]),
                    float(hand[p_a, 1] - hand[p_b, 1]),
                )
            idx += 1

    for hand in (lh, rh):
        for (p1, p2), (p3, p4) in FINGER_SPREAD_RAY_PAIRS:
            if (
                _valid_2d(hand, p1)
                and _valid_2d(hand, p2)
                and _valid_2d(hand, p3)
                and _valid_2d(hand, p4)
            ):
                out[idx] = _angle_2d(
                    float(hand[p2, 0] - hand[p1, 0]),
                    float(hand[p2, 1] - hand[p1, 1]),
                    float(hand[p4, 0] - hand[p3, 0]),
                    float(hand[p4, 1] - hand[p3, 1]),
                )
            idx += 1

    nose_2d = (
        (float(pose[POSE_SLOT_NOSE, 0]), float(pose[POSE_SLOT_NOSE, 1]))
        if _valid_2d(pose, POSE_SLOT_NOSE)
        else None
    )
    mouth_2d = _midpoint_2d(pose, POSE_SLOT_LEFT_MOUTH, POSE_SLOT_RIGHT_MOUTH)
    chest_2d = _midpoint_2d(pose, POSE_SLOT_LEFT_SHOULDER, POSE_SLOT_RIGHT_SHOULDER)

    for hand in (lh, rh):
        wrist_ok = _valid_2d(hand, 0)
        for anchor in (nose_2d, mouth_2d, chest_2d):
            if wrist_ok and anchor is not None:
                out[idx] = math.hypot(float(hand[0, 0]) - anchor[0], float(hand[0, 1]) - anchor[1])
            idx += 1

    prev_arr = np.asarray(prev_frame, dtype=np.float32) if prev_frame is not None else None
    prev_lh = (
        prev_arr[LEFT_HAND_START:LEFT_HAND_END].reshape((HAND_POINTS, COORDINATES_PER_POINT))
        if prev_arr is not None
        else None
    )
    prev_rh = (
        prev_arr[RIGHT_HAND_START:RIGHT_HAND_END].reshape((HAND_POINTS, COORDINATES_PER_POINT))
        if prev_arr is not None
        else None
    )

    valid_dt = dt_s is not None and math.isfinite(dt_s) and dt_s > 0.0

    for hand, prev_hand in ((lh, prev_lh), (rh, prev_rh)):
        if prev_hand is not None and valid_dt and _valid_2d(hand, 0) and _valid_2d(prev_hand, 0):
            dx = float(hand[0, 0] - prev_hand[0, 0])
            dy = float(hand[0, 1] - prev_hand[0, 1])
            disp = math.hypot(dx, dy)
            if disp > MIN_SEGMENT_NORM:
                out[idx] = disp / float(dt_s)
                out[idx + 1] = dx / disp
                out[idx + 2] = dy / disp
            else:
                out[idx] = 0.0
                out[idx + 1] = 0.0
                out[idx + 2] = 0.0
        idx += VELOCITY_CHANNELS_PER_HAND

    if _valid_2d(lh, 0) and _valid_2d(rh, 0):
        out[idx] = math.hypot(float(lh[0, 0] - rh[0, 0]), float(lh[0, 1] - rh[0, 1]))
    idx += TWO_HAND_DIST_COUNT

    # Untrusted z channels kept strictly separate at tail
    if prev_lh is not None and valid_dt and _valid_3d(lh, 0) and _valid_3d(prev_lh, 0):
        out[idx] = float(lh[0, 2] - prev_lh[0, 2]) / float(dt_s)
    idx += 1
    if prev_rh is not None and valid_dt and _valid_3d(rh, 0) and _valid_3d(prev_rh, 0):
        out[idx] = float(rh[0, 2] - prev_rh[0, 2]) / float(dt_s)
    idx += 1
    if _valid_3d(lh, 0) and _valid_3d(rh, 0):
        out[idx] = float(lh[0, 2] - rh[0, 2])

    return out


def derived_sequence(
    seq: np.ndarray,
    timestamps_s: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Compute derived features across every frame of a normalized (T, F) sequence."""
    seq_arr = np.asarray(seq, dtype=np.float32)
    num_frames = seq_arr.shape[0]
    ts_arr: np.ndarray | None = None
    if timestamps_s is not None:
        ts_arr = np.asarray(timestamps_s, dtype=np.float64)
        if ts_arr.shape != (num_frames,):
            raise ValueError(
                f"timestamps_s shape {ts_arr.shape} does not match num_frames ({num_frames},)"
            )
    out = np.empty((num_frames, DERIVED_FEATURES_PER_FRAME), dtype=np.float32)
    for t in range(num_frames):
        prev = seq_arr[t - 1] if t > 0 else None
        dt_s = float(ts_arr[t] - ts_arr[t - 1]) if (t > 0 and ts_arr is not None) else None
        out[t] = derived_features(seq_arr[t], prev_frame=prev, dt_s=dt_s)
    return out
