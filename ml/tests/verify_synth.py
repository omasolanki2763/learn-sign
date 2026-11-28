"""Synthetic, already-normalised sign sequences for the verifier tests.

Real clips need the Kaggle data, so the verifier is tested on hand-built sequences whose
handshape, location and movement we control exactly. Coordinates are in the normalised
frame of contract v2: origin at the shoulder midpoint, +x to the image right, +y DOWN,
one unit = one shoulder width.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from learnsign.features.schema import (
    COORDINATES_PER_POINT,
    FEATURES_PER_FRAME,
    HAND_POINTS,
    LEFT_HAND_START,
    POSE_POINTS,
    POSE_SLOT_LEFT_MOUTH,
    POSE_SLOT_LEFT_SHOULDER,
    POSE_SLOT_NOSE,
    POSE_SLOT_RIGHT_MOUTH,
    POSE_SLOT_RIGHT_SHOULDER,
    POSE_SLOT_RIGHT_WRIST,
    POSE_START,
    RIGHT_HAND_START,
)

FPS = 30.0

# Length of one finger segment in shoulder widths (a hand is roughly a third of a shoulder width).
SEGMENT = 0.035

# Direction (radians, image coordinates, -pi/2 = straight up) of each finger's first segment.
FINGER_BASE_ANGLES = (-0.6, -1.85, -1.6, -1.35, -1.1)  # thumb, index, middle, ring, pinky

# How much each later joint turns relative to the previous segment, per handshape and finger.
# 0 = straight finger; ~1.4 rad = curled into a fist.
SHAPE_CURLS: dict[str, tuple[float, float, float, float, float]] = {
    "open": (0.0, 0.0, 0.0, 0.0, 0.0),
    "fist": (0.9, 1.4, 1.4, 1.4, 1.4),
    "point": (0.9, 0.0, 1.4, 1.4, 1.4),  # index straight, others curled
}

Path = Callable[[float], tuple[float, float]]


def hand_points(wrist_xy: tuple[float, float], shape: str) -> np.ndarray:
    """Build 21 MediaPipe-ordered hand points for one frame.

    Args:
        wrist_xy: Wrist position in normalised coordinates.
        shape: One of SHAPE_CURLS.

    Returns:
        Array (21, 3) with z = 0.
    """
    pts = np.zeros((HAND_POINTS, COORDINATES_PER_POINT), dtype=np.float32)
    pts[0, :2] = wrist_xy
    curls = SHAPE_CURLS[shape]
    for finger, (base_angle, curl) in enumerate(zip(FINGER_BASE_ANGLES, curls, strict=True)):
        # MediaPipe order: each finger owns 4 consecutive points starting at 1 + 4*finger.
        x, y = wrist_xy
        angle = base_angle
        # Why the longer first segment: wrist-to-knuckle is longer than a finger segment.
        lengths = (SEGMENT * 2.0, SEGMENT, SEGMENT, SEGMENT)
        for joint, length in enumerate(lengths):
            if joint >= 2:
                angle += curl  # later segments turn by the curl amount
            x += length * math.cos(angle)
            y += length * math.sin(angle)
            pts[1 + 4 * finger + joint, :2] = (x, y)
    return pts


def base_pose() -> np.ndarray:
    """Upper-body pose in normalised coordinates (17 slots, NaN where unused)."""
    pose = np.full((POSE_POINTS, COORDINATES_PER_POINT), np.nan, dtype=np.float32)
    pose[POSE_SLOT_LEFT_SHOULDER] = (-0.5, 0.0, 0.0)
    pose[POSE_SLOT_RIGHT_SHOULDER] = (0.5, 0.0, 0.0)
    pose[POSE_SLOT_NOSE] = (0.0, -0.9, 0.0)
    pose[POSE_SLOT_LEFT_MOUTH] = (-0.08, -0.75, 0.0)
    pose[POSE_SLOT_RIGHT_MOUTH] = (0.08, -0.75, 0.0)
    return pose


def make_sequence(
    path: Path,
    shape: str = "open",
    n_frames: int = 30,
    left_path: Path | None = None,
    left_shape: str = "open",
) -> tuple[np.ndarray, np.ndarray]:
    """Render a normalised sequence where the right wrist follows `path(t)` for t in [0, 1].

    Args:
        path: Right-wrist trajectory.
        shape: Right handshape (constant over the sign).
        n_frames: Number of frames at FPS.
        left_path: Optional left-wrist trajectory; None means the left hand is absent (NaN).
        left_shape: Left handshape when present.

    Returns:
        (sequence float32[T, FEATURES_PER_FRAME], timestamps_s float64[T]).
    """
    seq = np.full((n_frames, FEATURES_PER_FRAME), np.nan, dtype=np.float32)
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        wrist = path(t)
        seq[i, RIGHT_HAND_START : RIGHT_HAND_START + HAND_POINTS * 3] = hand_points(
            wrist, shape
        ).ravel()
        if left_path is not None:
            seq[i, LEFT_HAND_START : LEFT_HAND_START + HAND_POINTS * 3] = hand_points(
                left_path(t), left_shape
            ).ravel()
        pose = base_pose()
        pose[POSE_SLOT_RIGHT_WRIST] = (wrist[0], wrist[1], 0.0)
        seq[i, POSE_START:] = pose.ravel()
    timestamps = np.arange(n_frames, dtype=np.float64) / FPS
    return seq, timestamps


def line(start: tuple[float, float], end: tuple[float, float]) -> Path:
    """Straight trajectory from start to end."""

    def _path(t: float) -> tuple[float, float]:
        return (start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)

    return _path


def jitter(seq: np.ndarray, scale: float, seed: int) -> np.ndarray:
    """Add small Gaussian noise to every coordinate (a different 'signer' of the same sign)."""
    rng = np.random.default_rng(seed)
    noisy = seq + rng.normal(0.0, scale, size=seq.shape).astype(np.float32)
    return noisy.astype(np.float32)
