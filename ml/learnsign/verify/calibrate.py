"""Measure the two face-to-shoulder proportions used by the eye fallback (contract v2).

When a shoulder leaves the frame, normalisation estimates the shoulder width as
eye distance x EYE_TO_SHOULDER_RATIO and the shoulder midpoint as
eye midpoint + shoulder width x EYE_TO_SHOULDER_Y_OFFSET (y points down). Both numbers are
assumptions (6.0 and 0.5) until measured here on real frames where eyes and both shoulders
are visible.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np

# MediaPipe Pose landmark ids (raw 33-point layout).
LEFT_EYE: Final[int] = 2
RIGHT_EYE: Final[int] = 5
LEFT_SHOULDER: Final[int] = 11
RIGHT_SHOULDER: Final[int] = 12

# Ignore frames with an eye distance this small (face too far, or a detection glitch).
MIN_EYE_DISTANCE: Final[float] = 1e-4


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
    }


def calibrate_eye_ratios(poses: np.ndarray, aspect_ratio: float = 1.0) -> dict[str, Any]:
    """Measure the ratios from raw pose frames.

    Args:
        poses: Raw MediaPipe pose landmarks, shape (N, 33, 3), image-normalised x/y.
        aspect_ratio: Video width / height; x is multiplied by it so distances are isotropic,
            matching the normalisation step.

    Returns:
        {"eye_to_shoulder_ratio": {median, p05, p95},
         "eye_to_shoulder_y_offset": {median, p05, p95}, "frames_used": int}
    """
    arr = np.asarray(poses, dtype=np.float64)
    xy = arr[:, :, :2].copy()
    xy[:, :, 0] *= aspect_ratio
    needed = xy[:, [LEFT_EYE, RIGHT_EYE, LEFT_SHOULDER, RIGHT_SHOULDER], :]
    usable = np.all(np.isfinite(needed), axis=(1, 2))
    xy = xy[usable]

    eye_dist = np.linalg.norm(xy[:, LEFT_EYE] - xy[:, RIGHT_EYE], axis=1)
    shoulder_w = np.linalg.norm(xy[:, LEFT_SHOULDER] - xy[:, RIGHT_SHOULDER], axis=1)
    eye_mid_y = (xy[:, LEFT_EYE, 1] + xy[:, RIGHT_EYE, 1]) / 2.0
    shoulder_mid_y = (xy[:, LEFT_SHOULDER, 1] + xy[:, RIGHT_SHOULDER, 1]) / 2.0

    ok = eye_dist > MIN_EYE_DISTANCE
    if not np.any(ok):
        raise ValueError("No frame has both eyes and both shoulders visible")
    ratio = shoulder_w[ok] / eye_dist[ok]
    offset = (shoulder_mid_y[ok] - eye_mid_y[ok]) / shoulder_w[ok]
    return {
        "eye_to_shoulder_ratio": _summary(ratio),
        "eye_to_shoulder_y_offset": _summary(offset),
        "frames_used": int(ok.sum()),
    }
