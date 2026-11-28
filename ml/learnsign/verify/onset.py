"""Onset gate: decides when the learner has started signing.

Why it exists: lifting the hands from the lap into position is not part of the sign. Judging
from the very first frame would mark every attempt red during that lift. The gate opens only
when the dominant wrist is inside signing space AND moving, and closes again once the hand has
gone back to rest, so a hand lying still in view (on a desk, or held between repetitions)
never starts an attempt on its own.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

from learnsign.features.schema import (
    COORDINATES_PER_POINT,
    POSE_SLOT_RIGHT_WRIST,
    POSE_START,
    RIGHT_HAND_START,
)

# Signing space ends this far below the shoulder line (normalised y points down; 1 unit = one
# shoulder width). Roughly the waist. Assumed value; checked on real data in the notebook.
SIGNING_SPACE_MAX_Y: Final[float] = 1.2

# Minimum wrist speed (shoulder widths per second) that counts as "moving".
MIN_ONSET_SPEED: Final[float] = 0.3

# Consecutive in-space, moving frames needed to open the gate (avoids single-frame jitter).
ONSET_FRAMES: Final[int] = 3

# Consecutive out-of-space frames needed to close it again (hysteresis).
RESET_FRAMES: Final[int] = 5


def dominant_wrist_xy(frame: np.ndarray) -> tuple[float, float] | None:
    """Dominant (right-slot) wrist position from the hand model, else from the pose model."""
    x, y = float(frame[RIGHT_HAND_START]), float(frame[RIGHT_HAND_START + 1])
    if math.isfinite(x) and math.isfinite(y):
        return x, y
    offset = POSE_START + POSE_SLOT_RIGHT_WRIST * COORDINATES_PER_POINT
    x, y = float(frame[offset]), float(frame[offset + 1])
    if math.isfinite(x) and math.isfinite(y):
        return x, y
    return None


class OnsetGate:
    """Streaming onset detector; call `update` once per frame, in time order."""

    def __init__(self) -> None:
        self.started = False
        self._moving_in_space = 0
        self._out_of_space = 0
        self._prev: tuple[float, float, float] | None = None  # (x, y, t)

    def update(self, frame: np.ndarray, timestamp_s: float) -> bool:
        """Feed one normalised frame.

        Args:
            frame: Normalised contract-v2 frame (FEATURES_PER_FRAME,).
            timestamp_s: Frame time in seconds.

        Returns:
            True while an attempt is in progress.
        """
        wrist = dominant_wrist_xy(frame)
        speed = 0.0
        if wrist is not None and self._prev is not None:
            dt = timestamp_s - self._prev[2]
            if dt > 0.0:
                speed = math.hypot(wrist[0] - self._prev[0], wrist[1] - self._prev[1]) / dt
        self._prev = (wrist[0], wrist[1], timestamp_s) if wrist is not None else None

        in_space = wrist is not None and wrist[1] < SIGNING_SPACE_MAX_Y

        if not self.started:
            moving = speed >= MIN_ONSET_SPEED
            self._moving_in_space = self._moving_in_space + 1 if (in_space and moving) else 0
            if self._moving_in_space >= ONSET_FRAMES:
                self.started = True
                self._out_of_space = 0
        else:
            self._out_of_space = 0 if in_space else self._out_of_space + 1
            if self._out_of_space >= RESET_FRAMES:
                self.started = False
                self._moving_in_space = 0
        return self.started
