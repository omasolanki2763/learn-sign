"""Which derived-feature channels describe which part of a sign.

The verifier judges a sign part by part (handshape, location, movement, hands used), so each
part needs a fixed list of channels in the contract-v2 derived vector. Everything here
assumes handedness has been canonicalised, so the dominant hand is always the RIGHT slot.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from learnsign.features.derived import (
    DERIVED_BEND_START,
    DERIVED_SPREAD_START,
    DERIVED_TIP_DIST_START,
    DERIVED_VELOCITY_START,
    DERIVED_WRIST_ANCHOR_START,
    FINGER_BEND_JOINTS_PER_HAND,
    FINGER_SPREADS_PER_HAND,
    FINGERTIP_DISTS_PER_HAND,
    VELOCITY_CHANNELS_PER_HAND,
    WRIST_ANCHORS_PER_HAND,
    derived_sequence,
)


def _right(start: int, per_hand: int) -> list[int]:
    """Channel indices of the right (dominant) hand block that starts at `start`.

    The derived layout stores each block as [left hand channels, right hand channels].
    """
    return list(range(start + per_hand, start + 2 * per_hand))


def _left(start: int, per_hand: int) -> list[int]:
    """Channel indices of the left (non-dominant) hand block that starts at `start`."""
    return list(range(start, start + per_hand))


# Part name -> derived-vector channel indices (dominant hand).
PART_CHANNELS: Final[dict[str, list[int]]] = {
    # Handshape = how each finger is bent and spread; distances between fingertips.
    "handshape": (
        _right(DERIVED_BEND_START, FINGER_BEND_JOINTS_PER_HAND)
        + _right(DERIVED_TIP_DIST_START, FINGERTIP_DISTS_PER_HAND)
        + _right(DERIVED_SPREAD_START, FINGER_SPREADS_PER_HAND)
    ),
    # Location = wrist distance to nose, mouth and chest centre (shoulder widths).
    "location": _right(DERIVED_WRIST_ANCHOR_START, WRIST_ANCHORS_PER_HAND),
    # Movement = wrist speed and unit direction.
    "movement": _right(DERIVED_VELOCITY_START, VELOCITY_CHANNELS_PER_HAND),
}

# Channels that say whether the second (non-dominant) hand is visible at all.
SECOND_HAND_PRESENCE_CHANNELS: Final[list[int]] = _left(
    DERIVED_BEND_START, FINGER_BEND_JOINTS_PER_HAND
)

PART_NAMES: Final[tuple[str, ...]] = ("handshape", "location", "movement")

# Per-part weight in the alignment distance. Why: the parts have different units (radians,
# shoulder widths, shoulder widths per second); these weights keep any one part from
# dominating the path. Assumed values, to be tuned on real data in the Task 8 notebook.
PART_ALIGNMENT_WEIGHTS: Final[dict[str, float]] = {
    "handshape": 1.0,
    "location": 3.0,
    "movement": 0.5,
}

# Movement channel layout inside PART_CHANNELS["movement"].
MOVEMENT_SPEED_INDEX: Final[int] = 0


def alignment_features(seq: np.ndarray, timestamps_s: np.ndarray) -> np.ndarray:
    """Compact per-frame vector used to line two recordings up in time.

    Concatenates the part channels (each scaled by its weight) so that alignment follows
    handshape, location and movement together.

    Args:
        seq: Normalised contract-v2 sequence (T, FEATURES_PER_FRAME).
        timestamps_s: Frame times in seconds (T,).

    Returns:
        float32 array (T, C).
    """
    derived = derived_sequence(seq, timestamps_s=timestamps_s)
    return alignment_features_from_derived(derived)


def alignment_features_from_derived(derived: np.ndarray) -> np.ndarray:
    """Same as `alignment_features`, starting from an already computed derived sequence."""
    blocks = [derived[:, PART_CHANNELS[p]] * PART_ALIGNMENT_WEIGHTS[p] for p in PART_NAMES]
    return np.concatenate(blocks, axis=1).astype(np.float32)


def second_hand_visible(derived_frame: np.ndarray) -> bool:
    """True when any non-dominant finger-bend channel is measured in this frame."""
    return bool(np.any(np.isfinite(derived_frame[SECOND_HAND_PRESENCE_CHANNELS])))
