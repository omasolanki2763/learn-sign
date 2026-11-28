"""Synthetic wrong signs: take a correct recording and break exactly one part.

Why: to test whether the verifier catches a mistake AND names the right part, we need wrong
attempts whose mistake we know. Real learner errors are not available, so we make them from
correct recordings of Deaf signers. Every function returns a new array; inputs are untouched.
All operate on normalised contract-v2 sequences with the dominant hand in the right slot.
"""

from __future__ import annotations

import numpy as np

from learnsign.features.schema import (
    COORDINATES_PER_POINT,
    HAND_POINTS,
    LEFT_HAND_END,
    LEFT_HAND_START,
    POSE_SLOT_LEFT_WRIST,
    POSE_SLOT_RIGHT_ELBOW,
    POSE_SLOT_RIGHT_WRIST,
    POSE_START,
    RIGHT_HAND_END,
    RIGHT_HAND_START,
)


def _right_hand(seq: np.ndarray) -> np.ndarray:
    """View of the dominant hand as (T, 21, 3)."""
    return seq[:, RIGHT_HAND_START:RIGHT_HAND_END].reshape(-1, HAND_POINTS, COORDINATES_PER_POINT)


def _pose_xy_offset(slot: int) -> int:
    return POSE_START + slot * COORDINATES_PER_POINT


def shift_location(seq: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Move the dominant hand (and its arm points) by (dx, dy) shoulder widths.

    Example: dy = -0.4 lifts a chin sign to the forehead (MOM -> DAD style location error).
    """
    out = np.array(seq, dtype=np.float32, copy=True)
    hand = _right_hand(out)
    hand[:, :, 0] += dx
    hand[:, :, 1] += dy
    for slot in (POSE_SLOT_RIGHT_WRIST, POSE_SLOT_RIGHT_ELBOW):
        off = _pose_xy_offset(slot)
        out[:, off] += dx
        out[:, off + 1] += dy
    return out


def swap_handshape(seq: np.ndarray, donor: np.ndarray) -> np.ndarray:
    """Replace the dominant handshape with the donor's, keeping the original wrist path.

    The donor is resampled to the same number of frames; each donor finger configuration is
    re-attached to the original wrist position, so location and movement stay unchanged.
    """
    out = np.array(seq, dtype=np.float32, copy=True)
    hand = _right_hand(out)
    donor_hand = _right_hand(np.asarray(donor, dtype=np.float32))
    pick = np.linspace(0, len(donor_hand) - 1, num=len(hand)).round().astype(int)
    donor_rel = donor_hand[pick] - donor_hand[pick, :1, :]  # fingers relative to donor wrist
    hand[:] = hand[:, :1, :] + donor_rel
    return out


def reverse_movement(seq: np.ndarray) -> np.ndarray:
    """Play the wrist path backwards while keeping each frame's handshape.

    Example: a sign that moves away from the body is made towards the body instead.
    """
    out = np.array(seq, dtype=np.float32, copy=True)
    hand = _right_hand(out)
    rel = hand - hand[:, :1, :]
    reversed_wrist = hand[::-1, :1, :].copy()
    hand[:] = reversed_wrist + rel
    wrist_off = _pose_xy_offset(POSE_SLOT_RIGHT_WRIST)
    out[:, wrist_off : wrist_off + 2] = seq[::-1, wrist_off : wrist_off + 2]
    return out


def drop_second_hand(seq: np.ndarray) -> np.ndarray:
    """Remove the non-dominant hand (and its pose wrist), as if signed one-handed."""
    out = np.array(seq, dtype=np.float32, copy=True)
    out[:, LEFT_HAND_START:LEFT_HAND_END] = np.nan
    off = _pose_xy_offset(POSE_SLOT_LEFT_WRIST)
    out[:, off : off + COORDINATES_PER_POINT] = np.nan
    return out
