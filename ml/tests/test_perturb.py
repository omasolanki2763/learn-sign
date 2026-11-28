"""Unit tests for synthetic wrong sign perturbations."""

from __future__ import annotations

import numpy as np
import pytest
from verify_synth import line, make_sequence

from learnsign.features.derived import derived_sequence
from learnsign.features.schema import LEFT_HAND_END, LEFT_HAND_START
from learnsign.verify.parts import PART_CHANNELS
from learnsign.verify.perturb import (
    drop_second_hand,
    reverse_movement,
    shift_location,
    swap_handshape,
)

CHIN_PATH = line((0.10, -0.70), (0.35, -0.30))
CHEST_PATH = line((-0.10, 0.30), (0.30, 0.30))


def _features(seq: np.ndarray, ts: np.ndarray) -> np.ndarray:
    return derived_sequence(seq, timestamps_s=ts)


def test_shift_location_moves_wrist_but_keeps_handshape() -> None:
    seq, ts = make_sequence(CHIN_PATH)
    moved = shift_location(seq, dx=0.0, dy=-0.4)
    f0, f1 = _features(seq, ts), _features(moved, ts)
    hs = PART_CHANNELS["handshape"]
    loc = PART_CHANNELS["location"]
    np.testing.assert_allclose(f0[:, hs], f1[:, hs], atol=1e-5)
    assert np.nanmax(np.abs(f0[:, loc] - f1[:, loc])) > 0.2


def test_swap_handshape_changes_bends_but_keeps_wrist() -> None:
    seq, ts = make_sequence(CHIN_PATH, shape="open")
    donor, _ = make_sequence(CHEST_PATH, shape="fist")
    swapped = swap_handshape(seq, donor)
    f0, f1 = _features(seq, ts), _features(swapped, ts)
    np.testing.assert_allclose(
        f0[:, PART_CHANNELS["location"]], f1[:, PART_CHANNELS["location"]], atol=1e-5
    )
    assert (
        np.nanmax(np.abs(f0[:, PART_CHANNELS["handshape"]] - f1[:, PART_CHANNELS["handshape"]]))
        > 0.5
    )


def test_reverse_movement_flips_direction_and_keeps_handshape() -> None:
    seq, ts = make_sequence(CHIN_PATH)
    rev = reverse_movement(seq)
    f0, f1 = _features(seq, ts), _features(rev, ts)
    np.testing.assert_allclose(
        f0[:, PART_CHANNELS["handshape"]], f1[:, PART_CHANNELS["handshape"]], atol=1e-5
    )
    mid = len(seq) // 2
    dir0 = f0[mid, PART_CHANNELS["movement"]][1:3]
    dir1 = f1[mid, PART_CHANNELS["movement"]][1:3]
    assert float(np.dot(dir0, dir1)) == pytest.approx(-1.0, abs=1e-3)


def test_drop_second_hand_removes_left_hand_only() -> None:
    seq, _ = make_sequence(CHIN_PATH, left_path=CHEST_PATH)
    out = drop_second_hand(seq)
    assert np.all(np.isnan(out[:, LEFT_HAND_START:LEFT_HAND_END]))
    np.testing.assert_array_equal(out[:, LEFT_HAND_END:], seq[:, LEFT_HAND_END:])
