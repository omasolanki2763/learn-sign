"""Unit tests for the verifier onset gate on synthetic sequences."""

from __future__ import annotations

import numpy as np
from verify_synth import line, make_sequence

from learnsign.verify.onset import OnsetGate


def test_onset_ignores_hand_resting_in_frame() -> None:
    seq, ts = make_sequence(line((0.2, 1.6), (0.2, 1.6)), n_frames=60)
    gate = OnsetGate()
    assert not any(gate.update(seq[i], ts[i]) for i in range(len(seq)))


def test_onset_ignores_hand_held_still_in_signing_space() -> None:
    seq, ts = make_sequence(line((0.2, 0.2), (0.2, 0.2)), n_frames=60)
    gate = OnsetGate()
    assert not any(gate.update(seq[i], ts[i]) for i in range(len(seq)))


def test_onset_fires_when_hand_rises_into_signing_space() -> None:
    seq, ts = make_sequence(line((0.2, 1.6), (0.2, -0.4)), n_frames=45)
    gate = OnsetGate()
    started = [gate.update(seq[i], ts[i]) for i in range(len(seq))]
    assert any(started)
    assert started.index(True) < len(seq) - 1


def test_onset_resets_after_hand_returns_to_rest() -> None:
    up, ts_up = make_sequence(line((0.2, 1.6), (0.2, -0.4)), n_frames=30)
    down, _ = make_sequence(line((0.2, -0.4), (0.2, 1.7)), n_frames=30)
    rest, _ = make_sequence(line((0.2, 1.7), (0.2, 1.7)), n_frames=30)
    seq = np.concatenate([up, down, rest])
    ts = np.arange(len(seq)) / 30.0
    gate = OnsetGate()
    states = [gate.update(seq[i], ts[i]) for i in range(len(seq))]
    assert any(states[:30])
    assert not states[-1]
