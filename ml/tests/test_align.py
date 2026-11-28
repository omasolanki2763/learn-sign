"""Unit tests for open-end and full DTW alignment on synthetic sequences."""

from __future__ import annotations

import numpy as np
import pytest
from verify_synth import line, make_sequence

from learnsign.verify.align import (
    OpenEndAligner,
    dtw_distance,
    dtw_path,
    frame_distance,
    frame_distance_matrix,
)
from learnsign.verify.parts import alignment_features

CHIN_PATH = line((0.10, -0.70), (0.35, -0.30))
CHEST_PATH = line((-0.10, 0.30), (0.30, 0.30))


def test_dtw_distance_zero_for_identical_and_positive_for_different() -> None:
    a = alignment_features(*make_sequence(CHIN_PATH))
    b = alignment_features(*make_sequence(CHEST_PATH, shape="fist"))
    assert dtw_distance(a, a) == pytest.approx(0.0, abs=1e-6)
    assert dtw_distance(a, b) > 0.1


def test_dtw_path_is_monotonic_and_spans_both_sequences() -> None:
    a = alignment_features(*make_sequence(CHIN_PATH, n_frames=30))
    b = alignment_features(*make_sequence(CHIN_PATH, n_frames=40))
    path = dtw_path(a, b)
    assert path[0] == (0, 0)
    assert path[-1] == (len(a) - 1, len(b) - 1)
    assert all(i2 >= i1 and j2 >= j1 for (i1, j1), (i2, j2) in zip(path, path[1:], strict=False))


def test_open_end_aligner_full_sequence_reaches_end_with_zero_cost() -> None:
    ref = alignment_features(*make_sequence(CHIN_PATH))
    aligner = OpenEndAligner(ref)
    state = None
    for frame in ref:
        state = aligner.step(frame)
    assert state is not None
    assert state.cost == pytest.approx(0.0, abs=1e-6)
    assert state.progress == pytest.approx(1.0, abs=1e-6)


def test_open_end_aligner_half_sequence_reports_half_progress() -> None:
    ref = alignment_features(*make_sequence(CHIN_PATH, n_frames=40))
    aligner = OpenEndAligner(ref)
    state = None
    for frame in ref[:20]:
        state = aligner.step(frame)
    assert state is not None
    assert state.progress == pytest.approx(0.5, abs=0.1)


def test_frame_distance_matrix_matches_scalar_distance_with_missing_channels() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=(5, 4))
    b = rng.normal(size=(6, 4))
    a[1, 2] = np.nan
    b[3, :] = np.nan
    mat = frame_distance_matrix(a, b)
    for i in range(5):
        for j in range(6):
            assert mat[i, j] == pytest.approx(frame_distance(a[i], b[j]))
