"""Unit tests for k-medoids clustering and per-phase ranges on synthetic sequences."""

from __future__ import annotations

import numpy as np
from verify_synth import jitter, line, make_sequence

from learnsign.verify.parts import alignment_features
from learnsign.verify.ranges import build_sign_model, cluster_references

CHIN_PATH = line((0.10, -0.70), (0.35, -0.30))


def _refs(path=CHIN_PATH, shape="open", n=6, seed0=0) -> list[tuple[np.ndarray, np.ndarray]]:
    base, ts = make_sequence(path, shape=shape)
    return [(jitter(base, 0.002, seed0 + k), ts) for k in range(n)]


def test_cluster_references_separates_two_variants() -> None:
    variant_a = _refs(CHIN_PATH, n=4, seed0=0)
    variant_b = _refs(line((0.10, -0.60), (0.60, -0.60)), n=4, seed0=10)
    feats = [alignment_features(s, t) for s, t in variant_a + variant_b]
    medoids, labels = cluster_references(feats, k=2)
    assert len(medoids) == 2
    assert len(set(labels[:4])) == 1
    assert len(set(labels[4:])) == 1
    assert labels[0] != labels[4]


def test_ranges_contain_most_frames_of_a_new_correct_signer() -> None:
    model = build_sign_model(_refs(n=8), k=1)
    held_seq, held_ts = _refs(n=1, seed0=99)[0]
    frac = model.fraction_frames_in_range(held_seq, held_ts)
    assert frac >= 0.9
