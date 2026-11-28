"""Per-sign reference model: variants (clusters) and allowed ranges per phase of the sign.

How a sign model is built from recordings of real signers:
1. Trim each recording to the part after onset (the lift into position is not the sign).
2. Group the recordings into variants: some signs are made in two or three correct ways, and
   one pooled range would be so wide that wrong signs pass. Grouping uses k-medoids on DTW
   distance; a medoid is the most typical recording of its group.
3. Align every recording of a group to its medoid and split the medoid into phases (e.g. 10
   slices of the sign). For each phase and channel, the allowed range is the middle 90% of the
   real signers' values, widened by a tolerance and never narrower than a floor.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from learnsign.features.derived import (
    FINGER_BEND_JOINTS_PER_HAND,
    FINGER_SPREADS_PER_HAND,
    FINGERTIP_DISTS_PER_HAND,
    derived_sequence,
)
from learnsign.verify.align import dtw_distance, dtw_path
from learnsign.verify.onset import OnsetGate
from learnsign.verify.parts import (
    PART_CHANNELS,
    PART_NAMES,
    alignment_features_from_derived,
    second_hand_visible,
)

N_PHASES: Final[int] = 10
LOW_QUANTILE: Final[float] = 0.05
HIGH_QUANTILE: Final[float] = 0.95
# Extra margin added on each side, as a fraction of the observed range width.
RANGE_TOLERANCE: Final[float] = 0.25
# A channel is "off" for a part if more than this fraction of the part's channels are outside.
PART_CHANNEL_FAIL_FRACTION: Final[float] = 0.3
# The second hand counts as required if it is visible in at least this fraction of frames.
SECOND_HAND_REQUIRED_FRACTION: Final[float] = 0.5
MAX_KMEDOIDS_ITERATIONS: Final[int] = 10

# Narrowest allowed half-margin per channel type (units differ, so floors differ).
ANGLE_FLOOR_RAD: Final[float] = 0.15
FINGER_DISTANCE_FLOOR: Final[float] = 0.03  # shoulder widths
LOCATION_FLOOR: Final[float] = 0.08  # shoulder widths
SPEED_FLOOR: Final[float] = 0.3  # shoulder widths per second
DIRECTION_FLOOR: Final[float] = 0.35  # unit-vector components


def _part_floors() -> dict[str, np.ndarray]:
    """Floor per channel, in the same order as PART_CHANNELS."""
    handshape = (
        [ANGLE_FLOOR_RAD] * FINGER_BEND_JOINTS_PER_HAND
        + [FINGER_DISTANCE_FLOOR] * FINGERTIP_DISTS_PER_HAND
        + [ANGLE_FLOOR_RAD] * FINGER_SPREADS_PER_HAND
    )
    floors = {
        "handshape": np.array(handshape),
        "location": np.full(len(PART_CHANNELS["location"]), LOCATION_FLOOR),
        "movement": np.array([SPEED_FLOOR, DIRECTION_FLOOR, DIRECTION_FLOOR]),
    }
    for part in PART_NAMES:
        assert len(floors[part]) == len(PART_CHANNELS[part]), part
    return floors


PART_FLOORS: Final[dict[str, np.ndarray]] = _part_floors()


def trim_to_onset(seq: np.ndarray, timestamps_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop the frames before the onset gate opens (keeps everything if it never opens)."""
    gate = OnsetGate()
    for i in range(len(seq)):
        if gate.update(seq[i], float(timestamps_s[i])):
            return seq[i:], timestamps_s[i:]
    return seq, timestamps_s


def cluster_references(features: list[np.ndarray], k: int) -> tuple[list[int], list[int]]:
    """k-medoids on DTW distance.

    Args:
        features: Alignment-feature sequences, one per recording.
        k: Number of variants (clipped to the number of recordings).

    Returns:
        (medoid indices into `features`, cluster label per recording).
    """
    n = len(features)
    k = max(1, min(k, n))
    dist = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            dist[a, b] = dist[b, a] = dtw_distance(features[a], features[b])

    # Start from the most central recording, then repeatedly add the one farthest from all.
    medoids = [int(np.argmin(dist.sum(axis=1)))]
    while len(medoids) < k:
        medoids.append(int(np.argmax(dist[:, medoids].min(axis=1))))

    labels = list(np.argmin(dist[:, medoids], axis=1))
    for _ in range(MAX_KMEDOIDS_ITERATIONS):
        new_medoids = []
        for c in range(k):
            members = [i for i, lab in enumerate(labels) if lab == c] or [medoids[c]]
            within = dist[np.ix_(members, members)].sum(axis=1)
            new_medoids.append(members[int(np.argmin(within))])
        new_labels = list(np.argmin(dist[:, new_medoids], axis=1))
        if new_medoids == medoids and new_labels == labels:
            break
        medoids, labels = new_medoids, new_labels
    return medoids, [int(x) for x in labels]


@dataclass
class VariantModel:
    """One variant of a sign: its medoid and per-phase ranges for every part."""

    medoid_alignment: np.ndarray  # (M, C) alignment features of the medoid
    low: dict[str, np.ndarray] = field(default_factory=dict)  # part -> (N_PHASES, channels)
    high: dict[str, np.ndarray] = field(default_factory=dict)

    def phase_of(self, ref_index: int) -> int:
        """Phase bucket for a medoid frame index."""
        return min(N_PHASES - 1, ref_index * N_PHASES // len(self.medoid_alignment))

    def off_parts(self, derived_frame: np.ndarray, phase: int) -> set[str]:
        """Parts whose channels are out of range at this phase."""
        off: set[str] = set()
        for part in PART_NAMES:
            values = derived_frame[PART_CHANNELS[part]]
            lo, hi = self.low[part][phase], self.high[part][phase]
            checked = np.isfinite(values) & np.isfinite(lo) & np.isfinite(hi)
            if not np.any(checked):
                continue
            outside = (values[checked] < lo[checked]) | (values[checked] > hi[checked])
            if outside.mean() > PART_CHANNEL_FAIL_FRACTION:
                off.add(part)
        return off


@dataclass
class SignModel:
    """All variants of one sign, plus whether it needs two hands."""

    variants: list[VariantModel]
    needs_second_hand: bool

    def best_variant_path(self, derived: np.ndarray) -> tuple[VariantModel, list[tuple[int, int]]]:
        """Variant closest to a complete recording, and the frame pairing with its medoid."""
        feats = alignment_features_from_derived(derived)
        scored = [(dtw_distance(feats, v.medoid_alignment), v) for v in self.variants]
        _, best = min(scored, key=lambda s: s[0])
        return best, dtw_path(feats, best.medoid_alignment)

    def fraction_frames_in_range(self, seq: np.ndarray, timestamps_s: np.ndarray) -> float:
        """Offline check: fraction of a complete recording's frames with no part off."""
        seq, timestamps_s = trim_to_onset(seq, timestamps_s)
        derived = derived_sequence(seq, timestamps_s=timestamps_s)
        variant, path = self.best_variant_path(derived)
        first_match: dict[int, int] = {}
        for i, j in path:
            first_match.setdefault(i, j)
        ok = [
            not variant.off_parts(derived[i], variant.phase_of(j)) for i, j in first_match.items()
        ]
        return float(np.mean(ok)) if ok else 0.0


def _widen(low: np.ndarray, high: np.ndarray, floor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    margin = np.maximum(RANGE_TOLERANCE * (high - low), floor)
    return low - margin, high + margin


def build_sign_model(references: list[tuple[np.ndarray, np.ndarray]], k: int = 1) -> SignModel:
    """Build a SignModel from (normalised sequence, timestamps) recordings of one sign."""
    trimmed = [trim_to_onset(s, t) for s, t in references]
    derived = [derived_sequence(s, timestamps_s=t) for s, t in trimmed]
    feats = [alignment_features_from_derived(d) for d in derived]
    medoids, labels = cluster_references(feats, k)

    variants: list[VariantModel] = []
    for c, medoid in enumerate(medoids):
        members = [i for i, lab in enumerate(labels) if lab == c]
        m_len = len(feats[medoid])
        variant = VariantModel(medoid_alignment=feats[medoid])
        bins: dict[str, list[list[np.ndarray]]] = {
            p: [[] for _ in range(N_PHASES)] for p in PART_NAMES
        }
        for i in members:
            for fi, mj in dtw_path(feats[i], feats[medoid]):
                phase = min(N_PHASES - 1, mj * N_PHASES // m_len)
                for p in PART_NAMES:
                    bins[p][phase].append(derived[i][fi, PART_CHANNELS[p]])
        for p in PART_NAMES:
            n_ch = len(PART_CHANNELS[p])
            low = np.full((N_PHASES, n_ch), np.nan)
            high = np.full((N_PHASES, n_ch), np.nan)
            for ph in range(N_PHASES):
                if bins[p][ph]:
                    stacked = np.stack(bins[p][ph])
                    with warnings.catch_warnings():
                        # Why: a channel never measured in this phase (e.g. velocity on the
                        # first frame) is all-NaN; NaN bounds are then skipped when checking.
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        low[ph] = np.nanquantile(stacked, LOW_QUANTILE, axis=0)
                        high[ph] = np.nanquantile(stacked, HIGH_QUANTILE, axis=0)
            variant.low[p], variant.high[p] = _widen(low, high, PART_FLOORS[p])
        variants.append(variant)

    visible = [second_hand_visible(frame) for d in derived for frame in d]
    needs_second = bool(np.mean(visible) >= SECOND_HAND_REQUIRED_FRACTION) if visible else False
    return SignModel(variants=variants, needs_second_hand=needs_second)
