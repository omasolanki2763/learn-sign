"""Dynamic time warping (DTW): lining two recordings of a sign up in time.

Two people sign at different speeds, so frame 10 of the learner is not frame 10 of the
reference. DTW finds the cheapest monotonic pairing of frames. Two flavours are used here:

- `dtw_distance` / `dtw_path`: both recordings complete (building references offline).
- `OpenEndAligner`: the learner's recording is still growing. After every new frame it says
  which reference frame the learner has reached ("progress") and how well it matches so far.
  "Open end" means the match may stop anywhere in the reference, not only at its last frame.

Costs are divided by the warping-path length, so short and long recordings compare fairly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

# Cost of a frame pair that shares no measured channel (e.g. the hand was not detected).
# Why a constant rather than infinity: one missing frame must not make a whole attempt fail.
MISSING_FRAME_COST: Final[float] = 1.0

# Default Sakoe-Chiba half-band in frames for the streaming aligner (None = no band).
# The browser uses a band for speed (Task 3 measured it); the offline spike defaults to no
# band so that learners who sign at a different speed are still aligned.
DEFAULT_BAND_RADIUS: Final[int | None] = None


def frame_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference over the channels measured in both frames."""
    both = np.isfinite(a) & np.isfinite(b)
    if not np.any(both):
        return MISSING_FRAME_COST
    return float(np.mean(np.abs(a[both] - b[both])))


def _in_band(i: int, j: int, band: int | None) -> bool:
    return band is None or abs(i - j) <= band


def frame_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """All pairwise `frame_distance` values at once, shape (len(a), len(b)).

    Why vectorised: building references compares every recording with every other one; doing
    the per-frame work in NumPy instead of a Python loop makes that minutes instead of hours.
    """
    diff = np.abs(a[:, None, :] - b[None, :, :])
    both = np.isfinite(diff)
    counts = both.sum(axis=2)
    sums = np.where(both, diff, 0.0).sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        dist = np.where(counts > 0, sums / np.maximum(counts, 1), MISSING_FRAME_COST)
    return dist


def _accumulate(
    a: np.ndarray, b: np.ndarray, band: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Full DTW tables: cumulative cost D and path length L, shape (len(a), len(b))."""
    n, m = len(a), len(b)
    local = frame_distance_matrix(a, b)
    cost = np.full((n, m), np.inf)
    length = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if not _in_band(i, j, band):
                continue
            d = local[i, j]
            if i == 0 and j == 0:
                cost[i, j], length[i, j] = d, 1
                continue
            # Predecessors: diagonal, from above, from the left (whichever exist).
            options = []
            if i > 0 and j > 0:
                options.append((cost[i - 1, j - 1], length[i - 1, j - 1]))
            if i > 0:
                options.append((cost[i - 1, j], length[i - 1, j]))
            if j > 0:
                options.append((cost[i, j - 1], length[i, j - 1]))
            best_cost, best_len = min(options, key=lambda o: o[0])
            cost[i, j], length[i, j] = best_cost + d, best_len + 1
    return cost, length


def dtw_distance(a: np.ndarray, b: np.ndarray, band: int | None = None) -> float:
    """Length-normalised DTW cost between two complete sequences (0 = identical)."""
    cost, length = _accumulate(a, b, band)
    return float(cost[-1, -1] / length[-1, -1])


def dtw_path(a: np.ndarray, b: np.ndarray, band: int | None = None) -> list[tuple[int, int]]:
    """Cheapest monotonic frame pairing [(i, j), ...] from (0, 0) to (len(a)-1, len(b)-1)."""
    cost, _ = _accumulate(a, b, band)
    i, j = len(a) - 1, len(b) - 1
    path = [(i, j)]
    while (i, j) != (0, 0):
        candidates = []
        if i > 0 and j > 0:
            candidates.append((cost[i - 1, j - 1], i - 1, j - 1))
        if i > 0:
            candidates.append((cost[i - 1, j], i - 1, j))
        if j > 0:
            candidates.append((cost[i, j - 1], i, j - 1))
        _, i, j = min(candidates, key=lambda c: c[0])
        path.append((i, j))
    path.reverse()
    return path


@dataclass(frozen=True)
class AlignState:
    """Where the learner is in the reference after the latest frame."""

    cost: float  # length-normalised cost of the best partial match so far
    progress: float  # fraction of the reference reached, 0..1
    ref_index: int  # reference frame matched to the latest learner frame


class OpenEndAligner:
    """Streaming open-end DTW against one reference; one `step` per learner frame.

    Keeps only the previous row of the DTW tables, so each step costs O(len(reference)).
    """

    def __init__(self, reference: np.ndarray, band: int | None = DEFAULT_BAND_RADIUS) -> None:
        self.reference = np.asarray(reference, dtype=np.float32)
        self.band = band
        self._row: np.ndarray | None = None  # cumulative cost for the previous learner frame
        self._len: np.ndarray | None = None  # path length for the previous learner frame
        self._i = -1

    def step(self, frame: np.ndarray) -> AlignState:
        """Add one learner frame (same channels as the reference) and return the best match."""
        self._i += 1
        m = len(self.reference)
        row = np.full(m, np.inf)
        length = np.zeros(m)
        local = frame_distance_matrix(np.asarray(frame)[None, :], self.reference)[0]
        for j in range(m):
            if not _in_band(self._i, j, self.band):
                continue
            d = local[j]
            options = []
            if self._row is not None and self._len is not None:
                if j > 0:
                    options.append((self._row[j - 1], self._len[j - 1]))
                options.append((self._row[j], self._len[j]))
            if j > 0:
                options.append((row[j - 1], length[j - 1]))
            if not options:  # very first cell (0, 0)
                row[j], length[j] = d, 1
                continue
            best_cost, best_len = min(options, key=lambda o: o[0])
            if np.isfinite(best_cost):
                row[j], length[j] = best_cost + d, best_len + 1
        self._row, self._len = row, length

        with np.errstate(invalid="ignore", divide="ignore"):
            normalised = np.where(length > 0, row / np.maximum(length, 1), np.inf)
        best_j = int(np.argmin(normalised))
        return AlignState(
            cost=float(normalised[best_j]),
            progress=(best_j + 1) / m,
            ref_index=best_j,
        )
