"""Scoring a candidate judge on labelled attempts (correct, wrong sign, minimal pair, perturbed).

Metrics (all on held-out signers in the real run):
- correct_pass_rate: correct attempts that pass (target >= 0.90).
- false_green_wrong / false_green_minimal: wrong attempts that pass (target <= 0.05); minimal
  pairs (signs differing in one part, e.g. MOM vs DAD) are reported separately because random
  wrong signs are far easier to reject.
- perturbation_caught: synthetic wrong signs that fail (target >= 0.90).
- part_named: of the caught perturbations, those whose failed parts include the one we broke
  (target >= 0.80).
- mean_first_verdict_s: seconds from onset to the first verdict tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from learnsign.verify.judges import Judge, Verdict

ExampleKind = Literal["correct", "wrong_sign", "minimal_pair", "perturbed"]


@dataclass(frozen=True)
class Example:
    """One labelled attempt at `target`."""

    target: str
    seq: np.ndarray
    timestamps_s: np.ndarray
    kind: ExampleKind
    perturbed_part: str | None = None  # set when kind == "perturbed"


@dataclass(frozen=True)
class Metrics:
    """Rates for one judge; a rate is None when no example of that kind was given."""

    correct_pass_rate: float | None
    false_green_wrong: float | None
    false_green_minimal: float | None
    perturbation_caught: float | None
    part_named: float | None
    mean_first_verdict_s: float | None
    pending: bool = False


def _rate(flags: list[bool]) -> float | None:
    return float(np.mean(flags)) if flags else None


def evaluate_judge(judge: Judge, examples: list[Example]) -> Metrics:
    """Run the judge on every example and compute the Task 8 metrics."""
    verdicts: list[tuple[Example, Verdict]] = []
    for ex in examples:
        verdict = judge.judge(ex.target, ex.seq, ex.timestamps_s)
        if verdict is None:
            return Metrics(None, None, None, None, None, None, pending=True)
        verdicts.append((ex, verdict))

    def passes(kind: str) -> list[bool]:
        return [v.passed for e, v in verdicts if e.kind == kind]

    perturbed = [(e, v) for e, v in verdicts if e.kind == "perturbed"]
    caught = [(e, v) for e, v in perturbed if not v.passed]
    first_times = [v.first_verdict_s for _, v in verdicts if v.first_verdict_s is not None]
    return Metrics(
        correct_pass_rate=_rate(passes("correct")),
        false_green_wrong=_rate(passes("wrong_sign")),
        false_green_minimal=_rate(passes("minimal_pair")),
        perturbation_caught=_rate([not v.passed for _, v in perturbed]),
        part_named=_rate([e.perturbed_part in v.failed_parts for e, v in caught]),
        mean_first_verdict_s=float(np.mean(first_times)) if first_times else None,
    )
