"""Candidate live judges compared in Task 8 (methodology M9).

Every judge answers the same question for one attempt: "is this the target sign, and if not,
which part is wrong?". Candidates:
 (a) RangesJudge  - data-only ranges measured from real signers (this module).
 (b) SpecJudge    - LLM-written declarative specs; pending until Task 9 delivers them.
 (c) ModelJudge   - the small trained model from Task 5 (whole-sign answer, no part named).
 (d) CombinedJudge - (a or b) together with (c): both must agree, the rules name the part.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Protocol

import numpy as np

from learnsign.features.derived import derived_sequence
from learnsign.verify.align import OpenEndAligner
from learnsign.verify.onset import OnsetGate
from learnsign.verify.parts import alignment_features_from_derived, second_hand_visible
from learnsign.verify.ranges import SignModel, build_sign_model

# Seconds between verdict ticks (methodology: a verdict every 100 ms).
TICK_PERIOD_S: Final[float] = 0.1
# A part fails the attempt if it is off on more than this fraction of ticks.
PART_TICK_FAIL_FRACTION: Final[float] = 0.3
# Part name used when the learner never started signing.
NO_SIGN: Final[str] = "no_sign"
# Part name for "the model recognised a different sign" (no specific part known).
WHOLE_SIGN: Final[str] = "sign"
# Part name for a missing second hand on a two-handed sign.
HANDS: Final[str] = "hands"


@dataclass
class Verdict:
    """Outcome of one attempt."""

    passed: bool
    failed_parts: list[str] = field(default_factory=list)
    first_verdict_s: float | None = None  # seconds from onset to the first tick
    ticks: int = 0


class Judge(Protocol):
    """Anything that can judge one attempt; None means 'not available yet'."""

    def judge(self, target: str, seq: np.ndarray, timestamps_s: np.ndarray) -> Verdict | None:
        """Judge one attempt at `target`."""
        ...


class RangesJudge:
    """Candidate (a): streaming check against ranges measured from real signers."""

    def __init__(
        self, references: dict[str, list[tuple[np.ndarray, np.ndarray]]], k: int = 1
    ) -> None:
        """Build one SignModel per gloss.

        Args:
            references: gloss -> list of (normalised sequence, timestamps) from TRAINING
                signers only (held-out signers are for scoring).
            k: Variants per sign (the notebook uses 3 on real data).
        """
        self.models: dict[str, SignModel] = {
            gloss: build_sign_model(refs, k=k) for gloss, refs in references.items()
        }

    def judge(self, target: str, seq: np.ndarray, timestamps_s: np.ndarray) -> Verdict:
        """Replay the attempt frame by frame, exactly as the browser will see it."""
        model = self.models[target]
        derived = derived_sequence(seq, timestamps_s=timestamps_s)
        feats = alignment_features_from_derived(derived)

        gate = OnsetGate()
        aligners: list[OpenEndAligner] | None = None
        onset_s: float | None = None
        last_tick_s = -np.inf
        ticks = 0
        off_counts: dict[str, int] = {}
        first_verdict_s: float | None = None

        for i in range(len(seq)):
            t = float(timestamps_s[i])
            if not gate.update(seq[i], t):
                continue
            if aligners is None:
                aligners = [OpenEndAligner(v.medoid_alignment) for v in model.variants]
                onset_s = t
            states = [a.step(feats[i]) for a in aligners]
            if t - last_tick_s < TICK_PERIOD_S - 1e-9:
                continue
            last_tick_s = t
            ticks += 1
            if first_verdict_s is None and onset_s is not None:
                first_verdict_s = t - onset_s
            best = int(np.argmin([s.cost for s in states]))
            variant = model.variants[best]
            off = variant.off_parts(derived[i], variant.phase_of(states[best].ref_index))
            if model.needs_second_hand and not second_hand_visible(derived[i]):
                off.add(HANDS)
            for part in off:
                off_counts[part] = off_counts.get(part, 0) + 1

        if ticks == 0:
            return Verdict(passed=False, failed_parts=[NO_SIGN])
        failed = sorted(p for p, n in off_counts.items() if n / ticks > PART_TICK_FAIL_FRACTION)
        return Verdict(
            passed=not failed, failed_parts=failed, first_verdict_s=first_verdict_s, ticks=ticks
        )


class SpecJudge:
    """Candidate (b): LLM-authored declarative specs (Task 9). Pending until specs exist."""

    def judge(self, target: str, seq: np.ndarray, timestamps_s: np.ndarray) -> Verdict | None:
        """Return None: no specs have been written yet."""
        return None


Predictor = Callable[[np.ndarray, np.ndarray], str]


class ModelJudge:
    """Candidate (c): a trained whole-sign classifier; passes when it predicts the target."""

    def __init__(self, predict: Predictor) -> None:
        self.predict = predict

    def judge(self, target: str, seq: np.ndarray, timestamps_s: np.ndarray) -> Verdict:
        """Pass if the predicted gloss equals the target (the model cannot name a part)."""
        ok = self.predict(seq, timestamps_s) == target
        return Verdict(passed=ok, failed_parts=[] if ok else [WHOLE_SIGN])


class CombinedJudge:
    """Candidate (d): rules and model must both agree; the rules explain what is wrong."""

    def __init__(self, rules: Judge, model: Judge) -> None:
        self.rules = rules
        self.model = model

    def judge(self, target: str, seq: np.ndarray, timestamps_s: np.ndarray) -> Verdict | None:
        """Combine two verdicts; None if either candidate is pending."""
        a = self.rules.judge(target, seq, timestamps_s)
        b = self.model.judge(target, seq, timestamps_s)
        if a is None or b is None:
            return None
        failed = sorted(set(a.failed_parts) | set(b.failed_parts))
        return Verdict(
            passed=a.passed and b.passed,
            failed_parts=failed,
            first_verdict_s=a.first_verdict_s,
            ticks=a.ticks,
        )


def load_quick_baseline_predictor(checkpoint: Path, gloss_names: list[str]) -> Predictor:
    """Wrap a Task 5 quick-baseline checkpoint (a state_dict) as a Predictor.

    Imported lazily so the verifier works without PyTorch installed.
    """
    import torch

    from learnsign.train.quick_baseline import QuickBaseline1DCNN, resample_sequence

    net = QuickBaseline1DCNN(num_classes=len(gloss_names))
    # Why weights_only: a full unpickle can execute code stored in the file; a state_dict
    # only needs tensors, so refuse anything else.
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    net.eval()

    def predict(seq: np.ndarray, timestamps_s: np.ndarray) -> str:
        x = np.nan_to_num(resample_sequence(seq), nan=0.0)
        with torch.no_grad():
            logits = net(torch.from_numpy(x[None].astype(np.float32)))
        return gloss_names[int(logits.argmax(dim=1))]

    return predict
