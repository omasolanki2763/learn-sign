"""Unit tests for judges, evaluation metrics, calibration, and Step 0 sign lists."""

from __future__ import annotations

import math

import numpy as np
import pytest
from verify_synth import jitter, line, make_sequence

from learnsign.verify.calibrate import calibrate_eye_ratios
from learnsign.verify.evaluate import Example, evaluate_judge
from learnsign.verify.judges import (
    CombinedJudge,
    ModelJudge,
    RangesJudge,
    SpecJudge,
    Verdict,
)
from learnsign.verify.perturb import (
    drop_second_hand,
    reverse_movement,
    shift_location,
    swap_handshape,
)
from learnsign.verify.signs import (
    GISLR_250_GLOSSES,
    load_step0_signs,
    validate_glosses,
)

CHIN_PATH = line((0.10, -0.70), (0.35, -0.30))
CHEST_PATH = line((-0.10, 0.30), (0.30, 0.30))


def _refs(path=CHIN_PATH, shape="open", n=6, seed0=0) -> list[tuple[np.ndarray, np.ndarray]]:
    base, ts = make_sequence(path, shape=shape)
    return [(jitter(base, 0.002, seed0 + k), ts) for k in range(n)]


@pytest.fixture(scope="module")
def judge() -> RangesJudge:
    return RangesJudge(
        {"chin_sign": _refs(n=8), "chest_sign": _refs(CHEST_PATH, "fist", n=8, seed0=50)}
    )


def test_ranges_judge_passes_correct_sign(judge: RangesJudge) -> None:
    seq, ts = _refs(n=1, seed0=123)[0]
    verdict = judge.judge("chin_sign", seq, ts)
    assert isinstance(verdict, Verdict)
    assert verdict.passed, verdict.failed_parts


def test_ranges_judge_rejects_wrong_sign(judge: RangesJudge) -> None:
    seq, ts = _refs(CHEST_PATH, "fist", n=1, seed0=321)[0]
    assert not judge.judge("chin_sign", seq, ts).passed


@pytest.mark.parametrize(
    ("part", "make"),
    [
        ("location", lambda s: shift_location(s, dx=0.0, dy=0.6)),
        ("handshape", lambda s: swap_handshape(s, make_sequence(CHEST_PATH, shape="fist")[0])),
        ("movement", reverse_movement),
    ],
)
def test_ranges_judge_names_the_perturbed_part(judge: RangesJudge, part: str, make) -> None:
    seq, ts = _refs(n=1, seed0=7)[0]
    verdict = judge.judge("chin_sign", make(seq), ts)
    assert not verdict.passed
    assert part in verdict.failed_parts


def test_ranges_judge_flags_missing_second_hand() -> None:
    two_handed = [
        (
            jitter(make_sequence(CHIN_PATH, left_path=CHEST_PATH)[0], 0.002, k),
            make_sequence(CHIN_PATH)[1],
        )
        for k in range(6)
    ]
    j = RangesJudge({"two": two_handed})
    seq, ts = two_handed[0]
    verdict = j.judge("two", drop_second_hand(seq), ts)
    assert not verdict.passed
    assert "hands" in verdict.failed_parts


def test_spec_judge_is_pending_until_task_9() -> None:
    seq, ts = make_sequence(CHIN_PATH)
    assert SpecJudge().judge("chin_sign", seq, ts) is None


def test_model_judge_and_combined_judge() -> None:
    def dummy_predict(seq: np.ndarray, ts: np.ndarray) -> str:
        return "chin_sign"

    model_j = ModelJudge(dummy_predict)
    seq, ts = make_sequence(CHIN_PATH)
    v_correct = model_j.judge("chin_sign", seq, ts)
    assert v_correct.passed
    assert v_correct.failed_parts == []

    v_wrong = model_j.judge("other_sign", seq, ts)
    assert not v_wrong.passed
    assert "sign" in v_wrong.failed_parts

    ranges_j = RangesJudge({"chin_sign": _refs(n=6)})
    combined = CombinedJudge(ranges_j, model_j)
    v_comb = combined.judge("chin_sign", seq, ts)
    assert v_comb is not None
    assert v_comb.passed

    combined_pending = CombinedJudge(SpecJudge(), model_j)
    assert combined_pending.judge("chin_sign", seq, ts) is None


def test_evaluate_judge_computes_rates(judge: RangesJudge) -> None:
    correct = _refs(n=2, seed0=200)
    wrong = _refs(CHEST_PATH, "fist", n=1, seed0=300)
    examples = [Example("chin_sign", s, t, "correct") for s, t in correct]
    examples += [Example("chin_sign", s, t, "wrong_sign") for s, t in wrong]
    s, t = correct[0]
    examples.append(
        Example("chin_sign", shift_location(s, dx=0.0, dy=0.6), t, "perturbed", "location")
    )
    m = evaluate_judge(judge, examples)
    assert m.correct_pass_rate == pytest.approx(1.0)
    assert m.false_green_wrong == pytest.approx(0.0)
    assert m.perturbation_caught == pytest.approx(1.0)
    assert m.part_named == pytest.approx(1.0)
    assert not m.pending
    assert m.mean_first_verdict_s is not None and m.mean_first_verdict_s < 1.0


def test_evaluate_marks_pending_judge() -> None:
    seq, ts = make_sequence(CHIN_PATH)
    m = evaluate_judge(SpecJudge(), [Example("chin_sign", seq, ts, "correct")])
    assert m.pending


def test_calibrate_eye_ratios_recovers_known_proportions() -> None:
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(50):
        pose = np.full((33, 3), np.nan, dtype=np.float32)
        eye_dist = 0.05 + rng.uniform(-0.005, 0.005)
        shoulder_w = eye_dist * 6.4
        cx, ey = 0.5, 0.3
        pose[2] = (cx + eye_dist / 2, ey, 0.0)
        pose[5] = (cx - eye_dist / 2, ey, 0.0)
        pose[11] = (cx + shoulder_w / 2, ey + 0.55 * shoulder_w, 0.0)
        pose[12] = (cx - shoulder_w / 2, ey + 0.55 * shoulder_w, 0.0)
        frames.append(pose)
    result = calibrate_eye_ratios(np.stack(frames))
    assert result["eye_to_shoulder_ratio"]["median"] == pytest.approx(6.4, rel=1e-3)
    assert result["eye_to_shoulder_y_offset"]["median"] == pytest.approx(0.55, rel=1e-3)
    assert result["frames_used"] == 50


def test_step0_sign_list_has_20_signs_and_minimal_pairs_inside_it() -> None:
    signs = load_step0_signs()
    assert len(signs["signs"]) == 20
    assert len(set(signs["signs"])) == 20
    for pair in signs["minimal_pairs"]:
        assert all(g in signs["signs"] for g in pair["signs"])
        assert pair["differs_in"] in {"location", "handshape", "movement", "hands"}
    assert math.isfinite(len(signs["two_handed"]))


def test_all_20_signs_exist_in_gislr_250_list() -> None:
    """Verifies that all 20 Task 8 signs exist in the canonical 250-gloss list."""
    missing = validate_glosses(GISLR_250_GLOSSES)
    assert missing == [], f"Task 8 signs missing from 250 list: {missing}"
    assert len(GISLR_250_GLOSSES) == 250
