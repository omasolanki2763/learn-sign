"""Unattended Task 8 run on GISLR data (designed for a Kaggle notebook in commit mode).

Usage (Kaggle):
    python -m learnsign.verify.run_spike \\
        --data-dir /kaggle/input/asl-signs \\
        --out reports/step0/verifier.md \\
        [--checkpoint /kaggle/working/quick_baseline.pt]

What it does:
1. Signer-independent split (Task 5 `signer_split`): references come from training signers,
   every scored attempt comes from held-out signers.
2. Calibrates the two eye-fallback proportions on raw pose frames.
3. Builds the examples: correct attempts, wrong signs, minimal pairs, and synthetic wrong
   signs (location shift, handshape swap, reversed movement, missing second hand).
4. Scores candidates (a) ranges, (b) specs [pending until Task 9], (c) model, (d) combined.
5. Writes a markdown report with the comparison table.

Assumption stated in the report: GISLR clips carry no timestamps; frames are taken as 30 fps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from learnsign.data.gislr import load_sequence
from learnsign.data.split import signer_split
from learnsign.verify.calibrate import calibrate_eye_ratios
from learnsign.verify.evaluate import Example, Metrics, evaluate_judge
from learnsign.verify.judges import (
    CombinedJudge,
    Judge,
    ModelJudge,
    RangesJudge,
    SpecJudge,
    load_quick_baseline_predictor,
)
from learnsign.verify.perturb import (
    drop_second_hand,
    reverse_movement,
    shift_location,
    swap_handshape,
)
from learnsign.verify.signs import load_step0_signs, validate_glosses

ASSUMED_FPS: Final[float] = 30.0
REFERENCES_PER_SIGN: Final[int] = 40
HELD_OUT_PER_SIGN: Final[int] = 10
VARIANTS_PER_SIGN: Final[int] = 3
CALIBRATION_CLIPS: Final[int] = 200
# Size of the synthetic location error (shoulder widths), about chin -> forehead.
LOCATION_SHIFT: Final[float] = 0.4
SEED: Final[int] = 42
POSE_POINTS_RAW: Final[int] = 33


def _timestamps(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / ASSUMED_FPS


def _load(data_dir: Path, rel_path: str) -> tuple[np.ndarray, np.ndarray]:
    seq = load_sequence(data_dir / rel_path, canonicalize=True, cache_dir=Path("cache"))
    return seq, _timestamps(len(seq))


def _raw_pose(data_dir: Path, rel_path: str) -> np.ndarray:
    """Raw (T, 33, 3) pose landmarks of one clip, for calibration."""
    cols = ["frame", "type", "landmark_index", "x", "y", "z"]
    df = pq.read_table(data_dir / rel_path, columns=cols).to_pandas()
    pose = df[df["type"] == "pose"].sort_values(["frame", "landmark_index"])
    return pose[["x", "y", "z"]].to_numpy(dtype=np.float32).reshape(-1, POSE_POINTS_RAW, 3)


def build_examples(
    data_dir: Path, held: pd.DataFrame, signs: dict
) -> list[Example]:
    """Correct, wrong-sign, minimal-pair and perturbed attempts from held-out signers."""
    rng = np.random.default_rng(SEED)
    glosses = signs["signs"]
    clips: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for gloss in glosses:
        rows = held[held["sign"] == gloss].head(HELD_OUT_PER_SIGN)
        clips[gloss] = [_load(data_dir, p) for p in rows["path"]]

    examples: list[Example] = []
    pair_of: dict[str, list[str]] = {}
    for pair in signs["minimal_pairs"]:
        a, b = pair["signs"]
        pair_of.setdefault(a, []).append(b)
        pair_of.setdefault(b, []).append(a)

    for gloss in glosses:
        for seq, ts in clips[gloss]:
            examples.append(Example(gloss, seq, ts, "correct"))
            others = [g for g in glosses if g != gloss and clips[g]]
            if others:
                wrong = str(rng.choice(others))
                wseq, wts = clips[wrong][0]
                examples.append(Example(gloss, wseq, wts, "wrong_sign"))
            examples.append(
                Example(
                    gloss,
                    shift_location(seq, 0.0, -LOCATION_SHIFT),
                    ts,
                    "perturbed",
                    "location",
                )
            )
            examples.append(Example(gloss, reverse_movement(seq), ts, "perturbed", "movement"))
            donor_gloss = others[0] if others else None
            if donor_gloss:
                examples.append(
                    Example(
                        gloss,
                        swap_handshape(seq, clips[donor_gloss][0][0]),
                        ts,
                        "perturbed",
                        "handshape",
                    )
                )
            if gloss in signs["two_handed"]:
                examples.append(Example(gloss, drop_second_hand(seq), ts, "perturbed", "hands"))
        for partner in pair_of.get(gloss, []):
            for pseq, pts in clips[partner]:
                examples.append(Example(gloss, pseq, pts, "minimal_pair"))
    return examples


def _fmt(value: float | None) -> str:
    return "pending" if value is None else f"{value:.3f}"


def format_report(results: dict[str, Metrics], calibration: dict, notes: list[str]) -> str:
    """Markdown report: one row per candidate judge plus the calibration result."""
    lines = [
        "# Step 0 Task 8 - live verifier spike",
        "",
        "| Judge | Correct pass | False-green (wrong) | False-green (minimal pair) | "
        "Perturbation caught | Right part named | First verdict (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in results.items():
        if m.pending:
            lines.append(f"| {name} | pending | pending | pending | pending | pending | pending |")
            continue
        lines.append(
            f"| {name} | {_fmt(m.correct_pass_rate)} | {_fmt(m.false_green_wrong)} | "
            f"{_fmt(m.false_green_minimal)} | {_fmt(m.perturbation_caught)} | "
            f"{_fmt(m.part_named)} | {_fmt(m.mean_first_verdict_s)} |"
        )
    lines += [
        "",
        "Targets: correct pass >= 0.90, false-green <= 0.05, perturbation caught >= 0.90, "
        "right part named >= 0.80.",
        "",
        "## Eye-fallback calibration",
        "",
        "```json",
        json.dumps(calibration, indent=2),
        "```",
        "",
        "## Notes",
        "",
    ]
    lines += [f"- {n}" for n in notes]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/step0/verifier.md"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    signs = load_step0_signs()
    missing = validate_glosses(train["sign"].unique())
    if missing:
        raise SystemExit(f"Task 8 signs missing from the dataset: {missing}")

    train_df, held_df = signer_split(train)
    notes = [
        f"References: training signers only, up to {REFERENCES_PER_SIGN} clips per sign; "
        f"scored attempts: held-out signers ({held_df['participant_id'].nunique()}).",
        f"Frame rate assumed {ASSUMED_FPS:.0f} fps (GISLR has no timestamps).",
    ]

    calib_rows = train_df.sample(n=min(CALIBRATION_CLIPS, len(train_df)), random_state=SEED)
    calibration = calibrate_eye_ratios(
        np.concatenate([_raw_pose(args.data_dir, p) for p in calib_rows["path"]])
    )

    references = {
        g: [
            _load(args.data_dir, p)
            for p in train_df[train_df["sign"] == g].head(REFERENCES_PER_SIGN)["path"]
        ]
        for g in signs["signs"]
    }
    examples = build_examples(args.data_dir, held_df, signs)

    ranges = RangesJudge(references, k=VARIANTS_PER_SIGN)
    candidates: dict[str, Judge] = {"(a) data ranges": ranges, "(b) LLM specs": SpecJudge()}
    if args.checkpoint is not None and args.checkpoint.exists():
        gloss_names = sorted(train["sign"].unique())
        model = ModelJudge(load_quick_baseline_predictor(args.checkpoint, gloss_names))
        candidates["(c) trained model"] = model
        candidates["(d) ranges + model"] = CombinedJudge(ranges, model)
    else:
        notes.append("No checkpoint given: candidates (c) and (d) not run.")

    results = {name: evaluate_judge(j, examples) for name, j in candidates.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(format_report(results, calibration, notes), encoding="utf-8")
    print(args.out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
