"""Parity decision gate comparing dataset landmarks vs MediaPipe Tasks re-extraction."""

from typing import Final, Literal

# Parity decision threshold from the Step 0 specification: 3.0 percentage points
PARITY_GAP_POINTS: Final[float] = 3.0

DecisionType = Literal["train_on_existing", "bulk_reextract"]


def compare(
    acc_dataset: float,
    acc_tasks: float,
    threshold_points: float = PARITY_GAP_POINTS,
) -> dict[str, float | str]:
    """Compare model validation accuracy on original dataset vs re-extracted landmarks.

    Decision Rule (Step 0 Spec):
    - Accuracy gap <= 3.0 points -> train on existing landmark files.
    - Accuracy gap > 3.0 points -> bulk re-extraction of clips via sharded Kaggle jobs.

    Args:
        acc_dataset: Top-1 validation accuracy on Kaggle dataset landmarks [0.0, 1.0].
        acc_tasks: Top-1 validation accuracy on MediaPipe Tasks re-extracted landmarks [0.0, 1.0].
        threshold_points: Maximum allowable gap in percentage points (default 3.0).

    Returns:
        Dictionary with acc_dataset, acc_tasks, gap_points, and decision.
    """
    gap_points = round(abs(acc_dataset - acc_tasks) * 100.0, 4)

    if gap_points <= threshold_points:
        decision: DecisionType = "train_on_existing"
    else:
        decision = "bulk_reextract"

    return {
        "acc_dataset": acc_dataset,
        "acc_tasks": acc_tasks,
        "gap_points": round(gap_points, 2),
        "decision": decision,
    }
