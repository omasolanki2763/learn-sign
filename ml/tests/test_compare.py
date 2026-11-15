"""Unit tests for the parity experiment decision rule."""

import pytest

from learnsign.parity.compare import PARITY_GAP_POINTS, compare


def test_compare_bulk_reextract_decision() -> None:
    """Step 1 test: Assert compare(0.70, 0.66) produces bulk_reextract (gap 4.0 > 3.0)."""
    result = compare(acc_dataset=0.70, acc_tasks=0.66)

    assert result["acc_dataset"] == 0.70
    assert result["acc_tasks"] == 0.66
    assert result["gap_points"] == pytest.approx(4.0, abs=1e-3)
    assert result["decision"] == "bulk_reextract"


def test_compare_train_on_existing_decision() -> None:
    """Step 1 test: Assert compare(0.70, 0.68) produces train_on_existing (gap 2.0 <= 3.0)."""
    result = compare(acc_dataset=0.70, acc_tasks=0.68)

    assert result["acc_dataset"] == 0.70
    assert result["acc_tasks"] == 0.68
    assert result["gap_points"] == pytest.approx(2.0, abs=1e-3)
    assert result["decision"] == "train_on_existing"


def test_compare_boundary_condition() -> None:
    """Assert exact boundary condition (gap == PARITY_GAP_POINTS) resolves to train_on_existing."""
    acc_dataset = 0.75
    acc_tasks = acc_dataset - (PARITY_GAP_POINTS / 100.0)
    result = compare(acc_dataset=acc_dataset, acc_tasks=acc_tasks)

    assert result["gap_points"] == pytest.approx(PARITY_GAP_POINTS, abs=1e-3)
    assert result["decision"] == "train_on_existing"


def test_compare_tasks_higher_than_dataset() -> None:
    """Assert compare(0.66, 0.70) produces train_on_existing with negative signed gap (-4.0 pts)."""
    result = compare(acc_dataset=0.66, acc_tasks=0.70)

    assert result["acc_dataset"] == 0.66
    assert result["acc_tasks"] == 0.70
    assert result["gap_points"] == pytest.approx(-4.0, abs=1e-3)
    assert result["decision"] == "train_on_existing"

