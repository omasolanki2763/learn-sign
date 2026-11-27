"""Tests for Step 0 Task 3 dummy ONNX model export and onnxruntime validation."""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from learnsign.bench.export_dummy_onnx import (
    DEFAULT_BENCH_FEATURES,
    DEFAULT_NUM_CLASSES,
    DEFAULT_WINDOW_FRAMES,
    MAX_EXPECTED_PARAMS,
    MIN_EXPECTED_PARAMS,
    export_dummy_onnx,
)
from learnsign.features.derived import DERIVED_FEATURES_PER_FRAME
from learnsign.features.schema import FEATURES_PER_FRAME

MAX_COMMITTED_ONNX_BYTES = 5_000_000


def test_default_bench_features_reads_v2_schema() -> None:
    """Verify dummy probe default input dimension uses v2 schema constants, never 144."""
    assert DEFAULT_BENCH_FEATURES == FEATURES_PER_FRAME + DERIVED_FEATURES_PER_FRAME


def test_export_dummy_onnx_param_count_size_and_ort_load(tmp_path: Path) -> None:
    """Assert exported single-file ONNX fits size budget, has 1.7M-2.0M params, and runs."""
    onnx_path = tmp_path / "dummy_model.onnx"
    param_count = export_dummy_onnx(onnx_path)

    assert MIN_EXPECTED_PARAMS <= param_count <= MAX_EXPECTED_PARAMS
    assert onnx_path.exists()
    assert not (tmp_path / "dummy_model.onnx.data").exists()
    assert 0 < onnx_path.stat().st_size <= MAX_COMMITTED_ONNX_BYTES

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sample = np.zeros((1, DEFAULT_WINDOW_FRAMES, DEFAULT_BENCH_FEATURES), dtype=np.float32)
    outputs = session.run(["logits"], {"features": sample})
    assert outputs[0].shape == (1, DEFAULT_NUM_CLASSES)
    assert np.isfinite(outputs[0]).all()
