"""Unit tests for MediaPipe Tasks video landmark extractor."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from learnsign.features.schema import (
    FEATURES_PER_FRAME,
    HAND_POINTS,
    LEFT_HAND_END,
    LEFT_HAND_START,
    POSE_END,
    POSE_START,
    RIGHT_HAND_END,
    RIGHT_HAND_START,
)
from learnsign.parity.extract_tasks import (
    SWAP_HANDEDNESS_LABELS,
    assign_hands,
    extract_tasks,
    extract_tasks_from_landmarks,
)


def test_swap_handedness_labels_constant_mirrors_browser() -> None:
    """Assert module exposes SWAP_HANDEDNESS_LABELS defaulting to False."""
    assert isinstance(SWAP_HANDEDNESS_LABELS, bool)
    assert SWAP_HANDEDNESS_LABELS is False


def test_extract_tasks_missing_files_raise(tmp_path: Path) -> None:
    """Assert missing video or missing .task model files raise FileNotFoundError."""
    video = tmp_path / "clip.mp4"
    hand_model = tmp_path / "hand_landmarker.task"
    pose_model = tmp_path / "pose_landmarker_lite.task"

    with pytest.raises(FileNotFoundError):
        extract_tasks(video, hand_model, pose_model)

    video.write_bytes(b"dummy")
    with pytest.raises(FileNotFoundError):
        extract_tasks(video, hand_model, pose_model)

    hand_model.write_bytes(b"hand")
    with pytest.raises(FileNotFoundError):
        extract_tasks(video, hand_model, pose_model)


_HandOutputs = list[tuple[list[list[tuple[str, float]]], list[list[tuple[float, float, float]]]]]


def _install_fake_mediapipe(
    monkeypatch: pytest.MonkeyPatch,
    hand_outputs: _HandOutputs,
    pose_outputs: list[list[list[tuple[float, float, float]]]],
    fps: float = 30.0,
) -> dict[str, list]:
    """Install fake mediapipe.tasks.python.vision (no mp.solutions) and cv2.VideoCapture."""
    recorded: dict[str, list] = {
        "hand_options": [],
        "pose_options": [],
        "hand_timestamps": [],
        "pose_timestamps": [],
        "released": [],
    }

    class _RunningMode:
        IMAGE = "IMAGE"
        VIDEO = "VIDEO"
        LIVE_STREAM = "LIVE_STREAM"

    class _ImageFormat:
        SRGB = "SRGB"

    class _FakeImage:
        def __init__(self, image_format: str, data: np.ndarray) -> None:
            self.image_format = image_format
            self.data = data

    class _BaseOptions:
        def __init__(self, model_asset_path: str) -> None:
            self.model_asset_path = model_asset_path

    class _HandLandmarkerOptions:
        def __init__(self, base_options: object, running_mode: str, num_hands: int = 2) -> None:
            self.base_options = base_options
            self.running_mode = running_mode
            self.num_hands = num_hands

    class _PoseLandmarkerOptions:
        def __init__(self, base_options: object, running_mode: str, num_poses: int = 1) -> None:
            self.base_options = base_options
            self.running_mode = running_mode
            self.num_poses = num_poses

    def _make_lm(x: float, y: float, z: float) -> SimpleNamespace:
        return SimpleNamespace(x=x, y=y, z=z)

    class _FakeHandLandmarker:
        def __init__(self, options: _HandLandmarkerOptions) -> None:
            self.options = options
            self._call_idx = 0

        @classmethod
        def create_from_options(cls, options: _HandLandmarkerOptions) -> "_FakeHandLandmarker":
            recorded["hand_options"].append(options)
            return cls(options)

        def __enter__(self) -> "_FakeHandLandmarker":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def detect_for_video(self, image: _FakeImage, timestamp_ms: int) -> SimpleNamespace:
            assert image.image_format == _ImageFormat.SRGB
            recorded["hand_timestamps"].append(timestamp_ms)
            handed_meta, lm_lists = hand_outputs[self._call_idx]
            self._call_idx += 1
            handedness = [
                [SimpleNamespace(category_name=name, score=score) for name, score in cats]
                for cats in handed_meta
            ]
            hand_landmarks = [[_make_lm(*pt) for pt in lms] for lms in lm_lists]
            return SimpleNamespace(handedness=handedness, hand_landmarks=hand_landmarks)

    class _FakePoseLandmarker:
        def __init__(self, options: _PoseLandmarkerOptions) -> None:
            self.options = options
            self._call_idx = 0

        @classmethod
        def create_from_options(cls, options: _PoseLandmarkerOptions) -> "_FakePoseLandmarker":
            recorded["pose_options"].append(options)
            return cls(options)

        def __enter__(self) -> "_FakePoseLandmarker":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def detect_for_video(self, image: _FakeImage, timestamp_ms: int) -> SimpleNamespace:
            assert image.image_format == _ImageFormat.SRGB
            recorded["pose_timestamps"].append(timestamp_ms)
            pose_lists = pose_outputs[self._call_idx]
            self._call_idx += 1
            pose_landmarks = [[_make_lm(*pt) for pt in lms] for lms in pose_lists]
            return SimpleNamespace(pose_landmarks=pose_landmarks)

    fake_vision = SimpleNamespace(
        RunningMode=_RunningMode,
        HandLandmarker=_FakeHandLandmarker,
        HandLandmarkerOptions=_HandLandmarkerOptions,
        PoseLandmarker=_FakePoseLandmarker,
        PoseLandmarkerOptions=_PoseLandmarkerOptions,
    )
    fake_tasks_python = SimpleNamespace(BaseOptions=_BaseOptions, vision=fake_vision)
    fake_tasks = SimpleNamespace(python=fake_tasks_python)
    # Intentionally omit `solutions` so any legacy mp.solutions call raises AttributeError
    fake_mp = SimpleNamespace(
        Image=_FakeImage,
        ImageFormat=_ImageFormat,
        tasks=fake_tasks,
    )

    monkeypatch.setitem(sys.modules, "mediapipe", fake_mp)
    monkeypatch.setitem(sys.modules, "mediapipe.tasks", fake_tasks)
    monkeypatch.setitem(sys.modules, "mediapipe.tasks.python", fake_tasks_python)
    monkeypatch.setitem(sys.modules, "mediapipe.tasks.python.vision", fake_vision)

    class _FakeCapture:
        def __init__(self, _path: str) -> None:
            self._idx = 0
            self._total = len(pose_outputs)

        def isOpened(self) -> bool:
            return True

        def get(self, _prop: int) -> float:
            return fps

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self._idx >= self._total:
                return False, None
            self._idx += 1
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

        def release(self) -> None:
            recorded["released"].append(True)

    fake_cv2 = SimpleNamespace(
        CAP_PROP_FPS=5,
        COLOR_BGR2RGB=4,
        VideoCapture=_FakeCapture,
        cvtColor=lambda frame, _code: frame,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    return recorded


def test_extract_tasks_uses_mediapipe_tasks_video_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assert extract_tasks uses Tasks VIDEO mode and strictly increasing timestamps."""
    video = tmp_path / "clip.mp4"
    hand_model = tmp_path / "hand_landmarker.task"
    pose_model = tmp_path / "pose_landmarker_lite.task"
    video.write_bytes(b"v")
    hand_model.write_bytes(b"h")
    pose_model.write_bytes(b"p")

    pose_33 = [(0.5, 0.5, 0.0) for _ in range(33)]
    pose_33[11] = (0.6, 0.4, 0.0)  # left shoulder
    pose_33[12] = (0.2, 0.4, 0.0)  # right shoulder => midpoint (0.4, 0.4), width 0.4

    lh_21 = [(0.6, 0.8, 0.1) for _ in range(HAND_POINTS)]
    rh_21 = [(0.2, 0.8, -0.1) for _ in range(HAND_POINTS)]

    hand_outputs = [
        ([[("Left", 0.95)], [("Right", 0.90)]], [lh_21, rh_21]),
        ([[("Right", 0.88)]], [rh_21]),
        ([], []),
    ]
    pose_outputs = [[pose_33], [pose_33], [pose_33]]

    # Use excessively high FPS (2000) to ensure timestamp monotonicity guard triggers
    recorded = _install_fake_mediapipe(monkeypatch, hand_outputs, pose_outputs, fps=2000.0)

    seq = extract_tasks(video, hand_model, pose_model, aspect_ratio=1.0)

    assert seq.shape == (3, FEATURES_PER_FRAME)
    assert seq.dtype == np.float32
    assert len(recorded["released"]) == 1

    # Verify Tasks API configuration
    hand_opt = recorded["hand_options"][0]
    pose_opt = recorded["pose_options"][0]
    assert hand_opt.running_mode == "VIDEO"
    assert hand_opt.base_options.model_asset_path == str(hand_model)
    assert pose_opt.running_mode == "VIDEO"
    assert pose_opt.base_options.model_asset_path == str(pose_model)

    # Strictly increasing integer timestamps
    for ts_list in (recorded["hand_timestamps"], recorded["pose_timestamps"]):
        assert len(ts_list) == 3
        assert all(isinstance(t, int) for t in ts_list)
        assert all(ts_list[i] > ts_list[i - 1] for i in range(1, len(ts_list)))

    # Frame 0: both hands present; frame 1: only right hand; frame 2: neither hand
    assert not np.isnan(seq[0, LEFT_HAND_START:LEFT_HAND_END]).any()
    assert not np.isnan(seq[0, RIGHT_HAND_START:RIGHT_HAND_END]).any()
    assert np.isnan(seq[1, LEFT_HAND_START:LEFT_HAND_END]).all()
    assert not np.isnan(seq[1, RIGHT_HAND_START:RIGHT_HAND_END]).any()
    assert np.isnan(seq[2, LEFT_HAND_START:RIGHT_HAND_END]).all()
    assert not np.isnan(seq[2, POSE_START:POSE_END]).any()


def test_assign_hands_confidence_tiebreak_and_swap() -> None:
    """Assert assign_hands keeps highest-scoring duplicate and respects swap_labels."""
    low_left = SimpleNamespace(category_name="Left", score=0.4)
    high_left = SimpleNamespace(category_name="Left", score=0.92)
    right = SimpleNamespace(category_name="Right", score=0.85)

    lms_low = [SimpleNamespace(x=0.1, y=0.1, z=0.0)] * HAND_POINTS
    lms_high = [SimpleNamespace(x=0.9, y=0.9, z=0.0)] * HAND_POINTS
    lms_right = [SimpleNamespace(x=0.5, y=0.5, z=0.0)] * HAND_POINTS

    lh, rh = assign_hands(
        [lms_low, lms_high, lms_right],
        [[low_left], [high_left], [right]],
        swap_labels=False,
    )
    assert lh is not None and rh is not None
    np.testing.assert_allclose(lh[0], [0.9, 0.9, 0.0])
    np.testing.assert_allclose(rh[0], [0.5, 0.5, 0.0])

    # Swap True flips Left <-> Right
    lh_swapped, rh_swapped = assign_hands(
        [lms_high],
        [[high_left]],
        swap_labels=True,
    )
    assert lh_swapped is None
    assert rh_swapped is not None
    np.testing.assert_allclose(rh_swapped[0], [0.9, 0.9, 0.0])


def test_extract_tasks_nonpositive_fps_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assert extract_tasks falls back cleanly when CAP_PROP_FPS returns 0."""
    video = tmp_path / "clip.mp4"
    hand_model = tmp_path / "hand_landmarker.task"
    pose_model = tmp_path / "pose_landmarker_lite.task"
    video.write_bytes(b"v")
    hand_model.write_bytes(b"h")
    pose_model.write_bytes(b"p")

    pose_33 = [(0.5, 0.5, 0.0) for _ in range(33)]
    pose_33[11] = (0.6, 0.4, 0.0)
    pose_33[12] = (0.2, 0.4, 0.0)

    recorded = _install_fake_mediapipe(
        monkeypatch,
        [([], []), ([], [])],
        [[pose_33], [pose_33]],
        fps=0.0,
    )
    seq = extract_tasks(video, hand_model, pose_model)
    assert seq.shape == (2, FEATURES_PER_FRAME)
    assert recorded["hand_timestamps"][1] > recorded["hand_timestamps"][0] >= 0


def test_extract_tasks_from_landmarks_shape() -> None:
    """Verify extract_tasks_from_landmarks outputs schema-sized (T, FEATURES_PER_FRAME) array."""
    pose = np.full((33, 3), np.nan, dtype=np.float32)
    pose[11] = [0.6, 0.4, 0.0]
    pose[12] = [0.2, 0.4, 0.0]
    seq = extract_tasks_from_landmarks([{"left_hand": None, "right_hand": None, "pose": pose}])
    assert seq.shape == (1, FEATURES_PER_FRAME)
    assert seq.dtype == np.float32
