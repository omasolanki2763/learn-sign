"""MediaPipe Tasks video extractor producing normalized landmark sequences."""

from pathlib import Path
from typing import Any, Final

import numpy as np

from learnsign.features.normalize import normalize_frame
from learnsign.features.schema import COORDINATES_PER_POINT, FEATURES_PER_FRAME, HAND_POINTS

# Matches web/src/landmarks/landmarkWorker.ts (`SWAP_HANDEDNESS_LABELS`).
# MediaPipe HandLandmarker documents handedness assuming a mirrored ("selfie")
# input image; un-mirrored frames may swap Left/Right depending on parity tests.
SWAP_HANDEDNESS_LABELS: Final[bool] = False

MAX_HANDS: Final[int] = 2
NUM_POSES: Final[int] = 1
FULL_POSE_POINTS: Final[int] = 33
DEFAULT_FPS: Final[float] = 30.0


def _landmarks_to_array(landmarks: Any, expected_count: int) -> np.ndarray:
    """Convert a sequence of MediaPipe NormalizedLandmark objects into a float32 array."""
    rows = []
    for lm in landmarks[:expected_count]:
        z_val = getattr(lm, "z", None)
        rows.append([lm.x, lm.y, 0.0 if z_val is None else z_val])
    return np.array(rows, dtype=np.float32)


def assign_hands(
    hand_landmarks_list: Any,
    handedness_list: Any,
    swap_labels: bool = SWAP_HANDEDNESS_LABELS,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Map MediaPipe Tasks hand detections onto anatomical (left_hand, right_hand) arrays.

    Mirrors `assignHands` in `web/src/landmarks/landmarkWorker.ts`: when two detections
    report the same handedness label, the higher-confidence score wins rather than
    letting list order decide.
    """
    best_left: tuple[float, np.ndarray] | None = None
    best_right: tuple[float, np.ndarray] | None = None

    for lms, categories in zip(hand_landmarks_list, handedness_list, strict=False):
        if len(lms) < HAND_POINTS:
            continue
        category = categories[0] if categories else None
        label = (getattr(category, "category_name", "") or "Right").lower()
        score = float(getattr(category, "score", 0.0))

        reported_left = label == "left"
        is_left = (not reported_left) if swap_labels else reported_left
        hand_pts = _landmarks_to_array(lms, HAND_POINTS)

        if is_left:
            if best_left is None or score > best_left[0]:
                best_left = (score, hand_pts)
        else:
            if best_right is None or score > best_right[0]:
                best_right = (score, hand_pts)

    return (
        best_left[1] if best_left is not None else None,
        best_right[1] if best_right is not None else None,
    )


def extract_tasks_from_landmarks(
    frames_landmarks: list[dict[str, Any]],
    aspect_ratio: float = 1.0,
) -> np.ndarray:
    """Convert raw MediaPipe Tasks frame detections into normalized (T, F) features.

    Args:
        frames_landmarks: List of dicts per frame, each containing 'left_hand' (21x3 or None),
            'right_hand' (21x3 or None), and 'pose' (33x3 or None).
        aspect_ratio: Video frame width / height ratio.

    Returns:
        float32 array of shape (T, FEATURES_PER_FRAME).
    """
    num_frames = len(frames_landmarks)
    seq = np.empty((num_frames, FEATURES_PER_FRAME), dtype=np.float32)

    for i, frame_data in enumerate(frames_landmarks):
        lh = frame_data.get("left_hand")
        rh = frame_data.get("right_hand")
        pose = frame_data.get("pose")
        seq[i] = normalize_frame(lh, rh, pose, aspect_ratio=aspect_ratio)

    return seq


def extract_tasks(
    video_path: str | Path,
    hand_model_path: str | Path,
    pose_model_path: str | Path,
    aspect_ratio: float = 1.0,
    swap_handedness_labels: bool = SWAP_HANDEDNESS_LABELS,
) -> np.ndarray:
    """Extract normalized landmarks from a video using MediaPipe Tasks Hand + Pose.

    Uses `mediapipe.tasks.python.vision` `HandLandmarker` and `PoseLandmarker` in
    `RunningMode.VIDEO` with strictly increasing millisecond timestamps, matching the
    browser inference worker and `.task` model bundles.

    Args:
        video_path: Path to video file.
        hand_model_path: Path to `hand_landmarker.task` bundle.
        pose_model_path: Path to `pose_landmarker_lite.task` (or full) bundle.
        aspect_ratio: Video frame width / height ratio (default 1.0).
        swap_handedness_labels: Override `SWAP_HANDEDNESS_LABELS` for parity testing.

    Returns:
        float32 NumPy array of shape (T, FEATURES_PER_FRAME).
    """
    path = Path(video_path)
    hand_model = Path(hand_model_path)
    pose_model = Path(pose_model_path)

    for required_path, kind in (
        (path, "Video file"),
        (hand_model, "HandLandmarker model"),
        (pose_model, "PoseLandmarker model"),
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"{kind} not found: {required_path}")

    # Lazy-import cv2 and mediapipe so lightweight tests run without native runtimes installed
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as err:
        raise ImportError(
            f"opencv-python and mediapipe are required for extract_tasks: {err}"
        ) from err

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {path}")

    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(hand_model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(pose_model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=NUM_POSES,
    )

    frames_landmarks: list[dict[str, Any]] = []
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0.0:
            fps = DEFAULT_FPS

        with (
            vision.HandLandmarker.create_from_options(hand_options) as hands,
            vision.PoseLandmarker.create_from_options(pose_options) as pose_detector,
        ):
            frame_idx = 0
            prev_timestamp_ms = -1
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                candidate_ms = int(round(frame_idx * 1000.0 / fps))
                # MediaPipe VIDEO mode raises if timestamps are non-increasing
                timestamp_ms = max(candidate_ms, prev_timestamp_ms + 1)
                prev_timestamp_ms = timestamp_ms
                frame_idx += 1

                hand_result = hands.detect_for_video(mp_image, timestamp_ms)
                pose_result = pose_detector.detect_for_video(mp_image, timestamp_ms)

                lh, rh = assign_hands(
                    hand_result.hand_landmarks or [],
                    hand_result.handedness or [],
                    swap_labels=swap_handedness_labels,
                )

                pose_arr: np.ndarray | None = None
                if (
                    pose_result.pose_landmarks
                    and len(pose_result.pose_landmarks[0]) >= FULL_POSE_POINTS
                ):
                    pose_arr = _landmarks_to_array(
                        pose_result.pose_landmarks[0], FULL_POSE_POINTS
                    ).reshape((FULL_POSE_POINTS, COORDINATES_PER_POINT))

                frames_landmarks.append({"left_hand": lh, "right_hand": rh, "pose": pose_arr})
    finally:
        cap.release()

    return extract_tasks_from_landmarks(frames_landmarks, aspect_ratio=aspect_ratio)
