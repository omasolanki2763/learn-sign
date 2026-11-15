"""MediaPipe Tasks video extractor producing normalized landmark sequences."""

from pathlib import Path
from typing import Any

import numpy as np

from learnsign.features.normalize import normalize_frame
from learnsign.features.schema import FEATURES_PER_FRAME


def extract_tasks_from_landmarks(
    frames_landmarks: list[dict[str, Any]],
    aspect_ratio: float = 1.0,
) -> np.ndarray:
    """Convert a sequence of raw MediaPipe Tasks frame detections into normalized (T, 144) features.

    Args:
        frames_landmarks: List of dicts per frame, each containing 'left_hand' (21x3 or None),
            'right_hand' (21x3 or None), and 'pose' (33x3 or None).
        aspect_ratio: Video frame width / height ratio.

    Returns:
        float32 array of shape (T, 144).
    """
    num_frames = len(frames_landmarks)
    seq = np.empty((num_frames, FEATURES_PER_FRAME), dtype=np.float32)

    for i, frame_data in enumerate(frames_landmarks):
        lh = frame_data.get("left_hand")
        rh = frame_data.get("right_hand")
        pose = frame_data.get("pose")
        seq[i] = normalize_frame(lh, rh, pose, aspect_ratio=aspect_ratio)

    return seq


def extract_tasks(video_path: str | Path, aspect_ratio: float = 1.0) -> np.ndarray:
    """Extract normalized landmarks from an MP4/video file using MediaPipe Tasks Hand + Pose.

    Designed for unattended execution in Kaggle environments.

    Args:
        video_path: Path to video file.
        aspect_ratio: Video frame width / height ratio (default 1.0).

    Returns:
        float32 NumPy array of shape (T, 144).
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    # Note: cv2 and mediapipe are lazy-imported so tests can run without full opencv video decoding
    try:
        import cv2
        import mediapipe as mp
    except ImportError as err:
        raise ImportError(
            f"opencv-python and mediapipe are required for extract_tasks: {err}"
        ) from err

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {path}")

    frames_landmarks: list[dict[str, Any]] = []

    # Initialize MediaPipe Solutions / Tasks API
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    with (
        mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
        ) as hands,
        mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
        ) as pose_detector,
    ):
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(rgb)
            pose_results = pose_detector.process(rgb)

            lh: np.ndarray | None = None
            rh: np.ndarray | None = None
            pose_arr: np.ndarray | None = None

            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                for h_lms, h_meta in zip(
                    hand_results.multi_hand_landmarks,
                    hand_results.multi_handedness,
                    strict=False,
                ):
                    label = h_meta.classification[0].label.lower()
                    h_pts = np.array(
                        [[lm.x, lm.y, lm.z] for lm in h_lms.landmark], dtype=np.float32
                    )
                    if "left" in label:
                        lh = h_pts
                    elif "right" in label:
                        rh = h_pts

            if pose_results.pose_landmarks:
                pose_arr = np.array(
                    [[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark],
                    dtype=np.float32,
                )

            frames_landmarks.append({"left_hand": lh, "right_hand": rh, "pose": pose_arr})

    cap.release()
    return extract_tasks_from_landmarks(frames_landmarks, aspect_ratio=aspect_ratio)
