"""Feature normalization, coordinate standardization, and handedness canonicalization."""

from typing import Any

import numpy as np

from learnsign.features.schema import (
    FEATURES_PER_FRAME,
    HAND_POINTS,
    LEFT_HAND_END,
    LEFT_HAND_START,
    POSE_END,
    POSE_IDS,
    POSE_START,
    RIGHT_HAND_END,
    RIGHT_HAND_START,
)

# Minimum shoulder width threshold to guard against division-by-zero on degenerate frames
MIN_SHOULDER_WIDTH: float = 1e-6


def normalize_frame(
    left_hand: np.ndarray | None,
    right_hand: np.ndarray | None,
    pose: np.ndarray | None,
    aspect_ratio: float = 1.0,
) -> np.ndarray:
    """Normalize a single frame's landmarks into the canonical 144-feature representation.

    Normalizes coordinate space using the torso anchor:
    - Origin: 2D midpoint between left and right shoulders (Mx, My).
    - Scale: 2D Euclidean distance between left and right shoulders (unit shoulder width).
    - z coordinates: divided by 2D shoulder width scale but NOT shifted, as hand z (wrist-relative)
      and pose z (torso-relative) exist in distinct coordinate reference frames.
    - aspect_ratio: optional width/height ratio applied to x coordinates before normalization
      to correct for non-square video capture frames.
    - Missing landmarks (or missing shoulders) are preserved as NaN.

    Args:
        left_hand: Array of shape (21, 3) or (63,) with left hand landmarks, or None.
        right_hand: Array of shape (21, 3) or (63,) with right hand landmarks, or None.
        pose: Array of shape (33, 3) or (6, 3) with upper-body pose landmarks, or None.
        aspect_ratio: Width divided by height of the video source (default 1.0).

    Returns:
        float32 array of shape (144,) representing normalized landmarks. If pose or
        shoulders are missing/zero-width, all 144 values are NaN.
    """
    output = np.full(FEATURES_PER_FRAME, np.nan, dtype=np.float32)

    if pose is None:
        # Why: Without pose landmarks, torso anchor (origin and scale) cannot be determined.
        return output

    pose_arr = np.asarray(pose, dtype=np.float32).copy()

    # Pose input can be the full 33-landmark skeleton or the 6 selected upper-body points
    if pose_arr.shape == (33, 3) or pose_arr.size == 99:
        pose_33 = pose_arr.reshape((33, 3))
        selected_pose = pose_33[POSE_IDS].copy()
    elif pose_arr.shape == (6, 3) or pose_arr.size == 18:
        selected_pose = pose_arr.reshape((6, 3)).copy()
    else:
        # Unexpected pose layout -> defensive NaN return
        return output

    # Apply aspect ratio scaling to x coordinates
    selected_pose[:, 0] *= float(aspect_ratio)

    left_shoulder = selected_pose[0]  # POSE_IDS index 11
    right_shoulder = selected_pose[1]  # POSE_IDS index 12

    # Check if either shoulder has NaN coordinates in x or y
    if np.any(np.isnan(left_shoulder[:2])) or np.any(np.isnan(right_shoulder[:2])):
        return output

    # 2D origin is the midpoint of both shoulders in x and y
    mid_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
    mid_y = (left_shoulder[1] + right_shoulder[1]) / 2.0

    # 2D scale is the Euclidean distance between shoulders in x and y
    dx = left_shoulder[0] - right_shoulder[0]
    dy = left_shoulder[1] - right_shoulder[1]
    shoulder_width_2d = float(np.sqrt(dx * dx + dy * dy))

    # Guard against zero or degenerate shoulder width to avoid division by zero
    if shoulder_width_2d <= MIN_SHOULDER_WIDTH or np.isnan(shoulder_width_2d):
        return output

    # Normalize left hand landmarks
    if left_hand is not None:
        lh_arr = np.asarray(left_hand, dtype=np.float32).copy().reshape((HAND_POINTS, 3))
        lh_arr[:, 0] *= float(aspect_ratio)
        lh_norm = np.empty_like(lh_arr)
        lh_norm[:, 0] = (lh_arr[:, 0] - mid_x) / shoulder_width_2d
        lh_norm[:, 1] = (lh_arr[:, 1] - mid_y) / shoulder_width_2d
        # Why: Hand z is relative to wrist; scale by torso width without shifting
        lh_norm[:, 2] = lh_arr[:, 2] / shoulder_width_2d
        output[LEFT_HAND_START:LEFT_HAND_END] = lh_norm.flatten()

    # Normalize right hand landmarks
    if right_hand is not None:
        rh_arr = np.asarray(right_hand, dtype=np.float32).copy().reshape((HAND_POINTS, 3))
        rh_arr[:, 0] *= float(aspect_ratio)
        rh_norm = np.empty_like(rh_arr)
        rh_norm[:, 0] = (rh_arr[:, 0] - mid_x) / shoulder_width_2d
        rh_norm[:, 1] = (rh_arr[:, 1] - mid_y) / shoulder_width_2d
        # Why: Hand z is relative to wrist; scale by torso width without shifting
        rh_norm[:, 2] = rh_arr[:, 2] / shoulder_width_2d
        output[RIGHT_HAND_START:RIGHT_HAND_END] = rh_norm.flatten()

    # Normalize selected pose landmarks
    pose_norm = np.empty_like(selected_pose)
    pose_norm[:, 0] = (selected_pose[:, 0] - mid_x) / shoulder_width_2d
    pose_norm[:, 1] = (selected_pose[:, 1] - mid_y) / shoulder_width_2d
    # Why: Pose z is relative to torso; scale by torso width without shifting
    pose_norm[:, 2] = selected_pose[:, 2] / shoulder_width_2d
    output[POSE_START:POSE_END] = pose_norm.flatten()

    return output


def canonicalize_handedness(seq: np.ndarray, is_left_dominant: bool) -> np.ndarray:
    """Canonicalize a landmark sequence to standard right-dominant signing orientation.

    When the signer is left-dominant:
    - Mirrors all x coordinates horizontally across the body midline (x -> -x).
    - Swaps left-hand and right-hand landmark feature slots so the active hand is in the
      right-hand slot.
    - Swaps bilateral pose pairs:
      - (11, 12) shoulders: slot 0 <-> slot 1
      - (13, 14) elbows:    slot 2 <-> slot 3
      - (15, 16) wrists:    slot 4 <-> slot 5

    Args:
        seq: Array of shape (T, 144) or (144,) representing normalized features.
        is_left_dominant: True if the sequence was produced by a left-dominant signer.

    Returns:
        float32 array of shape identical to seq, canonicalized if left-dominant.
    """
    output = np.array(seq, dtype=np.float32).copy()

    if not is_left_dominant:
        # Why: Right-handed signers are already canonical; pass through without mutation.
        return output

    # Mirror x coordinates across all points: in 3D (x, y, z), x is at stride 3 starting at 0
    output[..., 0::3] = -output[..., 0::3]

    # Swap left-hand slot (0..62) and right-hand slot (63..125)
    left_hand_slice = output[..., LEFT_HAND_START:LEFT_HAND_END].copy()
    right_hand_slice = output[..., RIGHT_HAND_START:RIGHT_HAND_END].copy()
    output[..., LEFT_HAND_START:LEFT_HAND_END] = right_hand_slice
    output[..., RIGHT_HAND_START:RIGHT_HAND_END] = left_hand_slice

    # Swap bilateral pose pairs in [126:144] to preserve skeleton graph automorphism
    # Pose: 6 landmarks * 3 coordinates = 18 elements
    # Slot 0 (left shoulder: 126..128) <-> Slot 1 (right shoulder: 129..131)
    left_shoulder = output[..., 126:129].copy()
    right_shoulder = output[..., 129:132].copy()
    output[..., 126:129] = right_shoulder
    output[..., 129:132] = left_shoulder

    # Slot 2 (left elbow: 132..134) <-> Slot 3 (right elbow: 135..138)
    left_elbow = output[..., 132:135].copy()
    right_elbow = output[..., 135:138].copy()
    output[..., 132:135] = right_elbow
    output[..., 135:138] = left_elbow

    # Slot 4 (left wrist: 138..140) <-> Slot 5 (right wrist: 141..144)
    left_wrist = output[..., 138:141].copy()
    right_wrist = output[..., 141:144].copy()
    output[..., 138:141] = right_wrist
    output[..., 141:144] = left_wrist

    return output


def estimate_dominant_hand(seq: np.ndarray) -> str:
    """Estimate dominant hand based on summed motion energy across the sequence.

    Args:
        seq: Array of shape (T, 144) representing normalized landmark frames over time.

    Returns:
        'left' if left hand carries strictly greater motion energy, otherwise 'right'
        (ties default to 'right').
    """
    arr = np.asarray(seq, dtype=np.float32)
    if arr.ndim == 1:
        # Single frame has 0 motion energy; ties default to right
        return "right"

    num_frames, num_features = arr.shape
    if num_frames <= 1 or num_features < RIGHT_HAND_END:
        return "right"

    # Reshape hand features to (T, 21, 3) for coordinate-wise displacement calculation
    left_hands = arr[:, LEFT_HAND_START:LEFT_HAND_END].reshape((num_frames, HAND_POINTS, 3))
    right_hands = arr[:, RIGHT_HAND_START:RIGHT_HAND_END].reshape((num_frames, HAND_POINTS, 3))

    # Compute inter-frame displacement deltas
    left_deltas = left_hands[1:] - left_hands[:-1]
    right_deltas = right_hands[1:] - right_hands[:-1]

    # Motion energy is the sum of squared 3D displacements across frames, ignoring NaNs
    left_sq_dist = np.sum(left_deltas ** 2, axis=-1)
    right_sq_dist = np.sum(right_deltas ** 2, axis=-1)

    left_energy = float(np.nansum(left_sq_dist))
    right_energy = float(np.nansum(right_sq_dist))

    # Ties default to 'right' per specification
    if left_energy > right_energy:
        return "left"
    return "right"


def from_gislr_frame(
    df_frame: Any,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Extract left_hand, right_hand, and pose arrays from a GISLR 543-row dataset frame.

    Args:
        df_frame: A pandas DataFrame or list of row dicts containing 'type',
            'landmark_index', 'x', 'y', and 'z'.

    Returns:
        Tuple of (left_hand, right_hand, pose):
        - left_hand: (21, 3) float32 array or None
        - right_hand: (21, 3) float32 array or None
        - pose: (33, 3) float32 array or None
    """
    if isinstance(df_frame, list):
        rows_by_type: dict[str, list[tuple[int, float, float, float]]] = {}
        for row in df_frame:
            t = row["type"]
            idx = int(row["landmark_index"])
            x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
            if t not in rows_by_type:
                rows_by_type[t] = []
            rows_by_type[t].append((idx, x, y, z))

        def _extract_from_dict(t: str, count: int) -> np.ndarray | None:
            if t not in rows_by_type or len(rows_by_type[t]) == 0:
                return None
            items = sorted(rows_by_type[t], key=lambda item: item[0])
            arr = np.full((count, 3), np.nan, dtype=np.float32)
            for idx, x, y, z in items:
                if 0 <= idx < count:
                    arr[idx] = [x, y, z]
            return arr

        left_hand = _extract_from_dict("left_hand", HAND_POINTS)
        right_hand = _extract_from_dict("right_hand", HAND_POINTS)
        pose = _extract_from_dict("pose", 33)
        return left_hand, right_hand, pose

    def _extract_from_df(t: str, count: int) -> np.ndarray | None:
        sub = df_frame[df_frame["type"] == t]
        if sub.empty:
            return None
        sub = sub.sort_values("landmark_index")
        arr = np.full((count, 3), np.nan, dtype=np.float32)
        indices = sub["landmark_index"].values.astype(int)
        coords = sub[["x", "y", "z"]].values.astype(np.float32)
        valid_idx = (indices >= 0) & (indices < count)
        arr[indices[valid_idx]] = coords[valid_idx]
        return arr

    left_hand = _extract_from_df("left_hand", HAND_POINTS)
    right_hand = _extract_from_df("right_hand", HAND_POINTS)
    pose = _extract_from_df("pose", 33)
    return left_hand, right_hand, pose
