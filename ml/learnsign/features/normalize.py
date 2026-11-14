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
) -> np.ndarray:
    """Normalize a single frame's landmarks into the canonical 144-feature representation.

    Normalizes coordinate space using the torso anchor:
    - Origin: midpoint between left and right shoulders.
    - Scale: Euclidean distance between left and right shoulders (unit shoulder width).
    - Missing landmarks (or missing shoulders) are preserved as NaN.

    Args:
        left_hand: Array of shape (21, 3) or (63,) with left hand landmarks, or None.
        right_hand: Array of shape (21, 3) or (63,) with right hand landmarks, or None.
        pose: Array of shape (33, 3) or (6, 3) with upper-body pose landmarks, or None.

    Returns:
        float32 array of shape (144,) representing normalized landmarks. If pose or
        shoulders are missing/zero-width, all 144 values are NaN.
    """
    output = np.full(FEATURES_PER_FRAME, np.nan, dtype=np.float32)

    if pose is None:
        # Why: Without pose landmarks, torso anchor (origin and scale) cannot be determined.
        return output

    pose_arr = np.asarray(pose, dtype=np.float32)

    # Pose input can be the full 33-landmark skeleton or the 6 selected upper-body points
    if pose_arr.shape == (33, 3) or pose_arr.size == 99:
        pose_33 = pose_arr.reshape((33, 3))
        left_shoulder = pose_33[11]
        right_shoulder = pose_33[12]
        selected_pose = pose_33[POSE_IDS]
    elif pose_arr.shape == (6, 3) or pose_arr.size == 18:
        # First two points in POSE_IDS are shoulders (11 and 12)
        pose_6 = pose_arr.reshape((6, 3))
        left_shoulder = pose_6[0]
        right_shoulder = pose_6[1]
        selected_pose = pose_6
    else:
        # Unexpected pose layout -> defensive NaN return
        return output

    # Check if either shoulder has NaN coordinates
    if np.any(np.isnan(left_shoulder)) or np.any(np.isnan(right_shoulder)):
        return output

    # Origin is the midpoint of both shoulders
    midpoint = (left_shoulder + right_shoulder) / 2.0

    # Scale is the 3D Euclidean distance between shoulders
    shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))

    # Guard against zero or degenerate shoulder width to avoid division by zero
    if shoulder_width <= MIN_SHOULDER_WIDTH or np.isnan(shoulder_width):
        return output

    # Normalize left hand landmarks
    if left_hand is not None:
        lh_arr = np.asarray(left_hand, dtype=np.float32).reshape((HAND_POINTS, 3))
        lh_norm = (lh_arr - midpoint) / shoulder_width
        output[LEFT_HAND_START:LEFT_HAND_END] = lh_norm.flatten()

    # Normalize right hand landmarks
    if right_hand is not None:
        rh_arr = np.asarray(right_hand, dtype=np.float32).reshape((HAND_POINTS, 3))
        rh_norm = (rh_arr - midpoint) / shoulder_width
        output[RIGHT_HAND_START:RIGHT_HAND_END] = rh_norm.flatten()

    # Normalize selected pose landmarks
    pose_norm = (selected_pose - midpoint) / shoulder_width
    output[POSE_START:POSE_END] = pose_norm.flatten()

    return output


def canonicalize_handedness(seq: np.ndarray, is_left_dominant: bool) -> np.ndarray:
    """Canonicalize a landmark sequence to standard right-dominant signing orientation.

    When the signer is left-dominant:
    - Mirrors all x coordinates horizontally across the body midline (x -> -x).
    - Swaps left-hand and right-hand landmark feature slots so the active hand is in the
      right-hand slot.

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
    # Convert list of dicts to a dict of lists if needed
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

    # If df_frame is a pandas DataFrame
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
