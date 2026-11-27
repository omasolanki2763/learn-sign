"""Feature normalization, coordinate standardization, and handedness canonicalization."""

from typing import Any

import numpy as np

from learnsign.features.schema import (
    COORDINATES_PER_POINT,
    EYE_TO_SHOULDER_RATIO,
    FEATURES_PER_FRAME,
    HAND_POINTS,
    LEFT_HAND_END,
    LEFT_HAND_START,
    POSE_BILATERAL_PAIRS,
    POSE_END,
    POSE_IDS,
    POSE_POINTS,
    POSE_SLOT_LEFT_EYE,
    POSE_SLOT_LEFT_SHOULDER,
    POSE_SLOT_RIGHT_EYE,
    POSE_SLOT_RIGHT_SHOULDER,
    POSE_START,
    RIGHT_HAND_END,
    RIGHT_HAND_START,
)

# Minimum 2D span threshold to guard against division-by-zero on degenerate frames
MIN_ANCHOR_SPAN: float = 1e-6


def _resolve_2d_anchor_pair(
    pt_a: np.ndarray,
    pt_b: np.ndarray,
    ratio: float = 1.0,
) -> tuple[float, float, float] | None:
    """Compute (mid_x, mid_y, span * ratio) from two 2D landmarks, or None if invalid."""
    if np.any(np.isnan(pt_a[:2])) or np.any(np.isnan(pt_b[:2])):
        return None
    dx = float(pt_a[0] - pt_b[0])
    dy = float(pt_a[1] - pt_b[1])
    span = float(np.sqrt(dx * dx + dy * dy))
    if span <= MIN_ANCHOR_SPAN or np.isnan(span):
        return None
    mid_x = float((pt_a[0] + pt_b[0]) / 2.0)
    mid_y = float((pt_a[1] + pt_b[1]) / 2.0)
    return mid_x, mid_y, span * ratio


def normalize_frame(
    left_hand: np.ndarray | None,
    right_hand: np.ndarray | None,
    pose: np.ndarray | None,
    aspect_ratio: float = 1.0,
) -> np.ndarray:
    """Normalize a single frame's landmarks into the canonical v2 feature representation.

    Normalizes coordinate space using primary shoulder anchor with eye-distance fallback:
    - Primary: origin = shoulder midpoint (11, 12), scale = 2D shoulder width.
    - Fallback (when either shoulder is missing/degenerate): origin = eye midpoint (2, 5),
      scale = eye distance * EYE_TO_SHOULDER_RATIO.
    - z coordinates: divided by scale but NOT shifted (hand z and pose z have distinct origins).
    - Missing landmarks (or missing both shoulder and eye anchors) are preserved as NaN.

    Args:
        left_hand: Array of shape (21, 3) or (63,) with left hand landmarks, or None.
        right_hand: Array of shape (21, 3) or (63,) with right hand landmarks, or None.
        pose: Array of shape (33, 3) or (POSE_POINTS, 3) with pose landmarks, or None.
        aspect_ratio: Width divided by height of the video source (default 1.0).

    Returns:
        float32 array of shape (FEATURES_PER_FRAME,) containing normalized landmarks.
    """
    output = np.full(FEATURES_PER_FRAME, np.nan, dtype=np.float32)

    if pose is None:
        return output

    pose_arr = np.asarray(pose, dtype=np.float32).copy()

    # Pose input can be full 33-landmark skeleton or the selected POSE_POINTS upper-body/head subset
    if pose_arr.shape == (33, COORDINATES_PER_POINT) or pose_arr.size == 33 * COORDINATES_PER_POINT:
        pose_33 = pose_arr.reshape((33, COORDINATES_PER_POINT))
        selected_pose = pose_33[POSE_IDS].copy()
    elif (
        pose_arr.shape == (POSE_POINTS, COORDINATES_PER_POINT)
        or pose_arr.size == POSE_POINTS * COORDINATES_PER_POINT
    ):
        selected_pose = pose_arr.reshape((POSE_POINTS, COORDINATES_PER_POINT)).copy()
    else:
        return output

    aspect = float(aspect_ratio)
    selected_pose[:, 0] *= aspect

    anchor = _resolve_2d_anchor_pair(
        selected_pose[POSE_SLOT_LEFT_SHOULDER],
        selected_pose[POSE_SLOT_RIGHT_SHOULDER],
    )
    if anchor is None:
        anchor = _resolve_2d_anchor_pair(
            selected_pose[POSE_SLOT_LEFT_EYE],
            selected_pose[POSE_SLOT_RIGHT_EYE],
            ratio=EYE_TO_SHOULDER_RATIO,
        )
    if anchor is None:
        return output

    mid_x, mid_y, scale_2d = anchor

    if left_hand is not None:
        lh_arr = np.asarray(left_hand, dtype=np.float32).copy().reshape((HAND_POINTS, 3))
        lh_arr[:, 0] *= aspect
        lh_norm = np.empty_like(lh_arr)
        lh_norm[:, 0] = (lh_arr[:, 0] - mid_x) / scale_2d
        lh_norm[:, 1] = (lh_arr[:, 1] - mid_y) / scale_2d
        lh_norm[:, 2] = lh_arr[:, 2] / scale_2d
        output[LEFT_HAND_START:LEFT_HAND_END] = lh_norm.flatten()

    if right_hand is not None:
        rh_arr = np.asarray(right_hand, dtype=np.float32).copy().reshape((HAND_POINTS, 3))
        rh_arr[:, 0] *= aspect
        rh_norm = np.empty_like(rh_arr)
        rh_norm[:, 0] = (rh_arr[:, 0] - mid_x) / scale_2d
        rh_norm[:, 1] = (rh_arr[:, 1] - mid_y) / scale_2d
        rh_norm[:, 2] = rh_arr[:, 2] / scale_2d
        output[RIGHT_HAND_START:RIGHT_HAND_END] = rh_norm.flatten()

    pose_norm = np.empty_like(selected_pose)
    pose_norm[:, 0] = (selected_pose[:, 0] - mid_x) / scale_2d
    pose_norm[:, 1] = (selected_pose[:, 1] - mid_y) / scale_2d
    pose_norm[:, 2] = selected_pose[:, 2] / scale_2d
    output[POSE_START:POSE_END] = pose_norm.flatten()

    return output


def normalize_sequence(
    left_hands: np.ndarray | None,
    right_hands: np.ndarray | None,
    poses: np.ndarray,
    aspect_ratio: float = 1.0,
) -> np.ndarray:
    """Vectorized anchor normalization across a sequence of T frames.

    Produces identical outputs to calling `normalize_frame` per frame (including
    shoulder primary anchor, eye-distance fallback, unshifted z scaling, and NaN rules).
    """
    poses_arr = np.asarray(poses, dtype=np.float32)
    num_frames = poses_arr.shape[0]
    output = np.full((num_frames, FEATURES_PER_FRAME), np.nan, dtype=np.float32)
    if num_frames == 0:
        return output

    if poses_arr.shape[1:] == (33, COORDINATES_PER_POINT):
        selected_pose = poses_arr[:, POSE_IDS, :].copy()
    elif poses_arr.shape[1:] == (POSE_POINTS, COORDINATES_PER_POINT):
        selected_pose = poses_arr.copy()
    else:
        return output

    aspect = float(aspect_ratio)
    selected_pose[:, :, 0] *= aspect

    def _pair_metrics(slot_a: int, slot_b: int, ratio: float = 1.0) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        pa = selected_pose[:, slot_a, :]
        pb = selected_pose[:, slot_b, :]
        mx = (pa[:, 0] + pb[:, 0]) / 2.0
        my = (pa[:, 1] + pb[:, 1]) / 2.0
        dx = pa[:, 0] - pb[:, 0]
        dy = pa[:, 1] - pb[:, 1]
        span = np.sqrt(dx * dx + dy * dy)
        valid = (
            ~np.isnan(pa[:, 0])
            & ~np.isnan(pa[:, 1])
            & ~np.isnan(pb[:, 0])
            & ~np.isnan(pb[:, 1])
            & ~np.isnan(span)
            & (span > MIN_ANCHOR_SPAN)
        )
        return mx, my, span * ratio, valid

    s_mx, s_my, s_scale, s_valid = _pair_metrics(
        POSE_SLOT_LEFT_SHOULDER, POSE_SLOT_RIGHT_SHOULDER
    )
    e_mx, e_my, e_scale, e_valid = _pair_metrics(
        POSE_SLOT_LEFT_EYE, POSE_SLOT_RIGHT_EYE, ratio=EYE_TO_SHOULDER_RATIO
    )

    anchor_valid = s_valid | e_valid
    mid_x = np.where(s_valid, s_mx, e_mx)
    mid_y = np.where(s_valid, s_my, e_my)
    scale_2d = np.where(s_valid, s_scale, e_scale)
    safe_scale = np.where(anchor_valid, scale_2d, np.float32(1.0))[:, None]

    def _normalize_part(pts: np.ndarray) -> np.ndarray:
        arr = np.asarray(pts, dtype=np.float32).copy()
        arr[:, :, 0] *= aspect
        norm = np.empty_like(arr)
        norm[:, :, 0] = (arr[:, :, 0] - mid_x[:, None]) / safe_scale
        norm[:, :, 1] = (arr[:, :, 1] - mid_y[:, None]) / safe_scale
        norm[:, :, 2] = arr[:, :, 2] / safe_scale
        return norm.reshape((num_frames, -1))

    if left_hands is not None:
        output[:, LEFT_HAND_START:LEFT_HAND_END] = _normalize_part(left_hands)
    if right_hands is not None:
        output[:, RIGHT_HAND_START:RIGHT_HAND_END] = _normalize_part(right_hands)
    output[:, POSE_START:POSE_END] = (
        np.stack(
            [
                (selected_pose[:, :, 0] - mid_x[:, None]) / safe_scale,
                (selected_pose[:, :, 1] - mid_y[:, None]) / safe_scale,
                selected_pose[:, :, 2] / safe_scale,
            ],
            axis=-1,
        )
        .astype(np.float32, copy=False)
        .reshape((num_frames, -1))
    )

    output[~anchor_valid] = np.nan
    return output


def canonicalize_handedness(seq: np.ndarray, is_left_dominant: bool) -> np.ndarray:
    """Canonicalize a landmark sequence to standard right-dominant signing orientation.

    When the signer is left-dominant:
    - Mirrors all x coordinates horizontally across the body midline (x -> -x).
    - Swaps left-hand and right-hand landmark feature slots.
    - Swaps all bilateral head and upper-body pose pairs defined in `POSE_BILATERAL_PAIRS`.

    Args:
        seq: Array of shape (T, FEATURES_PER_FRAME) or (FEATURES_PER_FRAME,).
        is_left_dominant: True if the sequence was produced by a left-dominant signer.

    Returns:
        float32 array of shape identical to seq.
    """
    output = np.array(seq, dtype=np.float32).copy()

    if not is_left_dominant:
        return output

    output[..., 0::3] = -output[..., 0::3]

    left_hand_slice = output[..., LEFT_HAND_START:LEFT_HAND_END].copy()
    right_hand_slice = output[..., RIGHT_HAND_START:RIGHT_HAND_END].copy()
    output[..., LEFT_HAND_START:LEFT_HAND_END] = right_hand_slice
    output[..., RIGHT_HAND_START:RIGHT_HAND_END] = left_hand_slice

    for left_slot, right_slot in POSE_BILATERAL_PAIRS:
        l_start = POSE_START + left_slot * COORDINATES_PER_POINT
        l_end = l_start + COORDINATES_PER_POINT
        r_start = POSE_START + right_slot * COORDINATES_PER_POINT
        r_end = r_start + COORDINATES_PER_POINT

        left_val = output[..., l_start:l_end].copy()
        right_val = output[..., r_start:r_end].copy()
        output[..., l_start:l_end] = right_val
        output[..., r_start:r_end] = left_val

    return output


def estimate_dominant_hand(seq: np.ndarray) -> str:
    """Estimate dominant hand based on summed motion energy across the sequence."""
    arr = np.asarray(seq, dtype=np.float32)
    if arr.ndim == 1:
        return "right"

    num_frames, num_features = arr.shape
    if num_frames <= 1 or num_features < RIGHT_HAND_END:
        return "right"

    left_hands = arr[:, LEFT_HAND_START:LEFT_HAND_END].reshape((num_frames, HAND_POINTS, 3))
    right_hands = arr[:, RIGHT_HAND_START:RIGHT_HAND_END].reshape((num_frames, HAND_POINTS, 3))

    left_deltas = left_hands[1:] - left_hands[:-1]
    right_deltas = right_hands[1:] - right_hands[:-1]

    left_sq_dist = np.sum(left_deltas**2, axis=-1)
    right_sq_dist = np.sum(right_deltas**2, axis=-1)

    left_energy = float(np.nansum(left_sq_dist))
    right_energy = float(np.nansum(right_sq_dist))

    if left_energy > right_energy:
        return "left"
    return "right"


def from_gislr_frame(
    df_frame: Any,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Extract left_hand, right_hand, and pose arrays from a GISLR 543-row dataset frame."""
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
