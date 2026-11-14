/**
 * Feature layout constants and schema shared between browser inference and ML models.
 */

// Number of landmarks per hand in the MediaPipe Hand model
export const HAND_POINTS = 21;

// Spatial coordinates per landmark: x, y, z
export const COORDINATES_PER_POINT = 3;

// Selected MediaPipe Pose landmark indices for upper-body gesture context:
// 11: left shoulder, 12: right shoulder
// 13: left elbow,    14: right elbow
// 15: left wrist,    16: right wrist
export const POSE_IDS = [11, 12, 13, 14, 15, 16] as const;

// Total landmarks per frame: left hand (21) + right hand (21) + upper-body pose (6)
export const TOTAL_LANDMARKS_PER_FRAME = HAND_POINTS + HAND_POINTS + POSE_IDS.length;

// Total flattened float features per frame: 48 landmarks * 3 coordinates = 144
export const FEATURES_PER_FRAME = TOTAL_LANDMARKS_PER_FRAME * COORDINATES_PER_POINT;

// Feature vector layout boundaries
export const LEFT_HAND_START = 0;
export const LEFT_HAND_END = HAND_POINTS * COORDINATES_PER_POINT; // 63

export const RIGHT_HAND_START = LEFT_HAND_END; // 63
export const RIGHT_HAND_END = RIGHT_HAND_START + HAND_POINTS * COORDINATES_PER_POINT; // 126

export const POSE_START = RIGHT_HAND_END; // 126
export const POSE_END = FEATURES_PER_FRAME; // 144
