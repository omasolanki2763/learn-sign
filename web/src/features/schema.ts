/**
 * Feature layout constants and contract v2 schema shared between browser inference and ML models.
 */

export const FEATURE_CONTRACT_VERSION = 2;

// Number of landmarks per hand in the MediaPipe Hand model
export const HAND_POINTS = 21;

// Spatial coordinates per landmark: x, y, z
export const COORDINATES_PER_POINT = 3;

// Selected MediaPipe Pose landmark indices (contract v2: 0-10 head anchors + 11-16 upper body)
export const POSE_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] as const;
export const POSE_POINTS = POSE_IDS.length;

// Slot indices inside the selected pose slice
export const POSE_SLOT_NOSE = 0;
export const POSE_SLOT_LEFT_EYE = 2;
export const POSE_SLOT_RIGHT_EYE = 5;
export const POSE_SLOT_LEFT_MOUTH = 9;
export const POSE_SLOT_RIGHT_MOUTH = 10;
export const POSE_SLOT_LEFT_SHOULDER = 11;
export const POSE_SLOT_RIGHT_SHOULDER = 12;
export const POSE_SLOT_LEFT_ELBOW = 13;
export const POSE_SLOT_RIGHT_ELBOW = 14;
export const POSE_SLOT_LEFT_WRIST = 15;
export const POSE_SLOT_RIGHT_WRIST = 16;

// Bilateral symmetric pose pairs swapped during left-dominant horizontal mirroring
export const POSE_BILATERAL_PAIRS = [
  [1, 4],
  [2, 5],
  [3, 6],
  [7, 8],
  [9, 10],
  [11, 12],
  [13, 14],
  [15, 16],
] as const;

// Anthropometric ratio (shoulder width / inter-eye distance) when shoulders leave frame
export const EYE_TO_SHOULDER_RATIO = 6.0;

// Total landmarks per frame: left hand (21) + right hand (21) + pose (17)
export const TOTAL_LANDMARKS_PER_FRAME = HAND_POINTS + HAND_POINTS + POSE_POINTS;

// Total flattened float features per frame
export const FEATURES_PER_FRAME = TOTAL_LANDMARKS_PER_FRAME * COORDINATES_PER_POINT;

// Feature vector layout boundaries
export const LEFT_HAND_START = 0;
export const LEFT_HAND_END = HAND_POINTS * COORDINATES_PER_POINT;

export const RIGHT_HAND_START = LEFT_HAND_END;
export const RIGHT_HAND_END = RIGHT_HAND_START + HAND_POINTS * COORDINATES_PER_POINT;

export const POSE_START = RIGHT_HAND_END;
export const POSE_END = FEATURES_PER_FRAME;
