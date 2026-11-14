"""Feature schema and layout constants shared across ML training and browser inference."""

from typing import Final

# Number of landmarks per hand in the MediaPipe Hand model
HAND_POINTS: Final[int] = 21

# Spatial coordinates per landmark: x, y, z
COORDINATES_PER_POINT: Final[int] = 3

# Selected MediaPipe Pose landmark indices for upper-body gesture context:
# 11: left shoulder, 12: right shoulder
# 13: left elbow,    14: right elbow
# 15: left wrist,    16: right wrist
POSE_IDS: Final[list[int]] = [11, 12, 13, 14, 15, 16]

# Total landmarks per frame: left hand (21) + right hand (21) + upper-body pose (6)
TOTAL_LANDMARKS_PER_FRAME: Final[int] = HAND_POINTS + HAND_POINTS + len(POSE_IDS)

# Total flattened float features per frame: 48 landmarks * 3 coordinates = 144
FEATURES_PER_FRAME: Final[int] = TOTAL_LANDMARKS_PER_FRAME * COORDINATES_PER_POINT

# Feature vector layout boundaries
LEFT_HAND_START: Final[int] = 0
LEFT_HAND_END: Final[int] = HAND_POINTS * COORDINATES_PER_POINT  # 63

RIGHT_HAND_START: Final[int] = LEFT_HAND_END  # 63
RIGHT_HAND_END: Final[int] = RIGHT_HAND_START + (HAND_POINTS * COORDINATES_PER_POINT)  # 126

POSE_START: Final[int] = RIGHT_HAND_END  # 126
POSE_END: Final[int] = FEATURES_PER_FRAME  # 144
