"""Feature schema and contract v2 layout constants shared across ML and browser inference."""

from typing import Final

# Frozen feature contract version for downstream models and golden checks.
FEATURE_CONTRACT_VERSION: Final[int] = 2

# Number of landmarks per hand in the MediaPipe Hand model
HAND_POINTS: Final[int] = 21

# Spatial coordinates per landmark: x, y, z
COORDINATES_PER_POINT: Final[int] = 3

# Selected MediaPipe Pose landmark indices (contract v2: 0-10 head anchors + 11-16 upper body):
# 0: nose
# 1..3: left eye inner / eye / outer
# 4..6: right eye inner / eye / outer
# 7, 8: left / right ear
# 9, 10: mouth left / right
# 11, 12: left / right shoulder
# 13, 14: left / right elbow
# 15, 16: left / right wrist
POSE_IDS: Final[list[int]] = list(range(17))
POSE_POINTS: Final[int] = len(POSE_IDS)

# Slot indices inside the selected pose slice (equal to MediaPipe IDs since POSE_IDS == 0..16)
POSE_SLOT_NOSE: Final[int] = 0
POSE_SLOT_LEFT_EYE: Final[int] = 2
POSE_SLOT_RIGHT_EYE: Final[int] = 5
POSE_SLOT_LEFT_MOUTH: Final[int] = 9
POSE_SLOT_RIGHT_MOUTH: Final[int] = 10
POSE_SLOT_LEFT_SHOULDER: Final[int] = 11
POSE_SLOT_RIGHT_SHOULDER: Final[int] = 12
POSE_SLOT_LEFT_ELBOW: Final[int] = 13
POSE_SLOT_RIGHT_ELBOW: Final[int] = 14
POSE_SLOT_LEFT_WRIST: Final[int] = 15
POSE_SLOT_RIGHT_WRIST: Final[int] = 16

# Bilateral symmetric pose pairs swapped during left-dominant horizontal mirroring.
POSE_BILATERAL_PAIRS: Final[tuple[tuple[int, int], ...]] = (
    (1, 4),
    (2, 5),
    (3, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
)

# Anthropometric ratio (shoulder width / inter-eye distance) used when shoulders leave frame
EYE_TO_SHOULDER_RATIO: Final[float] = 6.0

# Vertical offset from eye midpoint downward to estimated shoulder midpoint (in shoulder widths)
EYE_TO_SHOULDER_Y_OFFSET: Final[float] = 0.5

# Total landmarks per frame: left hand (21) + right hand (21) + pose (17)
TOTAL_LANDMARKS_PER_FRAME: Final[int] = HAND_POINTS + HAND_POINTS + POSE_POINTS

# Total flattened float features per frame: 59 landmarks * 3 coordinates = 177
FEATURES_PER_FRAME: Final[int] = TOTAL_LANDMARKS_PER_FRAME * COORDINATES_PER_POINT

# Feature vector layout boundaries
LEFT_HAND_START: Final[int] = 0
LEFT_HAND_END: Final[int] = HAND_POINTS * COORDINATES_PER_POINT

RIGHT_HAND_START: Final[int] = LEFT_HAND_END
RIGHT_HAND_END: Final[int] = RIGHT_HAND_START + (HAND_POINTS * COORDINATES_PER_POINT)

POSE_START: Final[int] = RIGHT_HAND_END
POSE_END: Final[int] = FEATURES_PER_FRAME
