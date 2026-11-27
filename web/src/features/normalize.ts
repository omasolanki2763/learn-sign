/**
 * Feature normalization, coordinate standardization, and handedness canonicalization for browser inference.
 */

import {
  COORDINATES_PER_POINT,
  EYE_TO_SHOULDER_RATIO,
  EYE_TO_SHOULDER_Y_OFFSET,
  FEATURES_PER_FRAME,
  HAND_POINTS,
  LEFT_HAND_END,
  LEFT_HAND_START,
  POSE_BILATERAL_PAIRS,
  POSE_IDS,
  POSE_POINTS,
  POSE_SLOT_LEFT_EYE,
  POSE_SLOT_LEFT_SHOULDER,
  POSE_SLOT_RIGHT_EYE,
  POSE_SLOT_RIGHT_SHOULDER,
  POSE_START,
  RIGHT_HAND_END,
  RIGHT_HAND_START,
} from './schema';

export type AnchorSource = 'shoulders' | 'eyes';

export interface NormalizedFrameResult {
  features: Float32Array;
  anchorSource: AnchorSource | null;
}

// Minimum 2D span threshold to guard against division-by-zero on degenerate frames
const MIN_ANCHOR_SPAN = 1e-6;

interface Anchor2D {
  midX: number;
  midY: number;
  scale2D: number;
}

function resolvePairAnchor(
  ax: number,
  ay: number,
  bx: number,
  by: number,
  ratio: number = 1.0
): Anchor2D | null {
  if (Number.isNaN(ax) || Number.isNaN(ay) || Number.isNaN(bx) || Number.isNaN(by)) {
    return null;
  }
  const dx = ax - bx;
  const dy = ay - by;
  const span = Math.sqrt(dx * dx + dy * dy);
  if (span <= MIN_ANCHOR_SPAN || Number.isNaN(span)) {
    return null;
  }
  return {
    midX: (ax + bx) / 2.0,
    midY: (ay + by) / 2.0,
    scale2D: span * ratio,
  };
}

/**
 * Normalizes a single frame's raw landmarks into the canonical v2 feature representation and
 * reports which anchor source resolved the frame (`shoulders`, `eyes`, or `null`).
 *
 * Uses primary shoulder anchor (11 <-> 12) and falls back to eye distance (2 <-> 5)
 * scaled by EYE_TO_SHOULDER_RATIO with origin shifted downward by
 * `scale2D * EYE_TO_SHOULDER_Y_OFFSET` to estimate the shoulder midpoint.
 */
export function normalizeFrameWithAnchor(
  leftHand: Float32Array | null,
  rightHand: Float32Array | null,
  pose: Float32Array | null,
  aspectRatio: number = 1.0
): NormalizedFrameResult {
  const output = new Float32Array(FEATURES_PER_FRAME).fill(NaN);

  if (!pose) {
    return { features: output, anchorSource: null };
  }

  const isFullPose = pose.length === 33 * COORDINATES_PER_POINT;
  const isSelectedPose = pose.length === POSE_POINTS * COORDINATES_PER_POINT;
  if (!isFullPose && !isSelectedPose) {
    return { features: output, anchorSource: null };
  }

  const poseOffset = (slot: number): number =>
    (isFullPose ? POSE_IDS[slot] : slot) * COORDINATES_PER_POINT;

  const lsOff = poseOffset(POSE_SLOT_LEFT_SHOULDER);
  const rsOff = poseOffset(POSE_SLOT_RIGHT_SHOULDER);
  let anchorSource: AnchorSource | null = null;
  let midX: number;
  let midY: number;
  let scale2D: number;

  const shoulderAnchor = resolvePairAnchor(
    pose[lsOff] * aspectRatio,
    pose[lsOff + 1],
    pose[rsOff] * aspectRatio,
    pose[rsOff + 1]
  );

  if (shoulderAnchor) {
    ({ midX, midY, scale2D } = shoulderAnchor);
    anchorSource = 'shoulders';
  } else {
    const leOff = poseOffset(POSE_SLOT_LEFT_EYE);
    const reOff = poseOffset(POSE_SLOT_RIGHT_EYE);
    const eyeAnchor = resolvePairAnchor(
      pose[leOff] * aspectRatio,
      pose[leOff + 1],
      pose[reOff] * aspectRatio,
      pose[reOff + 1],
      EYE_TO_SHOULDER_RATIO
    );
    if (!eyeAnchor) {
      return { features: output, anchorSource: null };
    }
    midX = eyeAnchor.midX;
    scale2D = eyeAnchor.scale2D;
    midY = eyeAnchor.midY + scale2D * EYE_TO_SHOULDER_Y_OFFSET;
    anchorSource = 'eyes';
  }

  if (leftHand && leftHand.length >= LEFT_HAND_END - LEFT_HAND_START) {
    for (let i = 0; i < HAND_POINTS; i++) {
      const srcIdx = i * COORDINATES_PER_POINT;
      const dstIdx = LEFT_HAND_START + srcIdx;
      output[dstIdx] = (leftHand[srcIdx] * aspectRatio - midX) / scale2D;
      output[dstIdx + 1] = (leftHand[srcIdx + 1] - midY) / scale2D;
      output[dstIdx + 2] = leftHand[srcIdx + 2] / scale2D;
    }
  }

  if (rightHand && rightHand.length >= RIGHT_HAND_END - RIGHT_HAND_START) {
    for (let i = 0; i < HAND_POINTS; i++) {
      const srcIdx = i * COORDINATES_PER_POINT;
      const dstIdx = RIGHT_HAND_START + srcIdx;
      output[dstIdx] = (rightHand[srcIdx] * aspectRatio - midX) / scale2D;
      output[dstIdx + 1] = (rightHand[srcIdx + 1] - midY) / scale2D;
      output[dstIdx + 2] = rightHand[srcIdx + 2] / scale2D;
    }
  }

  for (let p = 0; p < POSE_POINTS; p++) {
    const srcIdx = poseOffset(p);
    const dstIdx = POSE_START + p * COORDINATES_PER_POINT;
    output[dstIdx] = (pose[srcIdx] * aspectRatio - midX) / scale2D;
    output[dstIdx + 1] = (pose[srcIdx + 1] - midY) / scale2D;
    output[dstIdx + 2] = pose[srcIdx + 2] / scale2D;
  }

  return { features: output, anchorSource };
}

/**
 * Normalizes a single frame's raw landmarks into the canonical v2 feature representation.
 */
export function normalizeFrame(
  leftHand: Float32Array | null,
  rightHand: Float32Array | null,
  pose: Float32Array | null,
  aspectRatio: number = 1.0
): Float32Array {
  return normalizeFrameWithAnchor(leftHand, rightHand, pose, aspectRatio).features;
}

/**
 * Canonicalizes a sequence of frames to standard right-dominant signing orientation.
 */
export function canonicalizeHandedness(
  seq: Float32Array,
  numFrames: number,
  isLeftDominant: boolean
): Float32Array {
  const output = new Float32Array(seq);

  if (!isLeftDominant) {
    return output;
  }

  for (let f = 0; f < numFrames; f++) {
    const frameOffset = f * FEATURES_PER_FRAME;

    for (let i = 0; i < FEATURES_PER_FRAME; i += COORDINATES_PER_POINT) {
      output[frameOffset + i] = -output[frameOffset + i];
    }

    for (let i = 0; i < LEFT_HAND_END - LEFT_HAND_START; i++) {
      const lhIdx = frameOffset + LEFT_HAND_START + i;
      const rhIdx = frameOffset + RIGHT_HAND_START + i;
      const temp = output[lhIdx];
      output[lhIdx] = output[rhIdx];
      output[rhIdx] = temp;
    }

    for (const [leftSlot, rightSlot] of POSE_BILATERAL_PAIRS) {
      const leftOffset = frameOffset + POSE_START + leftSlot * COORDINATES_PER_POINT;
      const rightOffset = frameOffset + POSE_START + rightSlot * COORDINATES_PER_POINT;
      for (let c = 0; c < COORDINATES_PER_POINT; c++) {
        const temp = output[leftOffset + c];
        output[leftOffset + c] = output[rightOffset + c];
        output[rightOffset + c] = temp;
      }
    }
  }

  return output;
}

/**
 * Estimates dominant signing hand based on summed motion energy across the sequence.
 */
export function estimateDominantHand(seq: Float32Array, numFrames: number): 'left' | 'right' {
  if (numFrames <= 1) {
    return 'right';
  }

  let leftEnergy = 0.0;
  let rightEnergy = 0.0;

  for (let f = 0; f < numFrames - 1; f++) {
    const curFrame = f * FEATURES_PER_FRAME;
    const nextFrame = (f + 1) * FEATURES_PER_FRAME;

    for (let p = 0; p < HAND_POINTS; p++) {
      const ptOffset = p * COORDINATES_PER_POINT;
      const idxCur = curFrame + LEFT_HAND_START + ptOffset;
      const idxNext = nextFrame + LEFT_HAND_START + ptOffset;

      const dx = seq[idxNext] - seq[idxCur];
      const dy = seq[idxNext + 1] - seq[idxCur + 1];
      const dz = seq[idxNext + 2] - seq[idxCur + 2];

      if (!Number.isNaN(dx) && !Number.isNaN(dy) && !Number.isNaN(dz)) {
        leftEnergy += dx * dx + dy * dy + dz * dz;
      }
    }

    for (let p = 0; p < HAND_POINTS; p++) {
      const ptOffset = p * COORDINATES_PER_POINT;
      const idxCur = curFrame + RIGHT_HAND_START + ptOffset;
      const idxNext = nextFrame + RIGHT_HAND_START + ptOffset;

      const dx = seq[idxNext] - seq[idxCur];
      const dy = seq[idxNext + 1] - seq[idxCur + 1];
      const dz = seq[idxNext + 2] - seq[idxCur + 2];

      if (!Number.isNaN(dx) && !Number.isNaN(dy) && !Number.isNaN(dz)) {
        rightEnergy += dx * dx + dy * dy + dz * dz;
      }
    }
  }

  if (leftEnergy > rightEnergy) {
    return 'left';
  }
  return 'right';
}
