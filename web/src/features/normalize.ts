/**
 * Feature normalization, coordinate standardization, and handedness canonicalization for browser inference.
 */

import {
  COORDINATES_PER_POINT,
  FEATURES_PER_FRAME,
  HAND_POINTS,
  LEFT_HAND_END,
  LEFT_HAND_START,
  POSE_IDS,
  POSE_START,
  RIGHT_HAND_END,
  RIGHT_HAND_START,
} from './schema';

// Minimum shoulder width threshold to guard against division-by-zero on degenerate frames
const MIN_SHOULDER_WIDTH = 1e-6;

/**
 * Normalizes a single frame's raw landmarks into the canonical 144-feature representation.
 *
 * Normalizes coordinate space using the torso anchor:
 * - Origin: midpoint between left and right shoulders.
 * - Scale: Euclidean distance between left and right shoulders (unit shoulder width).
 * - Missing landmarks (or missing shoulders) are preserved as NaN.
 *
 * @param leftHand Flat Float32Array of length 63 (21*3) or null.
 * @param rightHand Flat Float32Array of length 63 (21*3) or null.
 * @param pose Flat Float32Array of length 99 (33*3) or length 18 (6*3), or null.
 * @returns Flat Float32Array of length 144 containing normalized coordinates.
 */
export function normalizeFrame(
  leftHand: Float32Array | null,
  rightHand: Float32Array | null,
  pose: Float32Array | null
): Float32Array {
  const output = new Float32Array(FEATURES_PER_FRAME).fill(NaN);

  if (!pose) {
    // Why: Without pose landmarks, torso anchor (origin and scale) cannot be determined.
    return output;
  }

  let lsX: number;
  let lsY: number;
  let lsZ: number;
  let rsX: number;
  let rsY: number;
  let rsZ: number;

  const isFullPose = pose.length === 99 || pose.length === 33 * COORDINATES_PER_POINT;
  const isSelectedPose = pose.length === 18 || pose.length === 6 * COORDINATES_PER_POINT;

  if (isFullPose) {
    const lsOffset = 11 * COORDINATES_PER_POINT;
    const rsOffset = 12 * COORDINATES_PER_POINT;
    lsX = pose[lsOffset];
    lsY = pose[lsOffset + 1];
    lsZ = pose[lsOffset + 2];
    rsX = pose[rsOffset];
    rsY = pose[rsOffset + 1];
    rsZ = pose[rsOffset + 2];
  } else if (isSelectedPose) {
    // In selected 6-pose array: slot 0 is left shoulder, slot 1 is right shoulder
    lsX = pose[0];
    lsY = pose[1];
    lsZ = pose[2];
    rsX = pose[3];
    rsY = pose[4];
    rsZ = pose[5];
  } else {
    // Unexpected pose length -> defensive NaN return
    return output;
  }

  // Check if either shoulder has NaN coordinates
  if (
    Number.isNaN(lsX) ||
    Number.isNaN(lsY) ||
    Number.isNaN(lsZ) ||
    Number.isNaN(rsX) ||
    Number.isNaN(rsY) ||
    Number.isNaN(rsZ)
  ) {
    return output;
  }

  // Torso origin: midpoint between shoulders
  const midX = (lsX + rsX) / 2.0;
  const midY = (lsY + rsY) / 2.0;
  const midZ = (lsZ + rsZ) / 2.0;

  // Torso scale: Euclidean distance between shoulders
  const dx = lsX - rsX;
  const dy = lsY - rsY;
  const dz = lsZ - rsZ;
  const shoulderWidth = Math.sqrt(dx * dx + dy * dy + dz * dz);

  // Guard against zero or degenerate shoulder width to avoid division by zero
  if (shoulderWidth <= MIN_SHOULDER_WIDTH || Number.isNaN(shoulderWidth)) {
    return output;
  }

  // Normalize left hand landmarks (indices 0..62)
  if (leftHand && leftHand.length >= LEFT_HAND_END - LEFT_HAND_START) {
    for (let i = 0; i < HAND_POINTS; i++) {
      const srcIdx = i * COORDINATES_PER_POINT;
      const dstIdx = LEFT_HAND_START + srcIdx;
      output[dstIdx] = (leftHand[srcIdx] - midX) / shoulderWidth;
      output[dstIdx + 1] = (leftHand[srcIdx + 1] - midY) / shoulderWidth;
      output[dstIdx + 2] = (leftHand[srcIdx + 2] - midZ) / shoulderWidth;
    }
  }

  // Normalize right hand landmarks (indices 63..125)
  if (rightHand && rightHand.length >= RIGHT_HAND_END - RIGHT_HAND_START) {
    for (let i = 0; i < HAND_POINTS; i++) {
      const srcIdx = i * COORDINATES_PER_POINT;
      const dstIdx = RIGHT_HAND_START + srcIdx;
      output[dstIdx] = (rightHand[srcIdx] - midX) / shoulderWidth;
      output[dstIdx + 1] = (rightHand[srcIdx + 1] - midY) / shoulderWidth;
      output[dstIdx + 2] = (rightHand[srcIdx + 2] - midZ) / shoulderWidth;
    }
  }

  // Normalize selected pose landmarks (indices 126..143)
  for (let p = 0; p < POSE_IDS.length; p++) {
    const srcIdx = isFullPose ? POSE_IDS[p] * COORDINATES_PER_POINT : p * COORDINATES_PER_POINT;
    const dstIdx = POSE_START + p * COORDINATES_PER_POINT;
    output[dstIdx] = (pose[srcIdx] - midX) / shoulderWidth;
    output[dstIdx + 1] = (pose[srcIdx + 1] - midY) / shoulderWidth;
    output[dstIdx + 2] = (pose[srcIdx + 2] - midZ) / shoulderWidth;
  }

  return output;
}

/**
 * Canonicalizes a sequence of frames to standard right-dominant signing orientation.
 *
 * When the signer is left-dominant:
 * - Mirrors all x coordinates horizontally across the body midline (x -> -x).
 * - Swaps left-hand and right-hand feature slots so the active hand is in the right-hand slot.
 *
 * @param seq Flat Float32Array of length (numFrames * 144).
 * @param numFrames Number of frames in the sequence.
 * @param isLeftDominant True if the signer was identified as left-dominant.
 * @returns New Float32Array containing canonicalized sequence.
 */
export function canonicalizeHandedness(
  seq: Float32Array,
  numFrames: number,
  isLeftDominant: boolean
): Float32Array {
  const output = new Float32Array(seq);

  if (!isLeftDominant) {
    // Why: Right-handed signers are already canonical; pass through without mutation.
    return output;
  }

  for (let f = 0; f < numFrames; f++) {
    const frameOffset = f * FEATURES_PER_FRAME;

    // Mirror x coordinates for every 3D coordinate point (stride 3)
    for (let i = 0; i < FEATURES_PER_FRAME; i += COORDINATES_PER_POINT) {
      output[frameOffset + i] = -output[frameOffset + i];
    }

    // Swap left-hand slot (0..62) with right-hand slot (63..125)
    for (let i = 0; i < LEFT_HAND_END - LEFT_HAND_START; i++) {
      const lhIdx = frameOffset + LEFT_HAND_START + i;
      const rhIdx = frameOffset + RIGHT_HAND_START + i;
      const temp = output[lhIdx];
      output[lhIdx] = output[rhIdx];
      output[rhIdx] = temp;
    }
  }

  return output;
}

/**
 * Estimates dominant signing hand based on summed motion energy across the sequence.
 *
 * @param seq Flat Float32Array of length (numFrames * 144).
 * @param numFrames Number of frames in the sequence.
 * @returns 'left' if left hand carries strictly greater motion energy, otherwise 'right'.
 */
export function estimateDominantHand(seq: Float32Array, numFrames: number): 'left' | 'right' {
  if (numFrames <= 1) {
    // Why: Single frame has 0 motion delta; ties default to right.
    return 'right';
  }

  let leftEnergy = 0.0;
  let rightEnergy = 0.0;

  for (let f = 0; f < numFrames - 1; f++) {
    const curFrame = f * FEATURES_PER_FRAME;
    const nextFrame = (f + 1) * FEATURES_PER_FRAME;

    // Left hand motion
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

    // Right hand motion
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

  // Ties default to 'right' per specification
  if (leftEnergy > rightEnergy) {
    return 'left';
  }
  return 'right';
}
