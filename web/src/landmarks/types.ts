/**
 * Types and constants for the browser landmark harness.
 *
 * The shapes here are the contract between three parties: the worker that runs
 * MediaPipe, the client that talks to it, and the feature normalisation in
 * `src/features/`. Keeping them in one file stops those three drifting apart.
 */
/** Verdict colour of the feedback box drawn over a detected hand. */
export type BoxState = 'idle' | 'green' | 'amber' | 'red';

/** Landmarks MediaPipe reports per hand. */
export const HAND_LANDMARK_COUNT = 21;

/** Landmarks MediaPipe reports for the full pose. */
export const POSE_LANDMARK_COUNT = 33;

/** Coordinates per landmark: x, y, z. */
export const COORDINATES_PER_LANDMARK = 3;

/** Flattened length of one hand: 21 * 3. */
export const FLAT_HAND_LENGTH = HAND_LANDMARK_COUNT * COORDINATES_PER_LANDMARK;

/** Flattened length of a full pose: 33 * 3. */
export const FLAT_POSE_LENGTH = POSE_LANDMARK_COUNT * COORDINATES_PER_LANDMARK;

/**
 * One frame of landmarks, in normalised image coordinates (0..1 for x and y).
 *
 * Why `null` rather than a NaN-filled array: an absent hand and a hand seen at
 * (0, 0, 0) are different facts. The detector reports absence here, and the
 * NaN filling that the model input needs happens later, in normalisation.
 */
export interface FrameLandmarks {
  /** Capture time of the frame, in milliseconds. */
  timestampMs: number;
  /** Left hand as 21 * 3 floats, or null when no left hand was detected. */
  leftHand: Float32Array | null;
  /** Right hand as 21 * 3 floats, or null when no right hand was detected. */
  rightHand: Float32Array | null;
  /** Full pose as 33 * 3 floats, or null when no pose was detected. */
  pose: Float32Array | null;
}

/** Messages the client sends to the worker. */
export type LandmarkRequest =
  | { type: 'init'; enableFace: boolean }
  | { type: 'process'; requestId: number; bitmap: ImageBitmap; timestampMs: number }
  | { type: 'dispose' };

/** Messages the worker sends back to the client. */
export type LandmarkResponse =
  | { type: 'ready' }
  | { type: 'result'; requestId: number; landmarks: FrameLandmarks }
  | { type: 'error'; requestId: number | null; message: string };
