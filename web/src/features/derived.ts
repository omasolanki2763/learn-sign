/**
 * Interpretable v2 geometric derived features matching ml/learnsign/features/derived.py.
 */

import {
  COORDINATES_PER_POINT,
  FEATURES_PER_FRAME,
  LEFT_HAND_START,
  POSE_SLOT_LEFT_MOUTH,
  POSE_SLOT_LEFT_SHOULDER,
  POSE_SLOT_NOSE,
  POSE_SLOT_RIGHT_MOUTH,
  POSE_SLOT_RIGHT_SHOULDER,
  POSE_START,
  RIGHT_HAND_START,
} from './schema';

const MIN_SEGMENT_NORM = 1e-6;

export const FINGER_BEND_TRIPLETS: ReadonlyArray<readonly [number, number, number]> = [
  [0, 1, 2],
  [1, 2, 3],
  [2, 3, 4],
  [0, 5, 6],
  [5, 6, 7],
  [6, 7, 8],
  [0, 9, 10],
  [9, 10, 11],
  [10, 11, 12],
  [0, 13, 14],
  [13, 14, 15],
  [14, 15, 16],
  [0, 17, 18],
  [17, 18, 19],
  [18, 19, 20],
];
export const FINGER_BEND_JOINTS_PER_HAND = FINGER_BEND_TRIPLETS.length;

const FINGERTIP_IDS = [4, 8, 12, 16, 20] as const;
const FINGERTIP_PAIRS: ReadonlyArray<readonly [number, number]> = (() => {
  const pairs: Array<readonly [number, number]> = [];
  for (let i = 0; i < FINGERTIP_IDS.length; i++) {
    for (let j = i + 1; j < FINGERTIP_IDS.length; j++) {
      pairs.push([FINGERTIP_IDS[i], FINGERTIP_IDS[j]]);
    }
  }
  return pairs;
})();
export const FINGERTIP_DISTS_PER_HAND = FINGERTIP_PAIRS.length;

const FINGER_SPREAD_RAY_PAIRS: ReadonlyArray<
  readonly [readonly [number, number], readonly [number, number]]
> = [
  [
    [0, 4],
    [5, 8],
  ],
  [
    [5, 8],
    [9, 12],
  ],
  [
    [9, 12],
    [13, 16],
  ],
  [
    [13, 16],
    [17, 20],
  ],
];
export const FINGER_SPREADS_PER_HAND = FINGER_SPREAD_RAY_PAIRS.length;

export const WRIST_ANCHORS_PER_HAND = 3;
export const VELOCITY_CHANNELS_PER_HAND = 3;
export const TWO_HAND_DIST_COUNT = 1;
export const DERIVED_Z_COUNT = 3;

export const DERIVED_BEND_START = 0;
export const DERIVED_TIP_DIST_START = DERIVED_BEND_START + FINGER_BEND_JOINTS_PER_HAND * 2;
export const DERIVED_SPREAD_START = DERIVED_TIP_DIST_START + FINGERTIP_DISTS_PER_HAND * 2;
export const DERIVED_WRIST_ANCHOR_START = DERIVED_SPREAD_START + FINGER_SPREADS_PER_HAND * 2;
export const DERIVED_VELOCITY_START = DERIVED_WRIST_ANCHOR_START + WRIST_ANCHORS_PER_HAND * 2;
export const DERIVED_HAND_DIST_START = DERIVED_VELOCITY_START + VELOCITY_CHANNELS_PER_HAND * 2;
export const DERIVED_PRIMARY_COUNT = DERIVED_HAND_DIST_START + TWO_HAND_DIST_COUNT;
export const DERIVED_Z_START = DERIVED_PRIMARY_COUNT;
export const DERIVED_FEATURES_PER_FRAME = DERIVED_Z_START + DERIVED_Z_COUNT;

function ptX(frame: Float32Array, base: number, pt: number): number {
  return frame[base + pt * COORDINATES_PER_POINT];
}

function ptY(frame: Float32Array, base: number, pt: number): number {
  return frame[base + pt * COORDINATES_PER_POINT + 1];
}

function ptZ(frame: Float32Array, base: number, pt: number): number {
  return frame[base + pt * COORDINATES_PER_POINT + 2];
}

function valid2D(x: number, y: number): boolean {
  return !Number.isNaN(x) && !Number.isNaN(y);
}

function valid3D(x: number, y: number, z: number): boolean {
  return !Number.isNaN(x) && !Number.isNaN(y) && !Number.isNaN(z);
}

function angle2D(ux: number, uy: number, vx: number, vy: number): number {
  const nu = Math.hypot(ux, uy);
  const nv = Math.hypot(vx, vy);
  if (nu <= MIN_SEGMENT_NORM || nv <= MIN_SEGMENT_NORM) {
    return NaN;
  }
  const cosVal = Math.max(-1.0, Math.min(1.0, (ux * vx + uy * vy) / (nu * nv)));
  return Math.acos(cosVal);
}

function poseMidpoint2D(
  frame: Float32Array,
  slotA: number,
  slotB: number
): readonly [number, number] | null {
  const ax = ptX(frame, POSE_START, slotA);
  const ay = ptY(frame, POSE_START, slotA);
  const bx = ptX(frame, POSE_START, slotB);
  const by = ptY(frame, POSE_START, slotB);
  if (!valid2D(ax, ay) || !valid2D(bx, by)) {
    return null;
  }
  return [(ax + bx) / 2.0, (ay + by) / 2.0];
}

/**
 * Computes interpretable 2D-primary derived features from a normalized v2 frame.
 */
export function derivedFeatures(
  frame: Float32Array,
  prevFrame: Float32Array | null = null
): Float32Array {
  const out = new Float32Array(DERIVED_FEATURES_PER_FRAME).fill(NaN);
  let idx = 0;

  const handBases = [LEFT_HAND_START, RIGHT_HAND_START] as const;

  for (const base of handBases) {
    for (const [a, b, c] of FINGER_BEND_TRIPLETS) {
      const ax = ptX(frame, base, a);
      const ay = ptY(frame, base, a);
      const bx = ptX(frame, base, b);
      const by = ptY(frame, base, b);
      const cx = ptX(frame, base, c);
      const cy = ptY(frame, base, c);
      if (valid2D(ax, ay) && valid2D(bx, by) && valid2D(cx, cy)) {
        out[idx] = angle2D(bx - ax, by - ay, cx - bx, cy - by);
      }
      idx++;
    }
  }

  for (const base of handBases) {
    for (const [pA, pB] of FINGERTIP_PAIRS) {
      const ax = ptX(frame, base, pA);
      const ay = ptY(frame, base, pA);
      const bx = ptX(frame, base, pB);
      const by = ptY(frame, base, pB);
      if (valid2D(ax, ay) && valid2D(bx, by)) {
        out[idx] = Math.hypot(ax - bx, ay - by);
      }
      idx++;
    }
  }

  for (const base of handBases) {
    for (const [[p1, p2], [p3, p4]] of FINGER_SPREAD_RAY_PAIRS) {
      const x1 = ptX(frame, base, p1);
      const y1 = ptY(frame, base, p1);
      const x2 = ptX(frame, base, p2);
      const y2 = ptY(frame, base, p2);
      const x3 = ptX(frame, base, p3);
      const y3 = ptY(frame, base, p3);
      const x4 = ptX(frame, base, p4);
      const y4 = ptY(frame, base, p4);
      if (valid2D(x1, y1) && valid2D(x2, y2) && valid2D(x3, y3) && valid2D(x4, y4)) {
        out[idx] = angle2D(x2 - x1, y2 - y1, x4 - x3, y4 - y3);
      }
      idx++;
    }
  }

  const nx = ptX(frame, POSE_START, POSE_SLOT_NOSE);
  const ny = ptY(frame, POSE_START, POSE_SLOT_NOSE);
  const nose2D: readonly [number, number] | null = valid2D(nx, ny) ? [nx, ny] : null;
  const mouth2D = poseMidpoint2D(frame, POSE_SLOT_LEFT_MOUTH, POSE_SLOT_RIGHT_MOUTH);
  const chest2D = poseMidpoint2D(frame, POSE_SLOT_LEFT_SHOULDER, POSE_SLOT_RIGHT_SHOULDER);
  const anchors = [nose2D, mouth2D, chest2D] as const;

  for (const base of handBases) {
    const wx = ptX(frame, base, 0);
    const wy = ptY(frame, base, 0);
    const wristOk = valid2D(wx, wy);
    for (const anchor of anchors) {
      if (wristOk && anchor !== null) {
        out[idx] = Math.hypot(wx - anchor[0], wy - anchor[1]);
      }
      idx++;
    }
  }

  for (const base of handBases) {
    const wx = ptX(frame, base, 0);
    const wy = ptY(frame, base, 0);
    if (prevFrame !== null && valid2D(wx, wy)) {
      const px = ptX(prevFrame, base, 0);
      const py = ptY(prevFrame, base, 0);
      if (valid2D(px, py)) {
        const dx = wx - px;
        const dy = wy - py;
        const speed = Math.hypot(dx, dy);
        if (speed > MIN_SEGMENT_NORM) {
          out[idx] = speed;
          out[idx + 1] = dx / speed;
          out[idx + 2] = dy / speed;
        } else {
          out[idx] = 0.0;
          out[idx + 1] = 0.0;
          out[idx + 2] = 0.0;
        }
      }
    }
    idx += VELOCITY_CHANNELS_PER_HAND;
  }

  const lwx = ptX(frame, LEFT_HAND_START, 0);
  const lwy = ptY(frame, LEFT_HAND_START, 0);
  const rwx = ptX(frame, RIGHT_HAND_START, 0);
  const rwy = ptY(frame, RIGHT_HAND_START, 0);
  if (valid2D(lwx, lwy) && valid2D(rwx, rwy)) {
    out[idx] = Math.hypot(lwx - rwx, lwy - rwy);
  }
  idx += TWO_HAND_DIST_COUNT;

  const lwz = ptZ(frame, LEFT_HAND_START, 0);
  const rwz = ptZ(frame, RIGHT_HAND_START, 0);
  if (prevFrame !== null) {
    const plwx = ptX(prevFrame, LEFT_HAND_START, 0);
    const plwy = ptY(prevFrame, LEFT_HAND_START, 0);
    const plwz = ptZ(prevFrame, LEFT_HAND_START, 0);
    if (valid3D(lwx, lwy, lwz) && valid3D(plwx, plwy, plwz)) {
      out[idx] = lwz - plwz;
    }
  }
  idx++;

  if (prevFrame !== null) {
    const prwx = ptX(prevFrame, RIGHT_HAND_START, 0);
    const prwy = ptY(prevFrame, RIGHT_HAND_START, 0);
    const prwz = ptZ(prevFrame, RIGHT_HAND_START, 0);
    if (valid3D(rwx, rwy, rwz) && valid3D(prwx, prwy, prwz)) {
      out[idx] = rwz - prwz;
    }
  }
  idx++;

  if (valid3D(lwx, lwy, lwz) && valid3D(rwx, rwy, rwz)) {
    out[idx] = lwz - rwz;
  }

  return out;
}

/**
 * Computes derived features across every frame of a flat normalized sequence.
 */
export function derivedSequence(seq: Float32Array, numFrames: number): Float32Array {
  const out = new Float32Array(numFrames * DERIVED_FEATURES_PER_FRAME);
  for (let t = 0; t < numFrames; t++) {
    const curStart = t * FEATURES_PER_FRAME;
    const frame = seq.subarray(curStart, curStart + FEATURES_PER_FRAME);
    const prev = t > 0 ? seq.subarray((t - 1) * FEATURES_PER_FRAME, curStart) : null;
    out.set(derivedFeatures(frame, prev), t * DERIVED_FEATURES_PER_FRAME);
  }
  return out;
}
