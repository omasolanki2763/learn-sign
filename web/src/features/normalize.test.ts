/**
 * Unit tests verifying v2 feature normalization and handedness canonicalization against golden cases.
 */
import { describe, it, expect } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  COORDINATES_PER_POINT,
  EYE_TO_SHOULDER_RATIO,
  EYE_TO_SHOULDER_Y_OFFSET,
  FEATURE_CONTRACT_VERSION,
  FEATURES_PER_FRAME,
  HAND_POINTS,
  LEFT_HAND_END,
  LEFT_HAND_START,
  POSE_BILATERAL_PAIRS,
  POSE_IDS,
  POSE_POINTS,
  POSE_SLOT_LEFT_SHOULDER,
  POSE_SLOT_NOSE,
  POSE_SLOT_RIGHT_SHOULDER,
  POSE_START,
  RIGHT_HAND_END,
  RIGHT_HAND_START,
} from './schema';
import {
  normalizeFrame,
  normalizeFrameWithAnchor,
  canonicalizeHandedness,
  estimateDominantHand,
  type AnchorSource,
} from './normalize';

interface GoldenCase {
  name: string;
  description: string;
  left_hand?: (number | null)[][] | null;
  right_hand?: (number | null)[][] | null;
  pose?: (number | null)[][] | null;
  aspect_ratio?: number;
  expected_anchor_source?: AnchorSource | null;
  expected_frame?: (number | null)[];
  sequence?: (number | null)[][];
  expected_dominant?: string;
  expected_canonicalized?: (number | null)[][];
}

type GoldenData = Record<string, GoldenCase>;

describe('Feature Contract v2 & Normalization', () => {
  const currentDirectory = path.dirname(fileURLToPath(import.meta.url));
  const goldenPath = path.resolve(currentDirectory, '../../../tests/golden/normalize_cases.json');
  const goldenCases: GoldenData = JSON.parse(fs.readFileSync(goldenPath, 'utf-8'));

  function toFloat32Array(data: (number | null)[] | undefined): Float32Array {
    if (!data) return new Float32Array(0);
    const arr = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
      arr[i] = data[i] === null ? NaN : (data[i] as number);
    }
    return arr;
  }

  function flatten2D(data: (number | null)[][] | null | undefined): Float32Array | null {
    if (!data) return null;
    const flat = new Float32Array(data.length * data[0].length);
    let idx = 0;
    for (let r = 0; r < data.length; r++) {
      for (let c = 0; c < data[r].length; c++) {
        const val = data[r][c];
        flat[idx++] = val === null ? NaN : val;
      }
    }
    return flat;
  }

  it('verifies schema constants contract v2', () => {
    expect(FEATURE_CONTRACT_VERSION).toBe(2);
    expect(HAND_POINTS).toBe(21);
    expect(POSE_IDS).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    expect(POSE_POINTS).toBe(POSE_IDS.length);
    expect(FEATURES_PER_FRAME).toBe((HAND_POINTS * 2 + POSE_POINTS) * COORDINATES_PER_POINT);
    expect(EYE_TO_SHOULDER_RATIO).toBeGreaterThan(1.0);
    expect(EYE_TO_SHOULDER_Y_OFFSET).toBeGreaterThan(0.0);
  });

  const singleFrameCases = [
    'both_hands',
    'left_only',
    'right_only',
    'no_hands',
    'no_pose',
    'shoulders_swapped',
    'zero_shoulder_width',
    'aspect_ratio_scaling',
    'head_anchors',
    'missing_shoulders_eye_fallback',
    'same_frame_with_shoulder',
    'same_frame_without_shoulder',
  ];

  for (const caseKey of singleFrameCases) {
    it(`normalizes frame matching golden case: ${caseKey}`, () => {
      const c = goldenCases[caseKey];
      const lh = flatten2D(c.left_hand);
      const rh = flatten2D(c.right_hand);
      const pose = flatten2D(c.pose);
      const aspectRatio = c.aspect_ratio ?? 1.0;
      const expected = toFloat32Array(c.expected_frame);

      const actual = normalizeFrame(lh, rh, pose, aspectRatio);
      const withAnchor = normalizeFrameWithAnchor(lh, rh, pose, aspectRatio);

      expect(withAnchor.anchorSource).toBe(c.expected_anchor_source ?? null);
      expect(actual.length).toBe(FEATURES_PER_FRAME);

      for (let i = 0; i < FEATURES_PER_FRAME; i++) {
        const isExpNaN = Number.isNaN(expected[i]);
        const isActNaN = Number.isNaN(actual[i]);
        expect(isActNaN, `NaN mismatch at index ${i} in ${caseKey}`).toBe(isExpNaN);
        expect(Number.isNaN(withAnchor.features[i])).toBe(isExpNaN);

        if (!isExpNaN) {
          expect(Math.abs(actual[i] - expected[i])).toBeLessThanOrEqual(1e-5);
          expect(Math.abs(withAnchor.features[i] - expected[i])).toBeLessThanOrEqual(1e-5);
        }
      }
    });
  }

  it('keeps normalized hands consistent when dropping a shoulder triggers eye fallback', () => {
    const withSh = goldenCases['same_frame_with_shoulder'];
    const withoutSh = goldenCases['same_frame_without_shoulder'];

    const resWith = normalizeFrameWithAnchor(
      flatten2D(withSh.left_hand),
      flatten2D(withSh.right_hand),
      flatten2D(withSh.pose)
    );
    const resWithout = normalizeFrameWithAnchor(
      flatten2D(withoutSh.left_hand),
      flatten2D(withoutSh.right_hand),
      flatten2D(withoutSh.pose)
    );

    expect(resWith.anchorSource).toBe('shoulders');
    expect(resWithout.anchorSource).toBe('eyes');

    for (let i = LEFT_HAND_START; i < LEFT_HAND_END; i++) {
      expect(Math.abs(resWith.features[i] - resWithout.features[i])).toBeLessThanOrEqual(1e-5);
    }
    for (let i = RIGHT_HAND_START; i < RIGHT_HAND_END; i++) {
      expect(Math.abs(resWith.features[i] - resWithout.features[i])).toBeLessThanOrEqual(1e-5);
    }
  });

  it('swaps bilateral head and torso pairs on handedness canonicalization', () => {
    const frame = new Float32Array(FEATURES_PER_FRAME);
    const noseOff = POSE_START + POSE_SLOT_NOSE * COORDINATES_PER_POINT;
    frame.set([0.5, -0.5, 0.1], noseOff);

    for (const [leftSlot, rightSlot] of POSE_BILATERAL_PAIRS) {
      frame.set([leftSlot + 1, 0.1, 0.2], POSE_START + leftSlot * COORDINATES_PER_POINT);
      frame.set([rightSlot + 1, -0.1, -0.2], POSE_START + rightSlot * COORDINATES_PER_POINT);
    }

    const canon = canonicalizeHandedness(frame, 1, true);

    expect(canon[noseOff]).toBeCloseTo(-0.5, 5);
    expect(canon[noseOff + 1]).toBeCloseTo(-0.5, 5);
    expect(canon[noseOff + 2]).toBeCloseTo(0.1, 5);

    for (const [leftSlot, rightSlot] of POSE_BILATERAL_PAIRS) {
      const lOff = POSE_START + leftSlot * COORDINATES_PER_POINT;
      const rOff = POSE_START + rightSlot * COORDINATES_PER_POINT;
      expect(canon[lOff]).toBeCloseTo(-(rightSlot + 1), 5);
      expect(canon[lOff + 1]).toBeCloseTo(-0.1, 5);
      expect(canon[rOff]).toBeCloseTo(-(leftSlot + 1), 5);
      expect(canon[rOff + 1]).toBeCloseTo(0.1, 5);
    }
  });

  it('divides z by 2D shoulder width without shifting', () => {
    const pose = new Float32Array(33 * COORDINATES_PER_POINT).fill(NaN);
    pose.set([0.6, 0.4, 0.8], POSE_SLOT_LEFT_SHOULDER * COORDINATES_PER_POINT);
    pose.set([0.2, 0.4, 0.4], POSE_SLOT_RIGHT_SHOULDER * COORDINATES_PER_POINT);

    const lh = new Float32Array(HAND_POINTS * COORDINATES_PER_POINT).fill(NaN);
    lh[0] = 0.6;
    lh[1] = 0.4;
    lh[2] = 0.2;

    const out = normalizeFrame(lh, null, pose);

    expect(out[2]).toBeCloseTo(0.2 / 0.4, 5);
    expect(out[POSE_START + POSE_SLOT_LEFT_SHOULDER * COORDINATES_PER_POINT + 2]).toBeCloseTo(
      0.8 / 0.4,
      5
    );
  });

  it('detects left-dominant signer and canonicalizes sequence', () => {
    const c = goldenCases['left_dominant'];
    const rawSeq = c.sequence!;
    const seq = new Float32Array(rawSeq.length * FEATURES_PER_FRAME);
    for (let f = 0; f < rawSeq.length; f++) {
      const frameArr = toFloat32Array(rawSeq[f]);
      seq.set(frameArr, f * FEATURES_PER_FRAME);
    }

    const detectedDom = estimateDominantHand(seq, rawSeq.length);
    expect(detectedDom).toBe(c.expected_dominant);

    const actualCanon = canonicalizeHandedness(seq, rawSeq.length, true);
    expect(actualCanon.length).toBe(seq.length);

    for (let f = 0; f < rawSeq.length; f++) {
      const expectedFrame = toFloat32Array(c.expected_canonicalized![f]);
      for (let i = 0; i < FEATURES_PER_FRAME; i++) {
        const idx = f * FEATURES_PER_FRAME + i;
        const isExpNaN = Number.isNaN(expectedFrame[i]);
        const isActNaN = Number.isNaN(actualCanon[idx]);
        expect(isActNaN).toBe(isExpNaN);
        if (!isExpNaN) {
          expect(Math.abs(actualCanon[idx] - expectedFrame[i])).toBeLessThanOrEqual(1e-5);
        }
      }
    }
  });

  it('keeps right-dominant sequence unchanged when isLeftDominant is false', () => {
    const numFrames = 3;
    const seq = new Float32Array(numFrames * FEATURES_PER_FRAME);
    for (let f = 1; f < numFrames; f++) {
      for (let i = RIGHT_HAND_START; i < RIGHT_HAND_END; i++) {
        seq[f * FEATURES_PER_FRAME + i] = f * 0.5;
      }
    }

    const detectedDom = estimateDominantHand(seq, numFrames);
    expect(detectedDom).toBe('right');

    const result = canonicalizeHandedness(seq, numFrames, false);
    for (let i = 0; i < seq.length; i++) {
      expect(result[i]).toBe(seq[i]);
    }
  });
});
