/**
 * Unit tests verifying feature normalization and handedness canonicalization against golden cases.
 */
import { describe, it, expect } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  HAND_POINTS,
  POSE_IDS,
  FEATURES_PER_FRAME,
} from './schema';
import {
  normalizeFrame,
  canonicalizeHandedness,
  estimateDominantHand,
} from './normalize';

interface GoldenCase {
  name: string;
  description: string;
  left_hand?: number[][] | null;
  right_hand?: number[][] | null;
  pose?: number[][] | null;
  aspect_ratio?: number;
  expected_frame?: (number | null)[];
  sequence?: (number | null)[][];
  expected_dominant?: string;
  expected_canonicalized?: (number | null)[][];
}

type GoldenData = Record<string, GoldenCase>;

describe('Feature Contract & Normalization', () => {
  // Why: Resolve path to golden cases file stored in repo root tests/golden/
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

  function flatten2D(data: number[][] | null | undefined): Float32Array | null {
    if (!data) return null;
    const flat = new Float32Array(data.length * data[0].length);
    let idx = 0;
    for (let r = 0; r < data.length; r++) {
      for (let c = 0; c < data[r].length; c++) {
        flat[idx++] = data[r][c];
      }
    }
    return flat;
  }

  it('verifies schema constants contract', () => {
    expect(HAND_POINTS).toBe(21);
    expect(POSE_IDS).toEqual([11, 12, 13, 14, 15, 16]);
    expect(FEATURES_PER_FRAME).toBe(144);
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

      expect(actual.length).toBe(FEATURES_PER_FRAME);

      for (let i = 0; i < FEATURES_PER_FRAME; i++) {
        const isExpNaN = Number.isNaN(expected[i]);
        const isActNaN = Number.isNaN(actual[i]);
        expect(isActNaN, `NaN mismatch at index ${i} in ${caseKey}`).toBe(isExpNaN);

        if (!isExpNaN) {
          expect(Math.abs(actual[i] - expected[i])).toBeLessThanOrEqual(1e-5);
        }
      }
    });
  }

  it('Blocker 1: swaps bilateral pose pairs on handedness canonicalization', () => {
    const frame = new Float32Array(FEATURES_PER_FRAME);
    // Pose: slot 0 (left shoulder: 126..128) = [1.0, 2.0, 3.0]
    //       slot 1 (right shoulder: 129..131) = [4.0, 5.0, 6.0]
    //       slot 2 (left elbow: 132..134) = [7.0, 8.0, 9.0]
    //       slot 3 (right elbow: 135..137) = [10.0, 11.0, 12.0]
    //       slot 4 (left wrist: 138..140) = [13.0, 14.0, 15.0]
    //       slot 5 (right wrist: 141..143) = [16.0, 17.0, 18.0]
    frame.set([1.0, 2.0, 3.0], 126);
    frame.set([4.0, 5.0, 6.0], 129);
    frame.set([7.0, 8.0, 9.0], 132);
    frame.set([10.0, 11.0, 12.0], 135);
    frame.set([13.0, 14.0, 15.0], 138);
    frame.set([16.0, 17.0, 18.0], 141);

    const canon = canonicalizeHandedness(frame, 1, true);

    // Slot 0 (left shoulder) should now hold mirrored previous slot 1 (right shoulder)
    expect(canon[126]).toBeCloseTo(-4.0, 5);
    expect(canon[127]).toBeCloseTo(5.0, 5);
    expect(canon[128]).toBeCloseTo(6.0, 5);

    // Slot 1 (right shoulder) should now hold mirrored previous slot 0 (left shoulder)
    expect(canon[129]).toBeCloseTo(-1.0, 5);
    expect(canon[130]).toBeCloseTo(2.0, 5);
    expect(canon[131]).toBeCloseTo(3.0, 5);

    // Slot 2 <-> 3 (elbows)
    expect(canon[132]).toBeCloseTo(-10.0, 5);
    expect(canon[135]).toBeCloseTo(-7.0, 5);

    // Slot 4 <-> 5 (wrists)
    expect(canon[138]).toBeCloseTo(-16.0, 5);
    expect(canon[141]).toBeCloseTo(-13.0, 5);
  });

  it('Blocker 2: divides z by 2D shoulder width without shifting', () => {
    const pose = new Float32Array(99).fill(NaN);
    // 2D distance: lsX=0.6, rsX=0.2, lsY=0.4, rsY=0.4 => 2D width = 0.4
    // Set shoulders with non-zero z values (lsZ = 0.8, rsZ = 0.4 => midpoint would be 0.6)
    pose[11 * 3] = 0.6;
    pose[11 * 3 + 1] = 0.4;
    pose[11 * 3 + 2] = 0.8;

    pose[12 * 3] = 0.2;
    pose[12 * 3 + 1] = 0.4;
    pose[12 * 3 + 2] = 0.4;

    const lh = new Float32Array(63).fill(NaN);
    lh[0] = 0.6;
    lh[1] = 0.4;
    lh[2] = 0.2; // z = 0.2

    const out = normalizeFrame(lh, null, pose);

    // Expected z for hand point 0: 0.2 / 0.4 = 0.5 (NOT (0.2 - 0.6) / 0.4 = -1.0)
    expect(out[2]).toBeCloseTo(0.2 / 0.4, 5);
    // Expected z for left shoulder: 0.8 / 0.4 = 2.0 (NOT (0.8 - 0.6) / 0.4 = 0.5)
    expect(out[126 + 2]).toBeCloseTo(0.8 / 0.4, 5);
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
      for (let i = 63; i < 126; i++) {
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
