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
  description: string;
  left_hand?: number[][] | null;
  right_hand?: number[][] | null;
  pose?: number[][] | null;
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
  ];

  for (const caseKey of singleFrameCases) {
    it(`normalizes frame matching golden case: ${caseKey}`, () => {
      const c = goldenCases[caseKey];
      const lh = flatten2D(c.left_hand);
      const rh = flatten2D(c.right_hand);
      const pose = flatten2D(c.pose);
      const expected = toFloat32Array(c.expected_frame);

      const actual = normalizeFrame(lh, rh, pose);

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
    // Right hand moves in frames 1 and 2 (indices 63..125)
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
