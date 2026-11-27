/**
 * Unit tests verifying v2 derived features match Python and shared golden cases within 1e-5.
 */
import { describe, it, expect } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  DERIVED_BEND_START,
  DERIVED_FEATURES_PER_FRAME,
  DERIVED_PRIMARY_COUNT,
  DERIVED_TIP_DIST_START,
  DERIVED_VELOCITY_START,
  DERIVED_WRIST_ANCHOR_START,
  DERIVED_Z_COUNT,
  DERIVED_Z_START,
  FINGER_BEND_JOINTS_PER_HAND,
  derivedFeatures,
  derivedSequence,
} from './derived';
import { FEATURES_PER_FRAME } from './schema';

interface DerivedGoldenCase {
  name: string;
  description: string;
  frame: (number | null)[];
  prev_frame: (number | null)[] | null;
  dt_s?: number | null;
  expected: (number | null)[];
}

type DerivedGoldenData = Record<string, DerivedGoldenCase>;

describe('Derived Features (Contract v2)', () => {
  const currentDirectory = path.dirname(fileURLToPath(import.meta.url));
  const goldenPath = path.resolve(currentDirectory, '../../../tests/golden/derived_cases.json');
  const goldenCases: DerivedGoldenData = JSON.parse(fs.readFileSync(goldenPath, 'utf-8'));

  function toFloat32Array(data: (number | null)[] | null): Float32Array | null {
    if (!data) return null;
    const arr = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
      arr[i] = data[i] === null ? NaN : (data[i] as number);
    }
    return arr;
  }

  it('keeps derived feature constants consistent', () => {
    expect(FINGER_BEND_JOINTS_PER_HAND).toBe(15);
    expect(DERIVED_Z_START).toBe(DERIVED_PRIMARY_COUNT);
    expect(DERIVED_Z_START + DERIVED_Z_COUNT).toBe(DERIVED_FEATURES_PER_FRAME);
  });

  for (const caseKey of [
    'straight_hand_at_nose',
    'fist_with_velocity',
    'missing_hands',
    'zero_dt_velocity_is_nan',
  ]) {
    it(`matches golden case within 1e-5: ${caseKey}`, () => {
      const c = goldenCases[caseKey];
      const frame = toFloat32Array(c.frame)!;
      const prev = toFloat32Array(c.prev_frame);
      const dtS = c.dt_s ?? null;
      const expected = toFloat32Array(c.expected)!;

      const actual = derivedFeatures(frame, prev, dtS);
      expect(actual.length).toBe(DERIVED_FEATURES_PER_FRAME);

      for (let i = 0; i < DERIVED_FEATURES_PER_FRAME; i++) {
        const expNaN = Number.isNaN(expected[i]);
        const actNaN = Number.isNaN(actual[i]);
        expect(actNaN, `NaN mismatch at index ${i} in ${caseKey}`).toBe(expNaN);
        if (!expNaN) {
          expect(Math.abs(actual[i] - expected[i])).toBeLessThanOrEqual(1e-5);
        }
      }
    });
  }

  it('computes derived semantic invariants (straight bend == 0, fist large bend, wrist at nose)', () => {
    const straight = toFloat32Array(goldenCases['straight_hand_at_nose'].frame)!;
    const feat = derivedFeatures(straight, null);

    for (let i = 0; i < FINGER_BEND_JOINTS_PER_HAND; i++) {
      expect(Math.abs(feat[DERIVED_BEND_START + i])).toBeLessThanOrEqual(1e-5);
      expect(feat[DERIVED_BEND_START + FINGER_BEND_JOINTS_PER_HAND + i]).toBeGreaterThan(1.0);
    }
    expect(Math.abs(feat[DERIVED_WRIST_ANCHOR_START])).toBeLessThanOrEqual(1e-5);
    expect(DERIVED_TIP_DIST_START).toBeGreaterThan(DERIVED_BEND_START);
  });

  it('computes derivedSequence across consecutive frames using timestamps', () => {
    const vel = goldenCases['fist_with_velocity'];
    const f0 = toFloat32Array(vel.prev_frame)!;
    const f1 = toFloat32Array(vel.frame)!;
    const flatSeq = new Float32Array(2 * FEATURES_PER_FRAME);
    flatSeq.set(f0, 0);
    flatSeq.set(f1, FEATURES_PER_FRAME);

    const out = derivedSequence(flatSeq, 2, [0.0, 0.1]);
    expect(out.length).toBe(2 * DERIVED_FEATURES_PER_FRAME);
    expect(Number.isNaN(out[DERIVED_VELOCITY_START])).toBe(true);
    expect(Number.isNaN(out[DERIVED_Z_START])).toBe(true);
    expect(out[DERIVED_FEATURES_PER_FRAME + DERIVED_VELOCITY_START + 3]).toBeCloseTo(5.0, 5);
    expect(out[DERIVED_FEATURES_PER_FRAME + DERIVED_VELOCITY_START + 4]).toBeCloseTo(0.6, 5);
    expect(out[DERIVED_FEATURES_PER_FRAME + DERIVED_VELOCITY_START + 5]).toBeCloseTo(-0.8, 5);
    expect(out[DERIVED_FEATURES_PER_FRAME + DERIVED_Z_START + 1]).toBeCloseTo(2.0, 5);
  });

  it('matches velocity across 15 FPS and 30 FPS for the same physical motion', () => {
    const vel = goldenCases['fist_with_velocity'];
    const base = toFloat32Array(vel.prev_frame)!;
    const target = toFloat32Array(vel.frame)!;
    const mid = new Float32Array(base.length);
    for (let i = 0; i < base.length; i++) {
      mid[i] = base[i] + 0.5 * (target[i] - base[i]);
    }

    const feat30 = derivedFeatures(mid, base, 1.0 / 30.0);
    const feat15 = derivedFeatures(target, base, 1.0 / 15.0);

    for (let i = 0; i < 6; i++) {
      expect(
        Math.abs(feat30[DERIVED_VELOCITY_START + i] - feat15[DERIVED_VELOCITY_START + i])
      ).toBeLessThanOrEqual(1e-5);
    }
    for (let i = 0; i < 2; i++) {
      expect(
        Math.abs(feat30[DERIVED_Z_START + i] - feat15[DERIVED_Z_START + i])
      ).toBeLessThanOrEqual(1e-5);
    }
  });

  it('returns NaN velocity and temporal dz when dtS is missing or non-positive', () => {
    const vel = goldenCases['fist_with_velocity'];
    const frame = toFloat32Array(vel.frame)!;
    const prev = toFloat32Array(vel.prev_frame)!;

    for (const badDt of [null, 0, -0.05, NaN]) {
      const out = derivedFeatures(frame, prev, badDt);
      for (let i = 0; i < 6; i++) {
        expect(Number.isNaN(out[DERIVED_VELOCITY_START + i])).toBe(true);
      }
      expect(Number.isNaN(out[DERIVED_Z_START])).toBe(true);
      expect(Number.isNaN(out[DERIVED_Z_START + 1])).toBe(true);
      expect(Number.isNaN(out[DERIVED_Z_START + 2])).toBe(false);
    }
  });
});
