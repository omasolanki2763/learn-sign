/**
 * Tests for banded incremental DTW alignment step (Step 1b timing and correctness).
 */
import { describe, expect, it } from 'vitest';

import { DERIVED_FEATURES_PER_FRAME } from '../features/derived';
import { FEATURES_PER_FRAME } from '../features/schema';
import { computePercentile } from './stopwatch';
import {
  AlignmentBench,
  BANDED_DTW_BAND_RADIUS,
  BANDED_DTW_NUM_REFS,
  BANDED_DTW_REF_FRAMES,
} from './alignmentBench';

describe('AlignmentBench (banded incremental DTW)', () => {
  it('uses Step 1b reference dimensions and v2 schema feature size', () => {
    const bench = AlignmentBench.createRandom();
    expect(bench.numRefs).toBe(BANDED_DTW_NUM_REFS);
    expect(bench.refFrames).toBe(BANDED_DTW_REF_FRAMES);
    expect(bench.bandRadius).toBe(BANDED_DTW_BAND_RADIUS);
    expect(bench.featureDim).toBe(FEATURES_PER_FRAME + DERIVED_FEATURES_PER_FRAME);
  });

  it('produces zero cost when query frames match a reference sequence', () => {
    const dim = 16;
    const refFrames = 10;
    const ref0 = new Float32Array(refFrames * dim);
    const ref1 = new Float32Array(refFrames * dim);
    for (let i = 0; i < ref0.length; i++) {
      ref0[i] = i * 0.01;
      ref1[i] = i * 0.01 + 2.0;
    }

    const bench = new AlignmentBench([ref0, ref1], refFrames, dim, 4);
    let lastCosts: Float32Array = new Float32Array(0);
    for (let t = 0; t < refFrames; t++) {
      const frame = ref0.subarray(t * dim, (t + 1) * dim);
      lastCosts = bench.step(frame);
    }

    expect(lastCosts[0]).toBeCloseTo(0, 5);
    expect(lastCosts[1]).toBeGreaterThan(10);
  });

  it('runs a full K=5 x 90 x v2-features alignment step with p95 <= 5 ms', () => {
    const bench = AlignmentBench.createRandom();
    const query = new Float32Array(bench.featureDim);
    for (let i = 0; i < bench.featureDim; i++) {
      query[i] = Math.sin(i * 0.1);
    }

    // Warm up JIT once before measuring 90 incremental frame steps
    bench.step(query);
    bench.reset();

    const stepMs: number[] = [];
    for (let t = 0; t < BANDED_DTW_REF_FRAMES; t++) {
      const start = performance.now();
      bench.step(query);
      stepMs.push(performance.now() - start);
    }

    expect(computePercentile(stepMs, 95)).toBeLessThanOrEqual(5.0);
  });
});
