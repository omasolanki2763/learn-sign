/**
 * Tests for the per-stage pipeline Stopwatch and percentile utilities.
 */
import { describe, expect, it } from 'vitest';

import { BENCH_STAGES, computePercentile, Stopwatch } from './stopwatch';

describe('computePercentile', () => {
  it('computes p50=50.5 and p95=95.05 on integers 1..100', () => {
    const values = Array.from({ length: 100 }, (_, i) => i + 1);
    expect(computePercentile(values, 50)).toBeCloseTo(50.5, 6);
    expect(computePercentile(values, 95)).toBeCloseTo(95.05, 6);
  });

  it('returns 0 for empty input and the single element for length 1', () => {
    expect(computePercentile([], 50)).toBe(0);
    expect(computePercentile([42], 95)).toBe(42);
  });
});

describe('Stopwatch', () => {
  it('includes all required pipeline stages including derived and alignment', () => {
    expect(BENCH_STAGES).toEqual([
      'landmarks',
      'features',
      'derived',
      'model',
      'render',
      'alignment',
    ]);
  });

  it('tracks p50, p95, and sample count per stage', () => {
    const sw = new Stopwatch();
    for (let i = 1; i <= 100; i++) {
      sw.mark('landmarks', i);
      sw.mark('derived', i * 0.1);
      sw.mark('alignment', i * 0.02);
    }

    const summary = sw.summary();
    expect(summary.landmarks.n).toBe(100);
    expect(summary.landmarks.p50).toBeCloseTo(50.5, 5);
    expect(summary.landmarks.p95).toBeCloseTo(95.05, 5);

    expect(summary.derived.n).toBe(100);
    expect(summary.derived.p50).toBeCloseTo(5.05, 5);
    expect(summary.derived.p95).toBeCloseTo(9.505, 5);

    expect(summary.alignment.n).toBe(100);
    expect(summary.alignment.p50).toBeCloseTo(1.01, 5);

    expect(summary.features).toEqual({ p50: 0, p95: 0, n: 0 });
  });

  it('computes FPS from recorded frame timestamps', () => {
    const sw = new Stopwatch();
    expect(sw.fps()).toBe(0);

    for (let f = 0; f <= 30; f++) {
      sw.markFrame(f * (1000 / 30));
    }
    expect(sw.fps()).toBeCloseTo(30, 3);
  });
});
