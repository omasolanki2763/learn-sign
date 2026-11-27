/**
 * Per-stage latency stopwatch and FPS meter for the live browser pipeline benchmark.
 */

/** Named pipeline stages timed on every frame or prediction tick. */
export const BENCH_STAGES = [
  'landmarks',
  'features',
  'derived',
  'model',
  'render',
  'alignment',
] as const;

export type BenchStage = (typeof BENCH_STAGES)[number];

/** Summary statistics computed for one recorded stage. */
export interface StageSummary {
  p50: number;
  p95: number;
  n: number;
}

const MILLISECONDS_PER_SECOND = 1000;
const MIN_PERCENTILE = 0;
const MAX_PERCENTILE = 100;

/**
 * Computes linear-interpolated percentile (R-7 / NumPy default) over numeric samples.
 *
 * @param values Unsorted numeric samples.
 * @param percentile Target percentile in [0, 100].
 * @returns Interpolated percentile value, or 0 when `values` is empty.
 */
export function computePercentile(values: readonly number[], percentile: number): number {
  if (values.length === 0) {
    return 0;
  }
  const clamped = Math.min(MAX_PERCENTILE, Math.max(MIN_PERCENTILE, percentile));
  const sorted = [...values].sort((a, b) => a - b);
  if (sorted.length === 1) {
    return sorted[0];
  }
  const index = (clamped / MAX_PERCENTILE) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const fraction = index - lower;
  return sorted[lower] * (1 - fraction) + sorted[upper] * fraction;
}

/**
 * Records per-stage durations and frame arrival timestamps during a benchmark run.
 */
export class Stopwatch {
  private readonly samples = new Map<BenchStage, number[]>();
  private frameCount = 0;
  private firstFrameMs: number | null = null;
  private lastFrameMs: number | null = null;

  constructor() {
    for (const stage of BENCH_STAGES) {
      this.samples.set(stage, []);
    }
  }

  /**
   * Records a single latency measurement in milliseconds for a pipeline stage.
   */
  mark(stage: BenchStage, ms: number): void {
    if (!Number.isFinite(ms) || ms < 0) {
      return;
    }
    this.samples.get(stage)?.push(ms);
  }

  /**
   * Records the arrival timestamp (ms) of a processed frame for FPS calculation.
   */
  markFrame(timestampMs: number): void {
    if (!Number.isFinite(timestampMs)) {
      return;
    }
    if (this.firstFrameMs === null) {
      this.firstFrameMs = timestampMs;
    }
    this.lastFrameMs = timestampMs;
    this.frameCount++;
  }

  /**
   * Returns p50, p95, and sample count for every stage in `BENCH_STAGES`.
   */
  summary(): Record<BenchStage, StageSummary> {
    const out = {} as Record<BenchStage, StageSummary>;
    for (const stage of BENCH_STAGES) {
      const stageSamples = this.samples.get(stage) ?? [];
      out[stage] = {
        p50: computePercentile(stageSamples, 50),
        p95: computePercentile(stageSamples, 95),
        n: stageSamples.length,
      };
    }
    return out;
  }

  /**
   * Returns average sustained FPS across recorded frames, or 0 before 2 frames arrive.
   */
  fps(): number {
    if (this.frameCount < 2 || this.firstFrameMs === null || this.lastFrameMs === null) {
      return 0;
    }
    const elapsedMs = this.lastFrameMs - this.firstFrameMs;
    if (elapsedMs <= 0) {
      return 0;
    }
    return ((this.frameCount - 1) * MILLISECONDS_PER_SECOND) / elapsedMs;
  }

  /**
   * Clears all stage samples and frame timestamps.
   */
  reset(): void {
    for (const stage of BENCH_STAGES) {
      this.samples.set(stage, []);
    }
    this.frameCount = 0;
    this.firstFrameMs = null;
    this.lastFrameMs = null;
  }
}
