/**
 * Banded incremental Dynamic Time Warping (DTW) alignment benchmark (Step 1b).
 *
 * Maintains a rolling Sakoe-Chiba banded cost row per reference template so each arriving
 * feature frame updates in bounded O(K * (2R + 1) * F) time without dynamic allocation.
 */
import { DERIVED_FEATURES_PER_FRAME } from '../features/derived';
import { FEATURES_PER_FRAME } from '../features/schema';

/** Number of reference templates evaluated per sign step (Step 1b spec). */
export const BANDED_DTW_NUM_REFS = 5;

/** Reference sequence length in frames (3 seconds at 30 FPS). */
export const BANDED_DTW_REF_FRAMES = 90;

/** Sakoe-Chiba half-band radius in frames (±15). */
export const BANDED_DTW_BAND_RADIUS = 15;

/** Default concatenated v2 frame feature dimension read from schema constants. */
export const DEFAULT_ALIGNMENT_FEATURE_DIM = FEATURES_PER_FRAME + DERIVED_FEATURES_PER_FRAME;

/** Step 1b alignment stage p95 latency gate in milliseconds. */
export const ALIGNMENT_P95_GATE_MS = 5.0;

/**
 * Generates deterministic pseudo-random reference templates for benchmark timing.
 */
function buildDeterministicRefs(
  numRefs: number,
  refFrames: number,
  featureDim: number
): Float32Array[] {
  const refs: Float32Array[] = [];
  let state = 0x12345678;
  for (let r = 0; r < numRefs; r++) {
    const buf = new Float32Array(refFrames * featureDim);
    for (let i = 0; i < buf.length; i++) {
      // Why integer LCG instead of Math.random(): keeps bench runs deterministic across engines.
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      buf[i] = (state / 0xffffffff) * 2 - 1;
    }
    refs.push(buf);
  }
  return refs;
}

/**
 * Computes L2 Euclidean distance between `query` and a slice of `ref` at `refFrame`.
 */
function frameL2Distance(
  query: Float32Array,
  ref: Float32Array,
  refFrame: number,
  featureDim: number
): number {
  const offset = refFrame * featureDim;
  let sumSq = 0;
  for (let i = 0; i < featureDim; i++) {
    const diff = query[i] - ref[offset + i];
    sumSq += diff * diff;
  }
  return Math.sqrt(sumSq);
}

/**
 * Incremental banded DTW stepper updating K rolling DP rows per incoming feature frame.
 */
export class AlignmentBench {
  readonly numRefs: number;
  readonly refFrames: number;
  readonly featureDim: number;
  readonly bandRadius: number;

  private readonly refs: Float32Array[];
  private prevRows: Float32Array[];
  private currRows: Float32Array[];
  private readonly latestCosts: Float32Array;
  private frameIndex = 0;

  constructor(
    refs: Float32Array[],
    refFrames: number,
    featureDim: number = DEFAULT_ALIGNMENT_FEATURE_DIM,
    bandRadius: number = BANDED_DTW_BAND_RADIUS
  ) {
    this.refs = refs;
    this.numRefs = refs.length;
    this.refFrames = refFrames;
    this.featureDim = featureDim;
    this.bandRadius = bandRadius;

    this.prevRows = refs.map(() => new Float32Array(refFrames).fill(Infinity));
    this.currRows = refs.map(() => new Float32Array(refFrames).fill(Infinity));
    this.latestCosts = new Float32Array(this.numRefs);
  }

  /**
   * Builds the default Step 1b benchmark (`K=5`, `M=90`, v2 `FEATURES_PER_FRAME + DERIVED_FEATURES_PER_FRAME`, band `±15`).
   */
  static createRandom(
    numRefs: number = BANDED_DTW_NUM_REFS,
    refFrames: number = BANDED_DTW_REF_FRAMES,
    featureDim: number = DEFAULT_ALIGNMENT_FEATURE_DIM,
    bandRadius: number = BANDED_DTW_BAND_RADIUS
  ): AlignmentBench {
    const refs = buildDeterministicRefs(numRefs, refFrames, featureDim);
    return new AlignmentBench(refs, refFrames, featureDim, bandRadius);
  }

  /**
   * Advances banded DTW by one incoming frame and returns each reference's diagonal/end cost.
   */
  step(frameFeatures: Float32Array): Float32Array {
    if (this.frameIndex >= this.refFrames) {
      // Why wrap: sustains steady-state rolling sign windows across 60s benchmark sessions.
      this.reset();
    }

    const t = this.frameIndex;
    const jStart = Math.max(0, t - this.bandRadius);
    const jEnd = Math.min(this.refFrames - 1, t + this.bandRadius);

    for (let r = 0; r < this.numRefs; r++) {
      const ref = this.refs[r];
      const prev = this.prevRows[r];
      const curr = this.currRows[r];

      for (let j = jStart; j <= jEnd; j++) {
        const localCost = frameL2Distance(frameFeatures, ref, j, this.featureDim);
        if (t === 0 && j === 0) {
          curr[j] = localCost;
          continue;
        }
        const fromPrevRow = prev[j];
        const fromDiag = j > 0 ? prev[j - 1] : Infinity;
        const fromLeft = j > jStart ? curr[j - 1] : Infinity;
        const bestPrev = Math.min(fromPrevRow, fromDiag, fromLeft);
        curr[j] = localCost + bestPrev;
      }

      const diagonalIdx = Math.min(t, this.refFrames - 1);
      this.latestCosts[r] = curr[diagonalIdx];

      // Reset spent row entries before swapping buffers so future steps see Infinity outside band.
      prev.fill(Infinity);
      this.prevRows[r] = curr;
      this.currRows[r] = prev;
    }

    this.frameIndex++;
    return this.latestCosts;
  }

  /**
   * Resets rolling cost rows back to frame 0.
   */
  reset(): void {
    for (let r = 0; r < this.numRefs; r++) {
      this.prevRows[r].fill(Infinity);
      this.currRows[r].fill(Infinity);
    }
    this.latestCosts.fill(0);
    this.frameIndex = 0;
  }
}
