/**
 * Timestamp-driven prediction tick scheduler.
 *
 * Drives the fixed 100 ms recognition tick from frame timestamps rather than frame counts
 * so variable-rate 15, 30, and 60 FPS sources produce identical cadence without phase drift.
 */

/** Default prediction tick interval in milliseconds. */
export const TICK_PERIOD_MS = 100;

const TIMESTAMP_TOLERANCE_MS = 1e-6;

/**
 * Emits prediction ticks on a fixed timestamp grid regardless of camera/video FPS.
 */
export class TickScheduler {
  private readonly periodMs: number;
  private nextTickMs: number | null = null;

  /**
   * @param periodMs Target interval between ticks in milliseconds.
   */
  constructor(periodMs: number = TICK_PERIOD_MS) {
    if (!Number.isFinite(periodMs) || periodMs <= 0) {
      throw new Error(`TickScheduler periodMs must be positive, received ${periodMs}`);
    }
    this.periodMs = periodMs;
  }

  /**
   * Evaluates whether a prediction tick should fire at `timestampMs`.
   *
   * @param timestampMs Monotonic frame timestamp in milliseconds.
   * @returns `true` if a tick boundary has been reached or crossed since the last tick.
   */
  shouldTick(timestampMs: number): boolean {
    if (!Number.isFinite(timestampMs)) {
      return false;
    }
    if (this.nextTickMs === null) {
      this.nextTickMs = timestampMs + this.periodMs;
      return false;
    }
    if (timestampMs + TIMESTAMP_TOLERANCE_MS < this.nextTickMs) {
      return false;
    }
    // Why advance on the fixed periodMs grid rather than `timestampMs + periodMs`:
    // variable FPS intervals (e.g. 66.67 ms at 15 FPS) overshoot exact boundaries by 33 ms,
    // which would otherwise slip every subsequent tick and drop ticks per second.
    const stepsCrossed =
      Math.floor((timestampMs + TIMESTAMP_TOLERANCE_MS - this.nextTickMs) / this.periodMs) + 1;
    this.nextTickMs += stepsCrossed * this.periodMs;
    return true;
  }

  /**
   * Resets internal tick anchor so the next source starts cleanly.
   */
  reset(): void {
    this.nextTickMs = null;
  }
}
