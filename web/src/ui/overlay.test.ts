/**
 * Tests for the skeleton overlay.
 *
 * A recording fake context is enough: what matters is how many points are
 * drawn per hand and which colour the verdict box uses.
 */
import { describe, it, expect } from 'vitest';
import { BOX_COLORS, drawOverlay } from './overlay';
import type { FrameLandmarks } from '../landmarks/types';

/** Canvas size used by every test. */
const CANVAS_WIDTH = 640;
const CANVAS_HEIGHT = 480;

/** Records the drawing calls the overlay makes. */
class RecordingContext {
  readonly canvas = { width: CANVAS_WIDTH, height: CANVAS_HEIGHT };
  fillStyle = '';
  strokeStyle = '';
  lineWidth = 0;
  arcCount = 0;
  readonly rectStyles: string[] = [];
  clearCount = 0;

  clearRect(): void {
    this.clearCount++;
  }

  beginPath(): void {}
  moveTo(): void {}
  lineTo(): void {}
  stroke(): void {}
  fill(): void {}

  arc(): void {
    this.arcCount++;
  }

  strokeRect(): void {
    this.rectStyles.push(this.strokeStyle);
  }
}

/**
 * Builds one hand with every landmark at a distinct position inside the frame.
 *
 * @param offset Shifts all points so two hands are told apart if needed.
 * @returns Flattened 21 * 3 landmarks.
 */
function makeHand(offset: number): Float32Array {
  const hand = new Float32Array(21 * 3);
  for (let point = 0; point < 21; point++) {
    hand[point * 3] = 0.1 + offset + point * 0.001;
    hand[point * 3 + 1] = 0.2 + offset + point * 0.001;
    hand[point * 3 + 2] = 0;
  }
  return hand;
}

/**
 * Builds a landmark frame.
 *
 * @param leftHand Left hand or null.
 * @param rightHand Right hand or null.
 * @returns Frame in the shared shape.
 */
function makeFrame(leftHand: Float32Array | null, rightHand: Float32Array | null): FrameLandmarks {
  return { timestampMs: 0, leftHand, rightHand, pose: null };
}

describe('drawOverlay', () => {
  it('draws 21 points for a single detected hand', () => {
    const ctx = new RecordingContext();

    drawOverlay(ctx as unknown as CanvasRenderingContext2D, makeFrame(makeHand(0), null), 'idle');

    expect(ctx.arcCount).toBe(21);
  });

  it('draws 42 points when both hands are detected', () => {
    const ctx = new RecordingContext();

    drawOverlay(ctx as unknown as CanvasRenderingContext2D, makeFrame(makeHand(0), makeHand(0.3)), 'idle');

    expect(ctx.arcCount).toBe(42);
  });

  it('draws no points and no box when no hand is detected', () => {
    const ctx = new RecordingContext();

    drawOverlay(ctx as unknown as CanvasRenderingContext2D, makeFrame(null, null), 'green');

    expect(ctx.arcCount).toBe(0);
    expect(ctx.rectStyles).toHaveLength(0);
  });

  it('colours the box with the verdict state', () => {
    for (const state of ['idle', 'green', 'amber', 'red'] as const) {
      const ctx = new RecordingContext();

      drawOverlay(ctx as unknown as CanvasRenderingContext2D, makeFrame(makeHand(0), null), state);

      expect(ctx.rectStyles).toEqual([BOX_COLORS[state]]);
    }
  });

  it('clears the canvas before drawing', () => {
    const ctx = new RecordingContext();

    drawOverlay(ctx as unknown as CanvasRenderingContext2D, makeFrame(makeHand(0), null), 'idle');

    expect(ctx.clearCount).toBe(1);
  });
});
