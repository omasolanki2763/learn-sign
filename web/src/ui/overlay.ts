/**
 * Canvas overlay: draws the detected skeleton and the verdict box.
 *
 * Kept separate from the harness so it can be drawn at any frequency the UI
 * chooses, and so it can be unit tested without a camera.
 */
import {
  COORDINATES_PER_LANDMARK,
  type BoxState,
  type FrameLandmarks,
  HAND_LANDMARK_COUNT,
} from '../landmarks/types';

/** Verdict colours. Grey means no verdict yet. */
export const BOX_COLORS: Record<BoxState, string> = {
  idle: '#9aa0a6',
  green: '#22c55e',
  amber: '#f59e0b',
  red: '#ef4444',
};

/** Colour of the skeleton itself, deliberately independent of the verdict. */
const SKELETON_COLOR = '#e5e7eb';

/** Skeleton edges of a MediaPipe hand, by landmark index. */
export const HAND_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

/** Landmark radius in pixels. */
const POINT_RADIUS_PX = 4;

/** Padding around a hand, as a fraction of its bounding box, so the box does not hug the fingers. */
const BOX_PADDING_RATIO = 0.15;

/**
 * Clears the canvas and draws every hand that was detected.
 *
 * @param ctx 2D context of the overlay canvas.
 * @param frame Landmarks for the current frame.
 * @param boxState Verdict state; decides the colour of the hand boxes.
 */
export function drawOverlay(ctx: CanvasRenderingContext2D, frame: FrameLandmarks, boxState: BoxState): void {
  const width = ctx.canvas.width;
  const height = ctx.canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.lineWidth = 2;

  const pose = frame.pose;
  if (pose) {
    drawPose(ctx, pose, width, height);
  }

  ctx.strokeStyle = BOX_COLORS[boxState];
  ctx.fillStyle = SKELETON_COLOR;

  for (const hand of [frame.leftHand, frame.rightHand]) {
    if (hand) {
      drawHand(ctx, hand, width, height);
      drawHandBox(ctx, hand, width, height);
    }
  }
}

/**
 * Draws the hands' landmarks and bones.
 *
 * @param ctx 2D context of the overlay canvas.
 * @param hand Flattened 21 * 3 hand landmarks in image coordinates.
 * @param width Canvas width in pixels.
 * @param height Canvas height in pixels.
 */
function drawHand(ctx: CanvasRenderingContext2D, hand: Float32Array, width: number, height: number): void {
  for (const [from, to] of HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(hand[from * COORDINATES_PER_LANDMARK] * width, hand[from * COORDINATES_PER_LANDMARK + 1] * height);
    ctx.lineTo(hand[to * COORDINATES_PER_LANDMARK] * width, hand[to * COORDINATES_PER_LANDMARK + 1] * height);
    ctx.stroke();
  }

  for (let point = 0; point < HAND_LANDMARK_COUNT; point++) {
    ctx.beginPath();
    ctx.arc(
      hand[point * COORDINATES_PER_LANDMARK] * width,
      hand[point * COORDINATES_PER_LANDMARK + 1] * height,
      POINT_RADIUS_PX,
      0,
      Math.PI * 2,
    );
    ctx.fill();
  }
}

/**
 * Draws the verdict box around one hand.
 *
 * @param ctx 2D context of the overlay canvas.
 * @param hand Flattened 21 * 3 hand landmarks in image coordinates.
 * @param width Canvas width in pixels.
 * @param height Canvas height in pixels.
 */
function drawHandBox(ctx: CanvasRenderingContext2D, hand: Float32Array, width: number, height: number): void {
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  for (let point = 0; point < HAND_LANDMARK_COUNT; point++) {
    const x = hand[point * COORDINATES_PER_LANDMARK] * width;
    const y = hand[point * COORDINATES_PER_LANDMARK + 1] * height;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }

  const paddingX = (maxX - minX) * BOX_PADDING_RATIO;
  const paddingY = (maxY - minY) * BOX_PADDING_RATIO;
  ctx.strokeRect(minX - paddingX, minY - paddingY, maxX - minX + paddingX * 2, maxY - minY + paddingY * 2);
}

/**
 * Draws a light outline of the upper body, so the signer can see they are framed.
 *
 * @param ctx 2D context of the overlay canvas.
 * @param pose Flattened 33 * 3 pose landmarks in image coordinates.
 * @param width Canvas width in pixels.
 * @param height Canvas height in pixels.
 */
function drawPose(ctx: CanvasRenderingContext2D, pose: Float32Array, width: number, height: number): void {
  const upperBodyEdges: ReadonlyArray<readonly [number, number]> = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  ];
  ctx.strokeStyle = '#3b82f6';
  for (const [from, to] of upperBodyEdges) {
    ctx.beginPath();
    ctx.moveTo(pose[from * COORDINATES_PER_LANDMARK] * width, pose[from * COORDINATES_PER_LANDMARK + 1] * height);
    ctx.lineTo(pose[to * COORDINATES_PER_LANDMARK] * width, pose[to * COORDINATES_PER_LANDMARK + 1] * height);
    ctx.stroke();
  }
}
