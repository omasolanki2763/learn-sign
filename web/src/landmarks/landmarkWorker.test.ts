/**
 * Tests for the landmark worker's hand assignment and its dispose path.
 *
 * MediaPipe itself is mocked: the point of these tests is the mapping logic,
 * not the model.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';

const closedDetectors: string[] = [];

vi.mock('@mediapipe/tasks-vision', () => {
  class FakeDetector {
    private readonly modelPath: string;

    constructor(modelPath: string) {
      this.modelPath = modelPath;
    }

    static async createFromOptions(_fileset: unknown, options: { baseOptions: { modelAssetPath: string } }) {
      return new FakeDetector(options.baseOptions.modelAssetPath);
    }

    detectForVideo() {
      return { handedness: [], landmarks: [] };
    }

    close() {
      closedDetectors.push(this.modelPath);
    }
  }

  return {
    FilesetResolver: { forVisionTasks: async () => ({}) },
    HandLandmarker: FakeDetector,
    PoseLandmarker: FakeDetector,
    FaceLandmarker: FakeDetector,
  };
});

const { assignHands, handleRequest } = await import('./landmarkWorker');

/** One detection as the worker sees it before assignment. */
const leftLabelHigh = { label: 'Left', score: 0.9, index: 0 };
const leftLabelLow = { label: 'Left', score: 0.4, index: 1 };
const rightLabel = { label: 'Right', score: 0.8, index: 2 };

describe('assignHands', () => {
  it('treats the "Left" label as the left hand by default', () => {
    const result = assignHands([leftLabelHigh], false);

    expect(result.left).toBe(leftLabelHigh);
    expect(result.right).toBeNull();
  });

  it('swaps the labels when the swap setting is on', () => {
    const result = assignHands([leftLabelHigh], true);

    expect(result.left).toBeNull();
    expect(result.right).toBe(leftLabelHigh);
  });

  it('keeps the higher scoring detection when two share a label', () => {
    const result = assignHands([leftLabelLow, leftLabelHigh], false);

    expect(result.left).toBe(leftLabelHigh);
  });

  it('assigns both hands when the labels differ', () => {
    const result = assignHands([leftLabelHigh, rightLabel], false);

    expect(result.left).toBe(leftLabelHigh);
    expect(result.right).toBe(rightLabel);
  });
});

describe('dispose', () => {
  beforeEach(() => {
    closedDetectors.length = 0;
  });

  it('closes every loaded detector and forgets them', async () => {
    await handleRequest({ type: 'init', enableFace: true });
    expect(closedDetectors).toHaveLength(0);

    await handleRequest({ type: 'dispose' });

    expect(closedDetectors.sort()).toEqual([
      '/models/face_landmarker.task',
      '/models/hand_landmarker.task',
      '/models/pose_landmarker_lite.task',
    ]);
  });

  it('rejects frames that arrive after dispose', async () => {
    await handleRequest({ type: 'init', enableFace: false });
    await handleRequest({ type: 'dispose' });

    const bitmap = { close: vi.fn() } as unknown as ImageBitmap;
    await handleRequest({ type: 'process', requestId: 7, bitmap, timestampMs: 1 });

    expect(bitmap.close).toHaveBeenCalled();
  });
});
