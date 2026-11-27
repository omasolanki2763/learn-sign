/**
 * Tests for DummyModelProbe ONNX runtime wrapper and v2 feature input contracts.
 */
import { describe, expect, it, vi } from 'vitest';

import { DERIVED_FEATURES_PER_FRAME } from '../features/derived';
import { FEATURES_PER_FRAME } from '../features/schema';
import {
  DUMMY_MODEL_FEATURE_DIM,
  DUMMY_MODEL_NUM_CLASSES,
  DUMMY_MODEL_URL,
  DUMMY_MODEL_WINDOW_FRAMES,
  DummyModelProbe,
  ORT_WASM_BASE_PATH,
  type InferenceSessionLike,
} from './dummyModel';

describe('DummyModelProbe', () => {
  it('derives input width from v2 schema constants instead of hardcoded 144', () => {
    expect(DUMMY_MODEL_FEATURE_DIM).toBe(FEATURES_PER_FRAME + DERIVED_FEATURES_PER_FRAME);
    expect(DUMMY_MODEL_FEATURE_DIM).not.toBe(144);
    expect(DUMMY_MODEL_WINDOW_FRAMES).toBe(64);
    expect(DUMMY_MODEL_URL).toBe('/bench/dummy_model.onnx');
    expect(ORT_WASM_BASE_PATH).toBe('/ort/');
  });

  it('initialises session with requested thread count and runs inference on (1, 64, F)', async () => {
    const runMock = vi.fn().mockResolvedValue({
      logits: {
        dims: [1, DUMMY_MODEL_NUM_CLASSES],
        data: new Float32Array(DUMMY_MODEL_NUM_CLASSES).fill(0.25),
      },
    });
    const releaseMock = vi.fn().mockResolvedValue(undefined);
    const fakeSession: InferenceSessionLike = {
      run: runMock,
      release: releaseMock,
    };
    const factory = vi.fn().mockResolvedValue(fakeSession);

    const probe = new DummyModelProbe(factory);
    await probe.init(4);

    expect(factory).toHaveBeenCalledWith(DUMMY_MODEL_URL, 4);
    // init performs one warmup inference
    expect(runMock).toHaveBeenCalledTimes(1);

    const windowBuf = new Float32Array(DUMMY_MODEL_WINDOW_FRAMES * DUMMY_MODEL_FEATURE_DIM).fill(1);
    const out = await probe.infer(windowBuf);
    expect(out.length).toBe(DUMMY_MODEL_NUM_CLASSES);
    expect(out[0]).toBeCloseTo(0.25, 5);
    expect(runMock).toHaveBeenCalledTimes(2);

    await probe.dispose();
    expect(releaseMock).toHaveBeenCalledTimes(1);
  });

  it('rejects inference before init()', async () => {
    const probe = new DummyModelProbe(vi.fn());
    await expect(probe.infer()).rejects.toThrow(/not initialised/i);
  });
});
