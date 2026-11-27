/**
 * Browser ONNX Runtime Web (WASM) probe for the ~1.85M-parameter dummy model.
 */
import * as ort from 'onnxruntime-web/wasm';

import { DERIVED_FEATURES_PER_FRAME } from '../features/derived';
import { FEATURES_PER_FRAME } from '../features/schema';

/** Public URL path for the static dummy ONNX probe committed under web/public/bench/. */
export const DUMMY_MODEL_URL = '/bench/dummy_model.onnx';

/** Same-origin directory where ORT WASM binaries and worker modules are served (COEP-safe). */
export const ORT_WASM_BASE_PATH = '/ort/';

/** Fixed temporal window length consumed by the dummy model probe. */
export const DUMMY_MODEL_WINDOW_FRAMES = 64;

/** Dummy output class count (matches GISLR 250 glosses). */
export const DUMMY_MODEL_NUM_CLASSES = 250;

/** Total per-frame feature width read directly from the v2 feature contract constants. */
export const DUMMY_MODEL_FEATURE_DIM = FEATURES_PER_FRAME + DERIVED_FEATURES_PER_FRAME;

export interface TensorLike {
  dims: readonly number[];
  data: Float32Array | ArrayLike<number>;
}

/** Minimal ORT InferenceSession interface kept injectable for unit tests. */
export interface InferenceSessionLike {
  run(feeds: Record<string, unknown>): Promise<Record<string, TensorLike>>;
  release(): Promise<void>;
}

/** Factory creating an ORT WASM inference session for a given URL and WASM thread count. */
export type InferenceSessionFactory = (
  modelUrl: string,
  threads: number
) => Promise<InferenceSessionLike>;

/**
 * Default ORT-Web session builder restricted strictly to the WASM execution provider.
 */
async function createWasmSession(
  modelUrl: string,
  threads: number
): Promise<InferenceSessionLike> {
  const safeThreads = Math.max(1, Math.floor(threads));
  ort.env.wasm.wasmPaths = ORT_WASM_BASE_PATH;
  ort.env.wasm.numThreads = safeThreads;
  const session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ['wasm'],
  });
  return session as unknown as InferenceSessionLike;
}

/**
 * Stateful probe managing one ORT WASM session and a reusable float32 input buffer.
 */
export class DummyModelProbe {
  private readonly sessionFactory: InferenceSessionFactory;
  private session: InferenceSessionLike | null = null;
  private readonly inputBuffer = new Float32Array(
    DUMMY_MODEL_WINDOW_FRAMES * DUMMY_MODEL_FEATURE_DIM
  );

  constructor(sessionFactory: InferenceSessionFactory = createWasmSession) {
    this.sessionFactory = sessionFactory;
  }

  /**
   * Loads the ONNX probe model, configures WASM thread count, and executes one warmup run.
   */
  async init(threads: number, modelUrl: string = DUMMY_MODEL_URL): Promise<void> {
    await this.dispose();
    this.session = await this.sessionFactory(modelUrl, threads);
    // Warm up WASM kernels before timed benchmark ticks begin.
    await this.infer(this.inputBuffer);
  }

  /**
   * Executes one forward pass on a `(1, 64, F)` float32 feature window.
   */
  async infer(featuresWindow?: Float32Array): Promise<Float32Array> {
    if (!this.session) {
      throw new Error('DummyModelProbe is not initialised');
    }
    if (featuresWindow) {
      if (featuresWindow.length !== this.inputBuffer.length) {
        throw new Error(
          `Expected ${this.inputBuffer.length} features, got ${featuresWindow.length}`
        );
      }
      this.inputBuffer.set(featuresWindow);
    }

    const inputTensor = new ort.Tensor('float32', this.inputBuffer, [
      1,
      DUMMY_MODEL_WINDOW_FRAMES,
      DUMMY_MODEL_FEATURE_DIM,
    ]);
    const results = await this.session.run({ features: inputTensor });
    const logits = results.logits;
    if (!logits) {
      throw new Error('Dummy ONNX model output missing expected "logits" tensor');
    }
    return logits.data instanceof Float32Array
      ? logits.data
      : Float32Array.from(logits.data);
  }

  /**
   * Releases underlying WASM session memory.
   */
  async dispose(): Promise<void> {
    if (this.session) {
      await this.session.release();
      this.session = null;
    }
  }
}
