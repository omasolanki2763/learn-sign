/**
 * Hosted Step 0 benchmark runner & UI page (`bench.html`).
 *
 * Measures sustained FPS, per-stage latencies (landmarks, features, derived, model,
 * render, alignment), and 100 ms tick compute p95 across the 2x2 configuration matrix:
 * `{face off, face on} x {1 thread, max threads}` for 60 s per configuration,
 * and exports downloadable `step0-bench.json` reports.
 */
import { derivedFeatures, DERIVED_FEATURES_PER_FRAME } from '../features/derived';
import { normalizeFrame } from '../features/normalize';
import { FEATURES_PER_FRAME } from '../features/schema';
import { LandmarkClient } from '../landmarks/landmarkClient';
import type { FrameLandmarks } from '../landmarks/types';
import { drawOverlay } from '../ui/overlay';
import { type FrameSourceKind, startSource } from '../ui/sourceSelect';
import { AlignmentBench } from './alignmentBench';
import { DUMMY_MODEL_FEATURE_DIM, DUMMY_MODEL_WINDOW_FRAMES, DummyModelProbe } from './dummyModel';
import {
  type BenchStage,
  computePercentile,
  type StageSummary,
  Stopwatch,
} from './stopwatch';
import { TICK_PERIOD_MS, TickScheduler } from './tickScheduler';

/** Benchmark duration per configuration in milliseconds (60 seconds). */
export const BENCH_DURATION_MS = 60_000;

const MILLISECONDS_PER_SECOND = 1000;
const FALLBACK_CONCURRENCY = 4;

/** Single benchmark matrix configuration descriptor. */
export interface BenchConfigSpec {
  id: string;
  label: string;
  enableFace: boolean;
  threads: 1 | 'max';
}

/** Complete 2x2 matrix specified for Step 0 Task 3. */
export const BENCH_CONFIGS: readonly BenchConfigSpec[] = [
  { id: 'face-off-t1', label: 'Face OFF · 1 thread', enableFace: false, threads: 1 },
  { id: 'face-off-tmax', label: 'Face OFF · max threads', enableFace: false, threads: 'max' },
  { id: 'face-on-t1', label: 'Face ON · 1 thread', enableFace: true, threads: 1 },
  { id: 'face-on-tmax', label: 'Face ON · max threads', enableFace: true, threads: 'max' },
];

/** Schema of each downloadable `step0-bench.json` output file. */
export interface BenchReport {
  userAgent: string;
  hardwareConcurrency: number;
  resolution: { width: number; height: number };
  enableFace: boolean;
  threads: number;
  fps: number;
  stages: Record<BenchStage, StageSummary>;
  tickComputeP95: number;
}

export interface BuildBenchReportInput {
  enableFace: boolean;
  threads: number;
  resolution: { width: number; height: number };
  stopwatch: Stopwatch;
  tickComputeSamplesMs: readonly number[];
}

/**
 * Resolves effective thread count (`1` or `navigator.hardwareConcurrency`) for a config.
 */
export function resolveThreads(threadsSpec: 1 | 'max'): number {
  if (threadsSpec === 1) {
    return 1;
  }
  const hw =
    typeof navigator !== 'undefined' && typeof navigator.hardwareConcurrency === 'number'
      ? navigator.hardwareConcurrency
      : FALLBACK_CONCURRENCY;
  return Math.max(1, hw);
}

/**
 * Produces the standardized download filename for a configuration run.
 */
export function buildBenchFilename(enableFace: boolean, threads: number): string {
  const faceTag = enableFace ? 'face-on' : 'face-off';
  return `step0-bench-${faceTag}-t${threads}.json`;
}

/**
 * Assembles a serializable `step0-bench.json` report object from collected run metrics.
 */
export function buildBenchReport(input: BuildBenchReportInput): BenchReport {
  const userAgent = typeof navigator !== 'undefined' ? navigator.userAgent : 'unknown';
  const hardwareConcurrency =
    typeof navigator !== 'undefined' && typeof navigator.hardwareConcurrency === 'number'
      ? navigator.hardwareConcurrency
      : 0;

  return {
    userAgent,
    hardwareConcurrency,
    resolution: input.resolution,
    enableFace: input.enableFace,
    threads: input.threads,
    fps: input.stopwatch.fps(),
    stages: input.stopwatch.summary(),
    tickComputeP95: computePercentile(input.tickComputeSamplesMs, 95),
  };
}

/**
 * Formats a compact one-line status summary for an executed bench configuration.
 */
export function formatReportSummary(report: BenchReport): string {
  const faceStr = report.enableFace ? 'face=on' : 'face=off';
  return (
    `${faceStr}, threads=${report.threads}: ` +
    `FPS=${report.fps.toFixed(1)}, tickP95=${report.tickComputeP95.toFixed(2)} ms ` +
    `(lm=${report.stages.landmarks.p95.toFixed(1)}, ` +
    `feat=${report.stages.features.p95.toFixed(2)}, ` +
    `der=${report.stages.derived.p95.toFixed(2)}, ` +
    `model=${report.stages.model.p95.toFixed(1)}, ` +
    `align=${report.stages.alignment.p95.toFixed(2)}, ` +
    `render=${report.stages.render.p95.toFixed(2)} ms)`
  );
}

interface BenchDomElements {
  videoCanvas: HTMLCanvasElement;
  overlayCanvas: HTMLCanvasElement;
  sourceCameraBtn: HTMLButtonElement;
  sourceFileInput: HTMLInputElement;
  runAllBtn: HTMLButtonElement;
  stopBtn: HTMLButtonElement;
  configSelect: HTMLSelectElement;
  runSingleBtn: HTMLButtonElement;
  statusEl: HTMLElement;
  isolationEl: HTMLElement;
  downloadsEl: HTMLElement;
}

function renderBenchMarkup(): string {
  const options = BENCH_CONFIGS.map(
    (c, idx) => `<option value="${idx}">${c.label}</option>`
  ).join('');

  return `
    <main class="learnsign-bench">
      <header>
        <h1>LearnSign — Step 0 Benchmark</h1>
        <p><a href="/">Back to Landmark Harness</a></p>
        <p id="bench-isolation"></p>
      </header>
      <section class="bench-controls">
        <div>
          <strong>1. Choose input source (for the active bench):</strong>
          <button id="bench-start-camera" type="button">Use Webcam</button>
          <label>
            Or Video File:
            <input id="bench-video-file" type="file" accept="video/*" />
          </label>
        </div>
        <div>
          <strong>2. Run matrix (60 s per config):</strong>
          <button id="bench-run-all" type="button">Run All 4 Configs (4 × 60 s)</button>
          <select id="bench-config-select">${options}</select>
          <button id="bench-run-single" type="button">Run Selected Config (60 s)</button>
          <button id="bench-stop" type="button">Stop</button>
        </div>
        <p id="bench-status" role="status">Ready. Choose Webcam or load a video file to start.</p>
      </section>
      <section class="bench-stage">
        <canvas id="bench-video-canvas" width="640" height="480"></canvas>
        <canvas id="bench-overlay-canvas" width="640" height="480"></canvas>
      </section>
      <section>
        <h2>Completed Reports (Download JSON)</h2>
        <div id="bench-downloads"></div>
      </section>
    </main>
  `;
}

function findBenchElements(root: HTMLElement): BenchDomElements | null {
  const videoCanvas = root.querySelector<HTMLCanvasElement>('#bench-video-canvas');
  const overlayCanvas = root.querySelector<HTMLCanvasElement>('#bench-overlay-canvas');
  const sourceCameraBtn = root.querySelector<HTMLButtonElement>('#bench-start-camera');
  const sourceFileInput = root.querySelector<HTMLInputElement>('#bench-video-file');
  const runAllBtn = root.querySelector<HTMLButtonElement>('#bench-run-all');
  const stopBtn = root.querySelector<HTMLButtonElement>('#bench-stop');
  const configSelect = root.querySelector<HTMLSelectElement>('#bench-config-select');
  const runSingleBtn = root.querySelector<HTMLButtonElement>('#bench-run-single');
  const statusEl = root.querySelector<HTMLElement>('#bench-status');
  const isolationEl = root.querySelector<HTMLElement>('#bench-isolation');
  const downloadsEl = root.querySelector<HTMLElement>('#bench-downloads');

  if (
    !videoCanvas ||
    !overlayCanvas ||
    !sourceCameraBtn ||
    !sourceFileInput ||
    !runAllBtn ||
    !stopBtn ||
    !configSelect ||
    !runSingleBtn ||
    !statusEl ||
    !isolationEl ||
    !downloadsEl
  ) {
    return null;
  }
  return {
    videoCanvas,
    overlayCanvas,
    sourceCameraBtn,
    sourceFileInput,
    runAllBtn,
    stopBtn,
    configSelect,
    runSingleBtn,
    statusEl,
    isolationEl,
    downloadsEl,
  };
}

/**
 * Triggers browser download of a JSON BenchReport object.
 */
function appendDownloadEntry(container: HTMLElement, report: BenchReport): void {
  const filename = buildBenchFilename(report.enableFace, report.threads);
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);

  const row = document.createElement('div');
  row.className = 'bench-report-row';

  const summary = document.createElement('span');
  summary.textContent = `${formatReportSummary(report)} `;

  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.textContent = `Download ${filename}`;

  row.appendChild(summary);
  row.appendChild(link);
  container.appendChild(row);
}

/**
 * Copies non-NaN v2 features into a scratch frame and shifts them into the rolling window.
 */
function pushFrameIntoWindow(
  normFrame: Float32Array,
  derivedFrame: Float32Array,
  scratchFrame: Float32Array,
  windowBuffer: Float32Array
): void {
  for (let i = 0; i < FEATURES_PER_FRAME; i++) {
    const val = normFrame[i];
    scratchFrame[i] = Number.isNaN(val) ? 0 : val;
  }
  for (let i = 0; i < DERIVED_FEATURES_PER_FRAME; i++) {
    const val = derivedFrame[i];
    scratchFrame[FEATURES_PER_FRAME + i] = Number.isNaN(val) ? 0 : val;
  }

  windowBuffer.copyWithin(0, DUMMY_MODEL_FEATURE_DIM);
  windowBuffer.set(
    scratchFrame,
    (DUMMY_MODEL_WINDOW_FRAMES - 1) * DUMMY_MODEL_FEATURE_DIM
  );
}

/**
 * Executes a single 60-second benchmark configuration against the selected source.
 */
async function runSingleConfig(
  elements: BenchDomElements,
  config: BenchConfigSpec,
  sourceKind: FrameSourceKind,
  sourceFile: File | undefined,
  shouldStop: () => boolean
): Promise<BenchReport> {
  const threads = resolveThreads(config.threads);
  const client = new LandmarkClient();
  const dummyModel = new DummyModelProbe();
  const stopwatch = new Stopwatch();
  const tickScheduler = new TickScheduler(TICK_PERIOD_MS);
  const alignmentBench = AlignmentBench.createRandom();
  const tickComputeSamplesMs: number[] = [];

  const scratchFrame = new Float32Array(DUMMY_MODEL_FEATURE_DIM);
  const windowBuffer = new Float32Array(DUMMY_MODEL_WINDOW_FRAMES * DUMMY_MODEL_FEATURE_DIM);
  let prevNormFrame: Float32Array | null = null;
  let prevTimestampMs: number | null = null;
  let width = 640;
  let height = 480;

  let frameIterator: AsyncIterator<{ bitmap: ImageBitmap; timestampMs: number }> | null = null;

  try {
    elements.statusEl.textContent = `Initialising ${config.label} (threads=${threads})...`;
    await client.init({ enableFace: config.enableFace });
    await dummyModel.init(threads);

    const stream = await startSource(sourceKind, sourceFile);
    frameIterator = stream[Symbol.asyncIterator]();

    const videoCtx = elements.videoCanvas.getContext('2d');
    const overlayCtx = elements.overlayCanvas.getContext('2d');
    if (!videoCtx || !overlayCtx) {
      throw new Error('2D canvas context unavailable');
    }

    const startWallMs = performance.now();
    for (;;) {
      if (shouldStop()) {
        break;
      }
      const elapsedMs = performance.now() - startWallMs;
      if (elapsedMs >= BENCH_DURATION_MS) {
        break;
      }

      const remainingSec = Math.max(0, Math.ceil((BENCH_DURATION_MS - elapsedMs) / 1000));
      elements.statusEl.textContent = `Running ${config.label} (${threads}t) — ${remainingSec}s remaining...`;

      const next = await frameIterator.next();
      if (next.done) {
        break;
      }

      const { bitmap, timestampMs } = next.value;
      width = bitmap.width;
      height = bitmap.height;
      if (elements.videoCanvas.width !== width || elements.videoCanvas.height !== height) {
        elements.videoCanvas.width = width;
        elements.videoCanvas.height = height;
        elements.overlayCanvas.width = width;
        elements.overlayCanvas.height = height;
      }
      videoCtx.drawImage(bitmap, 0, 0, width, height);

      const tLandmarks0 = performance.now();
      let landmarks: FrameLandmarks;
      try {
        landmarks = await client.process(bitmap, timestampMs);
      } catch (err) {
        bitmap.close();
        throw err;
      }
      stopwatch.mark('landmarks', performance.now() - tLandmarks0);
      stopwatch.markFrame(timestampMs);

      const aspectRatio = height > 0 ? width / height : 1.0;
      const tFeat0 = performance.now();
      const normFrame = normalizeFrame(
        landmarks.leftHand,
        landmarks.rightHand,
        landmarks.pose,
        aspectRatio
      );
      const featMs = performance.now() - tFeat0;
      stopwatch.mark('features', featMs);

      const dtS =
        prevTimestampMs !== null ? (timestampMs - prevTimestampMs) / MILLISECONDS_PER_SECOND : null;
      const tDerived0 = performance.now();
      const derivedFrame = derivedFeatures(normFrame, prevNormFrame, dtS);
      const derivedMs = performance.now() - tDerived0;
      stopwatch.mark('derived', derivedMs);

      prevNormFrame = normFrame;
      prevTimestampMs = timestampMs;

      pushFrameIntoWindow(normFrame, derivedFrame, scratchFrame, windowBuffer);

      const tAlign0 = performance.now();
      alignmentBench.step(scratchFrame);
      const alignMs = performance.now() - tAlign0;
      stopwatch.mark('alignment', alignMs);

      const tRender0 = performance.now();
      drawOverlay(overlayCtx, landmarks, 'idle');
      const renderMs = performance.now() - tRender0;
      stopwatch.mark('render', renderMs);

      if (tickScheduler.shouldTick(timestampMs)) {
        const tModel0 = performance.now();
        await dummyModel.infer(windowBuffer);
        const modelMs = performance.now() - tModel0;
        stopwatch.mark('model', modelMs);
        tickComputeSamplesMs.push(featMs + derivedMs + alignMs + modelMs + renderMs);
      }
    }
  } finally {
    await frameIterator?.return?.(undefined);
    await dummyModel.dispose();
    client.dispose();
  }

  return buildBenchReport({
    enableFace: config.enableFace,
    threads,
    resolution: { width, height },
    stopwatch,
    tickComputeSamplesMs,
  });
}

/**
 * Mounts the Step 0 Benchmark page into `rootElement`.
 */
export function mountBenchPage(rootElement: HTMLElement): void {
  rootElement.innerHTML = renderBenchMarkup();
  const elements = findBenchElements(rootElement);
  if (!elements) {
    return;
  }

  const isIsolated =
    typeof window !== 'undefined' && Boolean((window as Window).crossOriginIsolated);
  elements.isolationEl.textContent = isIsolated
    ? 'crossOriginIsolated=true (SharedArrayBuffer multi-threading enabled)'
    : 'crossOriginIsolated=false (multi-threaded WASM requires COOP/COEP headers)';

  let selectedKind: FrameSourceKind = 'webcam';
  let selectedFile: File | undefined;
  let stopRequested = false;
  let isRunning = false;

  elements.sourceCameraBtn.addEventListener('click', () => {
    selectedKind = 'webcam';
    selectedFile = undefined;
    elements.statusEl.textContent = 'Source set to Webcam. Click Run to start.';
  });

  elements.sourceFileInput.addEventListener('change', () => {
    const file = elements.sourceFileInput.files?.[0];
    if (file) {
      selectedKind = 'file';
      selectedFile = file;
      elements.statusEl.textContent = `Source set to file "${file.name}". Click Run to start.`;
    }
  });

  elements.stopBtn.addEventListener('click', () => {
    stopRequested = true;
    elements.statusEl.textContent = 'Stopping benchmark after current frame...';
  });

  const executeConfigs = async (configs: readonly BenchConfigSpec[]): Promise<void> => {
    if (isRunning) {
      return;
    }
    isRunning = true;
    stopRequested = false;
    try {
      for (const config of configs) {
        if (stopRequested) {
          break;
        }
        const report = await runSingleConfig(
          elements,
          config,
          selectedKind,
          selectedFile,
          () => stopRequested
        );
        appendDownloadEntry(elements.downloadsEl, report);
      }
      elements.statusEl.textContent = stopRequested
        ? 'Benchmark stopped.'
        : `Completed ${configs.length} configuration(s). Download JSON below.`;
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      elements.statusEl.textContent = `Benchmark failed: ${msg}`;
    } finally {
      isRunning = false;
    }
  };

  elements.runAllBtn.addEventListener('click', () => {
    void executeConfigs(BENCH_CONFIGS);
  });

  elements.runSingleBtn.addEventListener('click', () => {
    const idx = Number.parseInt(elements.configSelect.value, 10) || 0;
    const chosen = BENCH_CONFIGS[idx] ?? BENCH_CONFIGS[0];
    void executeConfigs([chosen]);
  });
}

if (typeof document !== 'undefined') {
  const root = document.getElementById('bench-app');
  if (root) {
    mountBenchPage(root);
  }
}
