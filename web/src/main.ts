/**
 * Main application entry point for LearnSign.
 *
 * Mounts the landmark harness: a camera or a video file on the left, the
 * detected skeleton drawn on top, and a status line. Everything runs locally in
 * the browser; no frame ever leaves the device.
 */
import { LandmarkClient } from './landmarks/landmarkClient';
import { drawOverlay } from './ui/overlay';
import { describeCameraError, type FrameSourceKind, startSource, type SourceFrame } from './ui/sourceSelect';

/** Status line text while the detectors are being fetched and compiled. */
const LOADING_MESSAGE = 'Loading landmark models…';

/** Status line text once frames are flowing. */
const RUNNING_MESSAGE = 'Tracking. Show one or both hands to the camera.';

/**
 * Returns the markup for the harness.
 *
 * @returns HTML for the app shell.
 */
function renderShell(): string {
  return `
    <main class="learnsign-container">
      <h1>LearnSign</h1>
      <p class="subtitle">Zero-cost, browser-only, real-time American Sign Language tutor</p>
      <section class="harness">
        <div class="stage">
          <canvas id="video-canvas" width="640" height="480"></canvas>
          <canvas id="overlay-canvas" width="640" height="480"></canvas>
        </div>
        <div class="controls">
          <button id="start-camera" type="button">Start camera</button>
          <label class="file-label">
            Load a video file
            <input id="video-file" type="file" accept="video/*" />
          </label>
        </div>
        <p id="status" role="status"></p>
      </section>
    </main>
  `;
}

/**
 * Initializes and mounts the LearnSign application to the DOM.
 *
 * Finds the `#app` element and sets up the root title and harness.
 */
export function mountApp(): void {
  const rootElement = document.getElementById('app');
  if (!rootElement) {
    // Why: Defensively handle the edge case where the DOM has not rendered #app yet.
    return;
  }

  rootElement.innerHTML = renderShell();
  const harness = createHarness(rootElement);
  if (harness) {
    harness.start();
  }
}

/** Elements the harness needs, resolved once. */
interface HarnessElements {
  videoCanvas: HTMLCanvasElement;
  overlayCanvas: HTMLCanvasElement;
  cameraButton: HTMLButtonElement;
  fileInput: HTMLInputElement;
  status: HTMLElement;
}

/**
 * Looks up the harness elements inside the mounted shell.
 *
 * @param root Mounted app root.
 * @returns The elements, or null when the shell is not present.
 */
function findHarnessElements(root: HTMLElement): HarnessElements | null {
  const videoCanvas = root.querySelector<HTMLCanvasElement>('#video-canvas');
  const overlayCanvas = root.querySelector<HTMLCanvasElement>('#overlay-canvas');
  const cameraButton = root.querySelector<HTMLButtonElement>('#start-camera');
  const fileInput = root.querySelector<HTMLInputElement>('#video-file');
  const status = root.querySelector<HTMLElement>('#status');

  if (!videoCanvas || !overlayCanvas || !cameraButton || !fileInput || !status) {
    return null;
  }
  return { videoCanvas, overlayCanvas, cameraButton, fileInput, status };
}

/**
 * Wires the buttons to the frame pipeline.
 *
 * @param root Mounted app root.
 * @returns A controller with a `start` method, or null when the shell is incomplete.
 */
function createHarness(root: HTMLElement): { start: () => void } | null {
  const elements = findHarnessElements(root);
  if (!elements) {
    return null;
  }

  const client = new LandmarkClient();
  let activeFrames: AsyncIterator<SourceFrame> | null = null;

  /**
   * Stops the current source, if any.
   */
  const stopActiveSource = async (): Promise<void> => {
    const running = activeFrames;
    activeFrames = null;
    // Why: calling return() runs the generator's cleanup, which releases the
    // camera tracks or the object URL.
    await running?.return?.(undefined);
  };

  /**
   * Switches the stage to one source and keeps drawing until it ends.
   *
   * @param kind Source kind.
   * @param file Video file for file mode.
   */
  const play = async (kind: FrameSourceKind, file?: File): Promise<void> => {
    await stopActiveSource();
    elements.status.textContent = LOADING_MESSAGE;

    try {
      const frames = await startSource(kind, file);
      const iterator = frames[Symbol.asyncIterator]();
      activeFrames = iterator;
      const videoContext = elements.videoCanvas.getContext('2d');
      const overlayContext = elements.overlayCanvas.getContext('2d');
      if (!videoContext || !overlayContext) {
        throw new Error('This browser cannot provide a 2D canvas context.');
      }

      elements.status.textContent = RUNNING_MESSAGE;
      for (;;) {
        const next = await iterator.next();
        if (next.done || activeFrames !== iterator) {
          break;
        }

        const { bitmap, timestampMs } = next.value;
        resizeCanvas(elements, bitmap.width, bitmap.height);
        // Why draw before process: process transfers the bitmap to the worker,
        // after which it can no longer be used here.
        videoContext.drawImage(bitmap, 0, 0, elements.videoCanvas.width, elements.videoCanvas.height);
        const landmarks = await client.process(bitmap, timestampMs);
        drawOverlay(overlayContext, landmarks, 'idle');
      }
    } catch (error) {
      elements.status.textContent = kind === 'webcam'
        ? describeCameraError(error)
        : 'That video could not be played. Try another file.';
    }
  };

  elements.cameraButton.addEventListener('click', () => {
    void play('webcam');
  });

  elements.fileInput.addEventListener('change', () => {
    const file = elements.fileInput.files?.[0];
    if (file) {
      void play('file', file);
    }
  });

  return {
    /**
     * Starts the harness without opening a source: both controls stay clickable
     * and the models are loaded on first use.
     */
    start: () => {
      elements.status.textContent = 'Choose the camera, or load a video file.';
    },
  };
}

/**
 * Matches both canvases to the incoming frame size.
 *
 * @param elements Harness elements.
 * @param width Frame width in pixels.
 * @param height Frame height in pixels.
 */
function resizeCanvas(elements: HarnessElements, width: number, height: number): void {
  for (const canvas of [elements.videoCanvas, elements.overlayCanvas]) {
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
  }
}

// Automatically mount when loaded in the browser document context
if (typeof document !== 'undefined') {
  mountApp();
}
