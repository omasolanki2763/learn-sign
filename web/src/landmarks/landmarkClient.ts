/**
 * Client side handle for the landmark worker.
 *
 * Everything MediaPipe related runs inside a dedicated worker, so the video
 * element and the canvas keep the main thread. This client owns the message
 * protocol: one request id per frame, promise settled when its own reply comes
 * back, and no work accepted once disposed.
 */
import {
  type FrameLandmarks,
  type LandmarkRequest,
  type LandmarkResponse,
} from './types';

/** The subset of the Worker API this client depends on, kept small so it can be faked in tests. */
export interface LandmarkWorkerLike {
  postMessage(message: LandmarkRequest, transfer?: Transferable[]): void;
  addEventListener(type: 'message', listener: (event: MessageEvent<LandmarkResponse>) => void): void;
  terminate(): void;
}

/** Creates a worker; injected in tests so no real worker or WebAssembly is needed. */
export type LandmarkWorkerFactory = () => LandmarkWorkerLike;

/** One in-flight frame request. */
interface PendingRequest {
  resolve: (landmarks: FrameLandmarks) => void;
  reject: (error: Error) => void;
}

/**
 * Builds the default worker: a module worker loaded by Vite.
 *
 * @returns A worker running `landmarkWorker.ts`.
 */
function createModuleWorker(): LandmarkWorkerLike {
  // Why the cast: DOM's Worker types its listener as a generic EventListener,
  // while this interface narrows it to the landmark message payload. The cast
  // is local and the runtime object is a genuine Worker.
  return new Worker(new URL('./landmarkWorker.ts', import.meta.url), { type: 'module' }) as unknown as LandmarkWorkerLike;
}

export class LandmarkClient {
  private readonly workerFactory: LandmarkWorkerFactory;
  private worker: LandmarkWorkerLike | null = null;
  private nextRequestId = 1;
  private readonly pendingRequests = new Map<number, PendingRequest>();
  private initWaiter: { resolve: () => void; reject: (error: Error) => void } | null = null;
  private isDisposed = false;

  /**
   * @param workerFactory Builds the worker; defaults to a Vite module worker.
   */
  constructor(workerFactory: LandmarkWorkerFactory = createModuleWorker) {
    this.workerFactory = workerFactory;
  }

  /**
   * Starts the worker and waits until its detectors are loaded.
   *
   * @param options.enableFace Also run the face detector. Its output is thrown
   *   away: it exists only so its frame cost can be measured.
   * @returns Resolves when the worker reports ready.
   */
  async init(options: { enableFace: boolean }): Promise<void> {
    if (this.isDisposed) {
      throw new Error('LandmarkClient has been disposed');
    }
    if (this.worker) {
      throw new Error('LandmarkClient has already been initialised');
    }

    const worker = this.workerFactory();
    this.worker = worker;
    worker.addEventListener('message', (event) => this.handleMessage(event.data));

    await new Promise<void>((resolve, reject) => {
      this.initWaiter = { resolve: () => resolve(), reject };
      worker.postMessage({ type: 'init', enableFace: options.enableFace });
    });
  }

  /**
   * Detects landmarks in one frame.
   *
   * @param frame Bitmap of the frame. Ownership is transferred to the worker,
   *   so the caller must not use it afterwards.
   * @param timestampMs Capture time of the frame in milliseconds.
   * @returns The landmarks for exactly this frame.
   */
  async process(frame: ImageBitmap, timestampMs: number): Promise<FrameLandmarks> {
    const worker = this.worker;
    if (this.isDisposed || !worker) {
      throw new Error('LandmarkClient has been disposed');
    }

    const requestId = this.nextRequestId++;
    return new Promise<FrameLandmarks>((resolve, reject) => {
      this.pendingRequests.set(requestId, { resolve, reject });
      worker.postMessage({ type: 'process', requestId, bitmap: frame, timestampMs }, [frame]);
    });
  }

  /**
   * Stops the worker and fails any frame still in flight.
   */
  dispose(): void {
    if (this.isDisposed) {
      return;
    }
    this.isDisposed = true;
    this.worker?.terminate();
    this.worker = null;

    const disposalError = new Error('LandmarkClient has been disposed');
    this.pendingRequests.forEach((request) => request.reject(disposalError));
    this.pendingRequests.clear();
    this.initWaiter?.reject(disposalError);
    this.initWaiter = null;
  }

  /**
   * Routes one message from the worker to whoever is waiting for it.
   *
   * @param response Message payload from the worker.
   */
  private handleMessage(response: LandmarkResponse): void {
    if (response.type === 'ready') {
      const waiter = this.initWaiter;
      this.initWaiter = null;
      waiter?.resolve();
      return;
    }

    if (response.type === 'result') {
      const request = this.pendingRequests.get(response.requestId);
      this.pendingRequests.delete(response.requestId);
      request?.resolve(response.landmarks);
      return;
    }

    const error = new Error(response.message);
    if (response.requestId === null) {
      // Why: a startup failure has no request id and can only fail init.
      this.initWaiter?.reject(error);
      this.initWaiter = null;
      return;
    }
    const request = this.pendingRequests.get(response.requestId);
    this.pendingRequests.delete(response.requestId);
    request?.reject(error);
  }
}

