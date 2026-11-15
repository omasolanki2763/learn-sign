/**
 * Frame sources: the webcam and a local video file.
 *
 * Both paths hand back bitmaps through one interface, so the harness does not
 * know or care where the pixels came from. Files are the fallback when a camera
 * is missing or refused, which is why the file path never touches
 * `getUserMedia`.
 */
/** Which kind of frame source to open. */
export type FrameSourceKind = 'webcam' | 'file';

/** Raised when the camera cannot be used, so the UI can fall back to a file. */
export class CameraUnavailableError extends Error {
  /**
   * @param message Text shown to the user.
   * @param options Standard error options; cause keeps the browser error.
   */
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = 'CameraUnavailableError';
  }
}

/** One frame ready for detection. */
export interface SourceFrame {
  bitmap: ImageBitmap;
  timestampMs: number;
}

/** Preferred webcam resolution; small enough to keep per-frame copies cheap. */
const WEBCAM_WIDTH = 640;
const WEBCAM_HEIGHT = 480;

/** Fallback frame interval when the browser lacks requestVideoFrameCallback. */
const FALLBACK_FRAME_INTERVAL_MS = 33;

/**
 * Creates the hidden video element both sources decode into.
 *
 * @returns A muted, inline video element that never plays audio.
 */
function createVideoElement(): HTMLVideoElement {
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  return video;
}

/**
 * Waits for the next presented video frame.
 *
 * @param video Video that is currently playing.
 * @returns Media time of the frame in seconds.
 */
function waitForVideoFrame(video: HTMLVideoElement): Promise<number> {
  // Why the feature test: requestVideoFrameCallback is absent in Safari and in
  // some test environments, and a timer is an acceptable, if less precise,
  // substitute there.
  const frameCallbackVideo = video as HTMLVideoElement & {
    requestVideoFrameCallback?: (callback: (now: number, metadata: { mediaTime: number }) => void) => number;
  };

  if (typeof frameCallbackVideo.requestVideoFrameCallback === 'function') {
    return new Promise<number>((resolve) => {
      frameCallbackVideo.requestVideoFrameCallback?.((_now, metadata) => resolve(metadata.mediaTime));
    });
  }

  return new Promise<number>((resolve) => {
    setTimeout(() => resolve(video.currentTime), FALLBACK_FRAME_INTERVAL_MS);
  });
}

/**
 * Turns a playing video into a stream of frames.
 *
 * @param video Playing video element.
 * @param onFinished Cleanup run when the stream ends or the consumer stops.
 * @returns Async iterable of bitmaps with capture times in milliseconds.
 */
async function* iterateFrames(video: HTMLVideoElement, onFinished: () => void): AsyncGenerator<SourceFrame> {
  try {
    while (!video.ended) {
      const mediaTimeSeconds = await waitForVideoFrame(video);
      const bitmap = await createImageBitmap(video);
      yield { bitmap, timestampMs: Math.round(mediaTimeSeconds * 1000) };
    }
  } finally {
    // Why: the stream, the object URL and the element must all be released even
    // when the consumer stops early with break.
    onFinished();
  }
}

/**
 * Starts a camera or a file as a frame source.
 *
 * @param kind 'webcam' for the camera, 'file' for the supplied video file.
 * @param file Video file; required for 'file'.
 * @returns Async iterable of frames.
 * @throws CameraUnavailableError when the camera is missing, blocked or fails.
 */
export async function startSource(kind: FrameSourceKind, file?: File): Promise<AsyncIterable<SourceFrame>> {
  const video = createVideoElement();

  if (kind === 'webcam') {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new CameraUnavailableError('This browser exposes no camera API. Load a video file instead.');
    }

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: WEBCAM_WIDTH }, height: { ideal: WEBCAM_HEIGHT } },
        audio: false,
      });
    } catch (cause) {
      // Why wrapped: the caller needs one error type to catch, whatever the
      // browser's DOMException said, plus the original cause for logging.
      throw new CameraUnavailableError(
        'Camera access was blocked. Allow it in the browser, or load a video file instead.',
        { cause },
      );
    }

    video.srcObject = stream;
    await video.play();
    return iterateFrames(video, () => {
      for (const track of stream.getTracks()) {
        track.stop();
      }
    });
  }

  if (!file) {
    throw new Error("startSource('file') needs a video file");
  }

  const objectUrl = URL.createObjectURL(file);
  video.src = objectUrl;
  video.loop = false;
  await video.play();

  return iterateFrames(video, () => {
    URL.revokeObjectURL(objectUrl);
    video.removeAttribute('src');
    video.load();
  });
}

/**
 * Turns a camera failure into text for the user.
 *
 * @param error Anything thrown by `startSource`.
 * @returns A sentence the UI can show as-is.
 */
export function describeCameraError(error: unknown): string {
  if (error instanceof CameraUnavailableError) {
    return error.message;
  }
  return 'The camera could not be started. Load a video file instead.';
}

