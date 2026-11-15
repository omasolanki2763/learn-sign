/**
 * Landmark worker: runs MediaPipe Tasks off the main thread.
 *
 * Why a worker at all: detection is the most expensive thing in the app. Run on
 * the main thread it would stutter the video and the overlay. Frames arrive as
 * transferred `ImageBitmap`s, which avoids copying pixels.
 *
 * The detector outputs are flattened here into `FrameLandmarks`, so the rest of
 * the app never touches MediaPipe result objects.
 */
import { FaceLandmarker, FilesetResolver, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision';
import {
  COORDINATES_PER_LANDMARK,
  FLAT_HAND_LENGTH,
  FLAT_POSE_LENGTH,
  type FrameLandmarks,
  type LandmarkRequest,
  type LandmarkResponse,
  POSE_LANDMARK_COUNT,
} from './types';

/** Where the downloaded `.task` bundles live, relative to the site root. */
const MODEL_BASE_PATH = '/models';

/** Detector settings. Two hands is all ASL needs here. */
const MAX_HANDS = 2;

/** MediaPipe's WASM runtime version must match the installed tasks-vision package. */
const WASM_BASE_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm';

/** The three detectors this worker can run, all optional except the first two. */
interface Detectors {
  hands: HandLandmarker;
  pose: PoseLandmarker;
  face: FaceLandmarker | null;
}

let detectors: Detectors | null = null;

/**
 * Sends a message to the client.
 *
 * @param message Payload matching the client protocol.
 * @param transfer Objects whose ownership moves to the client.
 */
function post(message: LandmarkResponse, transfer: Transferable[] = []): void {
  (self as unknown as Worker).postMessage(message, transfer);
}

/**
 * Copies one MediaPipe landmark list into a flat Float32Array.
 *
 * @param landmarks Landmark list from a MediaPipe result.
 * @returns Flattened x, y, z triples. MediaPipe omits z on some models, in
 *   which case zero is used so downstream code always sees a fixed layout.
 */
function flatten(landmarks: Array<{ x: number; y: number; z?: number }>): Float32Array {
  const flat = new Float32Array(landmarks.length * COORDINATES_PER_LANDMARK);
  for (let i = 0; i < landmarks.length; i++) {
    const point = landmarks[i];
    flat[i * COORDINATES_PER_LANDMARK] = point.x;
    flat[i * COORDINATES_PER_LANDMARK + 1] = point.y;
    flat[i * COORDINATES_PER_LANDMARK + 2] = point.z ?? 0;
  }
  return flat;
}

/**
 * Loads the detectors this worker needs.
 *
 * @param enableFace Whether to also load the face detector, for cost measurement.
 * @returns The ready detectors.
 */
async function loadDetectors(enableFace: boolean): Promise<Detectors> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE_PATH);

  const hands = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: `${MODEL_BASE_PATH}/hand_landmarker.task`, delegate: 'CPU' },
    runningMode: 'VIDEO',
    numHands: MAX_HANDS,
  });

  const pose = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: `${MODEL_BASE_PATH}/pose_landmarker_lite.task`, delegate: 'CPU' },
    runningMode: 'VIDEO',
    // Why one pose: the app tutors a single signer sitting in front of the camera.
    numPoses: 1,
  });

  // Why CPU delegate: the harness is pinned to the WASM backend, so WebGPU
  // must not be picked implicitly by the default delegate.
  const face = enableFace
    ? await FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: `${MODEL_BASE_PATH}/face_landmarker.task`, delegate: 'CPU' },
        runningMode: 'VIDEO',
        numFaces: 1,
      })
    : null;

  return { hands, pose, face };
}

/**
 * Detects landmarks for one frame.
 *
 * @param loaded Ready detectors.
 * @param bitmap The frame image.
 * @param timestampMs Capture time; MediaPipe requires strictly increasing values.
 * @returns Landmarks in the shared `FrameLandmarks` shape.
 */
function detect(loaded: Detectors, bitmap: ImageBitmap, timestampMs: number): FrameLandmarks {
  const handResult = loaded.hands.detectForVideo(bitmap, timestampMs);
  const poseResult = loaded.pose.detectForVideo(bitmap, timestampMs);
  if (loaded.face) {
    // Why the discarded result: face landmarks are not used by the app, but the
    // call must still run so the stopwatch can report what enabling it costs.
    loaded.face.detectForVideo(bitmap, timestampMs);
  }

  const leftHand = new Float32Array(FLAT_HAND_LENGTH);
  const rightHand = new Float32Array(FLAT_HAND_LENGTH);
  // Why handedness from the label rather than index: MediaPipe reports which
  // side each detected hand belongs to, and the order of the array is not
  // guaranteed once both hands are visible.
  let hasLeftHand = false;
  let hasRightHand = false;

  handResult.handedness.forEach((categories, index) => {
    const landmarks = handResult.landmarks[index];
    if (!landmarks) {
      return;
    }
    const label = categories[0]?.categoryName ?? '';
    const target = label === 'Left' ? leftHand : rightHand;
    target.set(flatten(landmarks).subarray(0, FLAT_HAND_LENGTH));
    if (label === 'Left') {
      hasLeftHand = true;
    } else {
      hasRightHand = true;
    }
  });

  const poseLandmarks = poseResult.landmarks[0];
  const pose = poseLandmarks && poseLandmarks.length >= POSE_LANDMARK_COUNT
    ? flatten(poseLandmarks).subarray(0, FLAT_POSE_LENGTH)
    : null;

  return {
    timestampMs,
    leftHand: hasLeftHand ? leftHand : null,
    rightHand: hasRightHand ? rightHand : null,
    pose: pose ? new Float32Array(pose) : null,
  };
}

/**
 * Handles one request from the client.
 *
 * @param request Message from the client.
 */
async function handleRequest(request: LandmarkRequest): Promise<void> {
  if (request.type === 'init') {
    try {
      detectors = await loadDetectors(request.enableFace);
      post({ type: 'ready' });
    } catch (error) {
      post({ type: 'error', requestId: null, message: error instanceof Error ? error.message : String(error) });
    }
    return;
  }

  if (request.type === 'process') {
    try {
      if (!detectors) {
        throw new Error('Landmark worker received a frame before init');
      }
      const landmarks = detect(detectors, request.bitmap, request.timestampMs);
      // Why close: the bitmap was transferred to us, so this worker owns it and
      // must release the memory; nobody else can.
      request.bitmap.close();
      post({ type: 'result', requestId: request.requestId, landmarks });
    } catch (error) {
      request.bitmap.close();
      post({
        type: 'error',
        requestId: request.requestId,
        message: error instanceof Error ? error.message : String(error),
      });
    }
  }
}

self.addEventListener('message', (event: MessageEvent<LandmarkRequest>) => {
  void handleRequest(event.data);
});
