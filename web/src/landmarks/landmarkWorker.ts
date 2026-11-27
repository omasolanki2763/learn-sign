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

/**
 * Whether to flip MediaPipe's handedness labels before using them.
 *
 * Why this exists: the hand model is documented as assuming a mirrored
 * ("selfie") input image - the MediaPipe hand landmarker documentation for
 * `HandLandmarker` states the handedness is "assumed to be of the mirrored
 * image". The camera frame the harness feeds in is not mirrored, so the model's
 * "Left" can be the person's right hand. The training data labels hands
 * anatomically, so which setting is right is a parity question, not a taste
 * question: Step 0 Task 6 measures both against the dataset and this constant is
 * the single place to flip the result.
 */
export const SWAP_HANDEDNESS_LABELS = false;

/** One detected hand, as reported by MediaPipe, before anatomical assignment. */
export interface HandedDetection {
  /** Label MediaPipe reported: 'Left' or 'Right'. */
  label: string;
  /** Confidence of that label, 0..1. */
  score: number;
  /** Index into the detector's landmark array. */
  index: number;
}

/** The chosen detection per anatomical hand. */
export interface AssignedHands {
  left: HandedDetection | null;
  right: HandedDetection | null;
}

let detectors: Detectors | null = null;

/**
 * Maps MediaPipe handedness labels onto anatomical left and right hands.
 *
 * Two detections can carry the same label when the model is unsure; keeping the
 * higher score is deliberate - silently letting the second one overwrite the
 * first would make the output depend on array order.
 *
 * @param detections Hands the detector reported for one frame.
 * @param swapLabels Whether to treat MediaPipe's 'Left' as the right hand.
 * @returns At most one detection per anatomical hand.
 */
export function assignHands(
  detections: readonly HandedDetection[],
  swapLabels: boolean = SWAP_HANDEDNESS_LABELS,
): AssignedHands {
  const assigned: AssignedHands = { left: null, right: null };

  for (const detection of detections) {
    const reportedLeft = detection.label === 'Left';
    const isAnatomicallyLeft = swapLabels ? !reportedLeft : reportedLeft;
    const slot = isAnatomicallyLeft ? 'left' : 'right';
    const current = assigned[slot];
    if (!current || detection.score > current.score) {
      assigned[slot] = detection;
    }
  }

  return assigned;
}

/**
 * True when this module really runs inside a dedicated worker.
 *
 * Why `window`: a dedicated worker has no `window`, and the project's tsconfig
 * carries the DOM lib rather than the WebWorker one, so this is both correct and
 * typechecks. The protocol tests import this file into a page-like environment,
 * where `postMessage` and `close` mean something else entirely.
 */
const isDedicatedWorker = typeof window === 'undefined';

/**
 * Sends a message to the client.
 *
 * @param message Payload matching the client protocol.
 * @param transfer Objects whose ownership moves to the client.
 */
function post(message: LandmarkResponse, transfer: Transferable[] = []): void {
  if (!isDedicatedWorker) {
    return;
  }
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

  const detections: HandedDetection[] = handResult.handedness.map((categories, index) => ({
    label: categories[0]?.categoryName ?? 'Right',
    score: categories[0]?.score ?? 0,
    index,
  }));
  const assigned = assignHands(detections);

  for (const slot of ['left', 'right'] as const) {
    const detection = assigned[slot];
    const landmarks = detection ? handResult.landmarks[detection.index] : undefined;
    if (!detection || !landmarks) {
      continue;
    }
    (slot === 'left' ? leftHand : rightHand).set(flatten(landmarks).subarray(0, FLAT_HAND_LENGTH));
  }

  const poseLandmarks = poseResult.landmarks[0];
  const pose = poseLandmarks && poseLandmarks.length >= POSE_LANDMARK_COUNT
    ? flatten(poseLandmarks).subarray(0, FLAT_POSE_LENGTH)
    : null;

  return {
    timestampMs,
    leftHand: assigned.left ? leftHand : null,
    rightHand: assigned.right ? rightHand : null,
    pose: pose ? new Float32Array(pose) : null,
  };
}

/**
 * Handles one request from the client.
 *
 * Exported so the protocol can be tested without a real MediaPipe runtime.
 *
 * @param request Message from the client.
 */
export async function handleRequest(request: LandmarkRequest): Promise<void> {
  if (request.type === 'dispose') {
    // Why: the detectors hold WASM heap that only `.close()` releases; dropping
    // the reference alone would leave the memory allocated until the worker dies.
    detectors?.hands.close();
    detectors?.pose.close();
    detectors?.face?.close();
    detectors = null;
    if (isDedicatedWorker) {
      (self as unknown as { close: () => void }).close();
    }
    return;
  }

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

if (isDedicatedWorker) {
  self.addEventListener('message', (event: MessageEvent<LandmarkRequest>) => {
    void handleRequest(event.data);
  });
}

