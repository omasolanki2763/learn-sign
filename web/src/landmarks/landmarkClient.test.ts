/**
 * Tests for the landmark worker client.
 *
 * The client is tested against a fake worker so the suite stays hermetic: no
 * WebAssembly, no model files and no camera are needed to verify the message
 * protocol and the shape of the data handed to the UI.
 */
import { describe, it, expect } from 'vitest';
import type { FrameLandmarks } from './types';
import { LandmarkClient, type LandmarkWorkerLike } from './landmarkClient';

/** Minimal stand-in for a dedicated Worker, replying on a later microtask. */
class FakeWorker implements LandmarkWorkerLike {
  readonly posted: unknown[] = [];
  isTerminated = false;
  private readonly messageListeners: Array<(event: MessageEvent) => void> = [];
  private respondToProcess: (requestId: number) => FrameLandmarks = () => ({
    timestampMs: 0,
    leftHand: null,
    rightHand: null,
    pose: null,
  });

  postMessage(message: unknown): void {
    this.posted.push(message);
    const typed = message as { type: string; requestId?: number };
    if (typed.type === 'init') {
      queueMicrotask(() => this.emit({ type: 'ready' }));
    }
    if (typed.type === 'process') {
      const requestId = typed.requestId ?? 0;
      // Why a microtask: a real worker never answers synchronously, and the
      // client must not depend on synchronous replies.
      queueMicrotask(() => this.emit({ type: 'result', requestId, landmarks: this.respondToProcess(requestId) }));
    }
  }

  addEventListener(type: 'message', listener: (event: MessageEvent) => void): void {
    if (type === 'message') {
      this.messageListeners.push(listener);
    }
  }

  terminate(): void {
    this.isTerminated = true;
  }

  /** Sets the landmarks every subsequent process request will return. */
  setProcessResponse(producer: (requestId: number) => FrameLandmarks): void {
    this.respondToProcess = producer;
  }

  private emit(data: unknown): void {
    for (const listener of this.messageListeners) {
      listener({ data } as MessageEvent);
    }
  }
}

describe('LandmarkClient', () => {
  it('resolves with leftHand null when the worker reports no left hand', async () => {
    const worker = new FakeWorker();
    const rightHand = new Float32Array(21 * 3).fill(0.5);
    worker.setProcessResponse(() => ({ timestampMs: 120, leftHand: null, rightHand, pose: null }));

    const client = new LandmarkClient(() => worker);
    await client.init({ enableFace: false });
    const frame = await client.process({} as ImageBitmap, 120);

    expect(frame.leftHand).toBeNull();
    expect(frame.rightHand).toBeInstanceOf(Float32Array);
    expect(frame.rightHand?.length).toBe(21 * 3);
    expect(frame.timestampMs).toBe(120);
  });

  it('keeps responses matched to requests when several are in flight', async () => {
    const worker = new FakeWorker();
    worker.setProcessResponse((requestId) => ({
      timestampMs: requestId,
      leftHand: null,
      rightHand: null,
      pose: null,
    }));

    const client = new LandmarkClient(() => worker);
    await client.init({ enableFace: false });
    const [first, second] = await Promise.all([
      client.process({} as ImageBitmap, 1),
      client.process({} as ImageBitmap, 2),
    ]);

    expect(first.timestampMs).toBeGreaterThan(0);
    expect(second.timestampMs).toBeGreaterThan(0);
    expect(first.timestampMs).not.toBe(second.timestampMs);
  });

  it('terminates the worker on dispose', async () => {
    const worker = new FakeWorker();
    const client = new LandmarkClient(() => worker);
    await client.init({ enableFace: false });

    client.dispose();

    expect(worker.isTerminated).toBe(true);
  });

  it('rejects process calls after dispose', async () => {
    const worker = new FakeWorker();
    const client = new LandmarkClient(() => worker);
    await client.init({ enableFace: false });
    client.dispose();

    await expect(client.process({} as ImageBitmap, 10)).rejects.toThrow(/disposed/i);
  });
});
