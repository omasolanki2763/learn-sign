/**
 * Tests for the frame sources.
 *
 * Only the camera failure path is covered here: it is the one behaviour that
 * decides whether a visitor without a camera can still use the site. The happy
 * paths need a real decoder and belong to the manual check.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { CameraUnavailableError, describeCameraError, startSource } from './sourceSelect';

const originalMediaDevices = Object.getOwnPropertyDescriptor(navigator, 'mediaDevices');

/**
 * Replaces `navigator.mediaDevices` for one test.
 *
 * @param mediaDevices Fake implementation to install.
 */
function stubMediaDevices(mediaDevices: unknown): void {
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: mediaDevices });
}

afterEach(() => {
  if (originalMediaDevices) {
    Object.defineProperty(navigator, 'mediaDevices', originalMediaDevices);
  } else {
    Reflect.deleteProperty(navigator as unknown as Record<string, unknown>, 'mediaDevices');
  }
});

describe('startSource', () => {
  it('rejects with CameraUnavailableError when getUserMedia rejects', async () => {
    stubMediaDevices({
      getUserMedia: vi.fn().mockRejectedValue(new Error('Permission denied')),
    });

    await expect(startSource('webcam')).rejects.toBeInstanceOf(CameraUnavailableError);
  });

  it('rejects with CameraUnavailableError when the browser has no camera API', async () => {
    stubMediaDevices(undefined);

    await expect(startSource('webcam')).rejects.toBeInstanceOf(CameraUnavailableError);
  });

  it('names the file fallback in the camera error message', async () => {
    stubMediaDevices({
      getUserMedia: vi.fn().mockRejectedValue(new Error('Permission denied')),
    });

    const error = await startSource('webcam').catch((caught: unknown) => caught);

    expect(describeCameraError(error)).toMatch(/video file/i);
  });

  it('refuses file mode without a file', async () => {
    await expect(startSource('file')).rejects.toThrow(/file/i);
  });
});
