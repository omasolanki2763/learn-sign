/**
 * Tests for the app shell and the harness wiring.
 *
 * The worker client and the frame source are mocked: what is under test is that
 * the harness initialises the client before the first frame, and that a broken
 * model is not reported to the user as a camera problem.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';

const initMock = vi.fn();
const processMock = vi.fn();
const startSourceMock = vi.fn();

vi.mock('./landmarks/landmarkClient', () => ({
  LandmarkClient: class {
    init = initMock;
    process = processMock;
  },
}));

vi.mock('./ui/sourceSelect', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./ui/sourceSelect')>();
  return { ...actual, startSource: startSourceMock };
});

/** Lets pending promise chains settle. */
async function flushPromises(): Promise<void> {
  for (let turn = 0; turn < 10; turn++) {
    await Promise.resolve();
  }
}

/**
 * Mounts the app into a fresh `#app` element.
 */
async function mountFreshApp(): Promise<void> {
  document.body.innerHTML = '<div id="app"></div>';
  vi.resetModules();
  await import('./main');
}

beforeEach(() => {
  initMock.mockReset().mockResolvedValue(undefined);
  processMock.mockReset();
  startSourceMock.mockReset();
  // Why the stub context: only a non-null 2D context is needed here; drawing is
  // covered by the overlay tests.
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({} as unknown as CanvasRenderingContext2D);
});

describe('main', () => {
  it('renders the app title', async () => {
    await mountFreshApp();

    expect(document.getElementById('app')?.textContent).toContain('LearnSign');
  });

  it('initialises the landmark client before requesting any frame', async () => {
    startSourceMock.mockResolvedValue((async function* empty() {})());
    await mountFreshApp();

    document.querySelector<HTMLButtonElement>('#start-camera')?.click();
    await flushPromises();

    expect(initMock).toHaveBeenCalledWith({ enableFace: false });
    expect(processMock).not.toHaveBeenCalled();
  });

  it('does not describe a broken model as a camera problem', async () => {
    // Why silenced: the harness logs the real cause on purpose; the assertion is
    // about what the user is told.
    vi.spyOn(console, 'error').mockImplementation(() => undefined);
    startSourceMock.mockResolvedValue((async function* empty() {})());
    initMock.mockRejectedValue(new Error('Failed to fetch /models/hand_landmarker.task'));
    await mountFreshApp();

    document.querySelector<HTMLButtonElement>('#start-camera')?.click();
    await flushPromises();

    const status = document.querySelector<HTMLElement>('#status')?.textContent ?? '';
    expect(status).not.toMatch(/camera access was blocked/i);
    expect(status).toMatch(/model|tracking|could not start/i);
  });
});

