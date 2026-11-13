/**
 * Tests for the web application main entry point.
 */
import { describe, it, expect, beforeEach } from 'vitest';

describe('main', () => {
  beforeEach(() => {
    // Why: Reset DOM body between test runs to ensure isolation.
    document.body.innerHTML = '<div id="app"></div>';
  });

  it('rendersAppTitle', async () => {
    // Dynamically import main to trigger DOM initialization
    await import('./main');
    const appElement = document.getElementById('app');
    expect(appElement).not.toBeNull();
    expect(appElement?.textContent).toContain('LearnSign');
  });
});
