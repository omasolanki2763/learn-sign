/**
 * Tests for timestamp-driven prediction TickScheduler across variable frame rates.
 */
import { describe, expect, it } from 'vitest';

import { TICK_PERIOD_MS, TickScheduler } from './tickScheduler';

describe('TickScheduler', () => {
  it('uses 100 ms as default prediction tick period', () => {
    expect(TICK_PERIOD_MS).toBe(100);
  });

  it.each([15, 30, 60])(
    'fires exactly 10 times across timestamps 0..1000 at %i FPS input',
    (fps) => {
      const scheduler = new TickScheduler(TICK_PERIOD_MS);
      const stepMs = 1000 / fps;
      let fireCount = 0;

      for (let i = 0; i <= fps; i++) {
        const ts = i * stepMs;
        if (scheduler.shouldTick(ts)) {
          fireCount++;
        }
      }

      expect(fireCount).toBe(10);
    }
  );

  it('resets schedule state cleanly when switching sources', () => {
    const scheduler = new TickScheduler(100);
    expect(scheduler.shouldTick(1000)).toBe(false);
    expect(scheduler.shouldTick(1100)).toBe(true);

    scheduler.reset();
    expect(scheduler.shouldTick(500)).toBe(false);
    expect(scheduler.shouldTick(550)).toBe(false);
    expect(scheduler.shouldTick(600)).toBe(true);
  });
});
