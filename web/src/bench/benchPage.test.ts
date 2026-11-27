/**
 * Tests for the Step 0 bench report builder, configuration matrix, and DOM page wiring.
 */
import { describe, expect, it } from 'vitest';

import {
  BENCH_CONFIGS,
  BENCH_DURATION_MS,
  buildBenchFilename,
  buildBenchReport,
  formatReportSummary,
  mountBenchPage,
} from './benchPage';
import { Stopwatch } from './stopwatch';

describe('benchPage', () => {
  it('defines all four face x thread configurations at 60 s each', () => {
    expect(BENCH_DURATION_MS).toBe(60_000);
    expect(BENCH_CONFIGS).toHaveLength(4);
    expect(BENCH_CONFIGS.map((c) => c.enableFace)).toEqual([false, false, true, true]);
    expect(BENCH_CONFIGS.filter((c) => c.threads === 1)).toHaveLength(2);
    expect(BENCH_CONFIGS.filter((c) => c.threads === 'max')).toHaveLength(2);
  });

  it('builds step0-bench.json payload matching schema and computes tickComputeP95', () => {
    const sw = new Stopwatch();
    for (let f = 0; f <= 30; f++) {
      sw.markFrame(f * (1000 / 30));
    }
    sw.mark('landmarks', 12);
    sw.mark('features', 0.4);
    sw.mark('derived', 0.5);
    sw.mark('model', 25);
    sw.mark('render', 1.2);
    sw.mark('alignment', 0.8);

    const tickTotals = Array.from({ length: 100 }, (_, i) => i + 1);
    const report = buildBenchReport({
      enableFace: false,
      threads: 1,
      resolution: { width: 640, height: 480 },
      stopwatch: sw,
      tickComputeSamplesMs: tickTotals,
    });

    expect(report.enableFace).toBe(false);
    expect(report.threads).toBe(1);
    expect(report.resolution).toEqual({ width: 640, height: 480 });
    expect(report.fps).toBeCloseTo(30, 3);
    expect(report.tickComputeP95).toBeCloseTo(95.05, 5);
    expect(Object.keys(report.stages).sort()).toEqual([
      'alignment',
      'derived',
      'features',
      'landmarks',
      'model',
      'render',
    ]);
    expect(buildBenchFilename(false, 1)).toBe('step0-bench-face-off-t1.json');
    expect(buildBenchFilename(true, 8)).toBe('step0-bench-face-on-t8.json');
    expect(formatReportSummary(report)).toContain('tickP95=');
  });

  it('mounts bench controls and download container into #bench-app', () => {
    document.body.innerHTML = '<div id="bench-app"></div>';
    const root = document.getElementById('bench-app')!;
    mountBenchPage(root);

    expect(root.textContent).toContain('Step 0 Benchmark');
    expect(root.querySelector('#bench-run-all')).not.toBeNull();
    expect(root.querySelector('#bench-start-camera')).not.toBeNull();
    expect(root.querySelector('#bench-video-file')).not.toBeNull();
    expect(root.querySelector('#bench-downloads')).not.toBeNull();
  });
});
