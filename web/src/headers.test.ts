/**
 * Tests verifying Cloudflare Pages security and cross-origin isolation headers.
 */
import { describe, it, expect } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';

describe('headers', () => {
  it('headersFileHasIsolation', () => {
    // Why: Ensure the static _headers file exists in public/ so Vite copies it to dist/.
    const headersPath = path.resolve(__dirname, '../public/_headers');
    expect(fs.existsSync(headersPath), 'public/_headers must exist').toBe(true);

    const content = fs.readFileSync(headersPath, 'utf-8');
    // Why: SharedArrayBuffer and high-precision timers require both COOP and COEP isolation headers.
    expect(content).toContain('Cross-Origin-Opener-Policy: same-origin');
    expect(content).toContain('Cross-Origin-Embedder-Policy: require-corp');
  });
});
