/**
 * Copies the ONNX Runtime Web WASM binary and worker script into `web/public/ort/`
 * so they are served same-origin under Cross-Origin-Embedder-Policy: require-corp.
 *
 * Only `ort-wasm-simd-threaded.{wasm,mjs}` (~14.8 MB + 24 KB) are copied; the unused
 * JSEP/Asyncify variants exceed Cloudflare Pages' 25 MB single-file upload limit.
 */
import { copyFile, mkdir } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const rootDir = join(dirname(fileURLToPath(import.meta.url)), '..');
const srcDir = join(rootDir, 'node_modules', 'onnxruntime-web', 'dist');
const outDir = join(rootDir, 'public', 'ort');

const ORT_RUNTIME_FILES = [
  'ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.mjs',
];

async function main() {
  await mkdir(outDir, { recursive: true });
  for (const filename of ORT_RUNTIME_FILES) {
    await copyFile(join(srcDir, filename), join(outDir, filename));
    console.log(`copied ${filename} -> public/ort/${filename}`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
