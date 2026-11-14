/**
 * Downloads the MediaPipe Task model files used by the browser harness.
 *
 * Why this is a script and not committed files: each `.task` bundle is several
 * megabytes of third-party weights. Committing them would bloat the repository
 * and, more importantly, mix vendor weights into our git history. They are
 * reproducibly fetched from Google's public model bucket instead, so
 * `web/public/models/` stays git-ignored.
 *
 * Usage: `npm run models:download`
 */
import { mkdir, stat, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const MODEL_BASE_URL = 'https://storage.googleapis.com/mediapipe-models';

/** One downloadable bundle: local file name and its upstream path. */
const MODELS = [
  { file: 'hand_landmarker.task', url: `${MODEL_BASE_URL}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task` },
  { file: 'pose_landmarker_lite.task', url: `${MODEL_BASE_URL}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task` },
  // Why the lightweight face bundle: it is only used to measure its FPS cost,
  // so accuracy is irrelevant and a smaller download keeps the check cheap.
  { file: 'face_landmarker.task', url: `${MODEL_BASE_URL}/face_landmarker/face_landmarker/float16/1/face_landmarker.task` },
];

const outputDirectory = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'models');

/**
 * Returns true when the file already exists with non-zero size.
 *
 * @param {string} path Absolute file path to test.
 * @returns {Promise<boolean>} Whether the download can be skipped.
 */
async function isAlreadyDownloaded(path) {
  try {
    const info = await stat(path);
    return info.size > 0;
  } catch {
    // Why: a missing file is the normal first-run case, not an error.
    return false;
  }
}

/**
 * Downloads every model that is not present yet.
 *
 * @returns {Promise<void>} Resolves when all models are on disk.
 */
async function main() {
  await mkdir(outputDirectory, { recursive: true });

  for (const model of MODELS) {
    const targetPath = join(outputDirectory, model.file);
    if (await isAlreadyDownloaded(targetPath)) {
      console.log(`skip   ${model.file} (already present)`);
      continue;
    }

    console.log(`fetch  ${model.file}`);
    const response = await fetch(model.url);
    if (!response.ok) {
      throw new Error(`Download failed for ${model.file}: HTTP ${response.status}`);
    }
    await writeFile(targetPath, Buffer.from(await response.arrayBuffer()));
    console.log(`saved  ${model.file}`);
  }

  console.log(`models ready in ${outputDirectory}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
