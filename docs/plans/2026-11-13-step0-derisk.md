# Step 0 — De-risk, Parity and Stopwatch: Implementation Plan

> **For agentic workers:** execute task-by-task. Each task has ONE owner and ONE reviewer (named per task). Steps use checkbox (`- [ ]`) syntax. Work on a task branch `step0/<task-slug>`, Conventional Commits, PR, green CI, rebase merge. Report every result with command output, not claims.

**Goal:** Prove, with measurements on the owner's laptop and on free Kaggle compute, that the browser pipeline is fast enough and that browser-extracted landmarks match the training data well enough, before any real model is built.

**Architecture:** A static web app (`web/`) runs MediaPipe Tasks landmarkers in a Web Worker, draws the hand skeleton, and records timings. A Python package (`ml/`) loads the GISLR landmark files, defines the single feature contract shared with the browser, trains a quick baseline, and measures the accuracy gap between dataset landmarks and landmarks re-extracted with MediaPipe Tasks. Outputs are JSON + Markdown reports under `reports/step0/`.

**Tech Stack:** Web: Vite 5, TypeScript 5 (strict), `@mediapipe/tasks-vision`, `onnxruntime-web` (WASM), Vitest, Playwright. ML: Python 3.11, PyTorch 2.x, `mediapipe` (Tasks API), pandas + pyarrow, scikit-learn, pytest, ruff. Hosting: Cloudflare Pages. CI: GitHub Actions.

**Spec:** `.notes/plan-merged.md` (Version 1 plan, final draft 4) and `.notes/requirements-v1.md`. Decisions: `.notes/merge-decisions.md` (D1-D35).

## Global Constraints

- All inference in the browser; no backend; no video leaves the device or is stored.
- onnxruntime-web on **WASM** (not WebGPU); test single-thread first.
- Face landmarker OFF by default; it is only measured here (FPS cost).
- Prediction tick every 100 ms; **per-tick compute p95 ≤ 50 ms**; pipeline **≥ 25 FPS** sustained.
- Signer-independent splits only (group by `participant_id`). Never report signer-dependent numbers.
- GISLR data (Kaggle `asl-signs`, CC-BY 4.0, rule 7.B): never commit or redistribute data files; `data/` is git-ignored. ASL Citizen: evaluation only.
- Every reported number carries the command that produced it. Estimates are labelled ASSUMED.
- Git: repo-local identity already set (`omasolanki2763`); no Co-Authored-By lines; no agent names in tracked files.
- Code style: docstring on every function and class; `why` comments on non-obvious lines; no magic numbers (name them in `constants`); booleans read as questions (`isReady`, `hasHand`).
- Parity decision rule: accuracy gap ≤ 3.0 points → train on existing landmark files; > 3.0 → bulk re-extraction as sharded unattended jobs.

## Review Focus

1. **No hand in frame / one hand only** → landmark arrays must be NaN-filled with a fixed shape, never dropped or zero-filled silently (Task 2, Task 4 tests).
2. **Left-handed signer** → canonicalisation mirrors x and swaps hand slots; right-handed input must pass through unchanged (Task 4 test).
3. **Camera permission denied or no webcam** → the page shows a clear message and file-upload mode still works (Task 2 test).
4. **Variable frame rate (15 / 30 / 60 FPS webcams and uploaded files)** → timestamps, not frame counts, drive the 100 ms tick (Task 3 test).
5. **Signer leakage in the split** → a test asserts no `participant_id` appears in both train and validation (Task 5 test).

---

### Task 1: Repository scaffold, CI and hosted empty page

**Owner:** Antigravity · **Reviewer:** OpenCode · **Second reviewer (CI/deploy):** Claude

**Files:**
- Create: `web/package.json`, `web/vite.config.ts`, `web/tsconfig.json`, `web/index.html`, `web/src/main.ts`, `web/public/_headers`
- Create: `ml/pyproject.toml`, `ml/learnsign/__init__.py`, `ml/tests/test_smoke.py`
- Create: `.github/workflows/ci.yml`, `.gitignore`, `README.md`, `reports/step0/.gitkeep`

**Interfaces:**
- Produces: `npm run dev|build|test|lint` in `web/`; `pytest` and `ruff check` in `ml/`; `web/dist/` as the Pages build output; `_headers` with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`.

- [ ] **Step 1:** Scaffold `web/` (Vite vanilla-ts, strict mode) and `ml/` (package `learnsign`, Python ≥ 3.11, ruff + pytest configured in `pyproject.toml`).
- [ ] **Step 2:** `.gitignore` covers `data/`, `node_modules/`, `dist/`, `*.parquet`, `*.onnx` under `data/`, `.venv/`, `reports/**/raw/`.
- [ ] **Step 3:** Write `web/src/main.test.ts::rendersAppTitle` (asserts `#app` contains "LearnSign") and `ml/tests/test_smoke.py::test_package_imports`. Run both; expect FAIL, then implement the minimum, expect PASS.
- [ ] **Step 4:** `ci.yml`: two jobs (`web`: npm ci, lint, test, build; `ml`: ruff, pytest) on push and pull_request.
- [ ] **Step 5:** Add a test `web/src/headers.test.ts::headersFileHasIsolation` asserting both header lines exist in `public/_headers`.
- [ ] **Step 6:** Commit; open PR; CI green. Output for the owner (Om): exact steps to connect the GitHub repo to Cloudflare Pages (build command `npm run build`, root `web`, output `dist`) — account creation and repo connection are Om's actions.
- [ ] **Step 7 (verification):** after deploy, `curl -sI <pages-url>` shows both isolation headers; paste the output in the PR.

### Task 2: Browser landmark harness (webcam + file, skeleton overlay)

**Owner:** OpenCode · **Reviewer:** Antigravity

**Files:**
- Create: `web/src/landmarks/landmarkWorker.ts`, `web/src/landmarks/landmarkClient.ts`, `web/src/landmarks/types.ts`, `web/src/ui/overlay.ts`, `web/src/ui/sourceSelect.ts`
- Test: `web/src/landmarks/landmarkClient.test.ts`, `web/src/ui/overlay.test.ts`

**Interfaces:**
- Produces:
  - `type FrameLandmarks = { timestampMs: number; leftHand: Float32Array | null; rightHand: Float32Array | null; pose: Float32Array | null }` (hands: 21×3, pose: 33×3, image-normalised x,y,z).
  - `class LandmarkClient { init(options: { enableFace: boolean }): Promise<void>; process(frame: ImageBitmap, timestampMs: number): Promise<FrameLandmarks>; dispose(): void }`
  - `drawOverlay(ctx: CanvasRenderingContext2D, frame: FrameLandmarks, boxState: 'idle' | 'green' | 'amber' | 'red'): void`
  - `startSource(kind: 'webcam' | 'file', file?: File): Promise<AsyncIterable<{ bitmap: ImageBitmap; timestampMs: number }>>`

- [ ] **Step 1:** Tests first: `process` resolves with `leftHand === null` when the worker reports no left hand; `drawOverlay` draws 21 points per present hand and a box coloured by `boxState`; `startSource('webcam')` rejects with `CameraUnavailableError` when `getUserMedia` rejects. Run: `npm test` → FAIL.
- [ ] **Step 2:** Implement the worker with `HandLandmarker` (numHands 2, running mode VIDEO) + `PoseLandmarker`, plus `FaceLandmarker` only when `enableFace` is true (its output is discarded; it exists to measure cost); models loaded from `web/public/models/`; frames transferred as `ImageBitmap`.
- [ ] **Step 3:** Implement overlay, source selection (webcam via `requestVideoFrameCallback`; file via a `<video>` element stepped by the same callback), and the error message UI for camera failure with file mode still usable.
- [ ] **Step 4:** `npm test` → PASS. Manual check on the dev server: skeleton follows both hands; attach a screenshot to the PR.
- [ ] **Step 5:** Commit; PR; CI green.

### Task 3: Stopwatch (FPS, per-stage timing, model-inference probe)

**Owner:** Antigravity · **Reviewer:** OpenCode

**Files:**
- Create: `web/src/bench/stopwatch.ts`, `web/src/bench/tickScheduler.ts`, `web/src/bench/dummyModel.ts`, `web/src/bench/benchPage.ts`, `ml/learnsign/bench/export_dummy_onnx.py`
- Test: `web/src/bench/stopwatch.test.ts`, `web/src/bench/tickScheduler.test.ts`, `ml/tests/test_export_dummy_onnx.py`

**Interfaces:**
- Consumes: `LandmarkClient`, `FrameLandmarks` (Task 2).
- Produces:
  - `class Stopwatch { mark(stage: 'landmarks' | 'features' | 'model' | 'render', ms: number): void; summary(): Record<string, { p50: number; p95: number; n: number }>; fps(): number }`
  - `class TickScheduler { constructor(periodMs: number); shouldTick(timestampMs: number): boolean }` — driven by timestamps, period `TICK_PERIOD_MS = 100`.
  - `export_dummy_onnx(path: str, params_target: int = 1_850_000, window: int = 64, features: int = 144) -> int` (returns actual parameter count; 1D-CNN + small Transformer, random weights).
  - Bench output file `step0-bench.json`: `{ userAgent, hardwareConcurrency, resolution, enableFace, threads, fps, stages, tickComputeP95 }`.

- [ ] **Step 1:** Tests: percentile maths on a known array (p50 of 1..100 = 50.5, p95 = 95.05); `shouldTick` fires 10 times across timestamps 0..1000 at 15, 30 and 60 FPS input; exported ONNX has 1.7M–2.0M parameters and loads in `onnxruntime`. → FAIL.
- [ ] **Step 2:** Implement; bench page runs 60 s per configuration: {face off, face on} × {1 thread, max threads}, and offers "Download JSON".
- [ ] **Step 3:** Tests PASS. Commit; PR; CI green.
- [ ] **Step 4 (owner action, Om):** open the hosted bench page on the laptop, run all four configurations, send the four JSON files. Agent writes `reports/step0/bench.md` (table + 5-line plain-English reading). **Gate:** face-off config has `fps ≥ 25` and `tickComputeP95 ≤ 50`.

### Task 4: Feature contract shared by Python and browser

**Owner:** OpenCode · **Reviewer:** Antigravity · **Second reviewer (core logic):** Claude

**Files:**
- Create: `ml/learnsign/features/schema.py`, `ml/learnsign/features/normalize.py`, `web/src/features/schema.ts`, `web/src/features/normalize.ts`, `tests/golden/normalize_cases.json`
- Test: `ml/tests/test_normalize.py`, `web/src/features/normalize.test.ts`

**Interfaces:**
- Produces (identical semantics in both languages):
  - Constants: `HAND_POINTS = 21`, `POSE_IDS = [11, 12, 13, 14, 15, 16]` (shoulders, elbows, wrists), `FEATURES_PER_FRAME = (21 + 21 + 6) * 3 = 144`.
  - `normalize_frame(left_hand, right_hand, pose) -> float32[144]`: origin = shoulder midpoint; scale = shoulder width; missing parts = NaN.
  - `canonicalize_handedness(seq: float32[T,144], is_left_dominant: bool) -> float32[T,144]`: mirror x and swap hand slots when left-dominant.
  - `estimate_dominant_hand(seq) -> 'left' | 'right'`: hand with larger summed motion energy; ties → right.
  - GISLR adapter `from_gislr_frame(df_frame) -> (left_hand, right_hand, pose)` mapping the 543-row layout to the parts above.

- [ ] **Step 1:** Write `tests/golden/normalize_cases.json` (8 cases: both hands, left only, right only, no hands, no pose, left-dominant, shoulders swapped, zero shoulder width → all NaN) with expected outputs computed by hand for 2 small cases and by the Python reference for the rest.
- [ ] **Step 2:** Tests in both languages load the same golden file and assert `allclose(atol=1e-5)` with NaN positions equal; plus: right-dominant input is unchanged by `canonicalize_handedness`. → FAIL.
- [ ] **Step 3:** Implement Python, then TypeScript. → PASS in both.
- [ ] **Step 4:** Commit; PR; CI green (CI runs both test suites against the one golden file).

### Task 5: Quick baseline and the parity experiment (Kaggle, unattended)

**Owner:** Antigravity · **Reviewer:** OpenCode · **Second reviewer (data + core logic):** Claude

**Files:**
- Create: `ml/learnsign/data/gislr.py`, `ml/learnsign/data/split.py`, `ml/learnsign/data/audit.py`, `ml/learnsign/train/quick_baseline.py`, `ml/learnsign/train/monitor.py`, `ml/learnsign/parity/extract_tasks.py`, `ml/learnsign/parity/compare.py`, `ml/notebooks/step0_parity.ipynb`
- Test: `ml/tests/test_split.py`, `ml/tests/test_audit.py`, `ml/tests/test_monitor.py`, `ml/tests/test_compare.py`

**Interfaces:**
- Consumes: `normalize_frame`, `from_gislr_frame`, `canonicalize_handedness` (Task 4).
- Produces:
  - `load_sequence(parquet_path) -> float32[T,144]`
  - `signer_split(train_csv, n_val_signers: int = 4, seed: int = 42) -> (train_df, val_df)`
  - `audit(train_csv, data_dir) -> dict` (class counts, sequence-length stats, missing-hand frame ratio, NaN count, duplicate paths) written to `reports/step0/data_audit.md`.
  - `class EpochMonitor`: `update(epoch, train_loss, val_loss, val_top1) -> str` returns one of `'ok' | 'overfitting' | 'underfitting' | 'stagnation'`; `should_stop() -> bool` (patience 5).
  - `extract_tasks(video_path) -> float32[T,144]` using Python MediaPipe Tasks Hand + Pose landmarkers.
  - `compare(acc_dataset: float, acc_tasks: float) -> { gap_points: float, decision: 'train_on_existing' | 'bulk_reextract' }` with threshold `PARITY_GAP_POINTS = 3.0`.

- [ ] **Step 1:** Tests: no `participant_id` in both splits; audit flags an injected NaN file and a duplicate; `EpochMonitor` returns `'overfitting'` for train loss falling 5 epochs while val loss rises, `'stagnation'` for flat val top-1, and `should_stop()` true after patience; `compare(0.70, 0.66)` → `bulk_reextract`, `compare(0.70, 0.68)` → `train_on_existing`. → FAIL.
- [ ] **Step 2:** Implement loaders, split, audit, monitor, compare. → PASS.
- [ ] **Step 3:** Quick baseline: small 1D-CNN (≤ 500k params), 64-frame resample, ≤ 15 epochs, TensorBoard logging per step (train loss, LR, grad norm) and per epoch (val loss, top-1, top-5); "overfit one batch" check runs first and aborts the job if loss does not drop below 0.1.
- [ ] **Step 4:** Parity sample: 2,000 PopSign clips (8 per sign) from validation signers; `extract_tasks` on each; evaluate the same checkpoint on (a) dataset landmarks of those signers and (b) Tasks-extracted landmarks.
- [ ] **Step 5:** Webcam gap: evaluate the checkpoint on ASL Citizen clips for glosses that overlap the 250 list (evaluation only; report the number of overlapping glosses and clips).
- [ ] **Step 6:** Run as a Kaggle notebook in commit mode (unattended, checkpoint each stage). Write `reports/step0/parity.md`: both accuracies, gap, decision, webcam-clip accuracy, training curves PNG, data audit summary, 5-line plain-English reading, and the exact commands.
- [ ] **Step 7:** Commit code + report (no data); PR; CI green.

### Task 6: Browser-vs-Python extractor distance and the Step 0 gate report

**Owner:** OpenCode · **Reviewer:** Antigravity

**Files:**
- Create: `web/src/bench/exportLandmarks.ts`, `ml/learnsign/parity/browser_distance.py`, `web/e2e/fileMode.spec.ts`, `reports/step0/GATE.md`
- Test: `ml/tests/test_browser_distance.py`

**Interfaces:**
- Consumes: `startSource('file')`, `LandmarkClient` (Task 2); `normalize_frame` (Task 4); `extract_tasks` (Task 5).
- Produces: `browser_distance(browser_json, python_npy) -> { mean_l2: float, p95_l2: float, frames_compared: int }`; Playwright test that loads a clip in file mode and downloads normalised landmarks as JSON.

- [ ] **Step 1:** Test: identical arrays → `mean_l2 == 0`; a known 0.01 offset on every coordinate → `mean_l2 ≈ 0.01 * sqrt(3)`; frames where either side is NaN are excluded and counted. → FAIL → implement → PASS.
- [ ] **Step 2:** Playwright spec runs 20 clips (local, git-ignored) through file mode in Chromium and saves JSON; the Python script compares against `extract_tasks` output.
- [ ] **Step 3:** Write `reports/step0/GATE.md`: one table — FPS (Task 3), tick compute p95 (Task 3), face-on FPS cost (Task 3), parity gap + decision (Task 5), webcam-clip accuracy (Task 5), browser-vs-Python distance (this task) — each with PASS/FAIL against the Global Constraints and the command that produced it. State the Slice 1 consequences (train on existing files or re-extract; face on or off; resolution).
- [ ] **Step 4:** Commit; PR; CI green. Planner (Claude) reruns the test suites and checks every number against its raw file before reporting to Om.

---

## Order and parallelism

Task 1 first. Then Tasks 2 and 4 in parallel (different owners' files do not overlap: OpenCode does 4 after 2). Task 3 after 2. Task 5 after 4. Task 6 last. Step 0 ends with `reports/step0/GATE.md` and a walkthrough for Om.

## Owner (Om) actions in Step 0

1. Create a GitHub repository and a Cloudflare Pages project (free) and connect them — instructions come with Task 1.
2. Kaggle account with phone verification (needed for GPU and internet in notebooks); accept the `asl-signs` competition rules.
3. Run the hosted bench page on the laptop (about 5 minutes) and send the JSON files.
