# LearnSign

LearnSign is a zero-cost, browser-only, real-time American Sign Language (ASL) tutor for hearing beginners across letters, words, and sentences.

## Key Principles

- **Zero Cost:** Hosted entirely as a static web application on Cloudflare Pages; models trained within free-tier GPU environments (Kaggle / Google Colab).
- **Privacy First:** All video processing and neural network inference occur strictly in local browser RAM using client-side WebAssembly. No video frames, audio, or biometric data ever leave the user's device.
- **Real-Time Performance:** Sustained $\ge 25\text{ FPS}$ pipeline with a prediction tick every 100 ms and a per-tick compute latency target of p95 $\le 50\text{ ms}$.

## Repository Structure

```text
learn-sign/
├── .github/workflows/    # Continuous integration pipelines
├── docs/plans/           # Architectural plans and step-by-step specifications
├── ml/                   # Machine learning package, feature contracts, parity experiments
│   ├── learnsign/        # Python library source
│   ├── tests/            # Python unit test suite
│   └── pyproject.toml    # Python project configuration
├── reports/              # Reproducible benchmark and parity reports
└── web/                  # Browser client application
    ├── public/           # Static assets and isolation headers (_headers)
    ├── src/              # TypeScript source code
    ├── index.html        # Web application entry page
    ├── package.json      # Frontend dependencies and npm scripts
    ├── tsconfig.json     # Strict TypeScript configuration
    └── vite.config.ts    # Vite bundler configuration
```

## Getting Started

### Web Application (`web/`)

Prerequisites: Node.js 20+ and npm.

```bash
cd web
npm install
npm run dev      # Start local development server
npm run build    # Build production bundle to dist/
npm test         # Run unit tests via Vitest
npm run lint     # Validate TypeScript types without emit
```

### Machine Learning (`ml/`)

Prerequisites: Python 3.11+.

```bash
cd ml
pip install -e ".[dev]"   # Install package in editable mode with test dependencies
ruff check .             # Run linter
python -m pytest         # Run unit test suite
```

## Security & Cross-Origin Isolation

LearnSign requires high-precision timers (`performance.now()`) and multithreaded WASM via `SharedArrayBuffer`. Cloudflare Pages serves the required isolation headers defined in `web/public/_headers`:

```http
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

## License

This project is open-source under the Apache 2.0 License.
