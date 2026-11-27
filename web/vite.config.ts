/// <reference types="vitest" />
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const currentDir = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: resolve(currentDir, 'index.html'),
        bench: resolve(currentDir, 'bench.html'),
      },
    },
  },
  test: {
    environment: 'happy-dom',
    globals: true,
  },
});
