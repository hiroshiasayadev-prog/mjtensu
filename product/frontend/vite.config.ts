import { fileURLToPath, URL } from 'node:url';

import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig(({ mode }) => ({
  plugins: [react()],
  publicDir: fileURLToPath(new URL('../../vendor/recognition-models', import.meta.url)),
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
      '@agari-wasm': fileURLToPath(
        new URL('../../vendor/agari-wasm', import.meta.url),
      ),
    },
  },
  server: {
    fs: {
      allow: [fileURLToPath(new URL('../..', import.meta.url))],
    },
  },
  build: mode === 'e2e'
    ? {
        rollupOptions: {
          input: {
            app: fileURLToPath(new URL('./index.html', import.meta.url)),
            fakeFlow: fileURLToPath(
              new URL('./test/e2e/fake-flow.html', import.meta.url),
            ),
            recognitionProductionArtifacts: fileURLToPath(
              new URL(
                './test/e2e/recognition-production-artifacts.html',
                import.meta.url,
              ),
            ),
          },
        },
      }
    : undefined,
}));
