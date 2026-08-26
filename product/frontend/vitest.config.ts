import { fileURLToPath, URL } from 'node:url';

import { defineConfig } from 'vitest/config';

export default defineConfig({
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./test/setup.ts'],
    include: ['./test/**/*.test.{ts,tsx}'],
    exclude: ['./test/e2e/**'],
    clearMocks: true,
    restoreMocks: true,
  },
});
