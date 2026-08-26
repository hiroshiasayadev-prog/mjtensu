import basicSsl from '@vitejs/plugin-basic-ssl';
import { defineConfig } from 'vite';
import { VitePWA } from 'vite-plugin-pwa';

const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Resource-Policy': 'same-origin',
};

export default defineConfig({
  base: './',
  plugins: [
    basicSsl({ name: 'mjtensu-nanodet-pwa-probe' }),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['app-icon.svg', 'app-icon-192.png', 'app-icon-512.png'],
      manifest: {
        name: 'Mjtensu NanoDet Probe',
        short_name: 'NanoDet Probe',
        description: 'Realtime NanoDet tile-region detector latency probe.',
        display: 'standalone',
        orientation: 'landscape',
        background_color: '#090b10',
        theme_color: '#090b10',
        icons: [
          {
            src: 'app-icon-192.png',
            sizes: '192x192',
            type: 'image/png',
            purpose: 'any',
          },
          {
            src: 'app-icon-512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'any maskable',
          },
        ],
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,svg,wasm,mjs,onnx,json}'],
        maximumFileSizeToCacheInBytes: 20 * 1024 * 1024,
        cleanupOutdatedCaches: true,
        clientsClaim: true,
        skipWaiting: true,
      },
      devOptions: {
        enabled: false,
      },
    }),
  ],
  server: {
    host: true,
    https: {},
    headers: isolationHeaders,
  },
  preview: {
    host: true,
    headers: isolationHeaders,
  },
});
