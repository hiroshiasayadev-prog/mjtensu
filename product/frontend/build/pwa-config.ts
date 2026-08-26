export const PRODUCTION_PWA_PRECACHE_GLOBS = [
  '**/*.{js,css,html,wasm,json,webmanifest,svg,png,ico,woff2}',
] as const;

export const PRODUCTION_PWA_PRECACHE_IGNORES = [
  '**/*.onnx',
  '**/provenance.json',
] as const;

export const PRODUCTION_PWA_WORKBOX_OPTIONS = {
  globPatterns: [...PRODUCTION_PWA_PRECACHE_GLOBS],
  globIgnores: [...PRODUCTION_PWA_PRECACHE_IGNORES],
  maximumFileSizeToCacheInBytes: 16 * 1024 * 1024,
  navigateFallback: 'index.html',
  cleanupOutdatedCaches: true,
  skipWaiting: false,
  clientsClaim: false,
};
