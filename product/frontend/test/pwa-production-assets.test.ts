import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import { loadProductionBuildAssetManifest } from '../build/production-assets';
import {
  PRODUCTION_PWA_PRECACHE_GLOBS,
  PRODUCTION_PWA_PRECACHE_IGNORES,
  PRODUCTION_PWA_WORKBOX_OPTIONS,
} from '../build/pwa-config';
import productionModelSetSource from '@/recognition/model-runtime/production-model-set.json';
import { PRODUCTION_RECOGNITION_MODEL_SET } from '@/recognition';
import { describe, expect, it } from 'vitest';

const AGARI_PROVENANCE_PATH = resolve(
  process.cwd(),
  '../../vendor/agari-wasm/provenance.json',
);

describe('production build asset manifest', () => {
  it('materializes one content-versioned build manifest from the pinned Recognition and Agari inputs', async () => {
    const materialized = loadProductionBuildAssetManifest();
    const agariProvenance = JSON.parse(
      await readFile(AGARI_PROVENANCE_PATH, 'utf8'),
    ) as Record<string, unknown>;

    expect(materialized.fileName).toBe(
      `production-assets-${materialized.manifest.buildAssetVersion}.json`,
    );
    expect(materialized.manifest).toMatchObject({
      schemaVersion: 1,
      recognitionModelSet: productionModelSetSource,
      agariWasm: {
        moduleSpecifier: '@agari-wasm/agari_wasm.js',
        artifact: 'agari_wasm_bg.wasm',
        provenance: agariProvenance,
      },
    });
    expect(agariProvenance).toMatchObject({
      forkCommit: 'fb362b6db416e67984cdb36f704d8ebf6657662e',
      abiVersion: 'v1',
      wasmSha256: '0e3297ed5f6807eac4d7369eb5846bc17e5ea4851470bf9d40c78ec6030e277c',
      wasmBytes: 200739,
    });
  });

  it('uses the same model-set source in runtime code and content-addresses every production model URL', () => {
    expect(PRODUCTION_RECOGNITION_MODEL_SET.modelSetVersion).toBe(
      productionModelSetSource.modelSetVersion,
    );

    for (const role of [
      'detector',
      'tile-classifier',
      'red-five-classifier',
    ] as const) {
      const runtimeModel = PRODUCTION_RECOGNITION_MODEL_SET.models[role];
      const sourceModel = productionModelSetSource.models[role];
      expect(runtimeModel.sha256).toBe(sourceModel.sha256);
      expect(runtimeModel.runtimeSpec).toBe(sourceModel.runtimeSpec);
      expect(runtimeModel.providerPreference).toEqual(sourceModel.providerPreference);
      expect(runtimeModel.url).toContain(sourceModel.url);
      expect(sourceModel.url).toContain(`sha256=${sourceModel.sha256}`);
    }
  });
});

describe('production PWA precache policy', () => {
  it('preloads the shell/build-owned WASM but never makes ONNX model payloads install-time precache assets', () => {
    expect(PRODUCTION_PWA_PRECACHE_GLOBS).toEqual([
      '**/*.{js,css,html,wasm,json,webmanifest,svg,png,ico,woff2}',
    ]);
    expect(PRODUCTION_PWA_PRECACHE_IGNORES).toContain('**/*.onnx');
    expect(PRODUCTION_PWA_WORKBOX_OPTIONS.maximumFileSizeToCacheInBytes).toBeGreaterThan(
      12.86 * 1024 * 1024,
    );
  });

  it('keeps normal service-worker waiting semantics instead of forcing active clients onto a new build', () => {
    expect(PRODUCTION_PWA_WORKBOX_OPTIONS.skipWaiting).toBe(false);
    expect(PRODUCTION_PWA_WORKBOX_OPTIONS.clientsClaim).toBe(false);
    expect(PRODUCTION_PWA_WORKBOX_OPTIONS.cleanupOutdatedCaches).toBe(true);
    expect(PRODUCTION_PWA_WORKBOX_OPTIONS.navigateFallback).toBe('index.html');
  });
});
