import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import {
  PRODUCTION_RECOGNITION_MODEL_SET,
  validateRecognitionModelSetManifest,
} from '@/recognition';
import { describe, expect, it } from 'vitest';

const VENDOR_MODEL_DIR = resolve(process.cwd(), '../../vendor/recognition-models');
const PROVENANCE_PATH = resolve(VENDOR_MODEL_DIR, 'provenance.json');

const expectedModels = {
  detector: {
    artifact: 'nanodet-plus-m-320-composite-augmented.onnx',
    sha256: '4768daa5cb44e7bee37fbb69c36063800164d9e9e8c852e5b3c77bc88ce9ac76',
    bytes: 5597449,
    runtimeSpec: 'nanodet-plus-m-320-v1',
  },
  'tile-classifier': {
    artifact: 'tile-c8-gray35-v3-jp189.onnx',
    sha256: 'b8a8fa3ff6c6d1e944a7593fa0afc947e0cd2513fb79ca46e5f8fcd6e19c97d0',
    bytes: 6261185,
    runtimeSpec: 'c8-tile-35-v1',
  },
  'red-five-classifier': {
    artifact: 'red-five-c8-rgb-warmaug.onnx',
    sha256: 'c2b780f682d84bf186db90290050f8b05016c3e8058de559eea679a28eeb80c6',
    bytes: 6256955,
    runtimeSpec: 'c8-red-five-v1',
  },
} as const;

describe('production Recognition model artifacts', () => {
  it('binds one valid production manifest to the pinned three-model set', () => {
    expect(validateRecognitionModelSetManifest(PRODUCTION_RECOGNITION_MODEL_SET)).toEqual(
      PRODUCTION_RECOGNITION_MODEL_SET,
    );
    expect(PRODUCTION_RECOGNITION_MODEL_SET.modelSetVersion).toBe(
      'recognition-v1-2026-08-27',
    );

    for (const [role, expected] of Object.entries(expectedModels)) {
      const model = PRODUCTION_RECOGNITION_MODEL_SET.models[
        role as keyof typeof expectedModels
      ];
      expect(model).toMatchObject({
        role,
        sha256: expected.sha256,
        runtimeSpec: expected.runtimeSpec,
        providerPreference: ['wasm-simd', 'wasm-threaded', 'webgl'],
      });
      expect(model.url).toContain(expected.artifact);
    }
  });

  it('matches every committed ONNX artifact to provenance SHA-256 and byte identity', async () => {
    const provenance = JSON.parse(await readFile(PROVENANCE_PATH, 'utf8')) as {
      schemaVersion: number;
      modelSetVersion: string;
      models: Record<string, { artifact: string; sha256: string; bytes: number }>;
    };

    expect(provenance.schemaVersion).toBe(1);
    expect(provenance.modelSetVersion).toBe(PRODUCTION_RECOGNITION_MODEL_SET.modelSetVersion);

    for (const [role, expected] of Object.entries(expectedModels)) {
      const bytes = await readFile(resolve(VENDOR_MODEL_DIR, expected.artifact));
      const sha256 = createHash('sha256').update(bytes).digest('hex');
      expect(bytes.byteLength, role).toBe(expected.bytes);
      expect(sha256, role).toBe(expected.sha256);
      expect(provenance.models[role], role).toMatchObject({
        artifact: expected.artifact,
        sha256: expected.sha256,
        bytes: expected.bytes,
      });
    }
  });
});
