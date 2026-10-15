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
    artifact: 'nanodet-plus-m-320-blender-v6-jp0.onnx',
    sha256: '4a88a7d09bb5dd505ee87fc66b9d6e078afe3d3208031583e29f4e3970d0c7c9',
    bytes: 4734123,
    runtimeSpec: 'nanodet-plus-m-320-v1',
    sourceRun:
      'nanodet/run-994bcd119779447ea85a98a050f387c3-trial-0001-train',
  },
  'tile-classifier': {
    artifact: 'tile-c8-gray35-narrow-stage234-dw-m2-m2-m2-v1.onnx',
    sha256: 'ede3fc3ee099415a616c5f526a3643de44f6706ad15a72fd464d73dcada1c4c0',
    bytes: 522705,
    runtimeSpec: 'gray64-tile-35-v1',
    sourceRun:
      'tile-classifier/run-bdcce50e63fb4665bebbd8d3b78ea243-trial-0003-train',
  },
  'red-five-classifier': {
    artifact: 'red-five-c8-rgb-aggressive-stage34-dw-m1-v1.onnx',
    sha256: '520abc54154e9618af1fbd3af7695945692d4dab5f170fa0fc5c6245469c498f',
    bytes: 662382,
    runtimeSpec: 'c8-red-five-v2',
    sourceRun:
      'tile-classifier/run-b18af9873dd84e788890646e4f81f3cc-trial-0003-train',
  },
} as const;

describe('production Recognition model artifacts', () => {
  it('binds one valid production manifest to the selected three-model set', () => {
    expect(validateRecognitionModelSetManifest(PRODUCTION_RECOGNITION_MODEL_SET)).toEqual(
      PRODUCTION_RECOGNITION_MODEL_SET,
    );
    expect(PRODUCTION_RECOGNITION_MODEL_SET.modelSetVersion).toBe(
      'recognition-v11-2026-10-16',
    );

    for (const [role, expected] of Object.entries(expectedModels)) {
      const model = PRODUCTION_RECOGNITION_MODEL_SET.models[
        role as keyof typeof expectedModels
      ];
      expect(model).toMatchObject({
        role,
        runtimeSpec: expected.runtimeSpec,
        providerPreference: ['wasm-simd', 'wasm-threaded', 'webgl'],
      });
      expect(model.url).toContain(expected.artifact);
      expect(model.url).toContain(`sha256=${model.sha256}`);
      expect(model.sha256).toMatch(/^[a-f0-9]{64}$/);
      if ('sha256' in expected) {
        expect(model.sha256).toBe(expected.sha256);
      }
    }
  });

  it('does not retain the superseded detector payload in the production vendor package', async () => {
    await expect(
      readFile(resolve(VENDOR_MODEL_DIR, 'nanodet-plus-m-320-composite-augmented.onnx')),
    ).rejects.toMatchObject({ code: 'ENOENT' });
  });

  it('matches every committed ONNX artifact to manifest/provenance identity', async () => {
    const provenance = JSON.parse(await readFile(PROVENANCE_PATH, 'utf8')) as {
      schemaVersion: number;
      modelSetVersion: string;
      models: Record<
        string,
        {
          artifact: string;
          sourceRun?: string;
          sha256: string;
          bytes: number;
        }
      >;
    };

    expect(provenance.schemaVersion).toBe(1);
    expect(provenance.modelSetVersion).toBe(PRODUCTION_RECOGNITION_MODEL_SET.modelSetVersion);

    for (const [role, expected] of Object.entries(expectedModels)) {
      const runtimeModel = PRODUCTION_RECOGNITION_MODEL_SET.models[
        role as keyof typeof expectedModels
      ];
      const bytes = await readFile(resolve(VENDOR_MODEL_DIR, expected.artifact));
      const sha256 = createHash('sha256').update(bytes).digest('hex');

      expect(bytes.byteLength, role).toBe(expected.bytes);
      expect(runtimeModel.sha256, role).toBe(sha256);
      expect(provenance.models[role], role).toMatchObject({
        artifact: expected.artifact,
        sha256,
        bytes: expected.bytes,
      });
      if ('sha256' in expected) {
        expect(sha256, role).toBe(expected.sha256);
      }
      if ('sourceRun' in expected) {
        expect(provenance.models[role]?.sourceRun, role).toBe(expected.sourceRun);
      }
    }
  });
});
