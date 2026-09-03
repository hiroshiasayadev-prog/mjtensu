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
    artifact: 'rotated-fcos-nano-s05-f64-obb120-gn.onnx',
    sha256: '6bb3b25ea2baa2dc641d5677fbfc8792e0a4e381bd2e0f4cf1dc05de4a38d1b0',
    bytes: 858340,
    runtimeSpec: 'rotated-fcos-nano-320-v1',
    sourceRun:
      '.local/recognition/rotated_fcos_runs/rfcos_nano_s05_f64_obb120_gn_seed42',
  },
  'tile-classifier': {
    artifact: 'mobile-tile-f8-r1.onnx',
    sha256: '5039c044a490b44e8c645ead5a3280293f78c3c43db9baabd9f07219ff883a7e',
    bytes: 3873724,
    runtimeSpec: 'gray64-tile-35-v1',
  },
  'red-five-classifier': {
    artifact: 'red-five-c8-rgb-warmaug.onnx',
    sha256: 'c2b780f682d84bf186db90290050f8b05016c3e8058de559eea679a28eeb80c6',
    bytes: 6256955,
    runtimeSpec: 'c8-red-five-v1',
  },
} as const;

describe('production Recognition model artifacts', () => {
  it('binds one valid production manifest to the selected three-model set', () => {
    expect(validateRecognitionModelSetManifest(PRODUCTION_RECOGNITION_MODEL_SET)).toEqual(
      PRODUCTION_RECOGNITION_MODEL_SET,
    );
    expect(PRODUCTION_RECOGNITION_MODEL_SET.modelSetVersion).toBe(
      'recognition-v8-2026-09-04',
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
