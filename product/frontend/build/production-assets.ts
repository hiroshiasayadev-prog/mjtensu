import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { basename, resolve } from 'node:path';

import type { Plugin } from 'vite';

import { validateRecognitionModelSetManifest } from '../src/recognition/model-runtime/manifest';
import type { RecognitionModelSetManifest } from '../src/recognition/model-runtime/types';

export interface AgariArtifactProvenanceV1 {
  readonly schemaVersion: 1;
  readonly upstreamRepository: string;
  readonly upstreamCommit: string;
  readonly forkRepository: string;
  readonly forkCommit: string;
  readonly abiVersion: string;
  readonly wasmSha256: string;
  readonly wasmBytes: number;
  readonly rustcVersion: string;
  readonly wasmPackVersion: string;
  readonly buildProfile: 'release';
  readonly generatedAt?: string;
}

export interface ProductionBuildAssetManifestV1 {
  readonly schemaVersion: 1;
  readonly buildAssetVersion: string;
  readonly recognitionModelSet: RecognitionModelSetManifest;
  readonly agariWasm: {
    readonly moduleSpecifier: '@agari-wasm/agari_wasm.js';
    readonly artifact: 'agari_wasm_bg.wasm';
    readonly provenance: AgariArtifactProvenanceV1;
  };
}

export interface MaterializedProductionBuildAssetManifest {
  readonly fileName: string;
  readonly source: string;
  readonly manifest: ProductionBuildAssetManifestV1;
}

const FRONTEND_ROOT = process.cwd();
const MODEL_SET_SOURCE_PATH = resolve(
  FRONTEND_ROOT,
  'src/recognition/model-runtime/production-model-set.json',
);
const RECOGNITION_MODEL_DIR = resolve(
  FRONTEND_ROOT,
  '../../vendor/recognition-models',
);
const AGARI_PROVENANCE_PATH = resolve(
  FRONTEND_ROOT,
  '../../vendor/agari-wasm/provenance.json',
);
const AGARI_WASM_PATH = resolve(
  FRONTEND_ROOT,
  '../../vendor/agari-wasm/agari_wasm_bg.wasm',
);
const SHA256_PATTERN = /^[a-f0-9]{64}$/i;
const FULL_GIT_SHA_PATTERN = /^[a-f0-9]{40}$/i;

export function loadProductionBuildAssetManifest(): MaterializedProductionBuildAssetManifest {
  const recognitionModelSet = validateRecognitionModelSetManifest(
    readJson(MODEL_SET_SOURCE_PATH),
  );
  verifyRecognitionModelArtifacts(recognitionModelSet);

  const agariProvenance = validateAgariProvenance(
    readJson(AGARI_PROVENANCE_PATH),
  );
  verifyAgariArtifact(agariProvenance);

  const semanticPayload = {
    schemaVersion: 1 as const,
    recognitionModelSet,
    agariWasm: {
      moduleSpecifier: '@agari-wasm/agari_wasm.js' as const,
      artifact: 'agari_wasm_bg.wasm' as const,
      provenance: agariProvenance,
    },
  };
  const semanticSource = JSON.stringify(semanticPayload);
  const buildAssetVersion = createHash('sha256')
    .update(semanticSource)
    .digest('hex')
    .slice(0, 16);
  const manifest: ProductionBuildAssetManifestV1 = {
    schemaVersion: 1,
    buildAssetVersion,
    recognitionModelSet,
    agariWasm: semanticPayload.agariWasm,
  };
  const source = `${JSON.stringify(manifest, null, 2)}\n`;

  return {
    fileName: `production-assets-${buildAssetVersion}.json`,
    source,
    manifest,
  };
}

export function productionAssetManifestPlugin(): Plugin {
  return {
    name: 'mjtensu-production-asset-manifest',
    apply: 'build',
    buildStart() {
      const materialized = loadProductionBuildAssetManifest();
      this.emitFile({
        type: 'asset',
        fileName: materialized.fileName,
        source: materialized.source,
      });
    },
  };
}

function verifyRecognitionModelArtifacts(
  manifest: RecognitionModelSetManifest,
): void {
  for (const model of Object.values(manifest.models)) {
    const [artifactName, query = ''] = model.url.split('?');
    if (
      artifactName === undefined ||
      artifactName.length === 0 ||
      basename(artifactName) !== artifactName
    ) {
      throw new Error(`Production model URL must name one vendored artifact: ${model.url}`);
    }
    const identity = new URLSearchParams(query).get('sha256');
    if (identity?.toLowerCase() !== model.sha256.toLowerCase()) {
      throw new Error(`Production model URL is not content-addressed: ${model.url}`);
    }

    const bytes = readFileSync(resolve(RECOGNITION_MODEL_DIR, artifactName));
    const actualSha256 = createHash('sha256').update(bytes).digest('hex');
    if (actualSha256 !== model.sha256.toLowerCase()) {
      throw new Error(
        `Production model integrity mismatch for ${model.role}: ${actualSha256}`,
      );
    }
  }
}

function validateAgariProvenance(value: unknown): AgariArtifactProvenanceV1 {
  if (!isRecord(value) || value.schemaVersion !== 1) {
    throw new Error('Agari provenance schemaVersion must be 1.');
  }
  const requiredStrings = [
    'upstreamRepository',
    'upstreamCommit',
    'forkRepository',
    'forkCommit',
    'abiVersion',
    'wasmSha256',
    'rustcVersion',
    'wasmPackVersion',
    'buildProfile',
  ] as const;
  for (const key of requiredStrings) {
    const field = value[key];
    if (typeof field !== 'string' || field.length === 0) {
      throw new Error(`Agari provenance ${key} is missing.`);
    }
  }
  if (
    !FULL_GIT_SHA_PATTERN.test(value.upstreamCommit as string) ||
    !FULL_GIT_SHA_PATTERN.test(value.forkCommit as string)
  ) {
    throw new Error('Agari provenance commits must be full Git SHAs.');
  }
  if (!SHA256_PATTERN.test(value.wasmSha256 as string)) {
    throw new Error('Agari provenance wasmSha256 is invalid.');
  }
  if (
    typeof value.wasmBytes !== 'number' ||
    !Number.isInteger(value.wasmBytes) ||
    value.wasmBytes <= 0
  ) {
    throw new Error('Agari provenance wasmBytes is invalid.');
  }
  if (value.buildProfile !== 'release') {
    throw new Error('Agari production artifact must use the release profile.');
  }
  if (value.generatedAt !== undefined && typeof value.generatedAt !== 'string') {
    throw new Error('Agari provenance generatedAt must be a string when present.');
  }

  return value as unknown as AgariArtifactProvenanceV1;
}

function verifyAgariArtifact(provenance: AgariArtifactProvenanceV1): void {
  const bytes = readFileSync(AGARI_WASM_PATH);
  const actualSha256 = createHash('sha256').update(bytes).digest('hex');
  if (
    actualSha256 !== provenance.wasmSha256.toLowerCase() ||
    bytes.byteLength !== provenance.wasmBytes
  ) {
    throw new Error('Agari WASM artifact does not match its production provenance.');
  }
}

function readJson(path: string): unknown {
  return JSON.parse(readFileSync(path, 'utf8')) as unknown;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
