import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import {
  loadProductionScoringService,
  PRODUCTION_AGARI_WASM_MODULE_PATH,
} from '@/scoring/agari/agari-wasm-loader';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { SCORING_GOLDEN_CORPUS_V1 } from './fixtures/scoring-golden-v1';
import { materializeScoringGoldenCaseV1 } from './support/scoring-golden-corpus';

const VENDOR_AGARI_WASM_DIR = resolve(
  process.cwd(),
  '../../vendor/agari-wasm',
);
const VENDOR_AGARI_WASM_PATH = resolve(
  VENDOR_AGARI_WASM_DIR,
  'agari_wasm_bg.wasm',
);
const VENDOR_PROVENANCE_PATH = resolve(
  VENDOR_AGARI_WASM_DIR,
  'provenance.json',
);

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('production Agari WASM artifact', () => {
  it('matches the committed provenance identity', async () => {
    const [wasmBytes, provenanceText] = await Promise.all([
      readFile(VENDOR_AGARI_WASM_PATH),
      readFile(VENDOR_PROVENANCE_PATH, 'utf8'),
    ]);
    const provenance = JSON.parse(provenanceText) as Record<string, unknown>;
    const sha256 = createHash('sha256').update(wasmBytes).digest('hex');

    expect(provenance).toMatchObject({
      schemaVersion: 1,
      upstreamRepository: 'https://github.com/agari-industries/agari',
      upstreamCommit: 'a0a9ce15cdf1bea6e7e158bbac1adb4e7a33a547',
      forkRepository: 'https://github.com/hiroshiasayadev-prog/mjtensu-agari.git',
      forkCommit: 'fb362b6db416e67984cdb36f704d8ebf6657662e',
      abiVersion: 'v1',
      wasmSha256: sha256,
      wasmBytes: wasmBytes.byteLength,
      rustcVersion: 'rustc 1.98.0 (88d9e12ae 2026-08-18)',
      wasmPackVersion: 'wasm-pack 0.15.0',
      buildProfile: 'release',
    });
    expect(sha256).toBe(
      '0e3297ed5f6807eac4d7369eb5846bc17e5ea4851470bf9d40c78ec6030e277c',
    );
    expect(wasmBytes.byteLength).toBe(200739);
  });

  it('loads the committed vendor module through the production loader and scores a golden case', async () => {
    const wasmBytes = Uint8Array.from(await readFile(VENDOR_AGARI_WASM_PATH));
    const fetchWasm = vi.fn(async () =>
      new Response(wasmBytes, {
        headers: { 'Content-Type': 'application/wasm' },
      }),
    );
    vi.stubGlobal('fetch', fetchWasm);

    expect(PRODUCTION_AGARI_WASM_MODULE_PATH).toBe(
      '@agari-wasm/agari_wasm.js',
    );

    const service = await loadProductionScoringService();
    const goldenCase = SCORING_GOLDEN_CORPUS_V1.cases.find(
      (candidate) => candidate.id === 'ordinary-tanyao-aka-on',
    );
    if (goldenCase === undefined) {
      throw new Error('ordinary-tanyao-aka-on golden case is missing');
    }
    const materialized = materializeScoringGoldenCaseV1(
      SCORING_GOLDEN_CORPUS_V1,
      goldenCase,
    );
    if (materialized.expected.status !== 'scored') {
      throw new Error('ordinary-tanyao-aka-on must be a scored golden case');
    }

    expect(
      service.calculate(materialized.input, materialized.ruleProfile),
    ).toEqual(materialized.expected.calculation);
    expect(fetchWasm).toHaveBeenCalled();
  });
});
