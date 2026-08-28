import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import initAgariWasm, {
  score_hand_v1,
} from '@agari-wasm/agari_wasm.js';
import { createAgariScoreRequest } from '@/scoring/agari/agari-input-adapter';
import { normalizeAgariCalculation } from '@/scoring/agari/agari-result-adapter';
import { beforeAll, describe, expect, it } from 'vitest';

import { SCORING_GOLDEN_CORPUS_V1 } from './fixtures/scoring-golden-v1';
import { materializeScoringGoldenCaseV1 } from './support/scoring-golden-corpus';

const VENDOR_AGARI_WASM_PATH = resolve(
  process.cwd(),
  '../../vendor/agari-wasm/agari_wasm_bg.wasm',
);

beforeAll(async () => {
  const wasmBytes = Uint8Array.from(await readFile(VENDOR_AGARI_WASM_PATH));
  await initAgariWasm({ module_or_path: wasmBytes });
});

describe('scoring golden corpus against committed real Agari WASM', () => {
  for (const goldenCase of SCORING_GOLDEN_CORPUS_V1.cases) {
    it(goldenCase.id, () => {
      const materialized = materializeScoringGoldenCaseV1(
        SCORING_GOLDEN_CORPUS_V1,
        goldenCase,
      );
      const request = createAgariScoreRequest(
        materialized.input,
        materialized.ruleProfile,
      );
      const outcome = score_hand_v1(request);

      if (materialized.expected.status !== 'scored') {
        expect(outcome.status).toBe(materialized.expected.status);
        return;
      }

      expect(outcome.status).toBe('scored');
      if (outcome.status !== 'scored') {
        throw new Error(
          `${goldenCase.id}: expected scored outcome, received ${outcome.status}`,
        );
      }

      const actualCalculation = normalizeAgariCalculation(
        outcome.result,
        materialized.input,
      );
      const {
        yaku: actualYaku,
        ...actualCalculationWithoutYaku
      } = actualCalculation;
      const {
        yaku: expectedYaku,
        ...expectedCalculationWithoutYaku
      } = materialized.expected.calculation;

      expect(actualCalculationWithoutYaku).toEqual(
        expectedCalculationWithoutYaku,
      );
      expect(actualYaku).toHaveLength(expectedYaku.length);
      expect(actualYaku).toEqual(expect.arrayContaining(expectedYaku));
    });
  }
});
