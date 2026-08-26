import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import type {
  RecognizedMeldGroup,
  RecognizedStructure,
  TileIdentity,
  TileInstance,
  TileInstanceId,
} from '@/domain';
import type {
  ScoringCalculation,
  ScoringDraft,
  ScoringInput,
  ScoringMeld,
  ScoringService,
  YakuEntry,
} from '@/scoring';
import { loadAgariScoringService } from '@/scoring/agari/agari-wasm-loader';
import { beforeAll, describe, expect, it } from 'vitest';

import * as realAgariWasm from '../../../external/agari/web/src/lib/wasm/agari_wasm.js';
import {
  REQUIRED_SCORING_GOLDEN_COVERAGE_V1,
  SCORING_GOLDEN_CORPUS_V1,
} from './fixtures/scoring-golden-v1';
import {
  materializeScoringGoldenCaseV1,
  validateScoringGoldenCorpusV1,
} from './support/scoring-golden-corpus';

const REAL_AGARI_WASM_PATH = resolve(
  process.cwd(),
  '../../external/agari/web/src/lib/wasm/agari_wasm_bg.wasm',
);

function tileInstance(
  id: string,
  tile: TileIdentity,
): TileInstance {
  return {
    id: id as TileInstanceId,
    tile,
  };
}

function recognizedMeld(
  caseId: string,
  meldIndex: number,
  meld: ScoringMeld,
): RecognizedMeldGroup {
  const instance = (tileIndex: number, tile: TileIdentity) =>
    tileInstance(`golden:${caseId}:meld:${meldIndex}:${tileIndex}`, tile);

  switch (meld.kind) {
    case 'chi':
    case 'pon':
      return {
        kind: meld.kind,
        tiles: [
          instance(0, meld.tiles[0]),
          instance(1, meld.tiles[1]),
          instance(2, meld.tiles[2]),
        ],
      };
    case 'open-kan':
    case 'concealed-kan':
      return {
        kind: meld.kind,
        tiles: [
          instance(0, meld.tiles[0]),
          instance(1, meld.tiles[1]),
          instance(2, meld.tiles[2]),
          instance(3, meld.tiles[3]),
        ],
      };
  }
}

function semanticYaku(entries: readonly YakuEntry[]): readonly YakuEntry[] {
  // The scoring-result contract permits any presentation-ready yaku ordering.
  // Golden compatibility therefore asserts identity/awarded-han semantics, not
  // the concrete engine's iteration order.
  return [...entries].sort((left, right) => {
    const leftKey = `${left.kind}:${left.id}:${left.kind === 'regular' ? left.han : ''}`;
    const rightKey = `${right.kind}:${right.id}:${right.kind === 'regular' ? right.han : ''}`;
    return leftKey.localeCompare(rightKey);
  });
}

function semanticCalculation(calculation: ScoringCalculation): ScoringCalculation {
  return {
    ...calculation,
    yaku: semanticYaku(calculation.yaku),
  };
}

function scoringDraft(caseId: string, input: ScoringInput): ScoringDraft {
  const structure: RecognizedStructure = {
    completedHand: input.completedHand,
    meldGroups: input.melds.map((meld, meldIndex) =>
      recognizedMeld(caseId, meldIndex, meld),
    ),
    doraIndicators: input.doraIndicators.map((tile, index) =>
      tileInstance(`golden:${caseId}:dora:${index}`, tile),
    ),
  };

  return {
    structure,
    winningTileId: input.winningTileId,
    conditions: input.conditions,
  };
}

describe('scoring golden corpus through real Agari WASM', () => {
  let service: ScoringService;

  beforeAll(async () => {
    const wasmBytes = Uint8Array.from(await readFile(REAL_AGARI_WASM_PATH));
    service = await loadAgariScoringService(async () => ({
      default: async () =>
        realAgariWasm.default({
          module_or_path: wasmBytes,
        }),
      score_hand_v1: realAgariWasm.score_hand_v1,
      validate_winning_shape_v1: realAgariWasm.validate_winning_shape_v1,
    }));
  });

  it('accepts the complete V1 corpus schema and coverage inventory', () => {
    expect(
      validateScoringGoldenCorpusV1(
        SCORING_GOLDEN_CORPUS_V1,
        REQUIRED_SCORING_GOLDEN_COVERAGE_V1,
      ),
    ).toEqual([]);
  });

  for (const goldenCase of SCORING_GOLDEN_CORPUS_V1.cases) {
    it(`matches ${goldenCase.id}`, () => {
      const materialized = materializeScoringGoldenCaseV1(
        SCORING_GOLDEN_CORPUS_V1,
        goldenCase,
      );
      const draft = scoringDraft(materialized.id, materialized.input);
      const shape = service.validateWinningStructure(draft.structure);
      const preview = service.preview(draft, materialized.ruleProfile);

      switch (materialized.expected.status) {
        case 'not-winning-shape':
          expect(shape).toEqual({ kind: 'not-winning-shape' });
          expect(preview).toEqual({ kind: 'invalid-winning-shape' });
          return;
        case 'no-yaku':
          expect(shape).toEqual({ kind: 'valid' });
          expect(preview).toEqual({ kind: 'no-yaku' });
          return;
        case 'scored': {
          expect(shape).toEqual({ kind: 'valid' });
          expect(preview.kind).toBe('ready');
          if (preview.kind !== 'ready') {
            throw new Error(`expected ready preview for ${materialized.id}`);
          }
          expect(semanticYaku(preview.yaku)).toEqual(
            semanticYaku(materialized.expected.calculation.yaku),
          );
          expect(
            semanticCalculation(
              service.calculate(materialized.input, materialized.ruleProfile),
            ),
          ).toEqual(semanticCalculation(materialized.expected.calculation));
          return;
        }
      }
    });
  }
});
