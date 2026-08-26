import {
  REQUIRED_SCORING_GOLDEN_COVERAGE_V1,
  SCORING_GOLDEN_CORPUS_V1,
} from './fixtures/scoring-golden-v1';
import {
  materializeScoringGoldenCaseV1,
  validateScoringGoldenCorpusV1,
  type ScoringGoldenCorpusV1,
} from './support/scoring-golden-corpus';
import { describe, expect, it } from 'vitest';

describe('scoring golden corpus v1', () => {
  it('passes schema and complete minimum-coverage validation', () => {
    expect(
      validateScoringGoldenCorpusV1(
        SCORING_GOLDEN_CORPUS_V1,
        REQUIRED_SCORING_GOLDEN_COVERAGE_V1,
      ),
    ).toEqual([]);
  });

  it('materializes every fixture into product ScoringInput without Agari notation', () => {
    for (const goldenCase of SCORING_GOLDEN_CORPUS_V1.cases) {
      const materialized = materializeScoringGoldenCaseV1(
        SCORING_GOLDEN_CORPUS_V1,
        goldenCase,
      );

      expect(materialized.id).toBe(goldenCase.id);
      expect(materialized.input.completedHand).toHaveLength(
        goldenCase.input.completedHand.length,
      );
      expect(materialized.input.melds).toHaveLength(goldenCase.input.melds.length);
      expect(materialized.input.doraIndicators).toHaveLength(
        goldenCase.input.doraIndicators.length,
      );
      expect(
        materialized.input.completedHand.some(
          (tile) => tile.id === materialized.input.winningTileId,
        ),
      ).toBe(true);
      expect(materialized.ruleProfile).toEqual(
        SCORING_GOLDEN_CORPUS_V1.ruleProfiles[goldenCase.ruleProfileId],
      );
    }
  });

  it('preserves red-five identity as product tile semantics', () => {
    const goldenCase = SCORING_GOLDEN_CORPUS_V1.cases.find(
      (candidate) => candidate.id === 'ordinary-tanyao-aka-on',
    );
    expect(goldenCase).toBeDefined();

    const materialized = materializeScoringGoldenCaseV1(
      SCORING_GOLDEN_CORPUS_V1,
      goldenCase!,
    );
    const redFive = materialized.input.completedHand.find(
      (tile) => tile.tile.red,
    );

    expect(redFive?.tile).toEqual({ kind: '5p', red: true });
  });

  it('keeps scored expectations explicit rather than snapshot-shaped', () => {
    const scoredCases = SCORING_GOLDEN_CORPUS_V1.cases.filter(
      (goldenCase) => goldenCase.expected.status === 'scored',
    );

    expect(scoredCases.length).toBeGreaterThan(0);
    for (const goldenCase of scoredCases) {
      if (goldenCase.expected.status !== 'scored') {
        throw new Error('unreachable');
      }

      const calculation = goldenCase.expected.calculation;
      expect(Array.isArray(calculation.yaku)).toBe(true);
      expect(calculation.dora).toEqual(
        expect.objectContaining({ dora: expect.any(Number), akaDora: expect.any(Number) }),
      );
      expect(calculation.payment.kind).toMatch(/^(ron|tsumo-dealer|tsumo-non-dealer)$/);
      expect(calculation.totalPoints).toEqual(expect.any(Number));
    }
  });

  it('rejects unknown profiles, duplicate case ids, and missing coverage', () => {
    const baseCase = SCORING_GOLDEN_CORPUS_V1.cases[0];
    expect(baseCase).toBeDefined();

    const invalidCorpus: ScoringGoldenCorpusV1<string> = {
      schemaVersion: 1,
      corpusId: 'invalid-fixture',
      ruleProfiles: SCORING_GOLDEN_CORPUS_V1.ruleProfiles,
      cases: [
        baseCase!,
        {
          ...baseCase!,
          ruleProfileId: 'missing-profile',
        },
      ],
    };

    const errors = validateScoringGoldenCorpusV1(invalidCorpus, [
      'ordinary-four-meld-pair',
      'intentionally-missing',
    ]);

    expect(errors).toEqual(
      expect.arrayContaining([
        expect.stringContaining('Duplicate scoring golden case id'),
        expect.stringContaining('unknown rule profile missing-profile'),
        expect.stringContaining('Missing required scoring golden coverage: intentionally-missing'),
      ]),
    );
  });
});
