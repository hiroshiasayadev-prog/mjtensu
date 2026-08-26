import {
  INITIAL_SCORING_CONDITIONS,
  createScoringSessionService,
} from '@/application';
import type { RecognizedStructure, TileInstance, TileInstanceId } from '@/domain';
import {
  DEFAULT_RULE_PROFILE,
  type ScoringCalculation,
  type ScoringDraft,
  type ScoringInput,
  type ScoringPreview,
  type ScoringRuleProfile,
  type ScoringService,
} from '@/scoring';
import { describe, expect, it } from 'vitest';

import { createFakeService } from './support';

const readyPreview: ScoringPreview = {
  kind: 'ready',
  yaku: [{ kind: 'regular', id: 'menzen-tsumo', han: 1 }],
};

const calculation: ScoringCalculation = {
  yaku: [{ kind: 'regular', id: 'menzen-tsumo', han: 1 }],
  dora: { dora: 0, akaDora: 0 },
  han: 1,
  fu: {
    kind: 'standard',
    base: 20,
    menzenRon: 0,
    tsumo: 2,
    melds: 0,
    pair: 0,
    wait: 0,
    rawTotal: 22,
    rounded: 30,
  },
  limit: null,
  winnerRole: 'dealer',
  winMethod: 'tsumo',
  payment: { kind: 'tsumo-dealer', eachOpponent: 500 },
  totalPoints: 1500,
};

const alternateRuleProfile: ScoringRuleProfile = {
  ...DEFAULT_RULE_PROFILE,
  akaDora: false,
  doubleWindPairFu: 2,
};

function tileId(value: string): TileInstanceId {
  return value as TileInstanceId;
}

function tile(id: string, kind = '5m'): TileInstance {
  return {
    id: tileId(id),
    tile: { kind: kind as TileInstance['tile']['kind'], red: false },
  };
}

function structure(
  completedHand: readonly TileInstance[],
  overrides: Partial<RecognizedStructure> = {},
): RecognizedStructure {
  return {
    completedHand,
    doraIndicators: [],
    meldGroups: [],
    ...overrides,
  };
}

function fakeScoringService(
  overrides: Partial<ScoringService> = {},
): ScoringService {
  return createFakeService<ScoringService>(
    {
      validateWinningStructure: () => ({ kind: 'valid' }),
      preview: () => readyPreview,
      calculate: () => calculation,
    },
    overrides,
  );
}

describe('scoring session service', () => {
  it('creates a session with initial conditions and the rightmost winning tile', () => {
    const service = createScoringSessionService(fakeScoringService());
    const committed = structure([tile('left'), tile('middle'), tile('right')]);

    const state = service.create(committed, DEFAULT_RULE_PROFILE);

    expect(state).toEqual({
      structure: committed,
      winningTileId: tileId('right'),
      conditions: INITIAL_SCORING_CONDITIONS,
      ruleProfile: DEFAULT_RULE_PROFILE,
      latestResult: null,
    });
  });

  it('rejects session creation without a completed-hand tile', () => {
    const service = createScoringSessionService(fakeScoringService());

    expect(() => service.create(structure([]), DEFAULT_RULE_PROFILE)).toThrow(
      RangeError,
    );
  });

  it('selects among duplicate tile identities by instance ID and rejects non-hand tiles', () => {
    const service = createScoringSessionService(fakeScoringService());
    const firstFive = tile('first-five', '5m');
    const secondFive = tile('second-five', '5m');
    const dora = tile('dora', '5m');
    const state = {
      ...service.create(
        structure([firstFive, secondFive], { doraIndicators: [dora] }),
        DEFAULT_RULE_PROFILE,
      ),
      latestResult: calculation,
    };

    const selected = service.update(state, {
      kind: 'select-winning-tile',
      tileId: firstFive.id,
    });

    expect(selected.winningTileId).toBe(firstFive.id);
    expect(selected.latestResult).toBeNull();
    expect(() =>
      service.update(state, {
        kind: 'select-winning-tile',
        tileId: dora.id,
      }),
    ).toThrow(RangeError);
  });

  it('preserves selected tile ID when replacement keeps the instance with corrected identity', () => {
    const service = createScoringSessionService(fakeScoringService());
    const selectedTile = tile('selected', '5m');
    const state = {
      ...service.create(structure([tile('left'), selectedTile]), DEFAULT_RULE_PROFILE),
      winningTileId: selectedTile.id,
      latestResult: calculation,
    };
    const correctedSelected = {
      ...selectedTile,
      tile: { kind: '6m' as const, red: false },
    };
    const replacement = structure([tile('new-left'), correctedSelected, tile('new-right')]);

    const next = service.update(state, {
      kind: 'replace-structure',
      structure: replacement,
    });

    expect(next.structure).toBe(replacement);
    expect(next.winningTileId).toBe(selectedTile.id);
    expect(next.latestResult).toBeNull();
  });

  it.each([
    {
      name: 'removed from completed hand',
      replacement: structure([tile('new-left'), tile('new-right')]),
    },
    {
      name: 'moved out of completed hand',
      replacement: structure([tile('new-left'), tile('new-right')], {
        doraIndicators: [tile('selected')],
      }),
    },
  ])('defaults to corrected rightmost when selected tile is $name', ({ replacement }) => {
    const service = createScoringSessionService(fakeScoringService());
    const selectedTile = tile('selected');
    const state = {
      ...service.create(structure([tile('left'), selectedTile]), DEFAULT_RULE_PROFILE),
      winningTileId: selectedTile.id,
      latestResult: calculation,
    };

    const next = service.update(state, {
      kind: 'replace-structure',
      structure: replacement,
    });

    expect(next.winningTileId).toBe(tileId('new-right'));
    expect(next.latestResult).toBeNull();
  });

  it('normalizes replaced conditions through the shared policy and invalidates result', () => {
    const service = createScoringSessionService(fakeScoringService());
    const state = {
      ...service.create(structure([tile('winning')]), DEFAULT_RULE_PROFILE),
      latestResult: calculation,
    };

    const next = service.update(state, {
      kind: 'replace-conditions',
      conditions: {
        ...INITIAL_SCORING_CONDITIONS,
        riichi: 'none',
        ippatsu: true,
      },
    });

    expect(next.conditions.ippatsu).toBe(false);
    expect(next.latestResult).toBeNull();
  });

  it('replaces the rule profile and invalidates result', () => {
    const service = createScoringSessionService(fakeScoringService());
    const state = {
      ...service.create(structure([tile('winning')]), DEFAULT_RULE_PROFILE),
      latestResult: calculation,
    };

    const next = service.update(state, {
      kind: 'replace-rule-profile',
      ruleProfile: alternateRuleProfile,
    });

    expect(next.ruleProfile).toBe(alternateRuleProfile);
    expect(next.latestResult).toBeNull();
  });

  it('delegates preview with the current scoring draft and rule profile', () => {
    const previewCalls: {
      readonly draft: ScoringDraft;
      readonly ruleProfile: ScoringRuleProfile;
    }[] = [];
    const service = createScoringSessionService(
      fakeScoringService({
        preview: (draft, ruleProfile) => {
          previewCalls.push({ draft, ruleProfile });
          return readyPreview;
        },
      }),
    );
    const state = service.create(structure([tile('winning')]), DEFAULT_RULE_PROFILE);

    const preview = service.preview(state);

    expect(preview).toBe(readyPreview);
    expect(previewCalls).toEqual([
      {
        draft: {
          structure: state.structure,
          winningTileId: state.winningTileId,
          conditions: state.conditions,
        },
        ruleProfile: DEFAULT_RULE_PROFILE,
      },
    ]);
  });

  it('delegates calculate with strict input and stores only successful results', () => {
    const calculateCalls: {
      readonly input: ScoringInput;
      readonly ruleProfile: ScoringRuleProfile;
    }[] = [];
    const service = createScoringSessionService(
      fakeScoringService({
        calculate: (input, ruleProfile) => {
          calculateCalls.push({ input, ruleProfile });
          return calculation;
        },
      }),
    );
    const completedHand = [tile('one'), tile('two')] as const;
    const doraIndicator = tile('dora', '1z');
    const meldTiles = [
      tile('meld-one', '2m'),
      tile('meld-two', '3m'),
      tile('meld-three', '4m'),
    ] as const;
    const state = service.create(
      structure(completedHand, {
        doraIndicators: [doraIndicator],
        meldGroups: [{ kind: 'chi', tiles: meldTiles }],
      }),
      DEFAULT_RULE_PROFILE,
    );

    const sessionCalculation = service.calculate(state);

    expect(sessionCalculation.result).toBe(calculation);
    expect(sessionCalculation.state.latestResult).toBe(calculation);
    expect(calculateCalls).toEqual([
      {
        input: {
          completedHand,
          melds: [
            {
              kind: 'chi',
              tiles: meldTiles.map(({ tile: tileIdentity }) => tileIdentity),
            },
          ],
          doraIndicators: [doraIndicator.tile],
          winningTileId: tileId('two'),
          conditions: INITIAL_SCORING_CONDITIONS,
        },
        ruleProfile: DEFAULT_RULE_PROFILE,
      },
    ]);
  });

  it('does not fabricate or install a result when calculation fails', () => {
    const service = createScoringSessionService(
      fakeScoringService({
        calculate: () => {
          throw new Error('invalid input');
        },
      }),
    );
    const state = {
      ...service.create(structure([tile('winning')]), DEFAULT_RULE_PROFILE),
      latestResult: calculation,
    };

    expect(() => service.calculate(state)).toThrow('invalid input');
    expect(state.latestResult).toBe(calculation);
  });
});
