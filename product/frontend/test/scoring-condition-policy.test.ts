import {
  getScoringConditionAvailability,
  normalizeScoringConditions,
  scoringConditionPolicy,
  type ScoringConditionKey,
} from '@/application';
import type {
  RiichiState,
  ScoringConditionsDraft,
  Wind,
  WinMethod,
} from '@/scoring';
import { describe, expect, it } from 'vitest';

const conditionKeys = [
  'ippatsu',
  'rinshan',
  'chankan',
  'haitei',
  'houtei',
  'tenhou',
  'chiihou',
] as const satisfies readonly ScoringConditionKey[];

const initialConditions: ScoringConditionsDraft = {
  winMethod: 'tsumo',
  roundWind: 'east',
  seatWind: 'east',
  riichi: 'none',
  ippatsu: false,
  rinshan: false,
  chankan: false,
  haitei: false,
  houtei: false,
  tenhou: false,
  chiihou: false,
};

function draft(
  overrides: Partial<ScoringConditionsDraft> = {},
): ScoringConditionsDraft {
  return {
    ...initialConditions,
    ...overrides,
  };
}

function withCondition(
  conditions: ScoringConditionsDraft,
  key: ScoringConditionKey,
  value: boolean,
): ScoringConditionsDraft {
  return {
    ...conditions,
    [key]: value,
  };
}

function selectedKeys(conditions: ScoringConditionsDraft): ScoringConditionKey[] {
  return conditionKeys.filter((key) => conditions[key]);
}

function draftSignature(conditions: ScoringConditionsDraft): string {
  return JSON.stringify(conditions);
}

function booleanVariants(): Pick<
  ScoringConditionsDraft,
  ScoringConditionKey
>[] {
  const variants: Pick<ScoringConditionsDraft, ScoringConditionKey>[] = [];

  for (let mask = 0; mask < 2 ** conditionKeys.length; mask += 1) {
    variants.push(
      Object.fromEntries(
        conditionKeys.map((key, index) => [key, (mask & (1 << index)) !== 0]),
      ) as Pick<ScoringConditionsDraft, ScoringConditionKey>,
    );
  }

  return variants;
}

describe('scoring condition policy', () => {
  const clearCases: {
    readonly name: string;
    readonly input: ScoringConditionsDraft;
    readonly expectedSelected: readonly ScoringConditionKey[];
  }[] = [
    {
      name: 'ron clears tsumo-only initial-draw and last-draw conditions',
      input: draft({
        winMethod: 'ron',
        rinshan: true,
        haitei: true,
        tenhou: true,
        chiihou: true,
      }),
      expectedSelected: [],
    },
    {
      name: 'tsumo clears ron-only conditions',
      input: draft({ winMethod: 'tsumo', chankan: true }),
      expectedSelected: [],
    },
    {
      name: 'tsumo clears houtei',
      input: draft({ winMethod: 'tsumo', houtei: true }),
      expectedSelected: [],
    },
    {
      name: 'missing win method clears method-dependent conditions',
      input: draft({ winMethod: null, rinshan: true, chankan: true }),
      expectedSelected: [],
    },
    {
      name: 'riichi none clears ippatsu',
      input: draft({ riichi: 'none', ippatsu: true }),
      expectedSelected: [],
    },
    {
      name: 'riichi clears tenhou and chiihou',
      input: draft({
        riichi: 'riichi',
        tenhou: true,
        chiihou: true,
      }),
      expectedSelected: [],
    },
    {
      name: 'double riichi clears tenhou and chiihou',
      input: draft({
        riichi: 'double-riichi',
        tenhou: true,
        chiihou: true,
      }),
      expectedSelected: [],
    },
    {
      name: 'east seat clears chiihou',
      input: draft({ seatWind: 'east', chiihou: true }),
      expectedSelected: [],
    },
    {
      name: 'non-east seat clears tenhou',
      input: draft({ seatWind: 'south', tenhou: true }),
      expectedSelected: [],
    },
    {
      name: 'missing seat clears initial-draw yakuman',
      input: draft({ seatWind: null, tenhou: true, chiihou: true }),
      expectedSelected: [],
    },
    {
      name: 'rinshan and haitei clear each other',
      input: draft({ rinshan: true, haitei: true }),
      expectedSelected: [],
    },
    {
      name: 'chankan and houtei clear each other',
      input: draft({ winMethod: 'ron', chankan: true, houtei: true }),
      expectedSelected: [],
    },
    {
      name: 'tenhou and chiihou clear each other',
      input: draft({ tenhou: true, chiihou: true }),
      expectedSelected: [],
    },
    {
      name: 'tenhou is isolated from other situational conditions',
      input: draft({ tenhou: true, rinshan: true }),
      expectedSelected: [],
    },
    {
      name: 'chiihou is isolated from other situational conditions',
      input: draft({
        seatWind: 'south',
        chiihou: true,
        haitei: true,
      }),
      expectedSelected: [],
    },
  ];

  it.each(clearCases)('$name', ({ input, expectedSelected }) => {
    const normalized = normalizeScoringConditions(input);

    expect(selectedKeys(normalized)).toEqual(expectedSelected);
    expect(normalizeScoringConditions(normalized)).toEqual(normalized);
  });

  const availabilityCases: {
    readonly name: string;
    readonly input: ScoringConditionsDraft;
    readonly available: readonly ScoringConditionKey[];
  }[] = [
    {
      name: 'tsumo east no-riichi baseline',
      input: draft(),
      available: ['rinshan', 'haitei', 'tenhou'],
    },
    {
      name: 'tsumo non-east no-riichi baseline',
      input: draft({ seatWind: 'south' }),
      available: ['rinshan', 'haitei', 'chiihou'],
    },
    {
      name: 'ron riichi baseline',
      input: draft({ winMethod: 'ron', riichi: 'riichi' }),
      available: ['ippatsu', 'chankan', 'houtei'],
    },
    {
      name: 'rinshan blocks haitei and initial-draw yakuman',
      input: draft({ rinshan: true }),
      available: ['rinshan'],
    },
    {
      name: 'chankan blocks houtei',
      input: draft({ winMethod: 'ron', chankan: true }),
      available: ['chankan'],
    },
    {
      name: 'tenhou blocks every other situational condition',
      input: draft({ tenhou: true }),
      available: ['tenhou'],
    },
    {
      name: 'chiihou blocks every other situational condition',
      input: draft({ seatWind: 'south', chiihou: true }),
      available: ['chiihou'],
    },
    {
      name: 'missing seat allows no initial-draw yakuman',
      input: draft({ seatWind: null }),
      available: ['rinshan', 'haitei'],
    },
  ];

  it.each(availabilityCases)('$name availability', ({ input, available }) => {
    const availability = getScoringConditionAvailability(input);

    expect(conditionKeys.filter((key) => availability[key])).toEqual(available);
  });

  it('keeps availability and normalization consistent for every condition draft', () => {
    const winMethods = ['ron', 'tsumo', null] as const satisfies readonly (
      | WinMethod
      | null
    )[];
    const winds = [
      'east',
      'south',
      'west',
      'north',
      null,
    ] as const satisfies readonly (Wind | null)[];
    const riichiStates = [
      'none',
      'riichi',
      'double-riichi',
    ] as const satisfies readonly RiichiState[];
    const booleans = booleanVariants();

    for (const winMethod of winMethods) {
      for (const roundWind of winds) {
        for (const seatWind of winds) {
          for (const riichi of riichiStates) {
            for (const booleanConditions of booleans) {
              const input = draft({
                winMethod,
                roundWind,
                seatWind,
                riichi,
                ...booleanConditions,
              });
              const normalized = scoringConditionPolicy.normalize(input);
              const secondNormalization =
                scoringConditionPolicy.normalize(normalized);
              const availability =
                scoringConditionPolicy.availability(normalized);

              if (draftSignature(secondNormalization) !== draftSignature(normalized)) {
                throw new Error(
                  `Normalization is not idempotent for ${draftSignature(input)}`,
                );
              }

              for (const key of conditionKeys) {
                if (normalized[key] && !availability[key]) {
                  throw new Error(
                    `Selected ${key} is unavailable for ${draftSignature(normalized)}`,
                  );
                }

                const selectedDraft = withCondition(normalized, key, true);
                const selectedNormalized =
                  scoringConditionPolicy.normalize(selectedDraft);

                if (selectedNormalized[key] !== availability[key]) {
                  throw new Error(
                    `${key} availability disagrees with normalization for ${draftSignature(normalized)}`,
                  );
                }
              }
            }
          }
        }
      }
    }
  });

  it('keeps dependent selections from mutating their ordinary prerequisites', () => {
    const tenhouDraft = draft({ tenhou: true });
    const chiihouDraft = draft({ seatWind: 'south', chiihou: true });
    const ippatsuDraft = draft({ riichi: 'double-riichi', ippatsu: true });

    expect(normalizeScoringConditions(tenhouDraft)).toEqual(tenhouDraft);
    expect(normalizeScoringConditions(chiihouDraft)).toEqual(chiihouDraft);
    expect(normalizeScoringConditions(ippatsuDraft)).toEqual(ippatsuDraft);
  });

  it('clears dependent specials when ordinary prerequisites change later', () => {
    const tenhou = normalizeScoringConditions(draft({ tenhou: true }));
    const chiihou = normalizeScoringConditions(
      draft({ seatWind: 'south', chiihou: true }),
    );
    const ippatsu = normalizeScoringConditions(
      draft({ riichi: 'riichi', ippatsu: true }),
    );

    expect(
      normalizeScoringConditions({ ...tenhou, winMethod: 'ron' }),
    ).toMatchObject({ winMethod: 'ron', tenhou: false });
    expect(
      normalizeScoringConditions({ ...chiihou, seatWind: 'east' }),
    ).toMatchObject({ seatWind: 'east', chiihou: false });
    expect(
      normalizeScoringConditions({ ...ippatsu, riichi: 'none' }),
    ).toMatchObject({ riichi: 'none', ippatsu: false });
  });

  it('does not implement structure-dependent scoring requirements', () => {
    const structureAwareDraft = {
      ...draft({
        tenhou: true,
      }),
      hasOpenMelds: true,
    };

    expect(normalizeScoringConditions(structureAwareDraft)).toMatchObject({
      tenhou: true,
    });
  });
});
