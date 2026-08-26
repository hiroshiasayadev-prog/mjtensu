import type { ScoringConditionsDraft, Wind } from '@/scoring';

export type ScoringConditionKey =
  | 'ippatsu'
  | 'rinshan'
  | 'chankan'
  | 'haitei'
  | 'houtei'
  | 'tenhou'
  | 'chiihou';

export interface ScoringConditionAvailability {
  readonly ippatsu: boolean;
  readonly rinshan: boolean;
  readonly chankan: boolean;
  readonly haitei: boolean;
  readonly houtei: boolean;
  readonly tenhou: boolean;
  readonly chiihou: boolean;
}

export interface ScoringConditionPolicy {
  normalize(conditions: ScoringConditionsDraft): ScoringConditionsDraft;
  availability(conditions: ScoringConditionsDraft): ScoringConditionAvailability;
}

interface ScoringConditionRule {
  readonly key: ScoringConditionKey;
  readonly selectableWhen: (conditions: ScoringConditionsDraft) => boolean;
}

type WritableScoringConditionAvailability = {
  -readonly [Key in keyof ScoringConditionAvailability]: ScoringConditionAvailability[Key];
};

const NON_EAST_WINDS = new Set<Wind>(['south', 'west', 'north']);

const INITIAL_DRAW_EXCLUDED_CONDITIONS = [
  'ippatsu',
  'rinshan',
  'chankan',
  'haitei',
  'houtei',
] as const satisfies readonly ScoringConditionKey[];

export const SCORING_CONDITION_RULES = [
  {
    key: 'ippatsu',
    selectableWhen: (conditions) =>
      conditions.riichi !== 'none' && !conditions.tenhou && !conditions.chiihou,
  },
  {
    key: 'rinshan',
    selectableWhen: (conditions) =>
      conditions.winMethod === 'tsumo' &&
      !conditions.haitei &&
      !conditions.tenhou &&
      !conditions.chiihou,
  },
  {
    key: 'chankan',
    selectableWhen: (conditions) =>
      conditions.winMethod === 'ron' &&
      !conditions.houtei &&
      !conditions.tenhou &&
      !conditions.chiihou,
  },
  {
    key: 'haitei',
    selectableWhen: (conditions) =>
      conditions.winMethod === 'tsumo' &&
      !conditions.rinshan &&
      !conditions.tenhou &&
      !conditions.chiihou,
  },
  {
    key: 'houtei',
    selectableWhen: (conditions) =>
      conditions.winMethod === 'ron' &&
      !conditions.chankan &&
      !conditions.tenhou &&
      !conditions.chiihou,
  },
  {
    key: 'tenhou',
    selectableWhen: (conditions) =>
      conditions.winMethod === 'tsumo' &&
      conditions.seatWind === 'east' &&
      conditions.riichi === 'none' &&
      !conditions.chiihou &&
      INITIAL_DRAW_EXCLUDED_CONDITIONS.every((key) => !conditions[key]),
  },
  {
    key: 'chiihou',
    selectableWhen: (conditions) =>
      conditions.winMethod === 'tsumo' &&
      conditions.seatWind !== null &&
      NON_EAST_WINDS.has(conditions.seatWind) &&
      conditions.riichi === 'none' &&
      !conditions.tenhou &&
      INITIAL_DRAW_EXCLUDED_CONDITIONS.every((key) => !conditions[key]),
  },
] as const satisfies readonly ScoringConditionRule[];

function selectedUnavailableConditions(
  conditions: ScoringConditionsDraft,
): ScoringConditionKey[] {
  const unavailable: ScoringConditionKey[] = [];

  for (const rule of SCORING_CONDITION_RULES) {
    if (conditions[rule.key] && !rule.selectableWhen(conditions)) {
      unavailable.push(rule.key);
    }
  }

  return unavailable;
}

export function normalizeScoringConditions(
  conditions: ScoringConditionsDraft,
): ScoringConditionsDraft {
  let normalized = { ...conditions };

  while (true) {
    const unavailable = selectedUnavailableConditions(normalized);
    if (unavailable.length === 0) {
      return normalized;
    }

    normalized = { ...normalized };
    for (const key of unavailable) {
      normalized[key] = false;
    }
  }
}

export function getScoringConditionAvailability(
  conditions: ScoringConditionsDraft,
): ScoringConditionAvailability {
  const normalized = normalizeScoringConditions(conditions);
  const availability: WritableScoringConditionAvailability = {
    ippatsu: false,
    rinshan: false,
    chankan: false,
    haitei: false,
    houtei: false,
    tenhou: false,
    chiihou: false,
  };

  for (const rule of SCORING_CONDITION_RULES) {
    availability[rule.key] = rule.selectableWhen(normalized);
  }

  return availability;
}

export const scoringConditionPolicy: ScoringConditionPolicy = {
  normalize: normalizeScoringConditions,
  availability: getScoringConditionAvailability,
};
