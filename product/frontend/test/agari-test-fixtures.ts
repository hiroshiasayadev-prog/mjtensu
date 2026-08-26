import type {
  TileIdentity,
  TileInstance,
  TileInstanceId,
  TileKind,
} from '@/domain';
import type {
  ScoringConditions,
  ScoringInput,
  ScoringRuleProfile,
} from '@/scoring';
import { DEFAULT_RULE_PROFILE } from '@/scoring';
import type { AgariScoredResultV1 } from '../src/scoring/agari/agari-abi';

export function tile(kind: TileKind, red = false): TileIdentity {
  return { kind, red };
}

export function instance(
  id: string,
  kind: TileKind,
  red = false,
): TileInstance {
  return { id: id as TileInstanceId, tile: tile(kind, red) };
}

export const BASE_CONDITIONS: ScoringConditions = {
  winMethod: 'ron',
  roundWind: 'east',
  seatWind: 'south',
  riichi: 'none',
  ippatsu: false,
  rinshan: false,
  chankan: false,
  haitei: false,
  houtei: false,
  tenhou: false,
  chiihou: false,
};

export function closedInput(
  conditions: ScoringConditions = BASE_CONDITIONS,
  ruleProfile: ScoringRuleProfile = DEFAULT_RULE_PROFILE,
): { readonly input: ScoringInput; readonly ruleProfile: ScoringRuleProfile } {
  const completedHand = [
    instance('t-1', '2m'),
    instance('t-2', '3m'),
    instance('t-3', '4m'),
    instance('t-4', '3p'),
    instance('t-5', '4p'),
    instance('t-6', '5p', true),
    instance('t-7', '4s'),
    instance('t-8', '5s'),
    instance('t-9', '6s'),
    instance('t-10', '6m'),
    instance('t-11', '7m'),
    instance('t-12', '8m'),
    instance('t-13', '6p'),
    instance('t-14', '6p'),
  ] as const;

  return {
    input: {
      completedHand,
      melds: [],
      doraIndicators: [tile('9m')],
      winningTileId: completedHand[13].id,
      conditions,
    },
    ruleProfile,
  };
}

export function standardScoredResult(
  overrides: Partial<AgariScoredResultV1> = {},
): AgariScoredResultV1 {
  return {
    han: 1,
    total_han: 1,
    yaku: [{ kind: 'regular', code: 'tanyao', han: 1 }],
    score_level: { kind: 'normal' },
    fu: {
      base: 20,
      menzen_ron: 10,
      tsumo: 0,
      melds: 0,
      pair: 0,
      wait: 0,
      raw_total: 30,
      rounded: 30,
    },
    dora: { regular: 0, ura: 0, aka: 0, total: 0 },
    payment: {
      total: 1000,
      from_discarder: 1000,
      from_dealer: null,
      from_non_dealer: null,
    },
    is_dealer: false,
    ...overrides,
  };
}
