import type {
  FuCalculation,
  LimitClassification,
  RegularYakuId,
  ScoringCalculation,
  ScoringInput,
  ScoringPayment,
  YakumanYakuId,
  YakuEntry,
} from '@/scoring';

import type {
  AgariDoraInfoV1,
  AgariFuBreakdownV1,
  AgariScoredResultV1,
  AgariScoreLevelV1,
  AgariYakuInfoV1,
} from './agari-abi';

const REGULAR_YAKU_MAP = {
  riichi: 'riichi',
  'double-riichi': 'double-riichi',
  ippatsu: 'ippatsu',
  'menzen-tsumo': 'menzen-tsumo',
  tanyao: 'tanyao',
  pinfu: 'pinfu',
  iipeikou: 'iipeikou',
  'yakuhai-east': 'yakuhai-east',
  'yakuhai-south': 'yakuhai-south',
  'yakuhai-west': 'yakuhai-west',
  'yakuhai-north': 'yakuhai-north',
  'yakuhai-white': 'yakuhai-white',
  'yakuhai-green': 'yakuhai-green',
  'yakuhai-red': 'yakuhai-red',
  'rinshan-kaihou': 'rinshan-kaihou',
  chankan: 'chankan',
  'haitei-raoyue': 'haitei',
  'houtei-raoyui': 'houtei',
  toitoi: 'toitoi',
  'sanshoku-doujun': 'sanshoku-doujun',
  'sanshoku-doukou': 'sanshoku-doukou',
  ittsu: 'ittsu',
  chiitoitsu: 'chiitoitsu',
  chanta: 'chanta',
  sanankou: 'sanankou',
  sankantsu: 'sankantsu',
  honroutou: 'honroutou',
  shousangen: 'shousangen',
  honitsu: 'honitsu',
  junchan: 'junchan',
  ryanpeikou: 'ryanpeikou',
  chinitsu: 'chinitsu',
} as const satisfies Record<string, RegularYakuId>;

const YAKUMAN_YAKU_MAP = {
  tenhou: 'tenhou',
  chiihou: 'chiihou',
  'kokushi-musou': 'kokushi-musou',
  'kokushi-13-wait': 'kokushi-13-wait',
  suuankou: 'suuankou',
  'suuankou-tanki': 'suuankou-tanki',
  daisangen: 'daisangen',
  shousuushii: 'shousuushii',
  daisuushii: 'daisuushii',
  tsuuiisou: 'tsuuiisou',
  chinroutou: 'chinroutou',
  ryuuiisou: 'ryuuiisou',
  'chuuren-poutou': 'chuuren-poutou',
  'junsei-chuuren-poutou': 'junsei-chuuren-poutou',
  suukantsu: 'suukantsu',
} as const satisfies Record<string, YakumanYakuId>;

export function normalizeAgariCalculation(
  result: AgariScoredResultV1,
  input: ScoringInput,
): ScoringCalculation {
  assertScoredResult(result, input);

  const yaku = normalizeAgariYaku(result.yaku);
  if (yaku.length === 0) {
    throw new RangeError('scored Agari result must contain at least one scoring yaku');
  }

  const limit = normalizeLimit(result.score_level);
  const actualYakuman = limit?.kind === 'yakuman' && !limit.counted;
  const countedYakuman = limit?.kind === 'yakuman' && limit.counted;
  const containsYakumanYaku = yaku.some((entry) => entry.kind === 'yakuman');

  if (actualYakuman !== containsYakumanYaku) {
    throw new RangeError('yakuman yaku and actual-yakuman score level disagree');
  }
  if (countedYakuman && containsYakumanYaku) {
    throw new RangeError('counted yakuman must not contain actual yakuman yaku');
  }

  const winnerRole = result.is_dealer ? 'dealer' : 'non-dealer';
  const expectedDealer = input.conditions.seatWind === 'east';
  if (result.is_dealer !== expectedDealer) {
    throw new RangeError('Agari dealer role disagrees with the scoring input seat wind');
  }

  return {
    yaku,
    dora: normalizeDora(result.dora),
    han: actualYakuman ? null : result.total_han,
    fu:
      actualYakuman || countedYakuman
        ? null
        : normalizeFu(result.fu, yaku),
    limit,
    winnerRole,
    winMethod: input.conditions.winMethod,
    payment: normalizePayment(result, input),
    totalPoints: result.payment.total,
  };
}

export function normalizeAgariYaku(
  entries: readonly AgariYakuInfoV1[],
): readonly YakuEntry[] {
  const normalized: YakuEntry[] = [];
  const regularIndexes = new Map<RegularYakuId, number>();
  const yakumanIds = new Set<YakumanYakuId>();

  for (const entry of entries) {
    if (entry.kind === 'regular') {
      assertPositiveInteger(entry.han, 'regular yaku han');
      const id = mapRegularYakuCode(entry.code);
      const existingIndex = regularIndexes.get(id);
      if (existingIndex === undefined) {
        regularIndexes.set(id, normalized.length);
        normalized.push({ kind: 'regular', id, han: entry.han });
        continue;
      }

      const existing = normalized[existingIndex];
      if (existing === undefined || existing.kind !== 'regular') {
        throw new RangeError('regular yaku aggregation index is invalid');
      }
      normalized[existingIndex] = {
        kind: 'regular',
        id,
        han: existing.han + entry.han,
      };
      continue;
    }

    if (entry.kind === 'yakuman') {
      const id = mapYakumanYakuCode(entry.code);
      if (yakumanIds.has(id)) {
        throw new RangeError(`duplicate yakuman yaku code: ${entry.code}`);
      }
      yakumanIds.add(id);
      normalized.push({ kind: 'yakuman', id });
      continue;
    }

    throw new RangeError('unknown Agari yaku entry kind');
  }

  return normalized;
}

export function normalizeLimit(
  level: AgariScoreLevelV1,
): LimitClassification | null {
  switch (level.kind) {
    case 'normal':
      return null;
    case 'mangan':
      return { kind: 'mangan', kiriage: level.kiriage };
    case 'haneman':
      return { kind: 'haneman' };
    case 'baiman':
      return { kind: 'baiman' };
    case 'sanbaiman':
      return { kind: 'sanbaiman' };
    case 'yakuman':
      assertPositiveInteger(level.units, 'yakuman units');
      if (level.counted && level.units !== 1) {
        throw new RangeError('counted yakuman must have exactly one unit');
      }
      return {
        kind: 'yakuman',
        units: level.units,
        counted: level.counted,
      };
  }

  throw new RangeError('unknown Agari score-level kind');
}

function normalizeFu(
  fu: AgariFuBreakdownV1,
  yaku: readonly YakuEntry[],
): FuCalculation {
  assertNonNegativeInteger(fu.base, 'fu.base');
  assertNonNegativeInteger(fu.menzen_ron, 'fu.menzen_ron');
  assertNonNegativeInteger(fu.tsumo, 'fu.tsumo');
  assertNonNegativeInteger(fu.melds, 'fu.melds');
  assertNonNegativeInteger(fu.pair, 'fu.pair');
  assertNonNegativeInteger(fu.wait, 'fu.wait');
  assertNonNegativeInteger(fu.raw_total, 'fu.raw_total');
  assertPositiveInteger(fu.rounded, 'fu.rounded');

  if (
    yaku.some((entry) => entry.kind === 'regular' && entry.id === 'chiitoitsu')
  ) {
    if (fu.rounded !== 25) {
      throw new RangeError('chiitoitsu result must use fixed 25 fu');
    }
    return { kind: 'chiitoitsu', fixed: 25 };
  }

  return {
    kind: 'standard',
    base: fu.base,
    menzenRon: fu.menzen_ron,
    tsumo: fu.tsumo,
    melds: fu.melds,
    pair: fu.pair,
    wait: fu.wait,
    rawTotal: fu.raw_total,
    rounded: fu.rounded,
  };
}

function normalizeDora(dora: AgariDoraInfoV1) {
  assertNonNegativeInteger(dora.regular, 'dora.regular');
  assertNonNegativeInteger(dora.ura, 'dora.ura');
  assertNonNegativeInteger(dora.aka, 'dora.aka');
  assertNonNegativeInteger(dora.total, 'dora.total');

  if (dora.total !== dora.regular + dora.ura + dora.aka) {
    throw new RangeError('Agari dora total is inconsistent with its breakdown');
  }

  return {
    dora: dora.regular + dora.ura,
    akaDora: dora.aka,
  };
}

function normalizePayment(
  result: AgariScoredResultV1,
  input: ScoringInput,
): ScoringPayment {
  const payment = result.payment;
  assertPositiveInteger(payment.total, 'payment.total');

  if (input.conditions.winMethod === 'ron') {
    if (
      payment.from_discarder === null ||
      payment.from_dealer !== null ||
      payment.from_non_dealer !== null
    ) {
      throw new RangeError('ron payment fields are inconsistent');
    }
    assertPositiveInteger(payment.from_discarder, 'payment.from_discarder');
    return { kind: 'ron', amount: payment.from_discarder };
  }

  if (result.is_dealer) {
    if (
      payment.from_discarder !== null ||
      payment.from_dealer !== null ||
      payment.from_non_dealer === null
    ) {
      throw new RangeError('dealer-tsumo payment fields are inconsistent');
    }
    assertPositiveInteger(payment.from_non_dealer, 'payment.from_non_dealer');
    return {
      kind: 'tsumo-dealer',
      eachOpponent: payment.from_non_dealer,
    };
  }

  if (
    payment.from_discarder !== null ||
    payment.from_dealer === null ||
    payment.from_non_dealer === null
  ) {
    throw new RangeError('non-dealer-tsumo payment fields are inconsistent');
  }
  assertPositiveInteger(payment.from_dealer, 'payment.from_dealer');
  assertPositiveInteger(payment.from_non_dealer, 'payment.from_non_dealer');
  return {
    kind: 'tsumo-non-dealer',
    dealerPays: payment.from_dealer,
    nonDealerPays: payment.from_non_dealer,
  };
}

function assertScoredResult(
  result: AgariScoredResultV1,
  input: ScoringInput,
): void {
  assertNonNegativeInteger(result.han, 'result.han');
  assertNonNegativeInteger(result.total_han, 'result.total_han');
  if (result.total_han < result.han) {
    throw new RangeError('total han cannot be lower than yaku-only han');
  }

  const containsYakumanYaku = result.yaku.some((entry) => entry.kind === 'yakuman');
  const regularHan = result.yaku.reduce(
    (total, entry) => total + (entry.kind === 'regular' ? entry.han : 0),
    0,
  );
  if (!containsYakumanYaku && regularHan !== result.han) {
    throw new RangeError('Agari yaku-only han disagrees with awarded regular yaku han');
  }
  if (
    !containsYakumanYaku &&
    result.total_han !== result.han + result.dora.total
  ) {
    throw new RangeError('Agari total han disagrees with yaku and dora contributions');
  }

  if (input.conditions.winMethod === 'ron' && result.payment.from_discarder === null) {
    throw new RangeError('ron result lacks discarder payment');
  }
}

function mapRegularYakuCode(code: string): RegularYakuId {
  if (!Object.hasOwn(REGULAR_YAKU_MAP, code)) {
    throw new RangeError(`unknown regular Agari yaku code: ${code}`);
  }
  return REGULAR_YAKU_MAP[code as keyof typeof REGULAR_YAKU_MAP];
}

function mapYakumanYakuCode(code: string): YakumanYakuId {
  if (!Object.hasOwn(YAKUMAN_YAKU_MAP, code)) {
    throw new RangeError(`unknown yakuman Agari yaku code: ${code}`);
  }
  return YAKUMAN_YAKU_MAP[code as keyof typeof YAKUMAN_YAKU_MAP];
}

function assertPositiveInteger(value: number, field: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new RangeError(`${field} must be a positive integer`);
  }
}

function assertNonNegativeInteger(value: number, field: string): void {
  if (!Number.isInteger(value) || value < 0) {
    throw new RangeError(`${field} must be a non-negative integer`);
  }
}
