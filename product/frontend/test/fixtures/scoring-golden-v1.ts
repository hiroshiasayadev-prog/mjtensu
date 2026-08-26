import type {
  FuCalculation,
  RegularYakuId,
  ScoringCalculation,
  ScoringConditions,
  ScoringRuleProfile,
  YakumanYakuId,
  YakuEntry,
} from '@/scoring';
import type {
  GoldenTileToken,
  ScoringGoldenCorpusV1,
  ScoringGoldenExpectedV1,
  ScoringGoldenInputV1,
  ScoringGoldenMeldV1,
} from '../support/scoring-golden-corpus';

export type ScoringGoldenCoverageV1 =
  | 'ordinary-four-meld-pair'
  | 'chiitoitsu'
  | 'kokushi-musou'
  | 'kokushi-13-wait'
  | 'aka-dora'
  | 'indicator-dora'
  | 'riichi'
  | 'double-riichi'
  | 'ippatsu'
  | 'menzen-tsumo'
  | 'rinshan-kaihou'
  | 'chankan'
  | 'haitei'
  | 'houtei'
  | 'tenhou'
  | 'chiihou'
  | 'kuisagari-open'
  | 'kuisagari-closed'
  | 'fu-base'
  | 'fu-menzen-ron'
  | 'fu-tsumo'
  | 'fu-melds'
  | 'fu-pair'
  | 'fu-wait'
  | 'fu-raw-rounded'
  | 'fu-chiitoitsu-25'
  | 'fu-pinfu-tsumo-20'
  | 'fu-open-minimum-30'
  | 'kiriage-4h30-on'
  | 'kiriage-4h30-off'
  | 'kiriage-3h60-on'
  | 'kiriage-3h60-off'
  | 'kazoe-yakuman-on'
  | 'kazoe-yakuman-off'
  | 'double-variant-kokushi-13-on'
  | 'double-variant-kokushi-13-off'
  | 'double-variant-suuankou-tanki-on'
  | 'double-variant-suuankou-tanki-off'
  | 'double-variant-daisuushii-on'
  | 'double-variant-daisuushii-off'
  | 'double-variant-junsei-chuuren-on'
  | 'double-variant-junsei-chuuren-off'
  | 'multiple-yakuman-on'
  | 'multiple-yakuman-off'
  | 'yakuman-policy-interaction-on-on'
  | 'yakuman-policy-interaction-on-off'
  | 'yakuman-policy-interaction-off-on'
  | 'yakuman-policy-interaction-off-off'
  | 'double-wind-pair-2'
  | 'double-wind-pair-4'
  | 'open-tanyao-on'
  | 'open-tanyao-off'
  | 'aka-dora-on'
  | 'aka-dora-off'
  | 'indicator-dora-on'
  | 'indicator-dora-off'
  | 'ippatsu-on'
  | 'ippatsu-off'
  | 'payment-ron-dealer'
  | 'payment-ron-non-dealer'
  | 'payment-tsumo-dealer'
  | 'payment-tsumo-non-dealer'
  | 'not-winning-shape'
  | 'no-yaku'
  | 'dora-only-no-yaku'
  | 'limit-normal'
  | 'limit-mangan'
  | 'limit-haneman'
  | 'limit-baiman'
  | 'limit-sanbaiman'
  | 'limit-yakuman-actual'
  | 'limit-yakuman-counted';

export const REQUIRED_SCORING_GOLDEN_COVERAGE_V1 = [
  'ordinary-four-meld-pair',
  'chiitoitsu',
  'kokushi-musou',
  'kokushi-13-wait',
  'aka-dora',
  'indicator-dora',
  'riichi',
  'double-riichi',
  'ippatsu',
  'menzen-tsumo',
  'rinshan-kaihou',
  'chankan',
  'haitei',
  'houtei',
  'tenhou',
  'chiihou',
  'kuisagari-open',
  'kuisagari-closed',
  'fu-base',
  'fu-menzen-ron',
  'fu-tsumo',
  'fu-melds',
  'fu-pair',
  'fu-wait',
  'fu-raw-rounded',
  'fu-chiitoitsu-25',
  'fu-pinfu-tsumo-20',
  'fu-open-minimum-30',
  'kiriage-4h30-on',
  'kiriage-4h30-off',
  'kiriage-3h60-on',
  'kiriage-3h60-off',
  'kazoe-yakuman-on',
  'kazoe-yakuman-off',
  'double-variant-kokushi-13-on',
  'double-variant-kokushi-13-off',
  'double-variant-suuankou-tanki-on',
  'double-variant-suuankou-tanki-off',
  'double-variant-daisuushii-on',
  'double-variant-daisuushii-off',
  'double-variant-junsei-chuuren-on',
  'double-variant-junsei-chuuren-off',
  'multiple-yakuman-on',
  'multiple-yakuman-off',
  'yakuman-policy-interaction-on-on',
  'yakuman-policy-interaction-on-off',
  'yakuman-policy-interaction-off-on',
  'yakuman-policy-interaction-off-off',
  'double-wind-pair-2',
  'double-wind-pair-4',
  'open-tanyao-on',
  'open-tanyao-off',
  'aka-dora-on',
  'aka-dora-off',
  'indicator-dora-on',
  'indicator-dora-off',
  'ippatsu-on',
  'ippatsu-off',
  'payment-ron-dealer',
  'payment-ron-non-dealer',
  'payment-tsumo-dealer',
  'payment-tsumo-non-dealer',
  'not-winning-shape',
  'no-yaku',
  'dora-only-no-yaku',
  'limit-normal',
  'limit-mangan',
  'limit-haneman',
  'limit-baiman',
  'limit-sanbaiman',
  'limit-yakuman-actual',
  'limit-yakuman-counted',
] as const satisfies readonly ScoringGoldenCoverageV1[];

const PRODUCT_DEFAULT_RULE_PROFILE_V1: ScoringRuleProfile = {
  openTanyao: true,
  akaDora: true,
  dora: true,
  ippatsu: true,
  kiriageMangan: true,
  kazoeYakuman: false,
  multipleYakuman: true,
  doubleYakumanVariants: false,
  doubleWindPairFu: 2,
  chiitoitsuFu: 25,
  pinfuTsumoFu: 20,
  openPinfuRonMinimumFu: 30,
};

const RULE_PROFILES = {
  'product-default': PRODUCT_DEFAULT_RULE_PROFILE_V1,
  'kiriage-off': {
    ...PRODUCT_DEFAULT_RULE_PROFILE_V1,
    kiriageMangan: false,
  },
  'kazoe-on': {
    ...PRODUCT_DEFAULT_RULE_PROFILE_V1,
    kazoeYakuman: true,
  },
  'double-variants-on': {
    ...PRODUCT_DEFAULT_RULE_PROFILE_V1,
    doubleYakumanVariants: true,
  },
  'multiple-off': {
    ...PRODUCT_DEFAULT_RULE_PROFILE_V1,
    multipleYakuman: false,
  },
  'double-variants-on-multiple-off': {
    ...PRODUCT_DEFAULT_RULE_PROFILE_V1,
    doubleYakumanVariants: true,
    multipleYakuman: false,
  },
  'double-wind-4': {
    ...PRODUCT_DEFAULT_RULE_PROFILE_V1,
    doubleWindPairFu: 4,
  },
  'double-wind-4-kiriage-off': {
    ...PRODUCT_DEFAULT_RULE_PROFILE_V1,
    doubleWindPairFu: 4,
    kiriageMangan: false,
  },
  'open-tanyao-off': {
    ...PRODUCT_DEFAULT_RULE_PROFILE_V1,
    openTanyao: false,
  },
  'aka-off': {
    ...PRODUCT_DEFAULT_RULE_PROFILE_V1,
    akaDora: false,
  },
  'dora-off': {
    ...PRODUCT_DEFAULT_RULE_PROFILE_V1,
    dora: false,
  },
  'kiriage-off-ippatsu-off': {
    ...PRODUCT_DEFAULT_RULE_PROFILE_V1,
    kiriageMangan: false,
    ippatsu: false,
  },
} as const satisfies Readonly<Record<string, ScoringRuleProfile>>;

const BASE_CONDITIONS: ScoringConditions = {
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

function conditions(overrides: Partial<ScoringConditions> = {}): ScoringConditions {
  return { ...BASE_CONDITIONS, ...overrides };
}

function input(
  completedHand: readonly GoldenTileToken[],
  winningTileIndex: number,
  conditionOverrides: Partial<ScoringConditions> = {},
  extras: {
    readonly melds?: readonly ScoringGoldenMeldV1[];
    readonly doraIndicators?: readonly GoldenTileToken[];
  } = {},
): ScoringGoldenInputV1 {
  return {
    completedHand,
    melds: extras.melds ?? [],
    doraIndicators: extras.doraIndicators ?? [],
    winningTileIndex,
    conditions: conditions(conditionOverrides),
  };
}

function regular(id: RegularYakuId, han: number): YakuEntry {
  return { kind: 'regular', id, han };
}

function yakuman(id: YakumanYakuId): YakuEntry {
  return { kind: 'yakuman', id };
}

function standardFu(
  base: number,
  menzenRon: number,
  tsumo: number,
  melds: number,
  pair: number,
  wait: number,
  rawTotal: number,
  rounded: number,
): FuCalculation {
  return {
    kind: 'standard',
    base,
    menzenRon,
    tsumo,
    melds,
    pair,
    wait,
    rawTotal,
    rounded,
  };
}

function scored(calculation: ScoringCalculation): ScoringGoldenExpectedV1 {
  return { status: 'scored', calculation };
}

const NO_DORA = { dora: 0, akaDora: 0 } as const;
const NO_LIMIT = null;

const HAND_TANYAO_RED = [
  '2m', '3m', '4m',
  '3p', '4p', 'red5p',
  '4s', '5s', '6s',
  '6m', '7m', '8m',
  '6p', '6p',
] as const satisfies readonly GoldenTileToken[];

const HAND_PINFU_TANYAO = [
  '2m', '3m', '4m',
  '5m', '6m', '7m',
  '2p', '3p', '4p',
  '5p', '6p', '7p',
  '2s', '2s',
] as const satisfies readonly GoldenTileToken[];

const HAND_OPEN_TANYAO = [
  '3p', '4p', '5p',
  '4s', '5s', '6s',
  '6m', '7m', '8m',
  '8p', '8p',
] as const satisfies readonly GoldenTileToken[];

const MELD_OPEN_234M = {
  kind: 'chi',
  tiles: ['2m', '3m', '4m'],
} as const satisfies ScoringGoldenMeldV1;

const HAND_ITTSU_CLOSED = [
  '1m', '2m', '3m',
  '4m', '5m', '6m',
  '7m', '8m', '9m',
  '7s', '8s', '9s',
  '5z', '5z',
] as const satisfies readonly GoldenTileToken[];

const HAND_ITTSU_OPEN = [
  '4m', '5m', '6m',
  '7m', '8m', '9m',
  '7s', '8s', '9s',
  '5z', '5z',
] as const satisfies readonly GoldenTileToken[];

const MELD_OPEN_123M = {
  kind: 'chi',
  tiles: ['1m', '2m', '3m'],
} as const satisfies ScoringGoldenMeldV1;

const HAND_CHIITOITSU = [
  '1m', '1m', '3m', '3m',
  '2p', '2p', '5p', '5p',
  '4s', '4s', '7s', '7s',
  '1z', '1z',
] as const satisfies readonly GoldenTileToken[];

const HAND_KOKUSHI = [
  '1m', '9m', '1p', '9p', '1s', '9s',
  '1z', '2z', '3z', '4z', '5z', '6z', '7z', '7z',
] as const satisfies readonly GoldenTileToken[];

const HAND_SITUATIONAL_YAKUMAN = [
  '2m', '3m', '4m',
  '4p', '5p', '6p',
  '7s', '8s', '9s',
  '3s', '4s', '5s',
  '6p', '6p',
] as const satisfies readonly GoldenTileToken[];

const HAND_RINSHAN = [
  '1p', '2p', '3p',
  '4p', '5p', '6p',
  '7s', '8s', '9s',
  '2z', '2z',
] as const satisfies readonly GoldenTileToken[];

const MELD_CONCEALED_5555M = {
  kind: 'concealed-kan',
  tiles: ['5m', '5m', '5m', '5m'],
} as const satisfies ScoringGoldenMeldV1;

const HAND_KIRIAGE_4H30 = [
  '2m', '3m', '4m',
  '3m', '4m', '5m',
  '4p', '5p', '6p',
  '6s', '7s', '8s',
  '2p', '2p',
] as const satisfies readonly GoldenTileToken[];

const HAND_KIRIAGE_3H = [
  '5z', '5z', '5z',
  '1m', '1m', '1m',
  '3p', '4p', '5p',
  '6s', '7s', '8s',
  '2z', '2z',
] as const satisfies readonly GoldenTileToken[];

const HAND_KAZOE = [
  '2s', '2s', '3s', '3s', '4s', '4s', '5s',
  '5s', '6s', '6s', '7s', '7s', '9s', '9s',
] as const satisfies readonly GoldenTileToken[];

const HAND_BAIMAN = [
  '1m', '2m', '3m',
  '3m', '4m', '5m',
  '5m', '6m', '7m',
  '7m', '8m', '9m',
  '5m', '5m',
] as const satisfies readonly GoldenTileToken[];

const HAND_SUUANKOU_TANKI = [
  '1m', '1m', '1m',
  '3m', '3m', '3m',
  '5p', '5p', '5p',
  '7s', '7s', '7s',
  '9s', '9s',
] as const satisfies readonly GoldenTileToken[];

const HAND_DAISUUSHII = [
  '1z', '1z', '1z',
  '2z', '2z', '2z',
  '3z', '3z', '3z',
  '4z', '4z', '4z',
  '9m', '9m',
] as const satisfies readonly GoldenTileToken[];

const HAND_JUNSEI_CHUUREN = [
  '1m', '1m', '1m',
  '2m', '3m', '4m',
  '5m', '5m',
  '6m', '7m', '8m',
  '9m', '9m', '9m',
] as const satisfies readonly GoldenTileToken[];

const HAND_MULTI_YAKUMAN = [
  '5z', '5z', '5z',
  '6z', '6z', '6z',
  '7z', '7z', '7z',
  '1z', '1z', '1z',
  '2z', '2z',
] as const satisfies readonly GoldenTileToken[];

const HAND_POLICY_INTERACTION = [
  '1z', '1z', '1z',
  '2z', '2z', '2z',
  '3z', '3z', '3z',
  '4z', '4z',
  '5z', '5z', '5z',
] as const satisfies readonly GoldenTileToken[];

const HAND_NO_YAKU = [
  '1m', '1m', '1m',
  '7m', '7m', '7m',
  '9p', '9p', '9p',
  '5z', '5z',
] as const satisfies readonly GoldenTileToken[];

const MELD_OPEN_678M = {
  kind: 'chi',
  tiles: ['6m', '7m', '8m'],
} as const satisfies ScoringGoldenMeldV1;

const HAND_NOT_WINNING = [
  '1m', '2m', '3m', '4m',
  '5p', '6p', '7p', '8p',
  '9s',
  '1z', '2z', '3z', '5z', '5z',
] as const satisfies readonly GoldenTileToken[];

function actualYakumanExpected(
  yaku: readonly YakuEntry[],
  units: number,
  totalPoints: number,
): ScoringGoldenExpectedV1 {
  return scored({
    yaku,
    dora: NO_DORA,
    han: null,
    fu: null,
    limit: { kind: 'yakuman', units, counted: false },
    winnerRole: 'non-dealer',
    winMethod: 'ron',
    payment: { kind: 'ron', amount: totalPoints },
    totalPoints,
  });
}

export const SCORING_GOLDEN_CORPUS_V1 = {
  schemaVersion: 1,
  corpusId: 'mjtensu-scoring-golden-v1',
  ruleProfiles: RULE_PROFILES,
  cases: [
    {
      id: 'ordinary-tanyao-aka-on',
      description: 'Closed ordinary hand: Tanyao plus one red-five aka dora.',
      coverage: [
        'ordinary-four-meld-pair',
        'aka-dora',
        'aka-dora-on',
        'fu-base',
        'fu-menzen-ron',
        'fu-wait',
        'fu-raw-rounded',
        'payment-ron-non-dealer',
        'limit-normal',
      ],
      ruleProfileId: 'product-default',
      input: input(HAND_TANYAO_RED, 12),
      expected: scored({
        yaku: [regular('tanyao', 1)],
        dora: { dora: 0, akaDora: 1 },
        han: 2,
        fu: standardFu(20, 10, 0, 0, 0, 2, 32, 40),
        limit: NO_LIMIT,
        winnerRole: 'non-dealer',
        winMethod: 'ron',
        payment: { kind: 'ron', amount: 2600 },
        totalPoints: 2600,
      }),
    },
    {
      id: 'ordinary-tanyao-aka-off',
      description: 'The same red-five hand with aka-dora scoring disabled.',
      coverage: ['aka-dora-off'],
      ruleProfileId: 'aka-off',
      input: input(HAND_TANYAO_RED, 12),
      expected: scored({
        yaku: [regular('tanyao', 1)],
        dora: { dora: 0, akaDora: 0 },
        han: 1,
        fu: standardFu(20, 10, 0, 0, 0, 2, 32, 40),
        limit: NO_LIMIT,
        winnerRole: 'non-dealer',
        winMethod: 'ron',
        payment: { kind: 'ron', amount: 1300 },
        totalPoints: 1300,
      }),
    },
    {
      id: 'indicator-dora-on',
      description: 'Indicator 5p makes both 6p pair tiles dora while aka remains separate.',
      coverage: ['indicator-dora', 'indicator-dora-on', 'limit-mangan'],
      ruleProfileId: 'product-default',
      input: input(HAND_TANYAO_RED, 12, {}, { doraIndicators: ['5p'] }),
      expected: scored({
        yaku: [regular('tanyao', 1)],
        dora: { dora: 2, akaDora: 1 },
        han: 4,
        fu: standardFu(20, 10, 0, 0, 0, 2, 32, 40),
        limit: { kind: 'mangan', kiriage: false },
        winnerRole: 'non-dealer',
        winMethod: 'ron',
        payment: { kind: 'ron', amount: 8000 },
        totalPoints: 8000,
      }),
    },
    {
      id: 'indicator-dora-off',
      description: 'The supplied indicator contributes zero when indicator dora is disabled.',
      coverage: ['indicator-dora-off'],
      ruleProfileId: 'dora-off',
      input: input(HAND_TANYAO_RED, 12, {}, { doraIndicators: ['5p'] }),
      expected: scored({
        yaku: [regular('tanyao', 1)],
        dora: { dora: 0, akaDora: 1 },
        han: 2,
        fu: standardFu(20, 10, 0, 0, 0, 2, 32, 40),
        limit: NO_LIMIT,
        winnerRole: 'non-dealer',
        winMethod: 'ron',
        payment: { kind: 'ron', amount: 2600 },
        totalPoints: 2600,
      }),
    },
    {
      id: 'dealer-tsumo-ordinary',
      description: 'Dealer closed tsumo preserves dealer payment shape.',
      coverage: ['payment-tsumo-dealer', 'menzen-tsumo'],
      ruleProfileId: 'product-default',
      input: input(HAND_TANYAO_RED, 12, {
        winMethod: 'tsumo',
        seatWind: 'east',
      }),
      expected: scored({
        yaku: [regular('menzen-tsumo', 1), regular('tanyao', 1)],
        dora: { dora: 0, akaDora: 1 },
        han: 3,
        fu: standardFu(20, 0, 2, 0, 0, 2, 24, 30),
        limit: NO_LIMIT,
        winnerRole: 'dealer',
        winMethod: 'tsumo',
        payment: { kind: 'tsumo-dealer', eachOpponent: 2000 },
        totalPoints: 6000,
      }),
    },
    {
      id: 'riichi-pinfu-tanyao-ron',
      description: 'Closed non-dealer ron with Riichi, Pinfu, and Tanyao.',
      coverage: ['riichi'],
      ruleProfileId: 'product-default',
      input: input(HAND_PINFU_TANYAO, 5, { riichi: 'riichi' }),
      expected: scored({
        yaku: [regular('riichi', 1), regular('tanyao', 1), regular('pinfu', 1)],
        dora: NO_DORA,
        han: 3,
        fu: standardFu(20, 10, 0, 0, 0, 0, 30, 30),
        limit: NO_LIMIT,
        winnerRole: 'non-dealer',
        winMethod: 'ron',
        payment: { kind: 'ron', amount: 3900 },
        totalPoints: 3900,
      }),
    },
    {
      id: 'double-riichi-pinfu-tsumo',
      description: 'Double Riichi is awarded as two han on a closed Pinfu tsumo.',
      coverage: ['double-riichi', 'fu-pinfu-tsumo-20', 'payment-tsumo-non-dealer'],
      ruleProfileId: 'product-default',
      input: input(HAND_PINFU_TANYAO, 5, {
        winMethod: 'tsumo',
        riichi: 'double-riichi',
      }),
      expected: scored({
        yaku: [
          regular('double-riichi', 2),
          regular('menzen-tsumo', 1),
          regular('tanyao', 1),
          regular('pinfu', 1),
        ],
        dora: NO_DORA,
        han: 5,
        fu: standardFu(20, 0, 0, 0, 0, 0, 20, 20),
        limit: { kind: 'mangan', kiriage: false },
        winnerRole: 'non-dealer',
        winMethod: 'tsumo',
        payment: {
          kind: 'tsumo-non-dealer',
          dealerPays: 4000,
          nonDealerPays: 2000,
        },
        totalPoints: 8000,
      }),
    },
    {
      id: 'ippatsu-rule-on',
      description: 'Ippatsu adds one awarded han when its rule switch is enabled.',
      coverage: ['ippatsu', 'ippatsu-on'],
      ruleProfileId: 'kiriage-off',
      input: input(HAND_PINFU_TANYAO, 5, {
        riichi: 'riichi',
        ippatsu: true,
      }),
      expected: scored({
        yaku: [
          regular('riichi', 1),
          regular('ippatsu', 1),
          regular('tanyao', 1),
          regular('pinfu', 1),
        ],
        dora: NO_DORA,
        han: 4,
        fu: standardFu(20, 10, 0, 0, 0, 0, 30, 30),
        limit: NO_LIMIT,
        winnerRole: 'non-dealer',
        winMethod: 'ron',
        payment: { kind: 'ron', amount: 7700 },
        totalPoints: 7700,
      }),
    },
    {
      id: 'ippatsu-rule-off',
      description: 'An active ippatsu condition awards nothing when the rule switch is disabled.',
      coverage: ['ippatsu-off'],
      ruleProfileId: 'kiriage-off-ippatsu-off',
      input: input(HAND_PINFU_TANYAO, 5, {
        riichi: 'riichi',
        ippatsu: true,
      }),
      expected: scored({
        yaku: [regular('riichi', 1), regular('tanyao', 1), regular('pinfu', 1)],
        dora: NO_DORA,
        han: 3,
        fu: standardFu(20, 10, 0, 0, 0, 0, 30, 30),
        limit: NO_LIMIT,
        winnerRole: 'non-dealer',
        winMethod: 'ron',
        payment: { kind: 'ron', amount: 3900 },
        totalPoints: 3900,
      }),
    },
    {
      id: 'haneman-double-riichi-ippatsu-tsumo',
      description: 'Six-han closed tsumo exercises Haneman classification.',
      coverage: ['limit-haneman'],
      ruleProfileId: 'product-default',
      input: input(HAND_PINFU_TANYAO, 5, {
        winMethod: 'tsumo',
        riichi: 'double-riichi',
        ippatsu: true,
      }),
      expected: scored({
        yaku: [
          regular('double-riichi', 2),
          regular('ippatsu', 1),
          regular('menzen-tsumo', 1),
          regular('tanyao', 1),
          regular('pinfu', 1),
        ],
        dora: NO_DORA,
        han: 6,
        fu: standardFu(20, 0, 0, 0, 0, 0, 20, 20),
        limit: { kind: 'haneman' },
        winnerRole: 'non-dealer',
        winMethod: 'tsumo',
        payment: {
          kind: 'tsumo-non-dealer',
          dealerPays: 6000,
          nonDealerPays: 3000,
        },
        totalPoints: 12000,
      }),
    },
    {
      id: 'chankan',
      description: 'Chankan is distinct from ordinary closed-ron composition yaku.',
      coverage: ['chankan'],
      ruleProfileId: 'product-default',
      input: input(HAND_PINFU_TANYAO, 5, { chankan: true }),
      expected: scored({
        yaku: [regular('chankan', 1), regular('tanyao', 1), regular('pinfu', 1)],
        dora: NO_DORA,
        han: 3,
        fu: standardFu(20, 10, 0, 0, 0, 0, 30, 30),
        limit: NO_LIMIT,
        winnerRole: 'non-dealer',
        winMethod: 'ron',
        payment: { kind: 'ron', amount: 3900 },
        totalPoints: 3900,
      }),
    },
    {
      id: 'haitei',
      description: 'Haitei is awarded on the last drawable tsumo tile.',
      coverage: ['haitei'],
      ruleProfileId: 'product-default',
      input: input(HAND_PINFU_TANYAO, 5, {
        winMethod: 'tsumo',
        haitei: true,
      }),
      expected: scored({
        yaku: [
          regular('menzen-tsumo', 1),
          regular('tanyao', 1),
          regular('pinfu', 1),
          regular('haitei', 1),
        ],
        dora: NO_DORA,
        han: 4,
        fu: standardFu(20, 0, 0, 0, 0, 0, 20, 20),
        limit: NO_LIMIT,
        winnerRole: 'non-dealer',
        winMethod: 'tsumo',
        payment: {
          kind: 'tsumo-non-dealer',
          dealerPays: 2600,
          nonDealerPays: 1300,
        },
        totalPoints: 5200,
      }),
    },
    {
      id: 'houtei',
      description: 'Houtei is awarded on ron from the final discard.',
      coverage: ['houtei'],
      ruleProfileId: 'product-default',
      input: input(HAND_PINFU_TANYAO, 5, { houtei: true }),
      expected: scored({
        yaku: [regular('tanyao', 1), regular('pinfu', 1), regular('houtei', 1)],
        dora: NO_DORA,
        han: 3,
        fu: standardFu(20, 10, 0, 0, 0, 0, 30, 30),
        limit: NO_LIMIT,
        winnerRole: 'non-dealer',
        winMethod: 'ron',
        payment: { kind: 'ron', amount: 3900 },
        totalPoints: 3900,
      }),
    },
    {
      id: 'rinshan-kaihou',
      description: 'Concealed-kan rinshan tsumo preserves tsumo, kan, pair, and wait fu categories.',
      coverage: ['rinshan-kaihou', 'fu-tsumo', 'fu-melds', 'fu-pair'],
      ruleProfileId: 'product-default',
      input: input(
        HAND_RINSHAN,
        2,
        {
          winMethod: 'tsumo',
          seatWind: 'south',
          rinshan: true,
        },
        { melds: [MELD_CONCEALED_5555M] },
      ),
      expected: scored({
        yaku: [regular('menzen-tsumo', 1), regular('rinshan-kaihou', 1)],
        dora: NO_DORA,
        han: 2,
        fu: standardFu(20, 0, 2, 16, 2, 2, 42, 50),
        limit: NO_LIMIT,
        winnerRole: 'non-dealer',
        winMethod: 'tsumo',
        payment: {
          kind: 'tsumo-non-dealer',
          dealerPays: 1600,
          nonDealerPays: 800,
        },
        totalPoints: 3200,
      }),
    },
    {
      id: 'chiitoitsu-25-fu',
      description: 'Seven pairs remains semantic fixed 25 fu.',
      coverage: ['chiitoitsu', 'fu-chiitoitsu-25'],
      ruleProfileId: 'product-default',
      input: input(HAND_CHIITOITSU, 0, { winMethod: 'tsumo' }),
      expected: scored({
        yaku: [regular('menzen-tsumo', 1), regular('chiitoitsu', 2)],
        dora: NO_DORA,
        han: 3,
        fu: { kind: 'chiitoitsu', fixed: 25 },
        limit: NO_LIMIT,
        winnerRole: 'non-dealer',
        winMethod: 'tsumo',
        payment: {
          kind: 'tsumo-non-dealer',
          dealerPays: 1600,
          nonDealerPays: 800,
        },
        totalPoints: 3200,
      }),
    },
    {
      id: 'open-tanyao-rule-on',
      description: 'Open Tanyao scores and the otherwise-20-fu open hand is raised to 30 fu.',
      coverage: [
        'open-tanyao-on',
        'fu-open-minimum-30',
        'payment-ron-dealer',
      ],
      ruleProfileId: 'product-default',
      input: input(
        HAND_OPEN_TANYAO,
        0,
        { roundWind: 'south', seatWind: 'east' },
        { melds: [MELD_OPEN_234M] },
      ),
      expected: scored({
        yaku: [regular('tanyao', 1)],
        dora: NO_DORA,
        han: 1,
        fu: standardFu(20, 0, 0, 0, 0, 0, 20, 30),
        limit: NO_LIMIT,
        winnerRole: 'dealer',
        winMethod: 'ron',
        payment: { kind: 'ron', amount: 1500 },
        totalPoints: 1500,
      }),
    },
    {
      id: 'open-tanyao-rule-off',
      description: 'The same open all-simples hand becomes no-yaku when open Tanyao is disabled.',
      coverage: ['open-tanyao-off'],
      ruleProfileId: 'open-tanyao-off',
      input: input(
        HAND_OPEN_TANYAO,
        0,
        { roundWind: 'south', seatWind: 'east' },
        { melds: [MELD_OPEN_234M] },
      ),
      expected: { status: 'no-yaku' },
    },
    {
      id: 'dora-only-does-not-create-yaku',
      description: 'Two indicator dora do not turn a rule-disabled open-Tanyao hand into a scoring hand.',
      coverage: ['dora-only-no-yaku'],
      ruleProfileId: 'open-tanyao-off',
      input: input(
        HAND_OPEN_TANYAO,
        0,
        { roundWind: 'south', seatWind: 'east' },
        { melds: [MELD_OPEN_234M], doraIndicators: ['7p'] },
      ),
      expected: { status: 'no-yaku' },
    },
    {
      id: 'closed-ittsu-kuisagari-reference',
      description: 'Closed Ittsu is awarded two han.',
      coverage: ['kuisagari-closed'],
      ruleProfileId: 'product-default',
      input: input(HAND_ITTSU_CLOSED, 11),
      expected: scored({
        yaku: [regular('ittsu', 2)],
        dora: NO_DORA,
        han: 2,
        fu: standardFu(20, 10, 0, 0, 2, 0, 32, 40),
        limit: NO_LIMIT,
        winnerRole: 'non-dealer',
        winMethod: 'ron',
        payment: { kind: 'ron', amount: 2600 },
        totalPoints: 2600,
      }),
    },
    {
      id: 'open-ittsu-kuisagari-reference',
      description: 'Open Ittsu is awarded one han without TypeScript-side kuisagari calculation.',
      coverage: ['kuisagari-open'],
      ruleProfileId: 'product-default',
      input: input(HAND_ITTSU_OPEN, 8, {}, { melds: [MELD_OPEN_123M] }),
      expected: scored({
        yaku: [regular('ittsu', 1)],
        dora: NO_DORA,
        han: 1,
        fu: standardFu(20, 0, 0, 0, 2, 0, 22, 30),
        limit: NO_LIMIT,
        winnerRole: 'non-dealer',
        winMethod: 'ron',
        payment: { kind: 'ron', amount: 1000 },
        totalPoints: 1000,
      }),
    },
    {
      id: 'kiriage-4han-30fu-on',
      description: '4 han 30 fu is kiriage mangan when the rule is enabled.',
      coverage: ['kiriage-4h30-on'],
      ruleProfileId: 'product-default',
      input: input(HAND_KIRIAGE_4H30, 0, {
        riichi: 'riichi',
        ippatsu: true,
      }),
      expected: scored({
        yaku: [
          regular('riichi', 1),
          regular('ippatsu', 1),
          regular('tanyao', 1),
          regular('pinfu', 1),
        ],
        dora: NO_DORA,
        han: 4,
        fu: standardFu(20, 10, 0, 0, 0, 0, 30, 30),
        limit: { kind: 'mangan', kiriage: true },
        winnerRole: 'non-dealer',
        winMethod: 'ron',
        payment: { kind: 'ron', amount: 8000 },
        totalPoints: 8000,
      }),
    },
    {
      id: 'kiriage-4han-30fu-off',
      description: '4 han 30 fu remains 7700 non-dealer ron when kiriage is disabled.',
      coverage: ['kiriage-4h30-off'],
      ruleProfileId: 'kiriage-off',
      input: input(HAND_KIRIAGE_4H30, 0, {
        riichi: 'riichi',
        ippatsu: true,
      }),
      expected: scored({
        yaku: [
          regular('riichi', 1),
          regular('ippatsu', 1),
          regular('tanyao', 1),
          regular('pinfu', 1),
        ],
        dora: NO_DORA,
        han: 4,
        fu: standardFu(20, 10, 0, 0, 0, 0, 30, 30),
        limit: NO_LIMIT,
        winnerRole: 'non-dealer',
        winMethod: 'ron',
        payment: { kind: 'ron', amount: 7700 },
        totalPoints: 7700,
      }),
    },
    {
      id: 'double-wind-pair-2-fu',
      description: 'Double-value South pair contributes 2 pair fu under the 2-fu policy.',
      coverage: ['double-wind-pair-2'],
      ruleProfileId: 'kiriage-off',
      input: input(HAND_KIRIAGE_3H, 7, {
        roundWind: 'south',
        seatWind: 'south',
        riichi: 'riichi',
        ippatsu: true,
      }),
      expected: scored({
        yaku: [regular('riichi', 1), regular('ippatsu', 1), regular('yakuhai-white', 1)],
        dora: NO_DORA,
        han: 3,
        fu: standardFu(20, 10, 0, 16, 2, 2, 50, 50),
        limit: NO_LIMIT,
        winnerRole: 'non-dealer',
        winMethod: 'ron',
        payment: { kind: 'ron', amount: 6400 },
        totalPoints: 6400,
      }),
    },
    {
      id: 'double-wind-pair-4-fu',
      description: 'The same hand contributes 4 pair fu under the 4-fu policy.',
      coverage: ['double-wind-pair-4'],
      ruleProfileId: 'double-wind-4-kiriage-off',
      input: input(HAND_KIRIAGE_3H, 7, {
        roundWind: 'south',
        seatWind: 'south',
        riichi: 'riichi',
        ippatsu: true,
      }),
      expected: scored({
        yaku: [regular('riichi', 1), regular('ippatsu', 1), regular('yakuhai-white', 1)],
        dora: NO_DORA,
        han: 3,
        fu: standardFu(20, 10, 0, 16, 4, 2, 52, 60),
        limit: NO_LIMIT,
        winnerRole: 'non-dealer',
        winMethod: 'ron',
        payment: { kind: 'ron', amount: 7700 },
        totalPoints: 7700,
      }),
    },
    {
      id: 'kiriage-3han-60fu-on',
      description: '3 han 60 fu is kiriage mangan when enabled.',
      coverage: ['kiriage-3h60-on'],
      ruleProfileId: 'double-wind-4',
      input: input(HAND_KIRIAGE_3H, 7, {
        roundWind: 'south',
        seatWind: 'south',
        riichi: 'riichi',
        ippatsu: true,
      }),
      expected: scored({
        yaku: [regular('riichi', 1), regular('ippatsu', 1), regular('yakuhai-white', 1)],
        dora: NO_DORA,
        han: 3,
        fu: standardFu(20, 10, 0, 16, 4, 2, 52, 60),
        limit: { kind: 'mangan', kiriage: true },
        winnerRole: 'non-dealer',
        winMethod: 'ron',
        payment: { kind: 'ron', amount: 8000 },
        totalPoints: 8000,
      }),
    },
    {
      id: 'kiriage-3han-60fu-off',
      description: '3 han 60 fu remains 7700 non-dealer ron when kiriage is disabled.',
      coverage: ['kiriage-3h60-off'],
      ruleProfileId: 'double-wind-4-kiriage-off',
      input: input(HAND_KIRIAGE_3H, 7, {
        roundWind: 'south',
        seatWind: 'south',
        riichi: 'riichi',
        ippatsu: true,
      }),
      expected: scored({
        yaku: [regular('riichi', 1), regular('ippatsu', 1), regular('yakuhai-white', 1)],
        dora: NO_DORA,
        han: 3,
        fu: standardFu(20, 10, 0, 16, 4, 2, 52, 60),
        limit: NO_LIMIT,
        winnerRole: 'non-dealer',
        winMethod: 'ron',
        payment: { kind: 'ron', amount: 7700 },
        totalPoints: 7700,
      }),
    },
    {
      id: 'kazoe-yakuman-on',
      description: '15 total non-yakuman han is one counted yakuman when kazoe is enabled.',
      coverage: ['kazoe-yakuman-on', 'limit-yakuman-counted'],
      ruleProfileId: 'kazoe-on',
      input: input(
        HAND_KAZOE,
        0,
        { winMethod: 'tsumo', riichi: 'riichi', ippatsu: true },
        { doraIndicators: ['1s'] },
      ),
      expected: scored({
        yaku: [
          regular('riichi', 1),
          regular('ippatsu', 1),
          regular('menzen-tsumo', 1),
          regular('pinfu', 1),
          regular('ryanpeikou', 3),
          regular('chinitsu', 6),
        ],
        dora: { dora: 2, akaDora: 0 },
        han: 15,
        fu: null,
        limit: { kind: 'yakuman', units: 1, counted: true },
        winnerRole: 'non-dealer',
        winMethod: 'tsumo',
        payment: {
          kind: 'tsumo-non-dealer',
          dealerPays: 16000,
          nonDealerPays: 8000,
        },
        totalPoints: 32000,
      }),
    },
    {
      id: 'kazoe-yakuman-off',
      description: 'The same 15-han non-yakuman hand is capped at sanbaiman when kazoe is disabled.',
      coverage: ['kazoe-yakuman-off', 'limit-sanbaiman'],
      ruleProfileId: 'product-default',
      input: input(
        HAND_KAZOE,
        0,
        { winMethod: 'tsumo', riichi: 'riichi', ippatsu: true },
        { doraIndicators: ['1s'] },
      ),
      expected: scored({
        yaku: [
          regular('riichi', 1),
          regular('ippatsu', 1),
          regular('menzen-tsumo', 1),
          regular('pinfu', 1),
          regular('ryanpeikou', 3),
          regular('chinitsu', 6),
        ],
        dora: { dora: 2, akaDora: 0 },
        han: 15,
        fu: standardFu(20, 0, 0, 0, 0, 0, 20, 20),
        limit: { kind: 'sanbaiman' },
        winnerRole: 'non-dealer',
        winMethod: 'tsumo',
        payment: {
          kind: 'tsumo-non-dealer',
          dealerPays: 12000,
          nonDealerPays: 6000,
        },
        totalPoints: 24000,
      }),
    },
    {
      id: 'baiman-chinitsu',
      description: 'Nine-han closed Pinfu tsumo exercises Baiman classification.',
      coverage: ['limit-baiman'],
      ruleProfileId: 'product-default',
      input: input(HAND_BAIMAN, 5, { winMethod: 'tsumo', riichi: 'riichi' }),
      expected: scored({
        yaku: [
          regular('riichi', 1),
          regular('menzen-tsumo', 1),
          regular('pinfu', 1),
          regular('chinitsu', 6),
        ],
        dora: NO_DORA,
        han: 9,
        fu: standardFu(20, 0, 0, 0, 0, 0, 20, 20),
        limit: { kind: 'baiman' },
        winnerRole: 'non-dealer',
        winMethod: 'tsumo',
        payment: {
          kind: 'tsumo-non-dealer',
          dealerPays: 8000,
          nonDealerPays: 4000,
        },
        totalPoints: 16000,
      }),
    },
    {
      id: 'kokushi-musou-normal-wait',
      description: 'Kokushi completed from a single missing terminal is ordinary Kokushi Musou.',
      coverage: ['kokushi-musou', 'limit-yakuman-actual'],
      ruleProfileId: 'product-default',
      input: input(HAND_KOKUSHI, 0),
      expected: actualYakumanExpected([yakuman('kokushi-musou')], 1, 32000),
    },
    {
      id: 'kokushi-13-wait-double-variant-off',
      description: 'Kokushi 13-sided wait preserves variant identity but scores one unit when double variants are disabled.',
      coverage: ['kokushi-13-wait', 'double-variant-kokushi-13-off'],
      ruleProfileId: 'product-default',
      input: input(HAND_KOKUSHI, 13),
      expected: actualYakumanExpected([yakuman('kokushi-13-wait')], 1, 32000),
    },
    {
      id: 'kokushi-13-wait-double-variant-on',
      description: 'Kokushi 13-sided wait scores two units when double variants are enabled.',
      coverage: ['double-variant-kokushi-13-on'],
      ruleProfileId: 'double-variants-on',
      input: input(HAND_KOKUSHI, 13),
      expected: actualYakumanExpected([yakuman('kokushi-13-wait')], 2, 64000),
    },
    {
      id: 'suuankou-tanki-double-variant-off',
      description: 'Suuankou tanki scores one unit with double variants disabled.',
      coverage: ['double-variant-suuankou-tanki-off'],
      ruleProfileId: 'product-default',
      input: input(HAND_SUUANKOU_TANKI, 12),
      expected: actualYakumanExpected([yakuman('suuankou-tanki')], 1, 32000),
    },
    {
      id: 'suuankou-tanki-double-variant-on',
      description: 'Suuankou tanki scores two units with double variants enabled.',
      coverage: ['double-variant-suuankou-tanki-on'],
      ruleProfileId: 'double-variants-on',
      input: input(HAND_SUUANKOU_TANKI, 12),
      expected: actualYakumanExpected([yakuman('suuankou-tanki')], 2, 64000),
    },
    {
      id: 'daisuushii-double-variant-off',
      description: 'Daisuushii identity is retained while its configured unit value is one.',
      coverage: ['double-variant-daisuushii-off'],
      ruleProfileId: 'product-default',
      input: input(HAND_DAISUUSHII, 0),
      expected: actualYakumanExpected([yakuman('daisuushii')], 1, 32000),
    },
    {
      id: 'daisuushii-double-variant-on',
      description: 'Daisuushii contributes two units when double variants are enabled.',
      coverage: ['double-variant-daisuushii-on'],
      ruleProfileId: 'double-variants-on',
      input: input(HAND_DAISUUSHII, 0),
      expected: actualYakumanExpected([yakuman('daisuushii')], 2, 64000),
    },
    {
      id: 'junsei-chuuren-double-variant-off',
      description: 'Junsei Chuuren Poutou scores one unit with double variants disabled.',
      coverage: ['double-variant-junsei-chuuren-off'],
      ruleProfileId: 'product-default',
      input: input(HAND_JUNSEI_CHUUREN, 6),
      expected: actualYakumanExpected([yakuman('junsei-chuuren-poutou')], 1, 32000),
    },
    {
      id: 'junsei-chuuren-double-variant-on',
      description: 'Junsei Chuuren Poutou scores two units with double variants enabled.',
      coverage: ['double-variant-junsei-chuuren-on'],
      ruleProfileId: 'double-variants-on',
      input: input(HAND_JUNSEI_CHUUREN, 6),
      expected: actualYakumanExpected([yakuman('junsei-chuuren-poutou')], 2, 64000),
    },
    {
      id: 'multiple-yakuman-on',
      description: 'Daisangen plus Tsuuiisou stacks to two actual-yakuman units.',
      coverage: ['multiple-yakuman-on'],
      ruleProfileId: 'product-default',
      input: input(HAND_MULTI_YAKUMAN, 9),
      expected: actualYakumanExpected(
        [yakuman('daisangen'), yakuman('tsuuiisou')],
        2,
        64000,
      ),
    },
    {
      id: 'multiple-yakuman-off',
      description: 'Both yakuman identities remain visible while multiple-yakuman scoring caps the result at one unit.',
      coverage: ['multiple-yakuman-off'],
      ruleProfileId: 'multiple-off',
      input: input(HAND_MULTI_YAKUMAN, 9),
      expected: actualYakumanExpected(
        [yakuman('daisangen'), yakuman('tsuuiisou')],
        1,
        32000,
      ),
    },
    {
      id: 'yakuman-policy-interaction-on-on',
      description: 'Stacking on + double variants on: Shousuushii + Tsuuiisou + double Suuankou tanki = four units.',
      coverage: ['yakuman-policy-interaction-on-on'],
      ruleProfileId: 'double-variants-on',
      input: input(HAND_POLICY_INTERACTION, 9),
      expected: actualYakumanExpected(
        [yakuman('shousuushii'), yakuman('tsuuiisou'), yakuman('suuankou-tanki')],
        4,
        128000,
      ),
    },
    {
      id: 'yakuman-policy-interaction-on-off',
      description: 'Stacking on + double variants off reduces the same detected set to three units.',
      coverage: ['yakuman-policy-interaction-on-off'],
      ruleProfileId: 'product-default',
      input: input(HAND_POLICY_INTERACTION, 9),
      expected: actualYakumanExpected(
        [yakuman('shousuushii'), yakuman('tsuuiisou'), yakuman('suuankou-tanki')],
        3,
        96000,
      ),
    },
    {
      id: 'yakuman-policy-interaction-off-on',
      description: 'Stacking off dominates double-variant units and caps the same detected set at one unit.',
      coverage: ['yakuman-policy-interaction-off-on'],
      ruleProfileId: 'double-variants-on-multiple-off',
      input: input(HAND_POLICY_INTERACTION, 9),
      expected: actualYakumanExpected(
        [yakuman('shousuushii'), yakuman('tsuuiisou'), yakuman('suuankou-tanki')],
        1,
        32000,
      ),
    },
    {
      id: 'yakuman-policy-interaction-off-off',
      description: 'Both policies off still preserve all detected yakuman identities while scoring one unit.',
      coverage: ['yakuman-policy-interaction-off-off'],
      ruleProfileId: 'multiple-off',
      input: input(HAND_POLICY_INTERACTION, 9),
      expected: actualYakumanExpected(
        [yakuman('shousuushii'), yakuman('tsuuiisou'), yakuman('suuankou-tanki')],
        1,
        32000,
      ),
    },
    {
      id: 'tenhou',
      description: 'Dealer initial-draw tsumo is one actual Tenhou yakuman under the product default variant policy.',
      coverage: ['tenhou'],
      ruleProfileId: 'product-default',
      input: input(HAND_SITUATIONAL_YAKUMAN, 0, {
        winMethod: 'tsumo',
        seatWind: 'east',
        tenhou: true,
      }),
      expected: scored({
        yaku: [yakuman('tenhou')],
        dora: NO_DORA,
        han: null,
        fu: null,
        limit: { kind: 'yakuman', units: 1, counted: false },
        winnerRole: 'dealer',
        winMethod: 'tsumo',
        payment: { kind: 'tsumo-dealer', eachOpponent: 16000 },
        totalPoints: 48000,
      }),
    },
    {
      id: 'chiihou',
      description: 'Non-dealer initial uninterrupted tsumo is one actual Chiihou yakuman.',
      coverage: ['chiihou'],
      ruleProfileId: 'product-default',
      input: input(HAND_SITUATIONAL_YAKUMAN, 0, {
        winMethod: 'tsumo',
        seatWind: 'south',
        chiihou: true,
      }),
      expected: scored({
        yaku: [yakuman('chiihou')],
        dora: NO_DORA,
        han: null,
        fu: null,
        limit: { kind: 'yakuman', units: 1, counted: false },
        winnerRole: 'non-dealer',
        winMethod: 'tsumo',
        payment: {
          kind: 'tsumo-non-dealer',
          dealerPays: 16000,
          nonDealerPays: 8000,
        },
        totalPoints: 32000,
      }),
    },
    {
      id: 'no-yaku-open-shape',
      description: 'A coherent completed shape with no scoring yaku is not a structural failure.',
      coverage: ['no-yaku'],
      ruleProfileId: 'product-default',
      input: input(HAND_NO_YAKU, 3, {}, { melds: [MELD_OPEN_678M] }),
      expected: { status: 'no-yaku' },
    },
    {
      id: 'not-winning-shape',
      description: 'A coherent 14-tile input that is not a completed winning shape remains distinct from no-yaku.',
      coverage: ['not-winning-shape'],
      ruleProfileId: 'product-default',
      input: input(HAND_NOT_WINNING, 8),
      expected: { status: 'not-winning-shape' },
    },
  ],
} as const satisfies ScoringGoldenCorpusV1<ScoringGoldenCoverageV1>;
