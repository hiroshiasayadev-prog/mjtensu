import type {
  RecognizedStructure,
  TileIdentity,
  TileInstance,
  TileInstanceId,
} from '@/domain';

export { loadProductionScoringService } from './agari/agari-wasm-loader';

export type Wind = 'east' | 'south' | 'west' | 'north';
export type WinMethod = 'ron' | 'tsumo';
export type RiichiState = 'none' | 'riichi' | 'double-riichi';

export interface ScoringConditionsDraft {
  readonly winMethod: WinMethod | null;
  readonly roundWind: Wind | null;
  readonly seatWind: Wind | null;

  readonly riichi: RiichiState;
  readonly ippatsu: boolean;

  readonly rinshan: boolean;
  readonly chankan: boolean;
  readonly haitei: boolean;
  readonly houtei: boolean;
  readonly tenhou: boolean;
  readonly chiihou: boolean;
}

export interface ScoringConditions {
  readonly winMethod: WinMethod;
  readonly roundWind: Wind;
  readonly seatWind: Wind;

  readonly riichi: RiichiState;
  readonly ippatsu: boolean;

  readonly rinshan: boolean;
  readonly chankan: boolean;
  readonly haitei: boolean;
  readonly houtei: boolean;
  readonly tenhou: boolean;
  readonly chiihou: boolean;
}

export interface ScoringDraft {
  readonly structure: RecognizedStructure;
  readonly winningTileId: TileInstanceId;
  readonly conditions: ScoringConditionsDraft;
}

export type ScoringMeld =
  | {
      readonly kind: 'chi';
      readonly tiles: readonly [TileIdentity, TileIdentity, TileIdentity];
    }
  | {
      readonly kind: 'pon';
      readonly tiles: readonly [TileIdentity, TileIdentity, TileIdentity];
    }
  | {
      readonly kind: 'open-kan';
      readonly tiles: readonly [
        TileIdentity,
        TileIdentity,
        TileIdentity,
        TileIdentity,
      ];
    }
  | {
      readonly kind: 'concealed-kan';
      readonly tiles: readonly [
        TileIdentity,
        TileIdentity,
        TileIdentity,
        TileIdentity,
      ];
    };

export interface ScoringInput {
  readonly completedHand: readonly TileInstance[];
  readonly melds: readonly ScoringMeld[];
  readonly doraIndicators: readonly TileIdentity[];
  readonly winningTileId: TileInstanceId;
  readonly conditions: ScoringConditions;
}

export interface ScoringRuleProfile {
  readonly openTanyao: boolean;
  readonly akaDora: boolean;
  readonly dora: boolean;
  readonly ippatsu: boolean;
  readonly kiriageMangan: boolean;
  readonly kazoeYakuman: boolean;
  readonly multipleYakuman: boolean;
  readonly doubleYakumanVariants: boolean;
  readonly doubleWindPairFu: 2 | 4;
  readonly chiitoitsuFu: 25;
  readonly pinfuTsumoFu: 20;
  readonly openPinfuRonMinimumFu: 30;
}

export const DEFAULT_RULE_PROFILE: ScoringRuleProfile = {
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

export type WinningStructureIssue =
  | {
      readonly kind: 'completed-hand-count';
    }
  | {
      readonly kind: 'completed-hand-tile';
      readonly tileIndex: number;
    }
  | {
      readonly kind: 'meld-group';
      readonly meldIndex: number;
    };

export type WinningStructureValidation =
  | {
      readonly kind: 'invalid-structure';
      readonly issues: readonly WinningStructureIssue[];
    }
  | {
      readonly kind: 'not-winning-shape';
    }
  | {
      readonly kind: 'valid';
    };

export type ScoringRequiredField = 'win-method' | 'round-wind' | 'seat-wind';

export type ScoringInputIssue =
  | {
      readonly kind: 'winning-tile-not-in-completed-hand';
    }
  | {
      readonly kind: 'unresolved-meld';
      readonly meldIndex: number;
    }
  | {
      readonly kind: 'invalid-meld';
      readonly meldIndex: number;
    }
  | {
      readonly kind: 'invalid-structure';
    }
  | {
      readonly kind: 'contradictory-conditions';
    };

export type ScoringPreview =
  | {
      readonly kind: 'incomplete';
      readonly missing: readonly ScoringRequiredField[];
    }
  | {
      readonly kind: 'invalid-input';
      readonly issues: readonly ScoringInputIssue[];
    }
  | {
      readonly kind: 'invalid-winning-shape';
    }
  | {
      readonly kind: 'no-yaku';
    }
  | {
      readonly kind: 'ready';
      readonly yaku: readonly YakuEntry[];
    };

export type RegularYakuId =
  | 'riichi'
  | 'double-riichi'
  | 'ippatsu'
  | 'menzen-tsumo'
  | 'tanyao'
  | 'pinfu'
  | 'iipeikou'
  | 'yakuhai-east'
  | 'yakuhai-south'
  | 'yakuhai-west'
  | 'yakuhai-north'
  | 'yakuhai-white'
  | 'yakuhai-green'
  | 'yakuhai-red'
  | 'rinshan-kaihou'
  | 'chankan'
  | 'haitei'
  | 'houtei'
  | 'toitoi'
  | 'sanshoku-doujun'
  | 'sanshoku-doukou'
  | 'ittsu'
  | 'chiitoitsu'
  | 'chanta'
  | 'sanankou'
  | 'sankantsu'
  | 'honroutou'
  | 'shousangen'
  | 'honitsu'
  | 'junchan'
  | 'ryanpeikou'
  | 'chinitsu';

export type YakumanYakuId =
  | 'tenhou'
  | 'chiihou'
  | 'kokushi-musou'
  | 'kokushi-13-wait'
  | 'suuankou'
  | 'suuankou-tanki'
  | 'daisangen'
  | 'shousuushii'
  | 'daisuushii'
  | 'tsuuiisou'
  | 'chinroutou'
  | 'ryuuiisou'
  | 'chuuren-poutou'
  | 'junsei-chuuren-poutou'
  | 'suukantsu';

export type YakuId = RegularYakuId | YakumanYakuId;

export type YakuEntry =
  | {
      readonly kind: 'regular';
      readonly id: RegularYakuId;
      readonly han: number;
    }
  | {
      readonly kind: 'yakuman';
      readonly id: YakumanYakuId;
    };

export interface DoraContribution {
  readonly dora: number;
  readonly akaDora: number;
}

export type FuCalculation =
  | {
      readonly kind: 'standard';
      readonly base: number;
      readonly menzenRon: number;
      readonly tsumo: number;
      readonly melds: number;
      readonly pair: number;
      readonly wait: number;
      readonly rawTotal: number;
      readonly rounded: number;
    }
  | {
      readonly kind: 'chiitoitsu';
      readonly fixed: 25;
    };

export type LimitClassification =
  | {
      readonly kind: 'mangan';
      readonly kiriage: boolean;
    }
  | {
      readonly kind: 'haneman';
    }
  | {
      readonly kind: 'baiman';
    }
  | {
      readonly kind: 'sanbaiman';
    }
  | {
      readonly kind: 'yakuman';
      readonly units: number;
      readonly counted: boolean;
    };

export type ScoringPayment =
  | {
      readonly kind: 'ron';
      readonly amount: number;
    }
  | {
      readonly kind: 'tsumo-dealer';
      readonly eachOpponent: number;
    }
  | {
      readonly kind: 'tsumo-non-dealer';
      readonly dealerPays: number;
      readonly nonDealerPays: number;
    };

export interface ScoringCalculation {
  readonly yaku: readonly YakuEntry[];
  readonly dora: DoraContribution;
  readonly han: number | null;
  readonly fu: FuCalculation | null;
  readonly limit: LimitClassification | null;
  readonly winnerRole: 'dealer' | 'non-dealer';
  readonly winMethod: WinMethod;
  readonly payment: ScoringPayment;
  readonly totalPoints: number;
}

export type ScoringError =
  | {
      readonly kind: 'input-contract-violation';
      readonly cause?: unknown;
    }
  | {
      readonly kind: 'adapter-failure';
      readonly cause: unknown;
    };

export interface ScoringService {
  validateWinningStructure(
    structure: RecognizedStructure,
  ): WinningStructureValidation;
  preview(
    draft: ScoringDraft,
    ruleProfile: ScoringRuleProfile,
  ): ScoringPreview;
  calculate(
    input: ScoringInput,
    ruleProfile: ScoringRuleProfile,
  ): ScoringCalculation;
}

export const REGULAR_YAKU_DISPLAY_NAMES = {
  riichi: '立直',
  'double-riichi': 'ダブル立直',
  ippatsu: '一発',
  'menzen-tsumo': '門前清自摸和',
  tanyao: '断么九',
  pinfu: '平和',
  iipeikou: '一盃口',
  'yakuhai-east': '役牌 東',
  'yakuhai-south': '役牌 南',
  'yakuhai-west': '役牌 西',
  'yakuhai-north': '役牌 北',
  'yakuhai-white': '役牌 白',
  'yakuhai-green': '役牌 發',
  'yakuhai-red': '役牌 中',
  'rinshan-kaihou': '嶺上開花',
  chankan: '槍槓',
  haitei: '海底摸月',
  houtei: '河底撈魚',
  toitoi: '対々和',
  'sanshoku-doujun': '三色同順',
  'sanshoku-doukou': '三色同刻',
  ittsu: '一気通貫',
  chiitoitsu: '七対子',
  chanta: '混全帯么九',
  sanankou: '三暗刻',
  sankantsu: '三槓子',
  honroutou: '混老頭',
  shousangen: '小三元',
  honitsu: '混一色',
  junchan: '純全帯么九',
  ryanpeikou: '二盃口',
  chinitsu: '清一色',
} as const satisfies Record<RegularYakuId, string>;

export const YAKUMAN_YAKU_DISPLAY_NAMES = {
  tenhou: '天和',
  chiihou: '地和',
  'kokushi-musou': '国士無双',
  'kokushi-13-wait': '国士無双十三面待ち',
  suuankou: '四暗刻',
  'suuankou-tanki': '四暗刻単騎',
  daisangen: '大三元',
  shousuushii: '小四喜',
  daisuushii: '大四喜',
  tsuuiisou: '字一色',
  chinroutou: '清老頭',
  ryuuiisou: '緑一色',
  'chuuren-poutou': '九蓮宝燈',
  'junsei-chuuren-poutou': '純正九蓮宝燈',
  suukantsu: '四槓子',
} as const satisfies Record<YakumanYakuId, string>;

export const YAKU_DISPLAY_NAMES = {
  ...REGULAR_YAKU_DISPLAY_NAMES,
  ...YAKUMAN_YAKU_DISPLAY_NAMES,
} as const satisfies Record<YakuId, string>;

export function getYakuDisplayName(id: YakuId): string {
  return YAKU_DISPLAY_NAMES[id];
}
