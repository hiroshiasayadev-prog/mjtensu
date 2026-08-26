/* tslint:disable */
/* eslint-disable */

export type AgariWindV1 = 'east' | 'south' | 'west' | 'north';

export interface AgariRuleConfigV1 {
    readonly open_tanyao: boolean;
    readonly aka_dora: boolean;
    readonly dora: boolean;
    readonly ippatsu: boolean;
    readonly kiriage_mangan: boolean;
    readonly kazoe_yakuman: boolean;
    readonly multiple_yakuman: boolean;
    readonly double_yakuman_variants: boolean;
    readonly double_wind_pair_fu: 2 | 4;
}

export interface AgariScoreRequestV1 {
    readonly hand: string;
    readonly winning_tile: string;
    readonly is_tsumo: boolean;
    readonly is_riichi: boolean;
    readonly is_double_riichi: boolean;
    readonly is_ippatsu: boolean;
    readonly round_wind: AgariWindV1;
    readonly seat_wind: AgariWindV1;
    readonly dora_indicators: readonly string[];
    readonly ura_dora_indicators: readonly string[];
    readonly is_last_tile: boolean;
    readonly is_rinshan: boolean;
    readonly is_chankan: boolean;
    readonly is_tenhou: boolean;
    readonly is_chiihou: boolean;
    readonly rules: AgariRuleConfigV1;
}

export type AgariYakuCodeV1 =
| 'riichi'
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
| 'haitei-raoyue'
| 'houtei-raoyui'
| 'double-riichi'
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
| 'chinitsu'
| 'tenhou'
| 'chiihou'
| 'kokushi-musou'
| 'suuankou'
| 'daisangen'
| 'shousuushii'
| 'daisuushii'
| 'tsuuiisou'
| 'chinroutou'
| 'ryuuiisou'
| 'chuuren-poutou'
| 'suukantsu'
| 'kokushi-13-wait'
| 'suuankou-tanki'
| 'junsei-chuuren-poutou';

export type AgariYakuInfoV1 =
| { readonly kind: 'regular'; readonly code: AgariYakuCodeV1; readonly han: number }
| { readonly kind: 'yakuman'; readonly code: AgariYakuCodeV1 };

export type AgariScoreLevelV1 =
| { readonly kind: 'normal' }
| { readonly kind: 'mangan'; readonly kiriage: boolean }
| { readonly kind: 'haneman' }
| { readonly kind: 'baiman' }
| { readonly kind: 'sanbaiman' }
| { readonly kind: 'yakuman'; readonly units: number; readonly counted: boolean };

export interface AgariFuBreakdownV1 {
    readonly base: number;
    readonly menzen_ron: number;
    readonly tsumo: number;
    readonly melds: number;
    readonly pair: number;
    readonly wait: number;
    readonly raw_total: number;
    readonly rounded: number;
}

export interface AgariDoraInfoV1 {
    readonly regular: number;
    readonly ura: number;
    readonly aka: number;
    readonly total: number;
}

export interface AgariPaymentInfoV1 {
    readonly total: number;
    readonly from_discarder: number | null;
    readonly from_dealer: number | null;
    readonly from_non_dealer: number | null;
}

export interface AgariScoredResultV1 {
    readonly han: number;
    readonly total_han: number;
    readonly yaku: readonly AgariYakuInfoV1[];
    readonly score_level: AgariScoreLevelV1;
    readonly fu: AgariFuBreakdownV1;
    readonly dora: AgariDoraInfoV1;
    readonly payment: AgariPaymentInfoV1;
    readonly is_dealer: boolean;
}

export interface AgariDiagnosticV1 {
    readonly code: string;
    readonly message: string;
}

export type AgariScoreOutcomeV1 =
| { readonly status: 'scored'; readonly result: AgariScoredResultV1 }
| { readonly status: 'not-winning-shape' }
| { readonly status: 'no-yaku' }
| { readonly status: 'invalid-request'; readonly error: AgariDiagnosticV1 }
| { readonly status: 'internal-error'; readonly error: AgariDiagnosticV1 };

export type AgariWinningShapeOutcomeV1 =
| { readonly status: 'winning' }
| { readonly status: 'not-winning-shape' }
| { readonly status: 'invalid-request'; readonly error: AgariDiagnosticV1 }
| { readonly status: 'internal-error'; readonly error: AgariDiagnosticV1 };

export function score_hand_v1(request: AgariScoreRequestV1): AgariScoreOutcomeV1;
export function validate_winning_shape_v1(hand: string): AgariWinningShapeOutcomeV1;



/**
 * Calculate shanten for a hand
 */
export function calculate_shanten_js(hand: string): any;

/**
 * Calculate ukeire (tile acceptance) for a hand
 */
export function calculate_ukeire_js(hand: string): any;

/**
 * Initialize panic hook for better error messages in the browser console
 */
export function init(): void;

/**
 * Score a mahjong hand
 *
 * Takes a JSON-serialized ScoreRequest and returns a JSON-serialized ScoreResponse
 */
export function score_hand(request_js: any): any;

/**
 * Validate a hand string without scoring
 */
export function validate_hand(hand: string): any;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly calculate_shanten_js: (a: number, b: number) => any;
    readonly calculate_ukeire_js: (a: number, b: number) => any;
    readonly init: () => void;
    readonly score_hand: (a: any) => any;
    readonly validate_hand: (a: number, b: number) => any;
    readonly score_hand_v1: (a: any) => any;
    readonly validate_winning_shape_v1: (a: number, b: number) => any;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
