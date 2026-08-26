import type { ScoringInput } from '@/scoring';
import { describe, expect, it } from 'vitest';

import type {
  AgariScoredResultV1,
  AgariYakuCodeV1,
} from '../src/scoring/agari/agari-abi';
import {
  normalizeAgariCalculation,
  normalizeAgariYaku,
  normalizeLimit,
} from '../src/scoring/agari/agari-result-adapter';
import {
  BASE_CONDITIONS,
  closedInput,
  standardScoredResult,
} from './agari-test-fixtures';

const REGULAR_YAKU_CASES = [
  ['riichi', 'riichi'],
  ['double-riichi', 'double-riichi'],
  ['ippatsu', 'ippatsu'],
  ['menzen-tsumo', 'menzen-tsumo'],
  ['tanyao', 'tanyao'],
  ['pinfu', 'pinfu'],
  ['iipeikou', 'iipeikou'],
  ['yakuhai-east', 'yakuhai-east'],
  ['yakuhai-south', 'yakuhai-south'],
  ['yakuhai-west', 'yakuhai-west'],
  ['yakuhai-north', 'yakuhai-north'],
  ['yakuhai-white', 'yakuhai-white'],
  ['yakuhai-green', 'yakuhai-green'],
  ['yakuhai-red', 'yakuhai-red'],
  ['rinshan-kaihou', 'rinshan-kaihou'],
  ['chankan', 'chankan'],
  ['haitei-raoyue', 'haitei'],
  ['houtei-raoyui', 'houtei'],
  ['toitoi', 'toitoi'],
  ['sanshoku-doujun', 'sanshoku-doujun'],
  ['sanshoku-doukou', 'sanshoku-doukou'],
  ['ittsu', 'ittsu'],
  ['chiitoitsu', 'chiitoitsu'],
  ['chanta', 'chanta'],
  ['sanankou', 'sanankou'],
  ['sankantsu', 'sankantsu'],
  ['honroutou', 'honroutou'],
  ['shousangen', 'shousangen'],
  ['honitsu', 'honitsu'],
  ['junchan', 'junchan'],
  ['ryanpeikou', 'ryanpeikou'],
  ['chinitsu', 'chinitsu'],
] as const satisfies readonly (readonly [AgariYakuCodeV1, string])[];

const YAKUMAN_YAKU_CASES = [
  'tenhou',
  'chiihou',
  'kokushi-musou',
  'kokushi-13-wait',
  'suuankou',
  'suuankou-tanki',
  'daisangen',
  'shousuushii',
  'daisuushii',
  'tsuuiisou',
  'chinroutou',
  'ryuuiisou',
  'chuuren-poutou',
  'junsei-chuuren-poutou',
  'suukantsu',
] as const satisfies readonly AgariYakuCodeV1[];

describe('Agari result adapter', () => {
  it('maps every supported regular and yakuman yaku code', () => {
    for (const [code, id] of REGULAR_YAKU_CASES) {
      expect(normalizeAgariYaku([{ kind: 'regular', code, han: 1 }])).toEqual([
        { kind: 'regular', id, han: 1 },
      ]);
    }

    for (const code of YAKUMAN_YAKU_CASES) {
      expect(normalizeAgariYaku([{ kind: 'yakuman', code }])).toEqual([
        { kind: 'yakuman', id: code },
      ]);
    }
  });

  it('preserves awarded han and combines duplicate same-ID yakuhai entries', () => {
    expect(
      normalizeAgariYaku([
        { kind: 'regular', code: 'yakuhai-east', han: 1 },
        { kind: 'regular', code: 'yakuhai-east', han: 1 },
        { kind: 'regular', code: 'honitsu', han: 2 },
      ]),
    ).toEqual([
      { kind: 'regular', id: 'yakuhai-east', han: 2 },
      { kind: 'regular', id: 'honitsu', han: 2 },
    ]);
  });

  it('normalizes every structured score level without han-derived inference', () => {
    expect(normalizeLimit({ kind: 'normal' })).toBeNull();
    expect(normalizeLimit({ kind: 'mangan', kiriage: false })).toEqual({
      kind: 'mangan',
      kiriage: false,
    });
    expect(normalizeLimit({ kind: 'mangan', kiriage: true })).toEqual({
      kind: 'mangan',
      kiriage: true,
    });
    expect(normalizeLimit({ kind: 'haneman' })).toEqual({ kind: 'haneman' });
    expect(normalizeLimit({ kind: 'baiman' })).toEqual({ kind: 'baiman' });
    expect(normalizeLimit({ kind: 'sanbaiman' })).toEqual({ kind: 'sanbaiman' });
    expect(normalizeLimit({ kind: 'yakuman', units: 3, counted: false })).toEqual({
      kind: 'yakuman',
      units: 3,
      counted: false,
    });
    expect(normalizeLimit({ kind: 'yakuman', units: 1, counted: true })).toEqual({
      kind: 'yakuman',
      units: 1,
      counted: true,
    });
  });

  it('normalizes standard fu, dora, and ron payment directly from the ABI', () => {
    const { input } = closedInput();
    const result = normalizeAgariCalculation(
      standardScoredResult({
        han: 2,
        total_han: 5,
        yaku: [{ kind: 'regular', code: 'honitsu', han: 2 }],
        dora: { regular: 2, ura: 0, aka: 1, total: 3 },
        payment: {
          total: 8000,
          from_discarder: 8000,
          from_dealer: null,
          from_non_dealer: null,
        },
      }),
      input,
    );

    expect(result.yaku).toEqual([{ kind: 'regular', id: 'honitsu', han: 2 }]);
    expect(result.dora).toEqual({ dora: 2, akaDora: 1 });
    expect(result.han).toBe(5);
    expect(result.fu).toEqual({
      kind: 'standard',
      base: 20,
      menzenRon: 10,
      tsumo: 0,
      melds: 0,
      pair: 0,
      wait: 0,
      rawTotal: 30,
      rounded: 30,
    });
    expect(result.payment).toEqual({ kind: 'ron', amount: 8000 });
    expect(result.totalPoints).toBe(8000);
  });

  it('normalizes chiitoitsu to fixed 25 fu', () => {
    const { input } = closedInput();
    const result = normalizeAgariCalculation(
      standardScoredResult({
        han: 2,
        total_han: 2,
        yaku: [{ kind: 'regular', code: 'chiitoitsu', han: 2 }],
        fu: {
          base: 25,
          menzen_ron: 0,
          tsumo: 0,
          melds: 0,
          pair: 0,
          wait: 0,
          raw_total: 25,
          rounded: 25,
        },
        payment: {
          total: 1600,
          from_discarder: 1600,
          from_dealer: null,
          from_non_dealer: null,
        },
      }),
      input,
    );

    expect(result.fu).toEqual({ kind: 'chiitoitsu', fixed: 25 });
  });

  it('uses score-level units as the authority for actual yakuman', () => {
    const { input } = closedInput();
    const result = normalizeAgariCalculation(
      standardScoredResult({
        han: 26,
        total_han: 26,
        yaku: [{ kind: 'yakuman', code: 'kokushi-13-wait' }],
        score_level: { kind: 'yakuman', units: 1, counted: false },
        payment: {
          total: 32000,
          from_discarder: 32000,
          from_dealer: null,
          from_non_dealer: null,
        },
      }),
      input,
    );

    expect(result.yaku).toEqual([{ kind: 'yakuman', id: 'kokushi-13-wait' }]);
    expect(result.limit).toEqual({ kind: 'yakuman', units: 1, counted: false });
    expect(result.han).toBeNull();
    expect(result.fu).toBeNull();
  });

  it('keeps counted yakuman han while suppressing irrelevant fu', () => {
    const { input } = closedInput();
    const result = normalizeAgariCalculation(
      standardScoredResult({
        han: 13,
        total_han: 13,
        yaku: [{ kind: 'regular', code: 'chinitsu', han: 6 }, { kind: 'regular', code: 'ryanpeikou', han: 3 }, { kind: 'regular', code: 'chiitoitsu', han: 2 }, { kind: 'regular', code: 'riichi', han: 1 }, { kind: 'regular', code: 'ippatsu', han: 1 }],
        score_level: { kind: 'yakuman', units: 1, counted: true },
        payment: {
          total: 32000,
          from_discarder: 32000,
          from_dealer: null,
          from_non_dealer: null,
        },
      }),
      input,
    );

    expect(result.han).toBe(13);
    expect(result.fu).toBeNull();
    expect(result.limit).toEqual({ kind: 'yakuman', units: 1, counted: true });
  });

  it('maps dealer and non-dealer tsumo payments without recalculation', () => {
    const dealerInput: ScoringInput = {
      ...closedInput({ ...BASE_CONDITIONS, winMethod: 'tsumo', seatWind: 'east' }).input,
    };
    const dealer = normalizeAgariCalculation(
      standardScoredResult({
        is_dealer: true,
        payment: {
          total: 6000,
          from_discarder: null,
          from_dealer: null,
          from_non_dealer: 2000,
        },
      }),
      dealerInput,
    );
    expect(dealer.payment).toEqual({ kind: 'tsumo-dealer', eachOpponent: 2000 });

    const nonDealerInput = closedInput({ ...BASE_CONDITIONS, winMethod: 'tsumo' }).input;
    const nonDealer = normalizeAgariCalculation(
      standardScoredResult({
        payment: {
          total: 4000,
          from_discarder: null,
          from_dealer: 2000,
          from_non_dealer: 1000,
        },
      }),
      nonDealerInput,
    );
    expect(nonDealer.payment).toEqual({
      kind: 'tsumo-non-dealer',
      dealerPays: 2000,
      nonDealerPays: 1000,
    });
  });

  it('fails explicitly on unknown yaku and impossible ABI combinations', () => {
    expect(() =>
      normalizeAgariYaku([
        { kind: 'regular', code: 'future-yaku' as AgariYakuCodeV1, han: 1 },
      ]),
    ).toThrow(/unknown regular/);

    const { input } = closedInput();
    const impossible = standardScoredResult({
      yaku: [{ kind: 'yakuman', code: 'daisangen' }],
      score_level: { kind: 'normal' },
    }) as AgariScoredResultV1;
    expect(() => normalizeAgariCalculation(impossible, input)).toThrow(/yakuman/);
  });
});
