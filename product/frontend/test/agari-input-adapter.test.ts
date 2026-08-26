import type { TileKind } from '@/domain';
import { DEFAULT_RULE_PROFILE } from '@/scoring';
import { describe, expect, it } from 'vitest';

import {
  createAgariRuleConfig,
  createAgariScoreRequest,
  serializeScoringHand,
  serializeTileIdentity,
} from '../src/scoring/agari/agari-input-adapter';
import {
  BASE_CONDITIONS,
  closedInput,
  instance,
  tile,
} from './agari-test-fixtures';

const TILE_KINDS = [
  '1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m',
  '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p',
  '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s',
  '1z', '2z', '3z', '4z', '5z', '6z', '7z',
] as const satisfies readonly TileKind[];

describe('Agari input adapter', () => {
  it('serializes every canonical tile kind and all red fives', () => {
    for (const kind of TILE_KINDS) {
      expect(serializeTileIdentity(tile(kind))).toBe(kind);
    }

    expect(serializeTileIdentity(tile('5m', true))).toBe('0m');
    expect(serializeTileIdentity(tile('5p', true))).toBe('0p');
    expect(serializeTileIdentity(tile('5s', true))).toBe('0s');
    expect(() => serializeTileIdentity(tile('4m', true))).toThrow(/suited five/);
    expect(() => serializeTileIdentity(tile('5z', true))).toThrow(/suited five/);
  });

  it('serializes chi, pon, open kan, and concealed kan notation', () => {
    const completedHand = [
      instance('h1', '1m'), instance('h2', '2m'), instance('h3', '3m'),
      instance('h4', '4p'), instance('h5', '5p'), instance('h6', '6p'),
      instance('h7', '7s'), instance('h8', '8s'), instance('h9', '9s'),
      instance('h10', '1z'), instance('h11', '1z'),
    ] as const;

    const base = {
      completedHand,
      doraIndicators: [],
      winningTileId: completedHand[10].id,
      conditions: BASE_CONDITIONS,
    } as const;

    expect(
      serializeScoringHand({
        ...base,
        melds: [{ kind: 'chi', tiles: [tile('3m'), tile('1m'), tile('2m')] }],
      }),
    ).toContain('(123m)');
    expect(
      serializeScoringHand({
        ...base,
        melds: [{ kind: 'pon', tiles: [tile('5p'), tile('5p', true), tile('5p')] }],
      }),
    ).toContain('(505p)');
    expect(
      serializeScoringHand({
        ...base,
        melds: [{
          kind: 'open-kan',
          tiles: [tile('7s'), tile('7s'), tile('7s'), tile('7s')],
        }],
      }),
    ).toContain('(7777s)');
    expect(
      serializeScoringHand({
        ...base,
        melds: [{
          kind: 'concealed-kan',
          tiles: [tile('2z'), tile('2z'), tile('2z'), tile('2z')],
        }],
      }),
    ).toContain('[2222z]');
  });

  it('maps the accepted default rule profile exactly', () => {
    expect(createAgariRuleConfig(DEFAULT_RULE_PROFILE)).toEqual({
      open_tanyao: true,
      aka_dora: true,
      dora: true,
      ippatsu: true,
      kiriage_mangan: true,
      kazoe_yakuman: false,
      multiple_yakuman: true,
      double_yakuman_variants: false,
      double_wind_pair_fu: 2,
    });
  });

  it('maps the full explicit rule profile without relying on Agari defaults', () => {
    const profile = {
      ...DEFAULT_RULE_PROFILE,
      openTanyao: false,
      akaDora: false,
      dora: false,
      ippatsu: false,
      kiriageMangan: false,
      kazoeYakuman: false,
      multipleYakuman: false,
      doubleYakumanVariants: false,
      doubleWindPairFu: 2 as const,
    };

    expect(createAgariRuleConfig(profile)).toEqual({
      open_tanyao: false,
      aka_dora: false,
      dora: false,
      ippatsu: false,
      kiriage_mangan: false,
      kazoe_yakuman: false,
      multiple_yakuman: false,
      double_yakuman_variants: false,
      double_wind_pair_fu: 2,
    });
  });

  it('maps winning tile, winds, riichi state, dora, and ura boundary explicitly', () => {
    const { input, ruleProfile } = closedInput({
      ...BASE_CONDITIONS,
      winMethod: 'tsumo',
      roundWind: 'west',
      seatWind: 'north',
      riichi: 'double-riichi',
      ippatsu: true,
      haitei: true,
    });
    const request = createAgariScoreRequest(input, ruleProfile);

    expect(request.winning_tile).toBe('6p');
    expect(request.is_tsumo).toBe(true);
    expect(request.round_wind).toBe('west');
    expect(request.seat_wind).toBe('north');
    expect(request.is_riichi).toBe(false);
    expect(request.is_double_riichi).toBe(true);
    expect(request.is_ippatsu).toBe(true);
    expect(request.is_last_tile).toBe(true);
    expect(request.dora_indicators).toEqual(['9m']);
    expect(request.ura_dora_indicators).toEqual([]);
  });

  it('maps situational conditions to their stable request fields', () => {
    const chankan = closedInput({ ...BASE_CONDITIONS, chankan: true });
    expect(createAgariScoreRequest(chankan.input, chankan.ruleProfile).is_chankan).toBe(true);

    const rinshanCompletedHand = [
      instance('r1', '1m'), instance('r2', '2m'), instance('r3', '3m'),
      instance('r4', '4p'), instance('r5', '5p'), instance('r6', '6p'),
      instance('r7', '7s'), instance('r8', '8s'), instance('r9', '9s'),
      instance('r10', '1z'), instance('r11', '1z'),
    ] as const;
    const rinshanRequest = createAgariScoreRequest(
      {
        completedHand: rinshanCompletedHand,
        melds: [{
          kind: 'concealed-kan',
          tiles: [tile('2z'), tile('2z'), tile('2z'), tile('2z')],
        }],
        doraIndicators: [tile('3z')],
        winningTileId: rinshanCompletedHand[10].id,
        conditions: { ...BASE_CONDITIONS, winMethod: 'tsumo', rinshan: true },
      },
      DEFAULT_RULE_PROFILE,
    );
    expect(rinshanRequest.is_rinshan).toBe(true);

    const houtei = closedInput({ ...BASE_CONDITIONS, houtei: true });
    expect(createAgariScoreRequest(houtei.input, houtei.ruleProfile).is_last_tile).toBe(true);

    const tenhou = closedInput({
      ...BASE_CONDITIONS,
      winMethod: 'tsumo',
      seatWind: 'east',
      tenhou: true,
    });
    expect(createAgariScoreRequest(tenhou.input, tenhou.ruleProfile).is_tenhou).toBe(true);

    const chiihou = closedInput({
      ...BASE_CONDITIONS,
      winMethod: 'tsumo',
      chiihou: true,
    });
    expect(createAgariScoreRequest(chiihou.input, chiihou.ruleProfile).is_chiihou).toBe(true);
  });

  it('uses ordinary tile kind for a selected red-five winning tile', () => {
    const { input, ruleProfile } = closedInput();
    const redFive = input.completedHand[5];
    const request = createAgariScoreRequest(
      { ...input, winningTileId: redFive.id },
      ruleProfile,
    );

    expect(redFive.tile).toEqual({ kind: '5p', red: true });
    expect(request.winning_tile).toBe('5p');
    expect(request.hand).toContain('0p');
  });

  it('rejects contradictory conditions before calling Agari', () => {
    const { input, ruleProfile } = closedInput({
      ...BASE_CONDITIONS,
      ippatsu: true,
    });

    expect(() => createAgariScoreRequest(input, ruleProfile)).toThrow(/ippatsu requires riichi/);
  });
});
