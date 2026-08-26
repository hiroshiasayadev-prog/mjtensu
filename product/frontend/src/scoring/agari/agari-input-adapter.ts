import type {
  RecognizedMeldGroup,
  RecognizedStructure,
  TileIdentity,
  TileInstance,
  TileKind,
} from '@/domain';
import type {
  ScoringConditions,
  ScoringInput,
  ScoringMeld,
  ScoringRuleProfile,
  WinningStructureIssue,
} from '@/scoring';

import type { AgariRuleConfigV1, AgariScoreRequestV1 } from './agari-abi';

const COMPLETE_HAND_LOGICAL_TILE_COUNT = 14;
const MELD_LOGICAL_TILE_COUNT = 3;
const MAX_TILE_COPIES = 4;

interface SuitedTileKind {
  readonly number: number;
  readonly suit: 'm' | 'p' | 's';
}

export function serializeTileIdentity(tile: TileIdentity): string {
  assertValidTileIdentity(tile);

  const suited = parseSuitedTileKind(tile.kind);
  if (tile.red) {
    if (suited === null || suited.number !== 5) {
      throw new RangeError('red tile identity must be a suited five');
    }
    return `0${suited.suit}`;
  }

  return tile.kind;
}

export function serializeScoringHand(input: ScoringInput): string {
  assertStrictInputStructure(input);
  return [
    ...input.completedHand.map(({ tile }) => serializeTileIdentity(tile)),
    ...input.melds.map(serializeScoringMeld),
  ].join('');
}

export function serializeWinningStructure(structure: RecognizedStructure): string {
  const issues = validateWinningStructureInput(structure);
  if (issues.length > 0) {
    throw new RangeError('winning structure is not serializable');
  }

  return [
    ...structure.completedHand.map(({ tile }) => serializeTileIdentity(tile)),
    ...structure.meldGroups.map((group) => serializeRecognizedMeld(group)),
  ].join('');
}

export function createAgariScoreRequest(
  input: ScoringInput,
  ruleProfile: ScoringRuleProfile,
): AgariScoreRequestV1 {
  assertStrictScoringContract(input, ruleProfile);

  const winningTile = input.completedHand.find(
    (tile) => tile.id === input.winningTileId,
  );
  if (winningTile === undefined) {
    throw new RangeError('winningTileId must identify a completed-hand tile');
  }

  return {
    hand: serializeScoringHand(input),
    winning_tile: winningTile.tile.kind,
    is_tsumo: input.conditions.winMethod === 'tsumo',
    is_riichi: input.conditions.riichi === 'riichi',
    is_double_riichi: input.conditions.riichi === 'double-riichi',
    is_ippatsu: input.conditions.ippatsu,
    round_wind: input.conditions.roundWind,
    seat_wind: input.conditions.seatWind,
    dora_indicators: input.doraIndicators.map(serializeTileIdentity),
    ura_dora_indicators: [],
    is_last_tile: input.conditions.haitei || input.conditions.houtei,
    is_rinshan: input.conditions.rinshan,
    is_chankan: input.conditions.chankan,
    is_tenhou: input.conditions.tenhou,
    is_chiihou: input.conditions.chiihou,
    rules: createAgariRuleConfig(ruleProfile),
  };
}

export function createAgariRuleConfig(
  ruleProfile: ScoringRuleProfile,
): AgariRuleConfigV1 {
  assertScoringRuleProfileContract(ruleProfile);

  return {
    open_tanyao: ruleProfile.openTanyao,
    aka_dora: ruleProfile.akaDora,
    dora: ruleProfile.dora,
    ippatsu: ruleProfile.ippatsu,
    kiriage_mangan: ruleProfile.kiriageMangan,
    kazoe_yakuman: ruleProfile.kazoeYakuman,
    multiple_yakuman: ruleProfile.multipleYakuman,
    double_yakuman_variants: ruleProfile.doubleYakumanVariants,
    double_wind_pair_fu: ruleProfile.doubleWindPairFu,
  };
}

export function validateWinningStructureInput(
  structure: RecognizedStructure,
): readonly WinningStructureIssue[] {
  const issues: WinningStructureIssue[] = [];
  const completedTileIssueIndexes = new Set<number>();
  const meldIssueIndexes = new Set<number>();
  const expectedCompletedHandCount =
    COMPLETE_HAND_LOGICAL_TILE_COUNT -
    structure.meldGroups.length * MELD_LOGICAL_TILE_COUNT;

  if (
    structure.meldGroups.length > 4 ||
    structure.completedHand.length !== expectedCompletedHandCount
  ) {
    issues.push({ kind: 'completed-hand-count' });
  }

  for (const [tileIndex, { tile }] of structure.completedHand.entries()) {
    if (!isValidTileIdentity(tile)) {
      completedTileIssueIndexes.add(tileIndex);
    }
  }

  for (const [meldIndex, group] of structure.meldGroups.entries()) {
    if (!isValidRecognizedMeld(group)) {
      meldIssueIndexes.add(meldIndex);
    }
  }

  const counts = new Map<TileKind, number>();
  for (const [tileIndex, { tile }] of structure.completedHand.entries()) {
    const count = (counts.get(tile.kind) ?? 0) + 1;
    counts.set(tile.kind, count);
    if (count > MAX_TILE_COPIES) {
      completedTileIssueIndexes.add(tileIndex);
    }
  }
  for (const [meldIndex, group] of structure.meldGroups.entries()) {
    for (const { tile } of group.tiles) {
      const count = (counts.get(tile.kind) ?? 0) + 1;
      counts.set(tile.kind, count);
      if (count > MAX_TILE_COPIES) {
        meldIssueIndexes.add(meldIndex);
      }
    }
  }

  for (const tileIndex of completedTileIssueIndexes) {
    issues.push({ kind: 'completed-hand-tile', tileIndex });
  }
  for (const meldIndex of meldIssueIndexes) {
    issues.push({ kind: 'meld-group', meldIndex });
  }

  return issues;
}

export function assertStrictScoringContract(
  input: ScoringInput,
  ruleProfile: ScoringRuleProfile,
): void {
  assertScoringRuleProfileContract(ruleProfile);
  assertStrictInputStructure(input);

  if (!input.completedHand.some((tile) => tile.id === input.winningTileId)) {
    throw new RangeError('winningTileId must identify a completed-hand tile');
  }

  assertUniqueCompletedHandIds(input.completedHand);
  assertScoringConditionsContract(input.conditions, input.melds);

  const allScoringTiles = [
    ...input.completedHand.map(({ tile }) => tile),
    ...input.melds.flatMap((meld) => [...meld.tiles]),
    ...input.doraIndicators,
  ];

  for (const tile of allScoringTiles) {
    assertValidTileIdentity(tile);
  }
  assertPhysicalMultiplicity(allScoringTiles);
}

function assertStrictInputStructure(input: ScoringInput): void {
  if (input.melds.length > 4) {
    throw new RangeError('a scoring hand cannot contain more than four melds');
  }

  const expectedCompletedHandCount =
    COMPLETE_HAND_LOGICAL_TILE_COUNT - input.melds.length * MELD_LOGICAL_TILE_COUNT;
  if (input.completedHand.length !== expectedCompletedHandCount) {
    throw new RangeError(
      `completed hand must contain ${expectedCompletedHandCount} tiles for ${input.melds.length} melds`,
    );
  }

  for (const meld of input.melds) {
    assertValidScoringMeld(meld);
  }
}

export function assertScoringRuleProfileContract(
  ruleProfile: ScoringRuleProfile,
): void {
  if (ruleProfile.doubleWindPairFu !== 2 && ruleProfile.doubleWindPairFu !== 4) {
    throw new RangeError('doubleWindPairFu must be 2 or 4');
  }
  if (ruleProfile.chiitoitsuFu !== 25) {
    throw new RangeError('chiitoitsuFu must remain fixed at 25');
  }
  if (ruleProfile.pinfuTsumoFu !== 20) {
    throw new RangeError('pinfuTsumoFu must remain fixed at 20');
  }
  if (ruleProfile.openPinfuRonMinimumFu !== 30) {
    throw new RangeError('openPinfuRonMinimumFu must remain fixed at 30');
  }
}

export function assertScoringConditionsContract(
  conditions: ScoringConditions,
  melds: readonly ScoringMeld[],
): void {
  const hasOpenMeld = melds.some((meld) => meld.kind !== 'concealed-kan');
  const hasKan = melds.some(
    (meld) => meld.kind === 'open-kan' || meld.kind === 'concealed-kan',
  );

  if (conditions.riichi !== 'none' && hasOpenMeld) {
    throw new RangeError('riichi requires a closed hand');
  }
  if (conditions.ippatsu && conditions.riichi === 'none') {
    throw new RangeError('ippatsu requires riichi');
  }
  if (conditions.rinshan && (conditions.winMethod !== 'tsumo' || !hasKan)) {
    throw new RangeError('rinshan requires a tsumo after a kan');
  }
  if (conditions.chankan && conditions.winMethod !== 'ron') {
    throw new RangeError('chankan requires ron');
  }
  if (conditions.haitei && conditions.winMethod !== 'tsumo') {
    throw new RangeError('haitei requires tsumo');
  }
  if (conditions.houtei && conditions.winMethod !== 'ron') {
    throw new RangeError('houtei requires ron');
  }
  if (conditions.rinshan && conditions.haitei) {
    throw new RangeError('rinshan and haitei cannot both be true');
  }
  if (conditions.chankan && conditions.houtei) {
    throw new RangeError('chankan and houtei cannot both be true');
  }
  if (conditions.tenhou && conditions.chiihou) {
    throw new RangeError('tenhou and chiihou cannot both be true');
  }

  if (conditions.tenhou) {
    if (
      conditions.winMethod !== 'tsumo' ||
      conditions.seatWind !== 'east' ||
      conditions.riichi !== 'none' ||
      melds.length !== 0 ||
      hasInitialDrawConflict(conditions)
    ) {
      throw new RangeError('tenhou conditions are contradictory');
    }
  }

  if (conditions.chiihou) {
    if (
      conditions.winMethod !== 'tsumo' ||
      conditions.seatWind === 'east' ||
      conditions.riichi !== 'none' ||
      melds.length !== 0 ||
      hasInitialDrawConflict(conditions)
    ) {
      throw new RangeError('chiihou conditions are contradictory');
    }
  }
}

function hasInitialDrawConflict(conditions: ScoringConditions): boolean {
  return (
    conditions.ippatsu ||
    conditions.rinshan ||
    conditions.chankan ||
    conditions.haitei ||
    conditions.houtei
  );
}

function assertUniqueCompletedHandIds(tiles: readonly TileInstance[]): void {
  const ids = new Set<string>();
  for (const tile of tiles) {
    if (ids.has(tile.id)) {
      throw new RangeError('completed-hand tile instance IDs must be unique');
    }
    ids.add(tile.id);
  }
}

function assertPhysicalMultiplicity(tiles: readonly TileIdentity[]): void {
  const counts = new Map<TileKind, number>();
  for (const tile of tiles) {
    const count = (counts.get(tile.kind) ?? 0) + 1;
    if (count > MAX_TILE_COPIES) {
      throw new RangeError(`tile ${tile.kind} exceeds physical multiplicity`);
    }
    counts.set(tile.kind, count);
  }
}

function assertValidTileIdentity(tile: TileIdentity): void {
  if (!isValidTileIdentity(tile)) {
    throw new RangeError('red tile identity must be a suited five');
  }
}

function isValidTileIdentity(tile: TileIdentity): boolean {
  if (!tile.red) {
    return true;
  }

  const suited = parseSuitedTileKind(tile.kind);
  return suited !== null && suited.number === 5;
}

function isValidRecognizedMeld(group: RecognizedMeldGroup): boolean {
  if (!group.tiles.every(({ tile }) => isValidTileIdentity(tile))) {
    return false;
  }

  switch (group.kind) {
    case 'unresolved':
      return false;
    case 'chi':
      return isSequence(group.tiles.map(({ tile }) => tile));
    case 'pon':
      return areEqualKinds(group.tiles.map(({ tile }) => tile));
    case 'open-kan':
    case 'concealed-kan':
      return areEqualKinds(group.tiles.map(({ tile }) => tile));
  }
}

function assertValidScoringMeld(meld: ScoringMeld): void {
  switch (meld.kind) {
    case 'chi':
      if (!isSequence(meld.tiles)) {
        throw new RangeError('chi must contain one suited three-tile sequence');
      }
      break;
    case 'pon':
      if (!areEqualKinds(meld.tiles)) {
        throw new RangeError('pon must contain three equal tile kinds');
      }
      break;
    case 'open-kan':
    case 'concealed-kan':
      if (!areEqualKinds(meld.tiles)) {
        throw new RangeError('kan must contain four equal tile kinds');
      }
      break;
  }

  for (const tile of meld.tiles) {
    assertValidTileIdentity(tile);
  }
}

function serializeRecognizedMeld(group: RecognizedMeldGroup): string {
  switch (group.kind) {
    case 'unresolved':
      throw new RangeError('unresolved meld cannot be serialized');
    case 'chi':
    case 'pon':
    case 'open-kan':
    case 'concealed-kan':
      return serializeMeld(group.kind, group.tiles.map(({ tile }) => tile));
  }
}

function serializeScoringMeld(meld: ScoringMeld): string {
  return serializeMeld(meld.kind, meld.tiles);
}

function serializeMeld(
  kind: Exclude<RecognizedMeldGroup['kind'], 'unresolved'>,
  tiles: readonly TileIdentity[],
): string {
  const ordered = kind === 'chi' ? [...tiles].sort(compareTileKinds) : [...tiles];
  const first = ordered[0];
  if (first === undefined) {
    throw new RangeError('meld cannot be empty');
  }

  const suffix = first.kind.at(1);
  if (suffix === undefined) {
    throw new RangeError('invalid meld tile kind');
  }

  const digits = ordered
    .map((tile) => {
      assertValidTileIdentity(tile);
      if (tile.kind.at(1) !== suffix) {
        throw new RangeError('meld tiles must use one suit');
      }
      return tile.red ? '0' : tile.kind.at(0);
    })
    .join('');

  const body = `${digits}${suffix}`;
  return kind === 'concealed-kan' ? `[${body}]` : `(${body})`;
}

function areEqualKinds(tiles: readonly TileIdentity[]): boolean {
  const first = tiles[0];
  return first !== undefined && tiles.every((tile) => tile.kind === first.kind);
}

function isSequence(tiles: readonly TileIdentity[]): boolean {
  if (tiles.length !== 3) {
    return false;
  }

  const parsed = tiles.map((tile) => parseSuitedTileKind(tile.kind));
  if (parsed.some((tile) => tile === null)) {
    return false;
  }

  const suited = [...(parsed as readonly SuitedTileKind[])].sort(
    (left, right) => left.number - right.number,
  );
  const [first, second, third] = suited;

  return (
    first !== undefined &&
    second !== undefined &&
    third !== undefined &&
    first.suit === second.suit &&
    second.suit === third.suit &&
    first.number + 1 === second.number &&
    second.number + 1 === third.number
  );
}

function parseSuitedTileKind(kind: TileKind): SuitedTileKind | null {
  const suit = kind.at(1);
  if (suit !== 'm' && suit !== 'p' && suit !== 's') {
    return null;
  }

  return {
    number: Number(kind.at(0)),
    suit,
  };
}

function compareTileKinds(left: TileIdentity, right: TileIdentity): number {
  return Number(left.kind.at(0)) - Number(right.kind.at(0));
}
