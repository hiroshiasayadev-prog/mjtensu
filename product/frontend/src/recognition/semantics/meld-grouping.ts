import type { TileIdentity, TileKind } from '@/domain';

import type {
  FrameMeldInterpretation,
  MeldGroupObservation,
  TileObservation,
} from './types';

const MAX_MELD_GROUPS = 4;
const MAX_GROUP_MEMBERS = 4;
const MIN_GROUP_MEMBERS = 2;
const MAX_COMMON_TILT_RADIANS = Math.PI / 4;
const ANGLE_EPSILON = 1e-9;
const ANGLE_CANDIDATE_DEDUP_RADIANS = Math.PI / 360;
const ROW_PERPENDICULAR_SPREAD_HEIGHT_FACTOR = 0.9;
const MAX_ADJACENT_GAP_WIDTH_FACTOR = 3;
const MIN_ROW_CENTER_SEPARATION_HEIGHT_FACTOR = 0.55;
const ROW_ANGLE_RESIDUAL_WEIGHT = 0.2;
const ROW_GAP_VARIANCE_WEIGHT = 0.15;
const ROW_LARGE_GAP_WEIGHT = 0.05;
const LARGE_GAP_WIDTH_FACTOR = 1.8;
const GROUP_COUNT_PENALTY = 0.02;
const PARTITION_AMBIGUITY_EPSILON = 1e-7;

export type MeldGroupingResult =
  | {
      readonly kind: 'stable';
      readonly groups: readonly MeldGroupObservation[];
      readonly commonAngleRadians: number | null;
    }
  | {
      readonly kind: 'unstable';
    };

interface ProjectedObservation {
  readonly index: number;
  readonly observation: TileObservation;
  readonly u: number;
  readonly v: number;
}

interface RowCandidate {
  readonly mask: number;
  readonly members: readonly ProjectedObservation[];
  readonly meanV: number;
  readonly meanY: number;
  readonly score: number;
}

interface CandidatePartition {
  readonly signature: string;
  readonly groups: readonly MeldGroupObservation[];
  readonly score: number;
  readonly commonAngleRadians: number;
}

export function groupMeldObservations(
  observations: readonly TileObservation[],
): MeldGroupingResult {
  if (observations.length === 0) {
    return { kind: 'stable', groups: [], commonAngleRadians: null };
  }

  if (
    observations.length < MIN_GROUP_MEMBERS ||
    observations.length > MAX_MELD_GROUPS * MAX_GROUP_MEMBERS ||
    observations.some((observation) =>
      observation.region !== 'melds' || observation.classification.kind !== 'tile',
    )
  ) {
    return { kind: 'unstable' };
  }

  const medianHeight = median(
    observations.map((observation) => observation.obb?.height ?? observation.bbox.height),
  );
  const medianWidth = median(
    observations.map((observation) => observation.obb?.width ?? observation.bbox.width),
  );
  if (!(medianHeight > 0) || !(medianWidth > 0)) {
    return { kind: 'unstable' };
  }

  const partitions = new Map<string, CandidatePartition>();
  for (const angle of candidateAngles(observations)) {
    collectPartitionsAtAngle(
      observations,
      angle,
      medianHeight,
      medianWidth,
      (partition) => {
        const previous = partitions.get(partition.signature);
        if (previous === undefined || partition.score < previous.score) {
          partitions.set(partition.signature, partition);
        }
      },
    );
  }

  const ranked = [...partitions.values()].sort((left, right) =>
    left.score - right.score || left.signature.localeCompare(right.signature),
  );
  const best = ranked[0];
  if (best === undefined) {
    return { kind: 'unstable' };
  }

  const competing = ranked[1];
  if (
    competing !== undefined &&
    Math.abs(competing.score - best.score) <= PARTITION_AMBIGUITY_EPSILON
  ) {
    return { kind: 'unstable' };
  }

  return {
    kind: 'stable',
    groups: best.groups,
    commonAngleRadians: best.commonAngleRadians,
  };
}

function candidateAngles(observations: readonly TileObservation[]): number[] {
  const angles = [0];

  for (let leftIndex = 0; leftIndex < observations.length; leftIndex += 1) {
    const left = observations[leftIndex];
    if (left === undefined) {
      continue;
    }
    for (
      let rightIndex = leftIndex + 1;
      rightIndex < observations.length;
      rightIndex += 1
    ) {
      const right = observations[rightIndex];
      if (right === undefined) {
        continue;
      }
      const angle = undirectedLineAngle(center(left), center(right));
      if (
        angle !== null &&
        Math.abs(angle) <= MAX_COMMON_TILT_RADIANS + ANGLE_EPSILON
      ) {
        angles.push(angle);
      }
    }
  }

  return angles
    .sort((left, right) => left - right)
    .filter(
      (angle, index, values) =>
        index === 0 ||
        Math.abs(angle - (values[index - 1] ?? 0)) >
          ANGLE_CANDIDATE_DEDUP_RADIANS,
    );
}

function collectPartitionsAtAngle(
  observations: readonly TileObservation[],
  angle: number,
  medianHeight: number,
  medianWidth: number,
  collect: (partition: CandidatePartition) => void,
): void {
  const projected = projectObservations(observations, angle);
  const rows = enumerateRowCandidates(
    projected,
    angle,
    medianHeight,
    medianWidth,
  );
  if (rows.length === 0) {
    return;
  }

  const rowsByObservation = Array.from(
    { length: observations.length },
    (): RowCandidate[] => [],
  );
  for (const row of rows) {
    for (const member of row.members) {
      rowsByObservation[member.index]?.push(row);
    }
  }

  const allMask = (1 << observations.length) - 1;
  const chosen: RowCandidate[] = [];

  const search = (remainingMask: number): void => {
    if (remainingMask === 0) {
      const partition = materializePartition(chosen, medianHeight, angle);
      if (partition !== null) {
        collect(partition);
      }
      return;
    }

    if (chosen.length >= MAX_MELD_GROUPS) {
      return;
    }

    const remainingCount = bitCount(remainingMask);
    const remainingGroupCapacity =
      (MAX_MELD_GROUPS - chosen.length) * MAX_GROUP_MEMBERS;
    if (
      remainingCount < MIN_GROUP_MEMBERS ||
      remainingCount > remainingGroupCapacity
    ) {
      return;
    }

    const firstIndex = firstSetBitIndex(remainingMask);
    const candidates = rowsByObservation[firstIndex] ?? [];
    for (const row of candidates) {
      if ((row.mask & remainingMask) !== row.mask) {
        continue;
      }
      chosen.push(row);
      search(remainingMask ^ row.mask);
      chosen.pop();
    }
  };

  search(allMask);
}

function projectObservations(
  observations: readonly TileObservation[],
  angle: number,
): ProjectedObservation[] {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return observations.map((observation, index): ProjectedObservation => {
    const point = center(observation);
    return {
      index,
      observation,
      u: point.x * cos + point.y * sin,
      v: -point.x * sin + point.y * cos,
    };
  });
}

function enumerateRowCandidates(
  projected: readonly ProjectedObservation[],
  commonAngle: number,
  medianHeight: number,
  medianWidth: number,
): RowCandidate[] {
  const rows: RowCandidate[] = [];

  for (
    let memberCount = MIN_GROUP_MEMBERS;
    memberCount <= MAX_GROUP_MEMBERS;
    memberCount += 1
  ) {
    enumerateCombinations(
      projected.length,
      memberCount,
      (indices) => {
        const row = createRowCandidate(
          indices.map((index) => projected[index]).filter(isDefined),
          commonAngle,
          medianHeight,
          medianWidth,
        );
        if (row !== null) {
          rows.push(row);
        }
      },
    );
  }

  return rows;
}

function createRowCandidate(
  membersInput: readonly ProjectedObservation[],
  commonAngle: number,
  medianHeight: number,
  medianWidth: number,
): RowCandidate | null {
  if (
    membersInput.length < MIN_GROUP_MEMBERS ||
    membersInput.length > MAX_GROUP_MEMBERS
  ) {
    return null;
  }

  const members = [...membersInput].sort(
    (left, right) => left.u - right.u || left.observation.id.localeCompare(right.observation.id),
  );
  const vValues = members.map((member) => member.v);
  const vSpread = Math.max(...vValues) - Math.min(...vValues);
  if (
    vSpread >
    medianHeight * ROW_PERPENDICULAR_SPREAD_HEIGHT_FACTOR + ANGLE_EPSILON
  ) {
    return null;
  }

  const gaps: number[] = [];
  for (let index = 1; index < members.length; index += 1) {
    const previous = members[index - 1];
    const current = members[index];
    if (previous === undefined || current === undefined) {
      return null;
    }
    const gap = current.u - previous.u;
    if (
      gap <= ANGLE_EPSILON ||
      gap > medianWidth * MAX_ADJACENT_GAP_WIDTH_FACTOR + ANGLE_EPSILON
    ) {
      return null;
    }
    gaps.push(gap);
  }

  const rowAngle = fittedRowAngle(members);
  if (rowAngle === null) {
    return null;
  }

  const meanV = average(vValues);
  let score = 0;
  for (const member of members) {
    const normalizedResidual = (member.v - meanV) / medianHeight;
    score += normalizedResidual * normalizedResidual;
  }

  const angleResidual = undirectedAngleDifference(rowAngle, commonAngle);
  score += ROW_ANGLE_RESIDUAL_WEIGHT * angleResidual * angleResidual;

  if (gaps.length > 1) {
    const meanGap = average(gaps);
    const normalizedGapVariance =
      gaps.reduce((sum, gap) => {
        const normalized = (gap - meanGap) / medianWidth;
        return sum + normalized * normalized;
      }, 0) / gaps.length;
    score += ROW_GAP_VARIANCE_WEIGHT * normalizedGapVariance;
  }

  for (const gap of gaps) {
    const excess = Math.max(0, gap / medianWidth - LARGE_GAP_WIDTH_FACTOR);
    score += ROW_LARGE_GAP_WEIGHT * excess * excess;
  }

  return {
    mask: members.reduce((mask, member) => mask | (1 << member.index), 0),
    members,
    meanV,
    meanY: average(members.map((member) => center(member.observation).y)),
    score,
  };
}

function materializePartition(
  rows: readonly RowCandidate[],
  medianHeight: number,
  commonAngleRadians: number,
): CandidatePartition | null {
  if (rows.length === 0 || rows.length > MAX_MELD_GROUPS) {
    return null;
  }

  const byProjectedRow = [...rows].sort(
    (left, right) => left.meanV - right.meanV,
  );
  for (let index = 1; index < byProjectedRow.length; index += 1) {
    const previous = byProjectedRow[index - 1];
    const current = byProjectedRow[index];
    if (
      previous === undefined ||
      current === undefined ||
      current.meanV - previous.meanV <
        medianHeight * MIN_ROW_CENTER_SEPARATION_HEIGHT_FACTOR - ANGLE_EPSILON
    ) {
      return null;
    }
  }

  const orderedRows = [...rows].sort(
    (left, right) => left.meanY - right.meanY || left.meanV - right.meanV,
  );
  const groups = orderedRows.map((row): MeldGroupObservation => {
    const orderedObservations = row.members.map((member) => member.observation);
    return {
      memberObservationIds: orderedObservations.map((member) => member.id),
      interpretation: interpretMeld(orderedObservations),
    };
  });
  const signature = groups
    .map((group) => group.memberObservationIds.join(','))
    .join('|');
  const score =
    rows.reduce((sum, row) => sum + row.score, 0) +
    rows.length * GROUP_COUNT_PENALTY;

  return { signature, groups, score, commonAngleRadians };
}

function fittedRowAngle(row: readonly ProjectedObservation[]): number | null {
  const points = row.map((member) => center(member.observation));
  const meanX = average(points.map((point) => point.x));
  const meanY = average(points.map((point) => point.y));
  let numerator = 0;
  let denominator = 0;

  for (const point of points) {
    const dx = point.x - meanX;
    numerator += dx * (point.y - meanY);
    denominator += dx * dx;
  }

  if (denominator <= ANGLE_EPSILON) {
    return null;
  }
  return Math.atan(numerator / denominator);
}

function undirectedLineAngle(
  left: { readonly x: number; readonly y: number },
  right: { readonly x: number; readonly y: number },
): number | null {
  const dx = right.x - left.x;
  const dy = right.y - left.y;
  if (Math.abs(dx) <= ANGLE_EPSILON && Math.abs(dy) <= ANGLE_EPSILON) {
    return null;
  }

  let angle = Math.atan2(dy, dx);
  if (angle > Math.PI / 2) {
    angle -= Math.PI;
  } else if (angle < -Math.PI / 2) {
    angle += Math.PI;
  }
  return angle;
}

function undirectedAngleDifference(left: number, right: number): number {
  let difference = left - right;
  while (difference > Math.PI / 2) {
    difference -= Math.PI;
  }
  while (difference < -Math.PI / 2) {
    difference += Math.PI;
  }
  return difference;
}

function enumerateCombinations(
  itemCount: number,
  choose: number,
  visit: (indices: readonly number[]) => void,
): void {
  const indices: number[] = [];
  const search = (start: number): void => {
    if (indices.length === choose) {
      visit(indices);
      return;
    }

    const needed = choose - indices.length;
    for (let index = start; index <= itemCount - needed; index += 1) {
      indices.push(index);
      search(index + 1);
      indices.pop();
    }
  };
  search(0);
}

function firstSetBitIndex(mask: number): number {
  for (let index = 0; index < 32; index += 1) {
    if ((mask & (1 << index)) !== 0) {
      return index;
    }
  }
  return -1;
}

function bitCount(mask: number): number {
  let value = mask >>> 0;
  let count = 0;
  while (value !== 0) {
    value &= value - 1;
    count += 1;
  }
  return count;
}

function average(values: readonly number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function isDefined<T>(value: T | undefined): value is T {
  return value !== undefined;
}

function interpretMeld(
  observations: readonly TileObservation[],
): FrameMeldInterpretation {
  const tiles = observations.map((observation) => {
    if (observation.classification.kind !== 'tile') {
      throw new Error('Meld interpretation requires tile-classified observations');
    }
    return observation.classification.tile;
  });

  if (tiles.length === 2 && sameBaseKind(tiles)) {
    const first = tiles[0];
    const second = tiles[1];
    if (first === undefined || second === undefined) {
      throw new Error('Concealed-kan reconstruction requires two visible tiles');
    }
    const hiddenLeft = ordinaryIdentity(first.kind);
    const hiddenRight = ordinaryIdentity(first.kind);
    return {
      kind: 'concealed-kan',
      tiles: [hiddenLeft, first, second, hiddenRight],
    };
  }

  if (tiles.length === 3) {
    const tuple = asThreeTiles(tiles);
    if (sameBaseKind(tiles)) {
      return { kind: 'pon', tiles: tuple };
    }
    if (isChi(tiles)) {
      return { kind: 'chi', tiles: tuple };
    }
    return { kind: 'unresolved', tiles };
  }

  if (tiles.length === 4) {
    const tuple = asFourTiles(tiles);
    if (sameBaseKind(tiles)) {
      return { kind: 'open-kan', tiles: tuple };
    }
    return { kind: 'unresolved', tiles };
  }

  return { kind: 'unresolved', tiles };
}

function sameBaseKind(tiles: readonly TileIdentity[]): boolean {
  const first = tiles[0];
  return first !== undefined && tiles.every((tile) => tile.kind === first.kind);
}

function isChi(tiles: readonly TileIdentity[]): boolean {
  if (tiles.length !== 3) {
    return false;
  }

  const parsed = tiles.map((tile) => parseSuitedTile(tile.kind));
  if (parsed.some((item) => item === null)) {
    return false;
  }

  const suited = parsed.filter(
    (item): item is { readonly suit: 'm' | 'p' | 's'; readonly rank: number } =>
      item !== null,
  );
  if (suited.length !== 3 || !suited.every((item) => item.suit === suited[0]?.suit)) {
    return false;
  }

  const ranks = suited.map((item) => item.rank).sort((left, right) => left - right);
  return ranks[1] === (ranks[0] ?? 0) + 1 && ranks[2] === (ranks[1] ?? 0) + 1;
}

function parseSuitedTile(
  kind: TileKind,
): { readonly suit: 'm' | 'p' | 's'; readonly rank: number } | null {
  const suit = kind.at(-1);
  if (suit !== 'm' && suit !== 'p' && suit !== 's') {
    return null;
  }
  return { suit, rank: Number(kind[0]) };
}

function ordinaryIdentity(kind: TileKind): TileIdentity {
  return { kind, red: false };
}

function asThreeTiles(
  tiles: readonly TileIdentity[],
): readonly [TileIdentity, TileIdentity, TileIdentity] {
  const [first, second, third] = tiles;
  if (first === undefined || second === undefined || third === undefined || tiles.length !== 3) {
    throw new Error('Expected exactly three tiles');
  }
  return [first, second, third];
}

function asFourTiles(
  tiles: readonly TileIdentity[],
): readonly [TileIdentity, TileIdentity, TileIdentity, TileIdentity] {
  const [first, second, third, fourth] = tiles;
  if (
    first === undefined ||
    second === undefined ||
    third === undefined ||
    fourth === undefined ||
    tiles.length !== 4
  ) {
    throw new Error('Expected exactly four tiles');
  }
  return [first, second, third, fourth];
}

function center(observation: TileObservation): { readonly x: number; readonly y: number } {
  return {
    x: observation.obb?.cx ?? observation.bbox.x + observation.bbox.width / 2,
    y: observation.obb?.cy ?? observation.bbox.y + observation.bbox.height / 2,
  };
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) {
    return sorted[middle] ?? 0;
  }
  return ((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2;
}
