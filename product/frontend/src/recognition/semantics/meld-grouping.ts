import type { TileIdentity, TileKind } from '@/domain';

import type {
  FrameMeldInterpretation,
  MeldGroupObservation,
  TileObservation,
} from './types';

const MAX_MELD_GROUPS = 4;
const MAX_GROUP_MEMBERS = 4;
const MIN_GROUP_MEMBERS = 2;
const MAX_TILT_RADIANS = Math.PI / 8;
const ANGLE_EPSILON = 1e-9;
const ROW_CLUSTER_HEIGHT_FACTOR = 0.35;
const PARTITION_AMBIGUITY_EPSILON = 1e-7;

export type MeldGroupingResult =
  | {
      readonly kind: 'stable';
      readonly groups: readonly MeldGroupObservation[];
    }
  | {
      readonly kind: 'unstable';
    };

interface ProjectedObservation {
  readonly observation: TileObservation;
  readonly u: number;
  readonly v: number;
}

interface CandidatePartition {
  readonly signature: string;
  readonly groups: readonly MeldGroupObservation[];
  readonly score: number;
}

export function groupMeldObservations(
  observations: readonly TileObservation[],
): MeldGroupingResult {
  if (observations.length === 0) {
    return { kind: 'stable', groups: [] };
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
    observations.map((observation) => observation.bbox.height),
  );
  if (!(medianHeight > 0)) {
    return { kind: 'unstable' };
  }

  const partitions = new Map<string, CandidatePartition>();
  for (const angle of candidateAngles(observations)) {
    const partition = partitionAtAngle(observations, angle, medianHeight);
    if (partition === null) {
      continue;
    }
    const previous = partitions.get(partition.signature);
    if (previous === undefined || partition.score < previous.score) {
      partitions.set(partition.signature, partition);
    }
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

  return { kind: 'stable', groups: best.groups };
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
      const leftCenter = center(left);
      const rightCenter = center(right);
      const dx = rightCenter.x - leftCenter.x;
      const dy = rightCenter.y - leftCenter.y;
      if (Math.abs(dx) <= ANGLE_EPSILON) {
        continue;
      }
      const angle = Math.atan2(dy * Math.sign(dx), Math.abs(dx));
      if (Math.abs(angle) <= MAX_TILT_RADIANS + ANGLE_EPSILON) {
        angles.push(angle);
      }
    }
  }

  return angles
    .sort((left, right) => left - right)
    .filter(
      (angle, index, values) =>
        index === 0 || Math.abs(angle - (values[index - 1] ?? 0)) > ANGLE_EPSILON,
    );
}

function partitionAtAngle(
  observations: readonly TileObservation[],
  angle: number,
  medianHeight: number,
): CandidatePartition | null {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const projected = observations
    .map((observation): ProjectedObservation => {
      const point = center(observation);
      return {
        observation,
        u: point.x * cos + point.y * sin,
        v: -point.x * sin + point.y * cos,
      };
    })
    .sort((left, right) => left.v - right.v || left.u - right.u);

  const tolerance = medianHeight * ROW_CLUSTER_HEIGHT_FACTOR;
  const rows: ProjectedObservation[][] = [];

  for (const item of projected) {
    const current = rows.at(-1);
    if (current === undefined) {
      rows.push([item]);
      continue;
    }

    const nextValues = [...current.map((member) => member.v), item.v];
    if (Math.max(...nextValues) - Math.min(...nextValues) <= tolerance * 2) {
      current.push(item);
    } else {
      rows.push([item]);
    }
  }

  if (
    rows.length === 0 ||
    rows.length > MAX_MELD_GROUPS ||
    rows.some(
      (row) =>
        row.length < MIN_GROUP_MEMBERS || row.length > MAX_GROUP_MEMBERS,
    )
  ) {
    return null;
  }

  const orderedRows = rows
    .map((row) => row.sort((left, right) => left.u - right.u))
    .sort((left, right) => averageY(left) - averageY(right));

  let score = 0;
  const groups: MeldGroupObservation[] = [];
  for (const row of orderedRows) {
    const rowAngle = fittedRowAngle(row);
    if (
      rowAngle === null ||
      Math.abs(rowAngle) > MAX_TILT_RADIANS + ANGLE_EPSILON
    ) {
      return null;
    }

    const meanV = row.reduce((sum, member) => sum + member.v, 0) / row.length;
    for (const member of row) {
      const normalizedResidual = (member.v - meanV) / medianHeight;
      score += normalizedResidual * normalizedResidual;
    }
    const angleResidual = rowAngle - angle;
    score += angleResidual * angleResidual;

    const orderedObservations = row.map((member) => member.observation);
    groups.push({
      memberObservationIds: orderedObservations.map((member) => member.id),
      interpretation: interpretMeld(orderedObservations),
    });
  }

  const signature = groups
    .map((group) => group.memberObservationIds.join(','))
    .join('|');

  return { signature, groups, score };
}

function fittedRowAngle(row: readonly ProjectedObservation[]): number | null {
  const points = row.map((member) => center(member.observation));
  const meanX = points.reduce((sum, point) => sum + point.x, 0) / points.length;
  const meanY = points.reduce((sum, point) => sum + point.y, 0) / points.length;
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
    x: observation.bbox.x + observation.bbox.width / 2,
    y: observation.bbox.y + observation.bbox.height / 2,
  };
}

function averageY(row: readonly ProjectedObservation[]): number {
  return row.reduce((sum, member) => sum + center(member.observation).y, 0) / row.length;
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) {
    return sorted[middle] ?? 0;
  }
  return ((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2;
}
