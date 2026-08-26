import type { TileIdentity, TileKind } from '@/domain';

export type BaseClassifierLabel =
  | '1m'
  | '2m'
  | '3m'
  | '4m'
  | '5m'
  | '6m'
  | '7m'
  | '8m'
  | '9m'
  | '1p'
  | '2p'
  | '3p'
  | '4p'
  | '5p'
  | '6p'
  | '7p'
  | '8p'
  | '9p'
  | '1s'
  | '2s'
  | '3s'
  | '4s'
  | '5s'
  | '6s'
  | '7s'
  | '8s'
  | '9s'
  | 'east'
  | 'south'
  | 'west'
  | 'north'
  | 'white'
  | 'green'
  | 'red'
  | 'invalid';

export type RedFiveClassifierLabel = 'normal' | 'red';

export type TileClassification =
  | {
      readonly kind: 'tile';
      readonly tile: TileIdentity;
    }
  | {
      readonly kind: 'invalid';
    };

export const BASE_CLASSIFIER_LABELS = [
  '1m',
  '2m',
  '3m',
  '4m',
  '5m',
  '6m',
  '7m',
  '8m',
  '9m',
  '1p',
  '2p',
  '3p',
  '4p',
  '5p',
  '6p',
  '7p',
  '8p',
  '9p',
  '1s',
  '2s',
  '3s',
  '4s',
  '5s',
  '6s',
  '7s',
  '8s',
  '9s',
  'east',
  'south',
  'west',
  'north',
  'white',
  'green',
  'red',
  'invalid',
] as const satisfies readonly BaseClassifierLabel[];

export const RED_FIVE_CLASSIFIER_LABELS = [
  'normal',
  'red',
] as const satisfies readonly RedFiveClassifierLabel[];

const HONOR_LABEL_TO_KIND = {
  east: '1z',
  south: '2z',
  west: '3z',
  north: '4z',
  white: '5z',
  green: '6z',
  red: '7z',
} as const satisfies Readonly<
  Record<
    Extract<
      BaseClassifierLabel,
      'east' | 'south' | 'west' | 'north' | 'white' | 'green' | 'red'
    >,
    TileKind
  >
>;

const RED_FIVE_BASE_KINDS = new Set<TileKind>(['5m', '5p', '5s']);

function assertLogitCount(
  logits: ArrayLike<number>,
  expected: number,
  classifierName: string,
): void {
  if (logits.length !== expected) {
    throw new Error(
      `${classifierName} returned ${logits.length} logits; expected ${expected}`,
    );
  }
}

function argmax(logits: ArrayLike<number>): number {
  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;

  for (let index = 0; index < logits.length; index += 1) {
    const value = logits[index];
    if (value === undefined || Number.isNaN(value)) {
      throw new Error(`Classifier logit ${index} is not a valid number`);
    }
    if (value > bestValue) {
      bestIndex = index;
      bestValue = value;
    }
  }

  return bestIndex;
}

export function isRedFiveSpecialistBaseKind(kind: TileKind): boolean {
  return RED_FIVE_BASE_KINDS.has(kind);
}

export function baseClassifierLabelToTile(
  label: BaseClassifierLabel,
): TileClassification {
  if (label === 'invalid') {
    return { kind: 'invalid' };
  }

  const kind =
    label in HONOR_LABEL_TO_KIND
      ? HONOR_LABEL_TO_KIND[label as keyof typeof HONOR_LABEL_TO_KIND]
      : (label as TileKind);

  return {
    kind: 'tile',
    tile: { kind, red: false },
  };
}

export function mapBaseClassifierLogits(
  logits: ArrayLike<number>,
): TileClassification {
  assertLogitCount(logits, BASE_CLASSIFIER_LABELS.length, 'base classifier');
  return baseClassifierLabelToTile(BASE_CLASSIFIER_LABELS[argmax(logits)]);
}

export function mapRedFiveLogits(
  baseTile: TileIdentity,
  logits: ArrayLike<number>,
): TileIdentity {
  if (!isRedFiveSpecialistBaseKind(baseTile.kind)) {
    return baseTile;
  }

  assertLogitCount(logits, RED_FIVE_CLASSIFIER_LABELS.length, 'red-five classifier');
  return {
    kind: baseTile.kind,
    red: RED_FIVE_CLASSIFIER_LABELS[argmax(logits)] === 'red',
  };
}
