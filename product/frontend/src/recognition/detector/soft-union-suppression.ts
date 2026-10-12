import type { Rect, RegionDetection } from './types';

export interface SoftUnionSuppressionOptions {
  readonly minimumScore: number;
  readonly softNmsSigma: number;
  readonly unionCoverageThreshold: number;
}

export function suppressDetectorSoftUnion(
  detections: readonly RegionDetection[],
  options: SoftUnionSuppressionOptions,
): RegionDetection[] {
  validateOptions(options);
  const byRegion = new Map<RegionDetection['region'], RegionDetection[]>();
  for (const detection of detections) {
    const group = byRegion.get(detection.region) ?? [];
    group.push(detection);
    byRegion.set(detection.region, group);
  }

  const retained: RegionDetection[] = [];
  for (const group of byRegion.values()) {
    const afterSoft = gaussianSoftNms(group, options);
    retained.push(
      ...suppressByUnionCoverage(afterSoft, options.unionCoverageThreshold),
    );
  }

  return retained.sort(compareDetectionOrder);
}

function gaussianSoftNms(
  detections: readonly RegionDetection[],
  options: SoftUnionSuppressionOptions,
): RegionDetection[] {
  let pending = detections.map((detection) => ({
    detection,
    score: detection.confidence,
  }));
  const kept: RegionDetection[] = [];

  while (pending.length > 0) {
    let bestIndex = 0;
    for (let index = 1; index < pending.length; index += 1) {
      if (
        (pending[index]?.score ?? Number.NEGATIVE_INFINITY) >
        (pending[bestIndex]?.score ?? Number.NEGATIVE_INFINITY)
      ) {
        bestIndex = index;
      }
    }

    const winner = pending.splice(bestIndex, 1)[0];
    if (winner === undefined || winner.score < options.minimumScore) {
      break;
    }
    kept.push(winner.detection);

    const nextPending: typeof pending = [];
    for (const candidate of pending) {
      const overlap = intersectionOverUnion(
        winner.detection.sourceBox,
        candidate.detection.sourceBox,
      );
      const score =
        candidate.score *
        Math.exp(-(overlap * overlap) / options.softNmsSigma);
      if (score >= options.minimumScore) {
        nextPending.push({ detection: candidate.detection, score });
      }
    }
    pending = nextPending;
  }

  return kept;
}

function suppressByUnionCoverage(
  detections: readonly RegionDetection[],
  threshold: number,
): RegionDetection[] {
  const active = [...detections];
  const suppressedIds = new Set<string>();
  const ordered = [...active].sort(compareConfidencePriority);

  for (const candidate of ordered) {
    if (suppressedIds.has(candidate.id)) {
      continue;
    }

    const others = active.filter(
      (other) =>
        other !== candidate &&
        !suppressedIds.has(other.id) &&
        intersectionArea(candidate.sourceBox, other.sourceBox) > 0,
    );
    if (
      others.length === 0 ||
      !others.every((other) => candidate.confidence < other.confidence)
    ) {
      continue;
    }

    if (unionCoverage(candidate.sourceBox, others) >= threshold) {
      suppressedIds.add(candidate.id);
    }
  }

  return active.filter((detection) => !suppressedIds.has(detection.id));
}

function unionCoverage(
  candidate: Rect,
  others: readonly RegionDetection[],
): number {
  const candidateArea = rectArea(candidate);
  if (candidateArea <= 0) {
    return 0;
  }

  const intersections = others
    .map((other) => intersectionRect(candidate, other.sourceBox))
    .filter((rect): rect is EdgeRect => rect !== null);
  return unionArea(intersections) / candidateArea;
}

interface EdgeRect {
  readonly x0: number;
  readonly y0: number;
  readonly x1: number;
  readonly y1: number;
}

function unionArea(rectangles: readonly EdgeRect[]): number {
  if (rectangles.length === 0) {
    return 0;
  }

  const xs = [...new Set(rectangles.flatMap((rect) => [rect.x0, rect.x1]))].sort(
    (left, right) => left - right,
  );
  let total = 0;
  for (let index = 0; index + 1 < xs.length; index += 1) {
    const left = xs[index];
    const right = xs[index + 1];
    if (left === undefined || right === undefined || right <= left) {
      continue;
    }

    const intervals = rectangles
      .filter((rect) => rect.x0 < right && rect.x1 > left && rect.y1 > rect.y0)
      .map((rect) => [rect.y0, rect.y1] as const)
      .sort((a, b) => a[0] - b[0]);
    if (intervals.length === 0) {
      continue;
    }

    let [start, end] = intervals[0]!;
    let covered = 0;
    for (const [y0, y1] of intervals.slice(1)) {
      if (y0 <= end) {
        end = Math.max(end, y1);
      } else {
        covered += end - start;
        start = y0;
        end = y1;
      }
    }
    covered += end - start;
    total += (right - left) * covered;
  }
  return total;
}

function intersectionRect(left: Rect, right: Rect): EdgeRect | null {
  const x0 = Math.max(left.x, right.x);
  const y0 = Math.max(left.y, right.y);
  const x1 = Math.min(left.x + left.width, right.x + right.width);
  const y1 = Math.min(left.y + left.height, right.y + right.height);
  return x1 > x0 && y1 > y0 ? { x0, y0, x1, y1 } : null;
}

function intersectionArea(left: Rect, right: Rect): number {
  const intersection = intersectionRect(left, right);
  return intersection === null
    ? 0
    : (intersection.x1 - intersection.x0) *
        (intersection.y1 - intersection.y0);
}

function intersectionOverUnion(left: Rect, right: Rect): number {
  const intersection = intersectionArea(left, right);
  if (intersection <= 0) {
    return 0;
  }
  const union = rectArea(left) + rectArea(right) - intersection;
  return union <= 0 ? 0 : intersection / union;
}

function rectArea(rect: Rect): number {
  return Math.max(0, rect.width) * Math.max(0, rect.height);
}

function compareConfidencePriority(
  left: RegionDetection,
  right: RegionDetection,
): number {
  return (
    right.confidence - left.confidence ||
    left.detectionIndex - right.detectionIndex ||
    left.id.localeCompare(right.id)
  );
}

function compareDetectionOrder(
  left: RegionDetection,
  right: RegionDetection,
): number {
  return (
    left.detectionIndex - right.detectionIndex ||
    left.id.localeCompare(right.id)
  );
}

function validateOptions(options: SoftUnionSuppressionOptions): void {
  assertUnitInterval(options.minimumScore, 'Soft-NMS minimum score');
  if (!Number.isFinite(options.softNmsSigma) || options.softNmsSigma <= 0) {
    throw new Error('Soft-NMS sigma must be finite and positive.');
  }
  assertUnitInterval(
    options.unionCoverageThreshold,
    'Union coverage threshold',
  );
}

function assertUnitInterval(value: number, label: string): void {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error(`${label} must be in [0, 1].`);
  }
}
