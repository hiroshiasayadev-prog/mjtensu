import type { Rect, RegionDetection } from './types';

export const DEFAULT_DUPLICATE_OVERLAP_THRESHOLD = 0.8;

export function suppressDetectorDuplicates(
  detections: readonly RegionDetection[],
  overlapThreshold = DEFAULT_DUPLICATE_OVERLAP_THRESHOLD,
): RegionDetection[] {
  assertThreshold(overlapThreshold);
  const byRegion = new Map<RegionDetection['region'], RegionDetection[]>();
  for (const detection of detections) {
    const group = byRegion.get(detection.region) ?? [];
    group.push(detection);
    byRegion.set(detection.region, group);
  }

  const winners: RegionDetection[] = [];
  for (const group of byRegion.values()) {
    const candidates = group.filter(
      (candidate, candidateIndex) =>
        !isMergedBridgeCandidate(candidate, candidateIndex, group, overlapThreshold),
    );
    candidates.sort(compareWinnerPriority);

    const kept: RegionDetection[] = [];
    for (const candidate of candidates) {
      const duplicatesKeptCandidate = kept.some(
        (winner) =>
          overlapOverSmallerBox(candidate.sourceBox, winner.sourceBox) >= overlapThreshold,
      );
      if (!duplicatesKeptCandidate) {
        kept.push(candidate);
      }
    }
    winners.push(...kept);
  }

  return winners.sort(
    (left, right) =>
      left.detectionIndex - right.detectionIndex || left.id.localeCompare(right.id),
  );
}

export function overlapOverSmallerBox(left: Rect, right: Rect): number {
  const leftArea = rectArea(left);
  const rightArea = rectArea(right);
  const smallerArea = Math.min(leftArea, rightArea);
  if (smallerArea <= 0) {
    return 0;
  }
  const intersectionWidth = Math.max(
    0,
    Math.min(left.x + left.width, right.x + right.width) - Math.max(left.x, right.x),
  );
  const intersectionHeight = Math.max(
    0,
    Math.min(left.y + left.height, right.y + right.height) - Math.max(left.y, right.y),
  );
  return (intersectionWidth * intersectionHeight) / smallerArea;
}

function isMergedBridgeCandidate(
  candidate: RegionDetection,
  candidateIndex: number,
  group: readonly RegionDetection[],
  overlapThreshold: number,
): boolean {
  const candidateArea = rectArea(candidate.sourceBox);
  if (candidateArea <= 0) {
    return false;
  }

  const coveredSmallerCandidates = group.filter((other, otherIndex) => {
    if (otherIndex === candidateIndex || rectArea(other.sourceBox) >= candidateArea) {
      return false;
    }
    return overlapOverSmallerBox(candidate.sourceBox, other.sourceBox) >= overlapThreshold;
  });

  for (let leftIndex = 0; leftIndex < coveredSmallerCandidates.length; leftIndex += 1) {
    const left = coveredSmallerCandidates[leftIndex];
    if (left === undefined) {
      continue;
    }
    for (
      let rightIndex = leftIndex + 1;
      rightIndex < coveredSmallerCandidates.length;
      rightIndex += 1
    ) {
      const right = coveredSmallerCandidates[rightIndex];
      if (
        right !== undefined &&
        overlapOverSmallerBox(left.sourceBox, right.sourceBox) < overlapThreshold
      ) {
        return true;
      }
    }
  }
  return false;
}

function rectArea(rect: Rect): number {
  return Math.max(0, rect.width) * Math.max(0, rect.height);
}

function compareWinnerPriority(left: RegionDetection, right: RegionDetection): number {
  return (
    right.confidence - left.confidence ||
    left.detectionIndex - right.detectionIndex ||
    left.id.localeCompare(right.id)
  );
}

function assertThreshold(value: number): void {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error('Detector duplicate overlap threshold must be in [0, 1].');
  }
}
