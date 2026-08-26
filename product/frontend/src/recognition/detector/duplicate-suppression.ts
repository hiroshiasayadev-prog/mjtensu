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
    const adjacency = group.map(() => new Set<number>());
    for (let leftIndex = 0; leftIndex < group.length; leftIndex += 1) {
      const left = group[leftIndex];
      if (left === undefined) {
        continue;
      }
      for (let rightIndex = leftIndex + 1; rightIndex < group.length; rightIndex += 1) {
        const right = group[rightIndex];
        if (
          right !== undefined &&
          overlapOverSmallerBox(left.sourceBox, right.sourceBox) >= overlapThreshold
        ) {
          adjacency[leftIndex]?.add(rightIndex);
          adjacency[rightIndex]?.add(leftIndex);
        }
      }
    }

    const visited = new Set<number>();
    for (let start = 0; start < group.length; start += 1) {
      if (visited.has(start)) {
        continue;
      }
      const component: RegionDetection[] = [];
      const stack = [start];
      visited.add(start);
      while (stack.length > 0) {
        const current = stack.pop();
        if (current === undefined) {
          continue;
        }
        const detection = group[current];
        if (detection !== undefined) {
          component.push(detection);
        }
        for (const neighbor of adjacency[current] ?? []) {
          if (!visited.has(neighbor)) {
            visited.add(neighbor);
            stack.push(neighbor);
          }
        }
      }

      component.sort(compareWinnerPriority);
      const winner = component[0];
      if (winner !== undefined) {
        winners.push(winner);
      }
    }
  }

  return winners.sort(
    (left, right) =>
      left.detectionIndex - right.detectionIndex || left.id.localeCompare(right.id),
  );
}

export function overlapOverSmallerBox(left: Rect, right: Rect): number {
  const leftArea = Math.max(0, left.width) * Math.max(0, left.height);
  const rightArea = Math.max(0, right.width) * Math.max(0, right.height);
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
