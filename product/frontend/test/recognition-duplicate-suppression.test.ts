import { describe, expect, it } from 'vitest';

import {
  overlapOverSmallerBox,
  suppressDetectorDuplicates,
} from '@/recognition/detector/duplicate-suppression';
import type { Rect, RegionDetection, SemanticRegion } from '@/recognition/detector/types';

describe('detector duplicate suppression', () => {
  it('uses intersection divided by the smaller box area', () => {
    expect(
      overlapOverSmallerBox(
        { x: 0, y: 0, width: 10, height: 10 },
        { x: 1, y: 0, width: 8, height: 10 },
      ),
    ).toBe(1);
  });

  it('keeps the highest-confidence member of a duplicate component', () => {
    const low = candidate('low', 0, 0.5, 0);
    const high = candidate('high', 1, 0.9, 2);
    expect(suppressDetectorDuplicates([low, high])).toEqual([high]);
  });

  it('uses detection order as a stable tie break', () => {
    const first = candidate('first', 3, 0.8, 0);
    const second = candidate('second', 4, 0.8, 1);
    expect(suppressDetectorDuplicates([second, first])).toEqual([first]);
  });

  it('treats transitively connected duplicates as one component', () => {
    const first = candidate('a', 0, 0.7, 0);
    const middle = candidate('b', 1, 0.8, 2);
    const last = candidate('c', 2, 0.9, 4);

    expect(suppressDetectorDuplicates([first, middle, last])).toEqual([last]);
  });

  it('preserves neighboring non-overlapping candidates', () => {
    const left = candidate('left', 0, 0.7, 0);
    const right = candidate('right', 1, 0.9, 10);

    expect(suppressDetectorDuplicates([right, left])).toEqual([left, right]);
  });

  it('does not connect overlapping boxes from different semantic regions', () => {
    const hand = candidate('hand', 0, 0.7, 0, 'completed_hand');
    const dora = candidate('dora', 1, 0.9, 0, 'dora_indicators');

    expect(suppressDetectorDuplicates([dora, hand])).toEqual([hand, dora]);
  });
});

function candidate(
  id: string,
  detectionIndex: number,
  confidence: number,
  x: number,
  region: SemanticRegion = 'completed_hand',
): RegionDetection {
  const box: Rect = { x, y: 0, width: 10, height: 10 };
  return {
    id,
    detectionIndex,
    classIndex: 0,
    confidence,
    box,
    region,
    sourceBox: box,
  };
}
