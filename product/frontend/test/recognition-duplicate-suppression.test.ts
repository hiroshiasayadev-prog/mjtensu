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

  it('keeps the highest-confidence member of a duplicate pair', () => {
    const low = candidate('low', 0, 0.5, 0);
    const high = candidate('high', 1, 0.9, 2);
    expect(suppressDetectorDuplicates([low, high])).toEqual([high]);
  });

  it('uses detection order as a stable tie break', () => {
    const first = candidate('first', 3, 0.8, 0);
    const second = candidate('second', 4, 0.8, 1);
    expect(suppressDetectorDuplicates([second, first])).toEqual([first]);
  });

  it('does not let a bridge candidate collapse two detections that are not duplicates', () => {
    const left = candidate('left', 0, 0.9, 0);
    const bridge = candidate('bridge', 1, 0.7, 2);
    const right = candidate('right', 2, 0.8, 4);

    expect(suppressDetectorDuplicates([left, bridge, right])).toEqual([left, right]);
  });

  it('removes a larger merged bridge even when it has the highest confidence', () => {
    const left = candidateRect('left', 0, 0.7, { x: 0, y: 0, width: 10, height: 10 });
    const right = candidateRect('right', 1, 0.8, { x: 12, y: 0, width: 10, height: 10 });
    const merged = candidateRect('merged', 2, 0.99, { x: 0, y: 0, width: 22, height: 10 });

    expect(suppressDetectorDuplicates([merged, right, left])).toEqual([left, right]);
  });

  it('uses confidence when overlapping smaller candidates still describe one duplicate', () => {
    const first = candidateRect('first', 0, 0.7, { x: 0, y: 0, width: 10, height: 10 });
    const second = candidateRect('second', 1, 0.8, { x: 1, y: 0, width: 10, height: 10 });
    const larger = candidateRect('larger', 2, 0.9, { x: 0, y: 0, width: 11, height: 10 });

    expect(suppressDetectorDuplicates([first, second, larger])).toEqual([larger]);
  });

  it('does not collapse the captured iPhone meld candidates to one transitive winner', () => {
    const captured = [
      candidateRect('1950', 1950, 0.594521, { x: 100.59, y: 240.54, width: 108.98, height: 50.13 }, 'melds'),
      candidateRect('1215', 1215, 0.534499, { x: 97.17, y: 228.14, width: 45.33, height: 30.96 }, 'melds'),
      candidateRect('1890', 1890, 0.502886, { x: 130.67, y: 215.05, width: 48.16, height: 46.88 }, 'melds'),
      candidateRect('1910', 1910, 0.493536, { x: 119.15, y: 215.27, width: 82.59, height: 53.57 }, 'melds'),
      candidateRect('1222', 1222, 0.435543, { x: 165.6, y: 214.08, width: 35.82, height: 42.67 }, 'melds'),
      candidateRect('2085', 2085, 0.390986, { x: 96.42, y: 211.2, width: 112.05, height: 87.1 }, 'melds'),
      candidateRect('1912', 1912, 0.352179, { x: 148.49, y: 216.03, width: 53.3, height: 49.32 }, 'melds'),
    ];

    expect(suppressDetectorDuplicates(captured).map((item) => item.id)).toEqual([
      '1215',
      '1222',
      '1890',
      '1950',
    ]);
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
  return candidateRect(
    id,
    detectionIndex,
    confidence,
    { x, y: 0, width: 10, height: 10 },
    region,
  );
}

function candidateRect(
  id: string,
  detectionIndex: number,
  confidence: number,
  box: Rect,
  region: SemanticRegion = 'completed_hand',
): RegionDetection {
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
