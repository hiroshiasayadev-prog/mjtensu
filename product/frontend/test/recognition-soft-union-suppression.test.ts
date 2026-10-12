import { describe, expect, it } from 'vitest';

import { suppressDetectorSoftUnion } from '@/recognition/detector/soft-union-suppression';
import type {
  Rect,
  RegionDetection,
  SemanticRegion,
} from '@/recognition/detector/types';

const options = {
  minimumScore: 0.35,
  softNmsSigma: 0.5,
  unionCoverageThreshold: 0.7,
} as const;

describe('NanoDet Soft-NMS plus union coverage suppression', () => {
  it('applies Gaussian Soft-NMS with the production minimum score', () => {
    const winner = candidate('winner', 0, 0.9, {
      x: 0,
      y: 0,
      width: 10,
      height: 10,
    });
    const duplicate = candidate('duplicate', 1, 0.5, {
      x: 1,
      y: 0,
      width: 10,
      height: 10,
    });
    const neighbor = candidate('neighbor', 2, 0.7, {
      x: 20,
      y: 0,
      width: 10,
      height: 10,
    });

    expect(
      suppressDetectorSoftUnion([duplicate, neighbor, winner], options).map(
        (item) => item.id,
      ),
    ).toEqual(['winner', 'neighbor']);
  });

  it('suppresses a lower-confidence box covered by the union of higher-confidence boxes', () => {
    const candidateBox = candidate('candidate', 2, 0.8, {
      x: 0,
      y: 0,
      width: 10,
      height: 10,
    });
    const left = candidate('left', 0, 0.95, {
      x: 0,
      y: 0,
      width: 3.5,
      height: 10,
    });
    const right = candidate('right', 1, 0.9, {
      x: 6.5,
      y: 0,
      width: 3.5,
      height: 10,
    });

    expect(
      suppressDetectorSoftUnion([candidateBox, right, left], options).map(
        (item) => item.id,
      ),
    ).toEqual(['left', 'right']);
  });

  it('keeps a union-covered box when any intersecting box has lower confidence', () => {
    const candidateBox = candidate('candidate', 2, 0.8, {
      x: 0,
      y: 0,
      width: 10,
      height: 10,
    });
    const left = candidate('left', 0, 0.95, {
      x: 0,
      y: 0,
      width: 3.5,
      height: 10,
    });
    const right = candidate('right', 1, 0.75, {
      x: 6.5,
      y: 0,
      width: 3.5,
      height: 10,
    });

    expect(
      suppressDetectorSoftUnion([candidateBox, right, left], options).map(
        (item) => item.id,
      ),
    ).toEqual(['left', 'candidate']);
  });

  it('never uses boxes from another semantic region for Soft-NMS or union coverage', () => {
    const hand = candidate(
      'hand',
      0,
      0.8,
      { x: 0, y: 0, width: 10, height: 10 },
      'completed_hand',
    );
    const meld = candidate(
      'meld',
      1,
      0.95,
      { x: 0, y: 0, width: 10, height: 10 },
      'melds',
    );

    expect(suppressDetectorSoftUnion([meld, hand], options)).toEqual([
      hand,
      meld,
    ]);
  });
});

function candidate(
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
