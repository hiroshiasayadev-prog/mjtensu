import { describe, expect, it } from 'vitest';

import { createNanoDetPostprocessor } from '@/recognition/detector/detection-postprocessor';
import {
  NANODET_OUTPUT_CHANNELS,
  NANODET_OUTPUT_POINTS,
} from '@/recognition/detector/nanodet';
import type { CaptureRegions, TensorOutput } from '@/recognition/detector/types';

const captureRegions: CaptureRegions = {
  completed_hand: {
    enabled: true,
    sourceRect: { x: 100, y: 200, width: 850, height: 200 },
  },
  dora_indicators: {
    enabled: true,
    sourceRect: { x: 100, y: 20, width: 850, height: 200 },
  },
  melds: {
    enabled: true,
    sourceRect: { x: 1000, y: 20, width: 400, height: 400 },
  },
};

describe('production NanoDet detection postprocess', () => {
  it('assigns regions, rejects separator candidates, and suppresses contained duplicates', () => {
    const values = new Float32Array(NANODET_OUTPUT_POINTS * NANODET_OUTPUT_CHANNELS);

    // Small hand box at prior (80, 40): [72, 24, 88, 56].
    setCandidate(values, 5 * 40 + 10, 0.7, [1, 2, 1, 2]);
    // Larger containing box at prior (88, 40): [64, 16, 96, 64].
    // IoU is below ordinary NMS, while overlap/smaller-area is 1.0.
    setCandidate(values, 5 * 40 + 11, 0.9, [3, 3, 1, 3]);
    // Its center is exactly in the y=72 separator and must be rejected.
    setCandidate(values, 9 * 40 + 20, 0.8, [1, 1, 1, 1]);

    const postprocessor = createNanoDetPostprocessor({
      confidenceThreshold: 0.35,
      nmsIouThreshold: 0.6,
      maximumDetections: 200,
      duplicateOverlapThreshold: 0.8,
    });
    const retained = postprocessor.process(tensor(values), captureRegions);

    expect(retained).toHaveLength(1);
    expect(retained[0]).toMatchObject({
      id: `nanodet:${5 * 40 + 11}`,
      confidence: expect.closeTo(0.9),
      classIndex: 0,
      region: 'completed_hand',
    });
    expect(retained[0]?.sourceBox.x).toBeCloseTo(100 + (64 - 7) * (850 / 306));
    expect(retained[0]?.sourceBox.y).toBeCloseTo(200 + 16 * (200 / 72));
  });
});

function setCandidate(
  values: Float32Array,
  pointIndex: number,
  confidence: number,
  distances: readonly [number, number, number, number],
): void {
  const offset = pointIndex * NANODET_OUTPUT_CHANNELS;
  values[offset] = confidence;
  for (const [side, distance] of distances.entries()) {
    const distributionOffset = offset + 1 + side * 8;
    values.fill(-50, distributionOffset, distributionOffset + 8);
    values[distributionOffset + distance] = 50;
  }
}

function tensor(data: Float32Array): TensorOutput {
  return {
    dims: [1, NANODET_OUTPUT_POINTS, NANODET_OUTPUT_CHANNELS],
    data,
    type: 'float32',
  };
}
