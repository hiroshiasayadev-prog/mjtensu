import { describe, expect, it } from 'vitest';

import {
  ROTATED_FCOS_INPUT_SIZE,
  decodeRotatedFcosOutputs,
  preprocessRotatedFcosCompositeRgba,
  rotatedIntersectionOverUnion,
} from '@/recognition/detector/rotated-fcos';
import type { OrientedRect, TensorOutput } from '@/recognition/detector/types';

describe('RotatedFCOSNano browser contract', () => {
  it('uses RGB ImageNet normalization in planar NCHW order', () => {
    const pixels = ROTATED_FCOS_INPUT_SIZE * ROTATED_FCOS_INPUT_SIZE;
    const rgba = new Uint8ClampedArray(pixels * 4);
    rgba[0] = 255;
    rgba[1] = 0;
    rgba[2] = 127;
    rgba[3] = 255;

    const output = preprocessRotatedFcosCompositeRgba(rgba);

    expect(output).toHaveLength(3 * pixels);
    expect(output[0]).toBeCloseTo((1 - 0.485) / 0.229, 6);
    expect(output[pixels]).toBeCloseTo((0 - 0.456) / 0.224, 6);
    expect(output[2 * pixels]).toBeCloseTo((127 / 255 - 0.406) / 0.225, 6);
  });

  it('decodes the three FCOS levels with half-stride points and doubled-angle regression', () => {
    const outputs = [outputLevel(40), outputLevel(20), outputLevel(10)];
    const stride8 = outputs[0];
    if (stride8 === undefined || !(stride8.data instanceof Float32Array)) {
      throw new Error('missing stride-8 fixture');
    }
    const width = 40;
    const plane = width * width;
    const pointIndex = 4 * width + 5;
    stride8.data[pointIndex] = 10;
    stride8.data[plane + pointIndex] = 10;
    stride8.data[2 * plane + pointIndex] = 0.25;
    stride8.data[3 * plane + pointIndex] = -0.5;
    stride8.data[4 * plane + pointIndex] = Math.log(3);
    stride8.data[5 * plane + pointIndex] = Math.log(5);
    stride8.data[6 * plane + pointIndex] = Math.sin(Math.PI / 3);
    stride8.data[7 * plane + pointIndex] = Math.cos(Math.PI / 3);

    const detections = decodeRotatedFcosOutputs(outputs, {
      confidenceThreshold: 0.3,
      nmsIouThreshold: 0.45,
      maximumDetections: 64,
    });

    expect(detections).toHaveLength(1);
    expect(detections[0]).toMatchObject({
      id: `rotated-fcos:0:${pointIndex}`,
      classIndex: 0,
      orientedBox: {
        cx: expect.closeTo(46, 5),
        cy: expect.closeTo(32, 5),
        width: expect.closeTo(24, 5),
        height: expect.closeTo(40, 5),
        angleDeg: expect.closeTo(30, 5),
      },
    });
    expect(detections[0]?.confidence).toBeGreaterThan(0.99);
  });

  it('computes 180-degree-periodic rotated IoU geometry', () => {
    const first: OrientedRect = {
      cx: 100,
      cy: 100,
      width: 24,
      height: 40,
      angleDeg: -35,
    };
    const equivalent = { ...first, angleDeg: 145 };
    const disjoint = { ...first, cx: 250, cy: 250 };

    expect(rotatedIntersectionOverUnion(first, equivalent)).toBeCloseTo(1, 6);
    expect(rotatedIntersectionOverUnion(first, disjoint)).toBe(0);
  });
});

function outputLevel(spatial: number): TensorOutput {
  const plane = spatial * spatial;
  const data = new Float32Array(8 * plane);
  for (let pointIndex = 0; pointIndex < plane; pointIndex += 1) {
    data[pointIndex] = -20;
    data[plane + pointIndex] = -20;
  }
  return {
    dims: [1, 8, spatial, spatial],
    data,
    type: 'float32',
  };
}
