import { describe, expect, it } from 'vitest';

import {
  decodeNanoDetOutput,
  NANODET_INPUT_SIZE,
  NANODET_OUTPUT_CHANNELS,
  NANODET_OUTPUT_POINTS,
  nonMaximumSuppression,
  preprocessCompositeRgba,
} from '@/recognition/detector/nanodet';
import type { DecodedDetection, TensorOutput } from '@/recognition/detector/types';

const PIXEL_COUNT = NANODET_INPUT_SIZE * NANODET_INPUT_SIZE;
const decodeOptions = {
  confidenceThreshold: 0.35,
  nmsIouThreshold: 0.6,
  maximumDetections: 200,
} as const;

describe('NanoDet preprocessing and decode', () => {
  it('normalizes RGBA pixels into BGR CHW input', () => {
    const rgba = new Uint8ClampedArray(PIXEL_COUNT * 4);
    rgba.set([123, 116, 104, 255], 0);
    rgba.set([255, 0, 10, 255], 4);

    const input = preprocessCompositeRgba(rgba);

    expect(input[0]).toBeCloseTo((104 - 103.53) / 57.375);
    expect(input[PIXEL_COUNT]).toBeCloseTo((116 - 116.28) / 57.12);
    expect(input[2 * PIXEL_COUNT]).toBeCloseTo((123 - 123.675) / 58.395);
    expect(input[1]).toBeCloseTo((10 - 103.53) / 57.375);
    expect(input[PIXEL_COUNT + 1]).toBeCloseTo((0 - 116.28) / 57.12);
    expect(input[2 * PIXEL_COUNT + 1]).toBeCloseTo((255 - 123.675) / 58.395);
  });

  it('decodes a fixed distribution fixture into the expected class, confidence, and box', () => {
    const values = new Float32Array(NANODET_OUTPUT_POINTS * NANODET_OUTPUT_CHANNELS);
    const pointIndex = 5 * 40 + 10;
    const offset = pointIndex * NANODET_OUTPUT_CHANNELS;
    values[offset] = 0.9;
    setDistribution(values, offset + 1, 1);
    setDistribution(values, offset + 9, 2);
    setDistribution(values, offset + 17, 3);
    setDistribution(values, offset + 25, 1);

    const detections = decodeNanoDetOutput(tensor(values), decodeOptions);

    expect(detections).toHaveLength(1);
    expect(detections[0]).toMatchObject({
      id: `nanodet:${pointIndex}`,
      detectionIndex: pointIndex,
      classIndex: 0,
      confidence: expect.closeTo(0.9),
    });
    expect(detections[0]?.box.x).toBeCloseTo(72);
    expect(detections[0]?.box.y).toBeCloseTo(24);
    expect(detections[0]?.box.width).toBeCloseTo(32);
    expect(detections[0]?.box.height).toBeCloseTo(24);
  });

  it('applies deterministic confidence-first IoU NMS', () => {
    const lower = detection('lower', 1, 0.8, { x: 1, y: 1, width: 10, height: 10 });
    const higher = detection('higher', 2, 0.9, { x: 0, y: 0, width: 10, height: 10 });
    const neighbor = detection('neighbor', 3, 0.7, { x: 10, y: 0, width: 10, height: 10 });

    expect(nonMaximumSuppression([lower, neighbor, higher], 0.6, 200)).toEqual([
      higher,
      neighbor,
    ]);
  });

  it('rejects incompatible output shape, type, and data length', () => {
    const values = new Float32Array(NANODET_OUTPUT_POINTS * NANODET_OUTPUT_CHANNELS);
    expect(() =>
      decodeNanoDetOutput({ dims: [1, 1, 33], data: values, type: 'float32' }, decodeOptions),
    ).toThrow(/output shape/);
    expect(() =>
      decodeNanoDetOutput(
        { dims: [1, 2125, 33], data: new Uint8Array(values.length), type: 'uint8' },
        decodeOptions,
      ),
    ).toThrow(/output type/);
    expect(() =>
      decodeNanoDetOutput(
        { dims: [1, 2125, 33], data: new Float32Array(1), type: 'float32' },
        decodeOptions,
      ),
    ).toThrow(/data length/);
  });
});

function setDistribution(values: Float32Array, offset: number, selectedBin: number): void {
  values.fill(-50, offset, offset + 8);
  values[offset + selectedBin] = 50;
}

function tensor(data: Float32Array): TensorOutput {
  return {
    dims: [1, NANODET_OUTPUT_POINTS, NANODET_OUTPUT_CHANNELS],
    data,
    type: 'float32',
  };
}

function detection(
  id: string,
  detectionIndex: number,
  confidence: number,
  box: DecodedDetection['box'],
): DecodedDetection {
  return { id, detectionIndex, confidence, box, classIndex: 0 };
}
