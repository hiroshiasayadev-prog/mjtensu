import type { DecodedDetection, Rect, TensorOutput } from './types';

export const NANODET_INPUT_SIZE = 320;
export const NANODET_OUTPUT_POINTS = 2125;
export const NANODET_OUTPUT_CHANNELS = 33;

type NanoDetFloatArray = Float32Array<ArrayBufferLike>;

const REG_MAX = 7;
const STRIDES = [8, 16, 32, 64] as const;
const PIXEL_COUNT = NANODET_INPUT_SIZE * NANODET_INPUT_SIZE;
const BGR_MEAN = [103.53, 116.28, 123.675] as const;
const BGR_STD = [57.375, 57.12, 58.395] as const;

interface CenterPrior {
  readonly x: number;
  readonly y: number;
  readonly stride: number;
}

export interface NanoDetDecodeOptions {
  readonly confidenceThreshold: number;
  readonly nmsIouThreshold: number;
  readonly maximumDetections: number;
}

const CENTER_PRIORS = buildCenterPriors();

export function preprocessCompositeRgba(
  rgba: Uint8ClampedArray,
  target: NanoDetFloatArray = new Float32Array(3 * PIXEL_COUNT),
): NanoDetFloatArray {
  if (rgba.length !== 4 * PIXEL_COUNT) {
    throw new Error(`Composite RGBA buffer length must be ${4 * PIXEL_COUNT}.`);
  }
  if (target.length !== 3 * PIXEL_COUNT) {
    throw new Error(`NanoDet input buffer length must be ${3 * PIXEL_COUNT}.`);
  }

  for (
    let pixelIndex = 0, rgbaIndex = 0;
    pixelIndex < PIXEL_COUNT;
    pixelIndex += 1, rgbaIndex += 4
  ) {
    const red = rgba[rgbaIndex] ?? 0;
    const green = rgba[rgbaIndex + 1] ?? 0;
    const blue = rgba[rgbaIndex + 2] ?? 0;
    target[pixelIndex] = (blue - BGR_MEAN[0]) / BGR_STD[0];
    target[PIXEL_COUNT + pixelIndex] = (green - BGR_MEAN[1]) / BGR_STD[1];
    target[2 * PIXEL_COUNT + pixelIndex] = (red - BGR_MEAN[2]) / BGR_STD[2];
  }
  return target;
}

export function preprocessCompositeCanvas(
  composite: HTMLCanvasElement,
  target?: NanoDetFloatArray,
): NanoDetFloatArray {
  if (
    composite.width !== NANODET_INPUT_SIZE ||
    composite.height !== NANODET_INPUT_SIZE
  ) {
    throw new Error(
      `Recognition composite must be ${NANODET_INPUT_SIZE} x ${NANODET_INPUT_SIZE}.`,
    );
  }
  const context = composite.getContext('2d', {
    alpha: false,
    willReadFrequently: true,
  });
  if (context === null) {
    throw new Error('Recognition composite 2D context is unavailable.');
  }
  return preprocessCompositeRgba(
    context.getImageData(0, 0, NANODET_INPUT_SIZE, NANODET_INPUT_SIZE).data,
    target,
  );
}

export function decodeNanoDetOutput(
  output: TensorOutput,
  options: NanoDetDecodeOptions,
): DecodedDetection[] {
  validateDecodeOptions(options);
  const values = validateOutput(output);
  const candidates: DecodedDetection[] = [];

  for (let pointIndex = 0; pointIndex < NANODET_OUTPUT_POINTS; pointIndex += 1) {
    const offset = pointIndex * NANODET_OUTPUT_CHANNELS;
    const confidence = values[offset] ?? 0;
    if (!Number.isFinite(confidence) || confidence <= options.confidenceThreshold) {
      continue;
    }

    const prior = CENTER_PRIORS[pointIndex];
    if (prior === undefined) {
      throw new Error(`Missing NanoDet center prior at output point ${pointIndex}.`);
    }
    const left = distributionExpectation(values, offset + 1) * prior.stride;
    const top = distributionExpectation(values, offset + 9) * prior.stride;
    const right = distributionExpectation(values, offset + 17) * prior.stride;
    const bottom = distributionExpectation(values, offset + 25) * prior.stride;
    const x1 = clamp(prior.x - left, 0, NANODET_INPUT_SIZE);
    const y1 = clamp(prior.y - top, 0, NANODET_INPUT_SIZE);
    const x2 = clamp(prior.x + right, 0, NANODET_INPUT_SIZE);
    const y2 = clamp(prior.y + bottom, 0, NANODET_INPUT_SIZE);
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    candidates.push({
      id: `nanodet:${pointIndex}`,
      detectionIndex: pointIndex,
      classIndex: 0,
      confidence,
      box: { x: x1, y: y1, width: x2 - x1, height: y2 - y1 },
    });
  }

  return nonMaximumSuppression(
    candidates,
    options.nmsIouThreshold,
    options.maximumDetections,
  );
}

export function nonMaximumSuppression(
  detections: readonly DecodedDetection[],
  iouThreshold: number,
  maximumDetections: number,
): DecodedDetection[] {
  assertUnitInterval(iouThreshold, 'NanoDet NMS IoU threshold');
  if (!Number.isInteger(maximumDetections) || maximumDetections < 1) {
    throw new Error('NanoDet maximum detections must be a positive integer.');
  }

  const retained: DecodedDetection[] = [];
  const sorted = [...detections].sort(
    (left, right) =>
      right.confidence - left.confidence ||
      left.detectionIndex - right.detectionIndex ||
      left.id.localeCompare(right.id),
  );
  for (const candidate of sorted) {
    if (retained.some((accepted) => intersectionOverUnion(candidate.box, accepted.box) > iouThreshold)) {
      continue;
    }
    retained.push(candidate);
    if (retained.length === maximumDetections) {
      break;
    }
  }
  return retained;
}

function validateOutput(output: TensorOutput): NanoDetFloatArray {
  const expected = [1, NANODET_OUTPUT_POINTS, NANODET_OUTPUT_CHANNELS];
  const dims = output.dims.map(Number);
  if (
    dims.length !== expected.length ||
    dims.some((dimension, index) => dimension !== expected[index])
  ) {
    throw new Error(
      `Unexpected NanoDet output shape [${dims.join(', ')}], expected [${expected.join(', ')}].`,
    );
  }
  if (!(output.data instanceof Float32Array)) {
    throw new Error(`Unexpected NanoDet output type: ${output.type}.`);
  }
  if (output.data.length !== NANODET_OUTPUT_POINTS * NANODET_OUTPUT_CHANNELS) {
    throw new Error('NanoDet output data length does not match its declared shape.');
  }
  return output.data;
}

function validateDecodeOptions(options: NanoDetDecodeOptions): void {
  assertUnitInterval(options.confidenceThreshold, 'NanoDet confidence threshold');
  assertUnitInterval(options.nmsIouThreshold, 'NanoDet NMS IoU threshold');
  if (!Number.isInteger(options.maximumDetections) || options.maximumDetections < 1) {
    throw new Error('NanoDet maximum detections must be a positive integer.');
  }
}

function buildCenterPriors(): CenterPrior[] {
  const priors: CenterPrior[] = [];
  for (const stride of STRIDES) {
    const featureSize = Math.ceil(NANODET_INPUT_SIZE / stride);
    for (let row = 0; row < featureSize; row += 1) {
      for (let column = 0; column < featureSize; column += 1) {
        priors.push({ x: column * stride, y: row * stride, stride });
      }
    }
  }
  if (priors.length !== NANODET_OUTPUT_POINTS) {
    throw new Error(
      `Generated ${priors.length} NanoDet center priors, expected ${NANODET_OUTPUT_POINTS}.`,
    );
  }
  return priors;
}

function distributionExpectation(values: NanoDetFloatArray, offset: number): number {
  let maximum = Number.NEGATIVE_INFINITY;
  for (let bin = 0; bin <= REG_MAX; bin += 1) {
    maximum = Math.max(maximum, values[offset + bin] ?? Number.NEGATIVE_INFINITY);
  }
  let denominator = 0;
  let numerator = 0;
  for (let bin = 0; bin <= REG_MAX; bin += 1) {
    const weight = Math.exp((values[offset + bin] ?? Number.NEGATIVE_INFINITY) - maximum);
    denominator += weight;
    numerator += weight * bin;
  }
  return denominator === 0 ? 0 : numerator / denominator;
}

function intersectionOverUnion(left: Rect, right: Rect): number {
  const intersectionWidth = Math.max(
    0,
    Math.min(left.x + left.width, right.x + right.width) - Math.max(left.x, right.x),
  );
  const intersectionHeight = Math.max(
    0,
    Math.min(left.y + left.height, right.y + right.height) - Math.max(left.y, right.y),
  );
  const intersection = intersectionWidth * intersectionHeight;
  if (intersection <= 0) {
    return 0;
  }
  const union = left.width * left.height + right.width * right.height - intersection;
  return union <= 0 ? 0 : intersection / union;
}

function assertUnitInterval(value: number, label: string): void {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error(`${label} must be in [0, 1].`);
  }
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}
