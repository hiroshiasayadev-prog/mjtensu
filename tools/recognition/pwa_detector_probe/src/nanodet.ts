export const INPUT_SIZE = 320;
export const OUTPUT_POINTS = 2125;
export const OUTPUT_CHANNELS = 33;
export const REG_MAX = 7;
export const STRIDES = [8, 16, 32, 64] as const;
export const NMS_IOU_THRESHOLD = 0.6;
export const MAX_DETECTIONS = 200;

const PIXEL_COUNT = INPUT_SIZE * INPUT_SIZE;
const BGR_MEAN = [103.53, 116.28, 123.675] as const;
const BGR_STD = [57.375, 57.12, 58.395] as const;

export interface Detection {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
}

export interface CenteredSquareCrop {
  x: number;
  y: number;
  size: number;
}

interface Prior {
  x: number;
  y: number;
  stride: number;
}

export interface TensorOutput {
  dims: readonly number[];
  data: unknown;
  type: string;
}

const PRIORS = buildPriors();

export function createInputBuffer(): Float32Array {
  return new Float32Array(3 * PIXEL_COUNT);
}

export function centeredSquareCrop(
  sourceWidth: number,
  sourceHeight: number,
): CenteredSquareCrop {
  if (
    !Number.isFinite(sourceWidth) ||
    !Number.isFinite(sourceHeight) ||
    sourceWidth <= 0 ||
    sourceHeight <= 0
  ) {
    throw new Error(
      `Source dimensions must be positive finite values, received ${sourceWidth} × ${sourceHeight}.`,
    );
  }

  const size = Math.min(sourceWidth, sourceHeight);
  return {
    x: (sourceWidth - size) / 2,
    y: (sourceHeight - size) / 2,
    size,
  };
}

export function preprocessVideoFrame(
  video: HTMLVideoElement,
  context: CanvasRenderingContext2D,
  target: Float32Array,
): Float32Array {
  if (target.length !== 3 * PIXEL_COUNT) {
    throw new Error(`Input buffer length must be ${3 * PIXEL_COUNT}.`);
  }

  const crop = centeredSquareCrop(video.videoWidth, video.videoHeight);
  context.drawImage(
    video,
    crop.x,
    crop.y,
    crop.size,
    crop.size,
    0,
    0,
    INPUT_SIZE,
    INPUT_SIZE,
  );
  const rgba = context.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;

  for (let pixelIndex = 0, rgbaIndex = 0; pixelIndex < PIXEL_COUNT; pixelIndex += 1, rgbaIndex += 4) {
    const red = rgba[rgbaIndex] ?? 0;
    const green = rgba[rgbaIndex + 1] ?? 0;
    const blue = rgba[rgbaIndex + 2] ?? 0;

    target[pixelIndex] = (blue - BGR_MEAN[0]) / BGR_STD[0];
    target[PIXEL_COUNT + pixelIndex] = (green - BGR_MEAN[1]) / BGR_STD[1];
    target[2 * PIXEL_COUNT + pixelIndex] = (red - BGR_MEAN[2]) / BGR_STD[2];
  }

  return target;
}

export function decodeNanoDetOutput(output: TensorOutput, confidenceThreshold: number): Detection[] {
  validateOutput(output);
  const values = output.data as Float32Array;
  const candidates: Detection[] = [];

  for (let pointIndex = 0; pointIndex < OUTPUT_POINTS; pointIndex += 1) {
    const rowOffset = pointIndex * OUTPUT_CHANNELS;
    const score = values[rowOffset] ?? 0;
    if (score <= confidenceThreshold) {
      continue;
    }

    const prior = PRIORS[pointIndex];
    if (prior === undefined) {
      throw new Error(`Missing center prior at output point ${pointIndex}.`);
    }

    const left = distributionExpectation(values, rowOffset + 1) * prior.stride;
    const top = distributionExpectation(values, rowOffset + 1 + 8) * prior.stride;
    const right = distributionExpectation(values, rowOffset + 1 + 16) * prior.stride;
    const bottom = distributionExpectation(values, rowOffset + 1 + 24) * prior.stride;

    const x1 = clamp(prior.x - left, 0, INPUT_SIZE);
    const y1 = clamp(prior.y - top, 0, INPUT_SIZE);
    const x2 = clamp(prior.x + right, 0, INPUT_SIZE);
    const y2 = clamp(prior.y + bottom, 0, INPUT_SIZE);

    if (x2 > x1 && y2 > y1) {
      candidates.push({ x1, y1, x2, y2, score });
    }
  }

  return nonMaximumSuppression(candidates, NMS_IOU_THRESHOLD, MAX_DETECTIONS);
}

function validateOutput(output: TensorOutput): void {
  const dims = output.dims.map(Number);
  const expected = [1, OUTPUT_POINTS, OUTPUT_CHANNELS];
  if (dims.length !== expected.length || dims.some((dimension, index) => dimension !== expected[index])) {
    throw new Error(`Unexpected NanoDet output shape [${dims.join(', ')}], expected [${expected.join(', ')}].`);
  }
  if (!(output.data instanceof Float32Array)) {
    throw new Error(`Unexpected NanoDet output type: ${output.type}.`);
  }
}

function buildPriors(): Prior[] {
  const priors: Prior[] = [];
  for (const stride of STRIDES) {
    const featureWidth = Math.ceil(INPUT_SIZE / stride);
    const featureHeight = Math.ceil(INPUT_SIZE / stride);
    for (let yIndex = 0; yIndex < featureHeight; yIndex += 1) {
      for (let xIndex = 0; xIndex < featureWidth; xIndex += 1) {
        priors.push({
          x: xIndex * stride,
          y: yIndex * stride,
          stride,
        });
      }
    }
  }

  if (priors.length !== OUTPUT_POINTS) {
    throw new Error(`Generated ${priors.length} priors, expected ${OUTPUT_POINTS}.`);
  }
  return priors;
}

function distributionExpectation(values: Float32Array, offset: number): number {
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

function nonMaximumSuppression(
  detections: Detection[],
  iouThreshold: number,
  maximumDetections: number,
): Detection[] {
  const sorted = [...detections].sort((left, right) => right.score - left.score);
  const retained: Detection[] = [];

  for (const candidate of sorted) {
    let suppressed = false;
    for (const accepted of retained) {
      if (intersectionOverUnion(candidate, accepted) > iouThreshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      retained.push(candidate);
      if (retained.length >= maximumDetections) {
        break;
      }
    }
  }
  return retained;
}

function intersectionOverUnion(left: Detection, right: Detection): number {
  const intersectionWidth = Math.max(0, Math.min(left.x2, right.x2) - Math.max(left.x1, right.x1));
  const intersectionHeight = Math.max(0, Math.min(left.y2, right.y2) - Math.max(left.y1, right.y1));
  const intersectionArea = intersectionWidth * intersectionHeight;
  if (intersectionArea <= 0) {
    return 0;
  }

  const leftArea = (left.x2 - left.x1) * (left.y2 - left.y1);
  const rightArea = (right.x2 - right.x1) * (right.y2 - right.y1);
  const unionArea = leftArea + rightArea - intersectionArea;
  return unionArea <= 0 ? 0 : intersectionArea / unionArea;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}
