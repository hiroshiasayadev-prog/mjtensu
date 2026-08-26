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

export interface TensorOutput {
  dims: readonly number[];
  data: unknown;
  type: string;
}

interface Prior { x: number; y: number; stride: number }
const PRIORS = buildPriors();

export function createInputBuffer(): Float32Array {
  return new Float32Array(3 * PIXEL_COUNT);
}

export function preprocessComposite(
  composite: HTMLCanvasElement,
  target: Float32Array,
): Float32Array {
  if (composite.width !== INPUT_SIZE || composite.height !== INPUT_SIZE) {
    throw new Error(`Composite must be ${INPUT_SIZE} x ${INPUT_SIZE}.`);
  }
  if (target.length !== 3 * PIXEL_COUNT) {
    throw new Error(`Input buffer length must be ${3 * PIXEL_COUNT}.`);
  }
  const context = composite.getContext('2d', { alpha: false, willReadFrequently: true });
  if (context === null) throw new Error('Composite 2D context is unavailable.');
  const rgba = context.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  for (let pixel = 0, rgbaIndex = 0; pixel < PIXEL_COUNT; pixel += 1, rgbaIndex += 4) {
    const red = rgba[rgbaIndex] ?? 0;
    const green = rgba[rgbaIndex + 1] ?? 0;
    const blue = rgba[rgbaIndex + 2] ?? 0;
    target[pixel] = (blue - BGR_MEAN[0]) / BGR_STD[0];
    target[PIXEL_COUNT + pixel] = (green - BGR_MEAN[1]) / BGR_STD[1];
    target[2 * PIXEL_COUNT + pixel] = (red - BGR_MEAN[2]) / BGR_STD[2];
  }
  return target;
}

export function decodeNanoDetOutput(output: TensorOutput, threshold: number): Detection[] {
  validateOutput(output);
  const values = output.data as Float32Array;
  const candidates: Detection[] = [];
  for (let pointIndex = 0; pointIndex < OUTPUT_POINTS; pointIndex += 1) {
    const offset = pointIndex * OUTPUT_CHANNELS;
    const score = values[offset] ?? 0;
    if (score <= threshold) continue;
    const prior = PRIORS[pointIndex];
    if (prior === undefined) throw new Error(`Missing center prior ${pointIndex}.`);
    const left = expectation(values, offset + 1) * prior.stride;
    const top = expectation(values, offset + 9) * prior.stride;
    const right = expectation(values, offset + 17) * prior.stride;
    const bottom = expectation(values, offset + 25) * prior.stride;
    const x1 = clamp(prior.x - left, 0, INPUT_SIZE);
    const y1 = clamp(prior.y - top, 0, INPUT_SIZE);
    const x2 = clamp(prior.x + right, 0, INPUT_SIZE);
    const y2 = clamp(prior.y + bottom, 0, INPUT_SIZE);
    if (x2 > x1 && y2 > y1) candidates.push({ x1, y1, x2, y2, score });
  }
  return nms(candidates, NMS_IOU_THRESHOLD, MAX_DETECTIONS);
}

function validateOutput(output: TensorOutput): void {
  const dims = output.dims.map(Number);
  if (dims.length !== 3 || dims[0] !== 1 || dims[1] !== OUTPUT_POINTS || dims[2] !== OUTPUT_CHANNELS) {
    throw new Error(`Unexpected output shape [${dims.join(', ')}].`);
  }
  if (!(output.data instanceof Float32Array)) throw new Error(`Unexpected output type: ${output.type}.`);
}

function buildPriors(): Prior[] {
  const priors: Prior[] = [];
  for (const stride of STRIDES) {
    const length = Math.ceil(INPUT_SIZE / stride);
    for (let row = 0; row < length; row += 1) {
      for (let column = 0; column < length; column += 1) {
        priors.push({ x: column * stride, y: row * stride, stride });
      }
    }
  }
  if (priors.length !== OUTPUT_POINTS) throw new Error(`Expected ${OUTPUT_POINTS} priors, got ${priors.length}.`);
  return priors;
}

function expectation(values: Float32Array, offset: number): number {
  let maximum = Number.NEGATIVE_INFINITY;
  for (let bin = 0; bin <= REG_MAX; bin += 1) maximum = Math.max(maximum, values[offset + bin] ?? maximum);
  let denominator = 0;
  let numerator = 0;
  for (let bin = 0; bin <= REG_MAX; bin += 1) {
    const weight = Math.exp((values[offset + bin] ?? Number.NEGATIVE_INFINITY) - maximum);
    denominator += weight;
    numerator += weight * bin;
  }
  return denominator === 0 ? 0 : numerator / denominator;
}

function nms(detections: Detection[], threshold: number, maximum: number): Detection[] {
  const retained: Detection[] = [];
  for (const candidate of [...detections].sort((left, right) => right.score - left.score)) {
    if (retained.some((accepted) => iou(candidate, accepted) > threshold)) continue;
    retained.push(candidate);
    if (retained.length >= maximum) break;
  }
  return retained;
}

function iou(left: Detection, right: Detection): number {
  const width = Math.max(0, Math.min(left.x2, right.x2) - Math.max(left.x1, right.x1));
  const height = Math.max(0, Math.min(left.y2, right.y2) - Math.max(left.y1, right.y1));
  const intersection = width * height;
  if (intersection <= 0) return 0;
  const leftArea = (left.x2 - left.x1) * (left.y2 - left.y1);
  const rightArea = (right.x2 - right.x1) * (right.y2 - right.y1);
  return intersection / (leftArea + rightArea - intersection);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}
