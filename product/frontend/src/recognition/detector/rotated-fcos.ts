import type { OrientedRect, TensorOutput } from './types';

export const ROTATED_FCOS_INPUT_SIZE = 320;
export const ROTATED_FCOS_OUTPUT_CHANNELS = 8;
export const ROTATED_FCOS_STRIDES = [8, 16, 32] as const;
export const ROTATED_FCOS_OUTPUT_NAMES = ['stride8', 'stride16', 'stride32'] as const;

const PIXEL_COUNT = ROTATED_FCOS_INPUT_SIZE * ROTATED_FCOS_INPUT_SIZE;
const IMAGE_MEAN = [0.485, 0.456, 0.406] as const;
const IMAGE_STD = [0.229, 0.224, 0.225] as const;
const PRE_NMS_TOP_K = 300;

export interface RotatedFcosDecodeOptions {
  readonly confidenceThreshold: number;
  readonly nmsIouThreshold: number;
  readonly maximumDetections: number;
}

export interface RotatedFcosDetection {
  readonly id: string;
  readonly detectionIndex: number;
  readonly classIndex: 0;
  readonly confidence: number;
  readonly orientedBox: OrientedRect;
}

type Candidate = RotatedFcosDetection;

export function preprocessRotatedFcosCompositeRgba(
  rgba: Uint8ClampedArray,
  target: Float32Array = new Float32Array(3 * PIXEL_COUNT),
): Float32Array {
  if (rgba.length !== 4 * PIXEL_COUNT) {
    throw new Error(`Composite RGBA buffer length must be ${4 * PIXEL_COUNT}.`);
  }
  if (target.length !== 3 * PIXEL_COUNT) {
    throw new Error(`Rotated FCOS input buffer length must be ${3 * PIXEL_COUNT}.`);
  }

  for (
    let pixelIndex = 0, rgbaIndex = 0;
    pixelIndex < PIXEL_COUNT;
    pixelIndex += 1, rgbaIndex += 4
  ) {
    const red = (rgba[rgbaIndex] ?? 0) / 255;
    const green = (rgba[rgbaIndex + 1] ?? 0) / 255;
    const blue = (rgba[rgbaIndex + 2] ?? 0) / 255;
    target[pixelIndex] = (red - IMAGE_MEAN[0]) / IMAGE_STD[0];
    target[PIXEL_COUNT + pixelIndex] = (green - IMAGE_MEAN[1]) / IMAGE_STD[1];
    target[2 * PIXEL_COUNT + pixelIndex] = (blue - IMAGE_MEAN[2]) / IMAGE_STD[2];
  }
  return target;
}

export function preprocessRotatedFcosCompositeCanvas(
  composite: HTMLCanvasElement,
  target?: Float32Array,
): Float32Array {
  if (
    composite.width !== ROTATED_FCOS_INPUT_SIZE ||
    composite.height !== ROTATED_FCOS_INPUT_SIZE
  ) {
    throw new Error(
      `Recognition composite must be ${ROTATED_FCOS_INPUT_SIZE} x ${ROTATED_FCOS_INPUT_SIZE}.`,
    );
  }
  const context = composite.getContext('2d', {
    alpha: false,
    willReadFrequently: true,
  });
  if (context === null) {
    throw new Error('Recognition composite 2D context is unavailable.');
  }
  return preprocessRotatedFcosCompositeRgba(
    context.getImageData(0, 0, ROTATED_FCOS_INPUT_SIZE, ROTATED_FCOS_INPUT_SIZE).data,
    target,
  );
}

export function decodeRotatedFcosOutputs(
  outputs: readonly TensorOutput[],
  options: RotatedFcosDecodeOptions,
): RotatedFcosDetection[] {
  validateDecodeOptions(options);
  if (outputs.length !== ROTATED_FCOS_STRIDES.length) {
    throw new Error(
      `Rotated FCOS requires ${ROTATED_FCOS_STRIDES.length} output levels, found ${outputs.length}.`,
    );
  }

  const candidates: Candidate[] = [];
  let globalPointIndex = 0;
  for (let level = 0; level < outputs.length; level += 1) {
    const output = outputs[level];
    const stride = ROTATED_FCOS_STRIDES[level];
    if (output === undefined || stride === undefined) {
      throw new Error(`Missing Rotated FCOS output level ${level}.`);
    }
    const { values, height, width } = validateOutput(output, stride);
    const plane = height * width;

    for (let row = 0; row < height; row += 1) {
      for (let column = 0; column < width; column += 1) {
        const pointIndex = row * width + column;
        const objectness = sigmoid(values[pointIndex] ?? Number.NaN);
        const centerness = sigmoid(values[plane + pointIndex] ?? Number.NaN);
        const confidence = objectness * centerness;
        if (!Number.isFinite(confidence) || confidence < options.confidenceThreshold) {
          globalPointIndex += 1;
          continue;
        }

        const dx = values[2 * plane + pointIndex] ?? Number.NaN;
        const dy = values[3 * plane + pointIndex] ?? Number.NaN;
        const logWidth = values[4 * plane + pointIndex] ?? Number.NaN;
        const logHeight = values[5 * plane + pointIndex] ?? Number.NaN;
        const sin2Theta = values[6 * plane + pointIndex] ?? Number.NaN;
        const cos2Theta = values[7 * plane + pointIndex] ?? Number.NaN;
        if (
          ![dx, dy, logWidth, logHeight, sin2Theta, cos2Theta].every(Number.isFinite)
        ) {
          globalPointIndex += 1;
          continue;
        }

        const pointX = (column + 0.5) * stride;
        const pointY = (row + 0.5) * stride;
        const orientedBox = canonicalizeOrientedRect({
          cx: pointX + dx * stride,
          cy: pointY + dy * stride,
          width: Math.exp(clamp(logWidth, -4, 4)) * stride,
          height: Math.exp(clamp(logHeight, -4, 4)) * stride,
          angleDeg: (0.5 * Math.atan2(sin2Theta, cos2Theta) * 180) / Math.PI,
        });
        candidates.push({
          id: `rotated-fcos:${level}:${pointIndex}`,
          detectionIndex: globalPointIndex,
          classIndex: 0,
          confidence,
          orientedBox,
        });
        globalPointIndex += 1;
      }
    }
  }

  const ranked = candidates.sort(compareCandidates).slice(0, PRE_NMS_TOP_K);
  return rotatedNonMaximumSuppression(
    ranked,
    options.nmsIouThreshold,
    options.maximumDetections,
  );
}

export function rotatedNonMaximumSuppression(
  detections: readonly RotatedFcosDetection[],
  iouThreshold: number,
  maximumDetections: number,
): RotatedFcosDetection[] {
  assertUnitInterval(iouThreshold, 'Rotated FCOS NMS IoU threshold');
  if (!Number.isInteger(maximumDetections) || maximumDetections < 1) {
    throw new Error('Rotated FCOS maximum detections must be a positive integer.');
  }

  const remaining = [...detections].sort(compareCandidates);
  const retained: RotatedFcosDetection[] = [];
  while (remaining.length > 0 && retained.length < maximumDetections) {
    const winner = remaining.shift();
    if (winner === undefined) {
      break;
    }
    retained.push(winner);
    for (let index = remaining.length - 1; index >= 0; index -= 1) {
      const candidate = remaining[index];
      if (
        candidate !== undefined &&
        rotatedIntersectionOverUnion(winner.orientedBox, candidate.orientedBox) >= iouThreshold
      ) {
        remaining.splice(index, 1);
      }
    }
  }
  return retained;
}

export function canonicalizeOrientedRect(box: OrientedRect): OrientedRect {
  let { width, height, angleDeg } = box;
  if (width > height) {
    [width, height] = [height, width];
    angleDeg += 90;
  }
  return {
    cx: box.cx,
    cy: box.cy,
    width,
    height,
    angleDeg: normalizeAngle180(angleDeg),
  };
}

export function orientedRectToAabb(box: OrientedRect): {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
} {
  const corners = orientedRectCorners(box);
  const xs = corners.map(([x]) => x);
  const ys = corners.map(([, y]) => y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

export function rotatedIntersectionOverUnion(
  left: OrientedRect,
  right: OrientedRect,
): number {
  const leftPolygon = orientedRectCorners(left);
  const rightPolygon = orientedRectCorners(right);
  const intersectionPolygon = polygonClip(leftPolygon, rightPolygon);
  const intersection = polygonArea(intersectionPolygon);
  const leftArea = Math.max(0, left.width) * Math.max(0, left.height);
  const rightArea = Math.max(0, right.width) * Math.max(0, right.height);
  const union = leftArea + rightArea - intersection;
  return union <= 0 ? 0 : intersection / union;
}

function validateOutput(
  output: TensorOutput,
  stride: number,
): { readonly values: Float32Array; readonly height: number; readonly width: number } {
  const expectedSpatial = ROTATED_FCOS_INPUT_SIZE / stride;
  const dims = output.dims.map(Number);
  if (
    dims.length !== 4 ||
    dims[0] !== 1 ||
    dims[1] !== ROTATED_FCOS_OUTPUT_CHANNELS ||
    dims[2] !== expectedSpatial ||
    dims[3] !== expectedSpatial ||
    !(output.data instanceof Float32Array) ||
    output.data.length !== ROTATED_FCOS_OUTPUT_CHANNELS * expectedSpatial * expectedSpatial
  ) {
    throw new Error(
      `Invalid Rotated FCOS stride-${stride} output contract: dims=${JSON.stringify(dims)} type=${output.type}.`,
    );
  }
  return { values: output.data, height: expectedSpatial, width: expectedSpatial };
}

function validateDecodeOptions(options: RotatedFcosDecodeOptions): void {
  assertUnitInterval(options.confidenceThreshold, 'Rotated FCOS confidence threshold');
  assertUnitInterval(options.nmsIouThreshold, 'Rotated FCOS NMS IoU threshold');
  if (!Number.isInteger(options.maximumDetections) || options.maximumDetections < 1) {
    throw new Error('Rotated FCOS maximum detections must be a positive integer.');
  }
}

function compareCandidates(
  left: Pick<RotatedFcosDetection, 'confidence' | 'detectionIndex' | 'id'>,
  right: Pick<RotatedFcosDetection, 'confidence' | 'detectionIndex' | 'id'>,
): number {
  return (
    right.confidence - left.confidence ||
    left.detectionIndex - right.detectionIndex ||
    left.id.localeCompare(right.id)
  );
}

function sigmoid(value: number): number {
  if (value >= 0) {
    const exp = Math.exp(-value);
    return 1 / (1 + exp);
  }
  const exp = Math.exp(value);
  return exp / (1 + exp);
}

function normalizeAngle180(angleDeg: number): number {
  let angle = ((angleDeg + 90) % 180 + 180) % 180 - 90;
  if (angle >= 90) {
    angle -= 180;
  }
  return angle;
}

function orientedRectCorners(box: OrientedRect): Array<readonly [number, number]> {
  const angle = (box.angleDeg * Math.PI) / 180;
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  const halfWidth = box.width / 2;
  const halfHeight = box.height / 2;
  const localCorners: readonly (readonly [number, number])[] = [
    [-halfWidth, -halfHeight],
    [halfWidth, -halfHeight],
    [halfWidth, halfHeight],
    [-halfWidth, halfHeight],
  ];
  return localCorners.map(([localX, localY]) => [
    box.cx + localX * cosine - localY * sine,
    box.cy + localX * sine + localY * cosine,
  ] as const);
}

function polygonArea(polygon: readonly (readonly [number, number])[]): number {
  if (polygon.length < 3) {
    return 0;
  }
  let sum = 0;
  for (let index = 0; index < polygon.length; index += 1) {
    const current = polygon[index];
    const next = polygon[(index + 1) % polygon.length];
    if (current === undefined || next === undefined) {
      continue;
    }
    sum += current[0] * next[1] - next[0] * current[1];
  }
  return Math.abs(sum) / 2;
}

function signedPolygonArea(polygon: readonly (readonly [number, number])[]): number {
  let sum = 0;
  for (let index = 0; index < polygon.length; index += 1) {
    const current = polygon[index];
    const next = polygon[(index + 1) % polygon.length];
    if (current === undefined || next === undefined) {
      continue;
    }
    sum += current[0] * next[1] - next[0] * current[1];
  }
  return sum / 2;
}

function polygonClip(
  subject: readonly (readonly [number, number])[],
  clipPolygon: readonly (readonly [number, number])[],
): Array<readonly [number, number]> {
  let output = [...subject];
  if (output.length === 0) {
    return [];
  }
  const orientation = signedPolygonArea(clipPolygon);
  for (let index = 0; index < clipPolygon.length; index += 1) {
    const edgeStart = clipPolygon[index];
    const edgeEnd = clipPolygon[(index + 1) % clipPolygon.length];
    if (edgeStart === undefined || edgeEnd === undefined) {
      continue;
    }
    const input = output;
    output = [];
    if (input.length === 0) {
      break;
    }
    let previous = input[input.length - 1];
    if (previous === undefined) {
      continue;
    }
    let previousInside = pointInsideEdge(previous, edgeStart, edgeEnd, orientation);
    for (const current of input) {
      const currentInside = pointInsideEdge(current, edgeStart, edgeEnd, orientation);
      if (currentInside) {
        if (!previousInside) {
          output.push(lineIntersection(previous, current, edgeStart, edgeEnd));
        }
        output.push(current);
      } else if (previousInside) {
        output.push(lineIntersection(previous, current, edgeStart, edgeEnd));
      }
      previous = current;
      previousInside = currentInside;
    }
  }
  return output;
}

function pointInsideEdge(
  point: readonly [number, number],
  edgeStart: readonly [number, number],
  edgeEnd: readonly [number, number],
  orientation: number,
): boolean {
  const cross =
    (edgeEnd[0] - edgeStart[0]) * (point[1] - edgeStart[1]) -
    (edgeEnd[1] - edgeStart[1]) * (point[0] - edgeStart[0]);
  return orientation >= 0 ? cross >= -1e-9 : cross <= 1e-9;
}

function lineIntersection(
  firstStart: readonly [number, number],
  firstEnd: readonly [number, number],
  secondStart: readonly [number, number],
  secondEnd: readonly [number, number],
): readonly [number, number] {
  const [x1, y1] = firstStart;
  const [x2, y2] = firstEnd;
  const [x3, y3] = secondStart;
  const [x4, y4] = secondEnd;
  const denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  if (Math.abs(denominator) < 1e-12) {
    return firstEnd;
  }
  const determinantFirst = x1 * y2 - y1 * x2;
  const determinantSecond = x3 * y4 - y3 * x4;
  return [
    (determinantFirst * (x3 - x4) - (x1 - x2) * determinantSecond) / denominator,
    (determinantFirst * (y3 - y4) - (y1 - y2) * determinantSecond) / denominator,
  ];
}

function assertUnitInterval(value: number, label: string): void {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error(`${label} must be within [0, 1].`);
  }
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}
