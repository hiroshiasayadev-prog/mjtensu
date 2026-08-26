export type ClassifierCropChannels = 1 | 3 | 4;

export interface ClassifierCropImage {
  readonly width: number;
  readonly height: number;
  readonly data: ArrayLike<number>;
  readonly channels?: ClassifierCropChannels;
}

export interface ClassifierNormalization {
  readonly mean: readonly number[];
  readonly std: readonly number[];
}

export interface ClassifierTensor {
  readonly data: Float32Array;
  readonly shape: readonly [number, number, number, number];
}

export const CLASSIFIER_IMAGE_SIZE = 64;

const LANCZOS_RADIUS = 3;

function validateCrop(crop: ClassifierCropImage): ClassifierCropChannels {
  const channels = crop.channels ?? 4;
  if (channels !== 1 && channels !== 3 && channels !== 4) {
    throw new Error(`Unsupported crop channel count: ${channels}`);
  }
  if (!Number.isInteger(crop.width) || crop.width <= 0) {
    throw new Error(`Invalid crop width: ${crop.width}`);
  }
  if (!Number.isInteger(crop.height) || crop.height <= 0) {
    throw new Error(`Invalid crop height: ${crop.height}`);
  }
  if (crop.data.length !== crop.width * crop.height * channels) {
    throw new Error(
      `Crop buffer length ${crop.data.length} does not match ${crop.width}x${crop.height}x${channels}`,
    );
  }
  return channels;
}

function validateImageSize(imageSize: number): void {
  if (!Number.isInteger(imageSize) || imageSize < 1) {
    throw new Error(`Invalid classifier image size: ${imageSize}`);
  }
}

function clampByte(value: number): number {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function median(values: readonly number[]): number {
  if (values.length === 0) {
    return 127;
  }

  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) {
    return sorted[middle] ?? 127;
  }

  return ((sorted[middle - 1] ?? 127) + (sorted[middle] ?? 127)) / 2;
}

function pixelOffset(
  width: number,
  x: number,
  y: number,
  channels: ClassifierCropChannels,
): number {
  return (y * width + x) * channels;
}

function readRgbPixel(
  crop: ClassifierCropImage,
  channels: ClassifierCropChannels,
  x: number,
  y: number,
): readonly [number, number, number] {
  const offset = pixelOffset(crop.width, x, y, channels);
  if (channels === 1) {
    const value = crop.data[offset] ?? 0;
    return [value, value, value];
  }

  return [
    crop.data[offset] ?? 0,
    crop.data[offset + 1] ?? 0,
    crop.data[offset + 2] ?? 0,
  ];
}

function readGrayPixel(
  crop: ClassifierCropImage,
  channels: ClassifierCropChannels,
  x: number,
  y: number,
): number {
  const offset = pixelOffset(crop.width, x, y, channels);
  if (channels === 1) {
    return crop.data[offset] ?? 0;
  }

  const red = crop.data[offset] ?? 0;
  const green = crop.data[offset + 1] ?? 0;
  const blue = crop.data[offset + 2] ?? 0;
  return clampByte(0.299 * red + 0.587 * green + 0.114 * blue);
}

function collectBorderCoordinates(width: number, height: number): [number, number][] {
  const coordinates: [number, number][] = [];
  if (width === 1 || height === 1) {
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        coordinates.push([x, y]);
      }
    }
    return coordinates;
  }

  for (let x = 0; x < width; x += 1) {
    coordinates.push([x, 0], [x, height - 1]);
  }
  for (let y = 1; y < height - 1; y += 1) {
    coordinates.push([0, y], [width - 1, y]);
  }
  return coordinates;
}

function grayBorderFill(
  crop: ClassifierCropImage,
  channels: ClassifierCropChannels,
): number {
  const values = collectBorderCoordinates(crop.width, crop.height).map(([x, y]) =>
    readGrayPixel(crop, channels, x, y),
  );
  return clampByte(median(values));
}

function rgbBorderFill(
  crop: ClassifierCropImage,
  channels: ClassifierCropChannels,
): readonly [number, number, number] {
  const red: number[] = [];
  const green: number[] = [];
  const blue: number[] = [];

  for (const [x, y] of collectBorderCoordinates(crop.width, crop.height)) {
    const pixel = readRgbPixel(crop, channels, x, y);
    red.push(pixel[0]);
    green.push(pixel[1]);
    blue.push(pixel[2]);
  }

  return [clampByte(median(red)), clampByte(median(green)), clampByte(median(blue))];
}

function lanczos(value: number): number {
  const distance = Math.abs(value);
  if (distance < 1e-7) {
    return 1;
  }
  if (distance >= LANCZOS_RADIUS) {
    return 0;
  }

  const piX = Math.PI * distance;
  return (
    (Math.sin(piX) / piX) *
    (Math.sin(piX / LANCZOS_RADIUS) / (piX / LANCZOS_RADIUS))
  );
}

function resampleChannel(
  source: Uint8Array,
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number,
): Uint8Array {
  if (sourceWidth === targetWidth && sourceHeight === targetHeight) {
    return new Uint8Array(source);
  }

  const target = new Uint8Array(targetWidth * targetHeight);
  const scaleX = sourceWidth / targetWidth;
  const scaleY = sourceHeight / targetHeight;

  for (let targetY = 0; targetY < targetHeight; targetY += 1) {
    const sourceY = (targetY + 0.5) * scaleY - 0.5;
    const minY = Math.ceil(sourceY - LANCZOS_RADIUS + 1);
    const maxY = Math.floor(sourceY + LANCZOS_RADIUS);

    for (let targetX = 0; targetX < targetWidth; targetX += 1) {
      const sourceX = (targetX + 0.5) * scaleX - 0.5;
      const minX = Math.ceil(sourceX - LANCZOS_RADIUS + 1);
      const maxX = Math.floor(sourceX + LANCZOS_RADIUS);
      let weightedSum = 0;
      let weightSum = 0;

      for (let y = minY; y <= maxY; y += 1) {
        const clampedY = Math.max(0, Math.min(sourceHeight - 1, y));
        const yWeight = lanczos(sourceY - y);
        for (let x = minX; x <= maxX; x += 1) {
          const clampedX = Math.max(0, Math.min(sourceWidth - 1, x));
          const weight = yWeight * lanczos(sourceX - x);
          weightedSum += source[clampedY * sourceWidth + clampedX] * weight;
          weightSum += weight;
        }
      }

      target[targetY * targetWidth + targetX] = clampByte(
        weightSum === 0 ? 0 : weightedSum / weightSum,
      );
    }
  }

  return target;
}

function resizedDimensions(
  sourceWidth: number,
  sourceHeight: number,
  imageSize: number,
): readonly [number, number] {
  const scale = Math.min(imageSize / sourceWidth, imageSize / sourceHeight);
  return [
    Math.max(1, Math.min(imageSize, Math.floor(sourceWidth * scale + 0.5))),
    Math.max(1, Math.min(imageSize, Math.floor(sourceHeight * scale + 0.5))),
  ];
}

function normalizeToTensor(
  channels: readonly Uint8Array[],
  imageSize: number,
  normalization: ClassifierNormalization,
): ClassifierTensor {
  if (
    normalization.mean.length !== channels.length ||
    normalization.std.length !== channels.length
  ) {
    throw new Error(
      `Normalization has ${normalization.mean.length}/${normalization.std.length} channels; expected ${channels.length}`,
    );
  }
  if (normalization.std.some((value) => value <= 0)) {
    throw new Error('Normalization std values must be positive');
  }

  const planeLength = imageSize * imageSize;
  const tensor = new Float32Array(channels.length * planeLength);

  for (let channelIndex = 0; channelIndex < channels.length; channelIndex += 1) {
    const channel = channels[channelIndex];
    const mean = normalization.mean[channelIndex] ?? 0;
    const std = normalization.std[channelIndex] ?? 1;
    const channelOffset = channelIndex * planeLength;
    for (let index = 0; index < planeLength; index += 1) {
      tensor[channelOffset + index] = ((channel[index] ?? 0) / 255 - mean) / std;
    }
  }

  return {
    data: tensor,
    shape: [1, channels.length, imageSize, imageSize],
  };
}

export function preprocessGrayClassifierCrop(
  crop: ClassifierCropImage,
  imageSize = CLASSIFIER_IMAGE_SIZE,
): Uint8Array {
  const channels = validateCrop(crop);
  validateImageSize(imageSize);
  const [resizedWidth, resizedHeight] = resizedDimensions(
    crop.width,
    crop.height,
    imageSize,
  );
  const graySource = new Uint8Array(crop.width * crop.height);

  for (let y = 0; y < crop.height; y += 1) {
    for (let x = 0; x < crop.width; x += 1) {
      graySource[y * crop.width + x] = readGrayPixel(crop, channels, x, y);
    }
  }

  const resized = resampleChannel(
    graySource,
    crop.width,
    crop.height,
    resizedWidth,
    resizedHeight,
  );
  const canvas = new Uint8Array(imageSize * imageSize);
  canvas.fill(grayBorderFill(crop, channels));
  const offsetX = Math.floor((imageSize - resizedWidth) / 2);
  const offsetY = Math.floor((imageSize - resizedHeight) / 2);

  for (let y = 0; y < resizedHeight; y += 1) {
    canvas.set(
      resized.subarray(y * resizedWidth, (y + 1) * resizedWidth),
      (offsetY + y) * imageSize + offsetX,
    );
  }

  return canvas;
}

export function preprocessRgbClassifierCrop(
  crop: ClassifierCropImage,
  imageSize = CLASSIFIER_IMAGE_SIZE,
): Uint8Array {
  const channels = validateCrop(crop);
  validateImageSize(imageSize);
  const [resizedWidth, resizedHeight] = resizedDimensions(
    crop.width,
    crop.height,
    imageSize,
  );
  const sourceChannels = [
    new Uint8Array(crop.width * crop.height),
    new Uint8Array(crop.width * crop.height),
    new Uint8Array(crop.width * crop.height),
  ];

  for (let y = 0; y < crop.height; y += 1) {
    for (let x = 0; x < crop.width; x += 1) {
      const sourceOffset = y * crop.width + x;
      const pixel = readRgbPixel(crop, channels, x, y);
      sourceChannels[0][sourceOffset] = pixel[0];
      sourceChannels[1][sourceOffset] = pixel[1];
      sourceChannels[2][sourceOffset] = pixel[2];
    }
  }

  const resizedChannels = sourceChannels.map((channel) =>
    resampleChannel(channel, crop.width, crop.height, resizedWidth, resizedHeight),
  );
  const fill = rgbBorderFill(crop, channels);
  const canvas = new Uint8Array(imageSize * imageSize * 3);
  const offsetX = Math.floor((imageSize - resizedWidth) / 2);
  const offsetY = Math.floor((imageSize - resizedHeight) / 2);

  for (let index = 0; index < imageSize * imageSize; index += 1) {
    canvas[index * 3] = fill[0];
    canvas[index * 3 + 1] = fill[1];
    canvas[index * 3 + 2] = fill[2];
  }
  for (let y = 0; y < resizedHeight; y += 1) {
    for (let x = 0; x < resizedWidth; x += 1) {
      const targetOffset = ((offsetY + y) * imageSize + offsetX + x) * 3;
      const sourceOffset = y * resizedWidth + x;
      canvas[targetOffset] = resizedChannels[0]?.[sourceOffset] ?? 0;
      canvas[targetOffset + 1] = resizedChannels[1]?.[sourceOffset] ?? 0;
      canvas[targetOffset + 2] = resizedChannels[2]?.[sourceOffset] ?? 0;
    }
  }

  return canvas;
}

export function makeBaseClassifierTensor(
  crop: ClassifierCropImage,
  normalization: ClassifierNormalization,
  imageSize = CLASSIFIER_IMAGE_SIZE,
): ClassifierTensor {
  return normalizeToTensor(
    [preprocessGrayClassifierCrop(crop, imageSize)],
    imageSize,
    normalization,
  );
}

export function makeRedFiveClassifierTensor(
  crop: ClassifierCropImage,
  normalization: ClassifierNormalization,
  imageSize = CLASSIFIER_IMAGE_SIZE,
): ClassifierTensor {
  const rgb = preprocessRgbClassifierCrop(crop, imageSize);
  const planeLength = imageSize * imageSize;
  const channels = [
    new Uint8Array(planeLength),
    new Uint8Array(planeLength),
    new Uint8Array(planeLength),
  ];

  for (let index = 0; index < planeLength; index += 1) {
    channels[0][index] = rgb[index * 3] ?? 0;
    channels[1][index] = rgb[index * 3 + 1] ?? 0;
    channels[2][index] = rgb[index * 3 + 2] ?? 0;
  }

  return normalizeToTensor(channels, imageSize, normalization);
}
