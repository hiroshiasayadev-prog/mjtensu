import {
  SEMANTIC_REGIONS,
  type CaptureRegions,
  type OrientedRect,
  type Rect,
  type SemanticRegion,
} from './types';

export const FIXED_COMPOSITE_LAYOUT = Object.freeze({
  width: 320,
  height: 320,
  paddingRgb: [0, 0, 0] as const,
  regions: Object.freeze({
    completed_hand: Object.freeze({ x: 7, y: 0, width: 306, height: 72 }),
    dora_indicators: Object.freeze({ x: 7, y: 74, width: 306, height: 72 }),
    melds: Object.freeze({ x: 74, y: 148, width: 172, height: 172 }),
  }),
});

const ASPECT_RATIO_TOLERANCE = 1e-6;

export function buildFixedComposite(
  source: CanvasImageSource,
  captureRegions: CaptureRegions,
): HTMLCanvasElement {
  validateCaptureRegions(captureRegions);

  const canvas = document.createElement('canvas');
  canvas.width = FIXED_COMPOSITE_LAYOUT.width;
  canvas.height = FIXED_COMPOSITE_LAYOUT.height;
  const context = canvas.getContext('2d', {
    alpha: false,
    willReadFrequently: true,
  });
  if (context === null) {
    throw new Error('Recognition composite 2D context is unavailable.');
  }

  const [red, green, blue] = FIXED_COMPOSITE_LAYOUT.paddingRgb;
  context.fillStyle = `rgb(${red}, ${green}, ${blue})`;
  context.fillRect(0, 0, canvas.width, canvas.height);

  for (const region of SEMANTIC_REGIONS) {
    const capture = captureRegions[region];
    if (!capture.enabled) {
      continue;
    }
    const destination = FIXED_COMPOSITE_LAYOUT.regions[region];
    const { sourceRect } = capture;
    context.drawImage(
      source,
      sourceRect.x,
      sourceRect.y,
      sourceRect.width,
      sourceRect.height,
      destination.x,
      destination.y,
      destination.width,
      destination.height,
    );
  }

  return canvas;
}

export function validateCaptureRegions(captureRegions: CaptureRegions): void {
  for (const region of SEMANTIC_REGIONS) {
    const source = captureRegions[region].sourceRect;
    assertRect(source, `${region} source rectangle`);
    if (!captureRegions[region].enabled) {
      continue;
    }
    assertCompatibleAspectRatio(source, region);
  }
}

export function assignSemanticRegion(
  box: Rect,
  captureRegions?: CaptureRegions,
): SemanticRegion | null {
  assertRect(box, 'detection rectangle');
  return assignSemanticRegionByCenter(
    box.x + box.width / 2,
    box.y + box.height / 2,
    captureRegions,
  );
}

export function assignSemanticRegionByCenter(
  centerX: number,
  centerY: number,
  captureRegions?: CaptureRegions,
): SemanticRegion | null {
  if (!Number.isFinite(centerX) || !Number.isFinite(centerY)) {
    throw new Error('Detection center must be finite.');
  }

  for (const region of SEMANTIC_REGIONS) {
    if (captureRegions !== undefined && !captureRegions[region].enabled) {
      continue;
    }
    if (containsPoint(FIXED_COMPOSITE_LAYOUT.regions[region], centerX, centerY)) {
      return region;
    }
  }
  return null;
}

export function sourceRectToComposite(
  sourceBox: Rect,
  region: SemanticRegion,
  captureRegions: CaptureRegions,
): Rect {
  const source = captureRegions[region].sourceRect;
  const destination = FIXED_COMPOSITE_LAYOUT.regions[region];
  assertRect(sourceBox, 'source detection rectangle');
  assertRect(source, `${region} source rectangle`);
  assertCompatibleAspectRatio(source, region);

  const clipped = intersectRects(sourceBox, source);
  const scaleX = destination.width / source.width;
  const scaleY = destination.height / source.height;
  return {
    x: destination.x + (clipped.x - source.x) * scaleX,
    y: destination.y + (clipped.y - source.y) * scaleY,
    width: clipped.width * scaleX,
    height: clipped.height * scaleY,
  };
}

export function compositeRectToSource(
  compositeBox: Rect,
  region: SemanticRegion,
  captureRegions: CaptureRegions,
): Rect {
  const source = captureRegions[region].sourceRect;
  const destination = FIXED_COMPOSITE_LAYOUT.regions[region];
  assertRect(compositeBox, 'composite detection rectangle');
  assertRect(source, `${region} source rectangle`);
  assertCompatibleAspectRatio(source, region);

  const clipped = intersectRects(compositeBox, destination);
  const scaleX = source.width / destination.width;
  const scaleY = source.height / destination.height;
  return {
    x: source.x + (clipped.x - destination.x) * scaleX,
    y: source.y + (clipped.y - destination.y) * scaleY,
    width: clipped.width * scaleX,
    height: clipped.height * scaleY,
  };
}

export function compositeOrientedRectToSource(
  compositeBox: OrientedRect,
  region: SemanticRegion,
  captureRegions: CaptureRegions,
): OrientedRect {
  const source = captureRegions[region].sourceRect;
  const destination = FIXED_COMPOSITE_LAYOUT.regions[region];
  assertOrientedRect(compositeBox, 'composite oriented detection rectangle');
  assertRect(source, `${region} source rectangle`);
  assertCompatibleAspectRatio(source, region);

  const scaleX = source.width / destination.width;
  const scaleY = source.height / destination.height;
  if (Math.abs(scaleX - scaleY) > ASPECT_RATIO_TOLERANCE) {
    throw new Error(`${region} composite mapping must use a uniform scale for OBB geometry.`);
  }
  const scale = (scaleX + scaleY) / 2;
  return {
    cx: source.x + (compositeBox.cx - destination.x) * scale,
    cy: source.y + (compositeBox.cy - destination.y) * scale,
    width: compositeBox.width * scale,
    height: compositeBox.height * scale,
    angleDeg: compositeBox.angleDeg,
  };
}

function containsPoint(rect: Rect, x: number, y: number): boolean {
  return (
    x >= rect.x &&
    y >= rect.y &&
    x < rect.x + rect.width &&
    y < rect.y + rect.height
  );
}

function intersectRects(left: Rect, right: Rect): Rect {
  const x = Math.max(left.x, right.x);
  const y = Math.max(left.y, right.y);
  const farX = Math.max(x, Math.min(left.x + left.width, right.x + right.width));
  const farY = Math.max(y, Math.min(left.y + left.height, right.y + right.height));
  return { x, y, width: farX - x, height: farY - y };
}

function assertOrientedRect(rect: OrientedRect, label: string): void {
  if (
    !Number.isFinite(rect.cx) ||
    !Number.isFinite(rect.cy) ||
    !Number.isFinite(rect.width) ||
    !Number.isFinite(rect.height) ||
    !Number.isFinite(rect.angleDeg) ||
    rect.width <= 0 ||
    rect.height <= 0
  ) {
    throw new Error(`${label} must have finite coordinates, angle, and positive dimensions.`);
  }
}

function assertRect(rect: Rect, label: string): void {
  if (
    !Number.isFinite(rect.x) ||
    !Number.isFinite(rect.y) ||
    !Number.isFinite(rect.width) ||
    !Number.isFinite(rect.height) ||
    rect.width <= 0 ||
    rect.height <= 0
  ) {
    throw new Error(`${label} must have finite coordinates and positive dimensions.`);
  }
}

function assertCompatibleAspectRatio(source: Rect, region: SemanticRegion): void {
  const destination = FIXED_COMPOSITE_LAYOUT.regions[region];
  const sourceAspect = source.width / source.height;
  const destinationAspect = destination.width / destination.height;
  if (Math.abs(sourceAspect - destinationAspect) > ASPECT_RATIO_TOLERANCE) {
    throw new Error(
      `${region} source rectangle must preserve aspect ratio ${destination.width}:${destination.height}.`,
    );
  }
}
