import type { CaptureLayoutDocument, Rect, RegionKey } from './types';

export interface VideoCoverGeometry {
  element: DOMRect;
  sourceWidth: number;
  sourceHeight: number;
  scale: number;
  offsetX: number;
  offsetY: number;
}

export async function loadCaptureLayout(): Promise<CaptureLayoutDocument> {
  const response = await fetch(new URL('capture_layout.v1.json', document.baseURI));
  if (!response.ok) throw new Error(`Failed to load capture layout: HTTP ${response.status}`);
  return response.json() as Promise<CaptureLayoutDocument>;
}

export function computeDisplayRegionRects(
  viewportWidth: number,
  viewportHeight: number,
): Record<RegionKey, Rect> {
  const margin = 14;
  const headerReserve = 54;
  const footerReserve = 78;
  const gap = 10;
  const availableWidth = Math.max(1, viewportWidth - margin * 2);
  const availableHeight = Math.max(1, viewportHeight - headerReserve - footerReserve);

  let square = availableHeight;
  let rowHeight = (square - gap) / 2;
  let rowWidth = rowHeight * 17 / 4;
  let totalWidth = rowWidth + gap + square;
  if (totalWidth > availableWidth) {
    const factor = availableWidth / totalWidth;
    square *= factor;
    rowHeight *= factor;
    rowWidth *= factor;
    totalWidth = availableWidth;
  }

  const left = (viewportWidth - totalWidth) / 2;
  const top = headerReserve + (availableHeight - square) / 2;
  return {
    dora_indicators: { x: left, y: top, width: rowWidth, height: rowHeight },
    completed_hand: { x: left, y: top + rowHeight + gap, width: rowWidth, height: rowHeight },
    melds: { x: left + rowWidth + gap, y: top, width: square, height: square },
  };
}

export function calculateVideoCoverGeometry(video: HTMLVideoElement): VideoCoverGeometry {
  const element = video.getBoundingClientRect();
  if (video.videoWidth <= 0 || video.videoHeight <= 0 || element.width <= 0 || element.height <= 0) {
    throw new Error('Video geometry is not ready.');
  }
  const scale = Math.max(element.width / video.videoWidth, element.height / video.videoHeight);
  return {
    element,
    sourceWidth: video.videoWidth,
    sourceHeight: video.videoHeight,
    scale,
    offsetX: (element.width - video.videoWidth * scale) / 2,
    offsetY: (element.height - video.videoHeight * scale) / 2,
  };
}

export function displayRectToSource(rect: Rect, geometry: VideoCoverGeometry): Rect {
  const x = (rect.x - geometry.element.left - geometry.offsetX) / geometry.scale;
  const y = (rect.y - geometry.element.top - geometry.offsetY) / geometry.scale;
  const width = rect.width / geometry.scale;
  const height = rect.height / geometry.scale;
  return clipRect({ x, y, width, height }, geometry.sourceWidth, geometry.sourceHeight);
}

export function sourceRectToDisplay(rect: Rect, geometry: VideoCoverGeometry): Rect {
  return {
    x: geometry.element.left + geometry.offsetX + rect.x * geometry.scale,
    y: geometry.element.top + geometry.offsetY + rect.y * geometry.scale,
    width: rect.width * geometry.scale,
    height: rect.height * geometry.scale,
  };
}

export function normalizeRect(rect: Rect, width: number, height: number): Rect {
  return {
    x: rect.x / width,
    y: rect.y / height,
    width: rect.width / width,
    height: rect.height / height,
  };
}

function clipRect(rect: Rect, maximumWidth: number, maximumHeight: number): Rect {
  const left = Math.min(maximumWidth, Math.max(0, rect.x));
  const top = Math.min(maximumHeight, Math.max(0, rect.y));
  const right = Math.min(maximumWidth, Math.max(left, rect.x + rect.width));
  const bottom = Math.min(maximumHeight, Math.max(top, rect.y + rect.height));
  return { x: left, y: top, width: right - left, height: bottom - top };
}
