import type { CaptureLayoutDocument, Rect } from './types';

export interface VideoContainGeometry {
  element: DOMRect;
  sourceWidth: number;
  sourceHeight: number;
  scale: number;
  offsetX: number;
  offsetY: number;
}

export async function loadCaptureLayout(): Promise<CaptureLayoutDocument> {
  const response = await fetch(new URL('tile_catalog_layout.v2.json', document.baseURI));
  if (!response.ok) throw new Error(`Failed to load tile catalog layout: HTTP ${response.status}`);
  return response.json() as Promise<CaptureLayoutDocument>;
}

export function calculateVideoContainGeometry(video: HTMLVideoElement): VideoContainGeometry {
  const element = video.getBoundingClientRect();
  if (video.videoWidth <= 0 || video.videoHeight <= 0 || element.width <= 0 || element.height <= 0) {
    throw new Error('Video geometry is not ready.');
  }
  const scale = Math.min(element.width / video.videoWidth, element.height / video.videoHeight);
  return {
    element,
    sourceWidth: video.videoWidth,
    sourceHeight: video.videoHeight,
    scale,
    offsetX: (element.width - video.videoWidth * scale) / 2,
    offsetY: (element.height - video.videoHeight * scale) / 2,
  };
}

export function fullSourceRect(geometry: VideoContainGeometry): Rect {
  return {
    x: 0,
    y: 0,
    width: geometry.sourceWidth,
    height: geometry.sourceHeight,
  };
}

export function sourceRectToDisplay(rect: Rect, geometry: VideoContainGeometry): Rect {
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
