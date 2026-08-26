import type { Detection } from './nanodet';
import type { CaptureLayoutDocument, DetectionRecord, Rect, RegionKey } from './types';

export interface CompositeBuildResult {
  compositeCanvas: HTMLCanvasElement;
  regionCanvases: Partial<Record<RegionKey, HTMLCanvasElement>>;
}

export function buildComposite(
  source: CanvasImageSource,
  sourceRects: Record<RegionKey, Rect>,
  enabled: Record<RegionKey, boolean>,
  layout: CaptureLayoutDocument,
  createRegionCanvases = true,
): CompositeBuildResult {
  const compositeCanvas = document.createElement('canvas');
  compositeCanvas.width = layout.composite.width;
  compositeCanvas.height = layout.composite.height;
  const context = requireContext(compositeCanvas);
  const [red, green, blue] = layout.composite.paddingRgb;
  context.fillStyle = `rgb(${red}, ${green}, ${blue})`;
  context.fillRect(0, 0, compositeCanvas.width, compositeCanvas.height);

  const regionCanvases: Partial<Record<RegionKey, HTMLCanvasElement>> = {};
  for (const key of regionKeys()) {
    if (!enabled[key]) continue;
    const sourceRect = sourceRects[key];
    const destination = layout.regions[key].destination;
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

    if (createRegionCanvases) {
      const crop = document.createElement('canvas');
      crop.width = Math.max(1, Math.round(sourceRect.width));
      crop.height = Math.max(1, Math.round(sourceRect.height));
      requireContext(crop).drawImage(
        source,
        sourceRect.x,
        sourceRect.y,
        sourceRect.width,
        sourceRect.height,
        0,
        0,
        crop.width,
        crop.height,
      );
      regionCanvases[key] = crop;
    }
  }
  return { compositeCanvas, regionCanvases };
}

export function mapDetections(
  detections: Detection[],
  sourceRects: Record<RegionKey, Rect>,
  enabled: Record<RegionKey, boolean>,
  layout: CaptureLayoutDocument,
): DetectionRecord[] {
  return detections.map((detection, detectionIndex) => {
    const centerX = (detection.x1 + detection.x2) / 2;
    const centerY = (detection.y1 + detection.y2) / 2;
    const region = regionAt(centerX, centerY, enabled, layout);
    const composite = {
      x: detection.x1,
      y: detection.y1,
      width: detection.x2 - detection.x1,
      height: detection.y2 - detection.y1,
    };
    return {
      detectionIndex,
      region,
      confidence: detection.score,
      composite,
      original: region === 'invalid'
        ? null
        : compositeRectToSource(composite, layout.regions[region].destination, sourceRects[region]),
      preview: null,
    };
  });
}

export async function canvasToBlob(
  canvas: HTMLCanvasElement,
  type: 'image/jpeg' | 'image/png',
  quality?: number,
): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => blob === null ? reject(new Error(`Canvas encoding failed for ${type}.`)) : resolve(blob),
      type,
      quality,
    );
  });
}

export function regionKeys(): RegionKey[] {
  return ['completed_hand', 'dora_indicators', 'melds'];
}

function regionAt(
  x: number,
  y: number,
  enabled: Record<RegionKey, boolean>,
  layout: CaptureLayoutDocument,
): RegionKey | 'invalid' {
  for (const key of regionKeys()) {
    if (!enabled[key]) continue;
    const rect = layout.regions[key].destination;
    if (x >= rect.x && y >= rect.y && x < rect.x + rect.width && y < rect.y + rect.height) return key;
  }
  return 'invalid';
}

function compositeRectToSource(composite: Rect, destination: Rect, source: Rect): Rect {
  const left = Math.max(composite.x, destination.x);
  const top = Math.max(composite.y, destination.y);
  const right = Math.min(composite.x + composite.width, destination.x + destination.width);
  const bottom = Math.min(composite.y + composite.height, destination.y + destination.height);
  const scaleX = source.width / destination.width;
  const scaleY = source.height / destination.height;
  return {
    x: source.x + (left - destination.x) * scaleX,
    y: source.y + (top - destination.y) * scaleY,
    width: Math.max(0, right - left) * scaleX,
    height: Math.max(0, bottom - top) * scaleY,
  };
}

function requireContext(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const context = canvas.getContext('2d', { alpha: false, willReadFrequently: true });
  if (context === null) throw new Error('2D canvas is unavailable.');
  return context;
}
