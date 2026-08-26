import { describe, expect, it, vi } from 'vitest';

import {
  assignSemanticRegion,
  buildFixedComposite,
  compositeRectToSource,
  FIXED_COMPOSITE_LAYOUT,
  sourceRectToComposite,
} from '@/recognition/detector/fixed-composite';
import type { CaptureRegions, Rect } from '@/recognition/detector/types';

const captureRegions: CaptureRegions = {
  completed_hand: {
    enabled: true,
    sourceRect: { x: 100, y: 200, width: 850, height: 200 },
  },
  dora_indicators: {
    enabled: true,
    sourceRect: { x: 100, y: 20, width: 850, height: 200 },
  },
  melds: {
    enabled: true,
    sourceRect: { x: 1000, y: 20, width: 400, height: 400 },
  },
};

describe('fixed NanoDet composite geometry', () => {
  it('uses the exact ADR-002 layout', () => {
    expect(FIXED_COMPOSITE_LAYOUT).toEqual({
      width: 320,
      height: 320,
      paddingRgb: [0, 0, 0],
      regions: {
        completed_hand: { x: 7, y: 0, width: 306, height: 72 },
        dora_indicators: { x: 7, y: 74, width: 306, height: 72 },
        melds: { x: 74, y: 148, width: 172, height: 172 },
      },
    });
  });

  it('draws every enabled camera region into its exact fixed destination', () => {
    const fillRect = vi.fn();
    const drawImage = vi.fn();
    const context = { fillRect, drawImage, fillStyle: '' };
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(
      context as unknown as CanvasRenderingContext2D,
    );
    const source = document.createElement('img');

    const composite = buildFixedComposite(source, captureRegions);

    expect(composite.width).toBe(320);
    expect(composite.height).toBe(320);
    expect(context.fillStyle).toBe('rgb(0, 0, 0)');
    expect(fillRect).toHaveBeenCalledWith(0, 0, 320, 320);
    expect(drawImage).toHaveBeenNthCalledWith(
      1,
      source,
      100,
      200,
      850,
      200,
      7,
      0,
      306,
      72,
    );
    expect(drawImage).toHaveBeenNthCalledWith(
      2,
      source,
      100,
      20,
      850,
      200,
      7,
      74,
      306,
      72,
    );
    expect(drawImage).toHaveBeenNthCalledWith(
      3,
      source,
      1000,
      20,
      400,
      400,
      74,
      148,
      172,
      172,
    );
  });

  it('maps camera coordinates to the fixed composite and back', () => {
    const sourceBox: Rect = { x: 525, y: 250, width: 85, height: 100 };
    const compositeBox = sourceRectToComposite(
      sourceBox,
      'completed_hand',
      captureRegions,
    );

    expect(compositeBox.x).toBeCloseTo(160);
    expect(compositeBox.y).toBeCloseTo(18);
    expect(compositeBox.width).toBeCloseTo(30.6);
    expect(compositeBox.height).toBeCloseTo(36);
    const roundTrip = compositeRectToSource(
      compositeBox,
      'completed_hand',
      captureRegions,
    );
    expect(roundTrip.x).toBeCloseTo(sourceBox.x);
    expect(roundTrip.y).toBeCloseTo(sourceBox.y);
    expect(roundTrip.width).toBeCloseTo(sourceBox.width);
    expect(roundTrip.height).toBeCloseTo(sourceBox.height);
  });

  it.each([
    [{ x: 7, y: 0, width: 2, height: 2 }, 'completed_hand'],
    [{ x: 311, y: 70, width: 2, height: 2 }, 'completed_hand'],
    [{ x: 7, y: 74, width: 2, height: 2 }, 'dora_indicators'],
    [{ x: 311, y: 144, width: 2, height: 2 }, 'dora_indicators'],
    [{ x: 74, y: 148, width: 2, height: 2 }, 'melds'],
    [{ x: 244, y: 318, width: 2, height: 2 }, 'melds'],
    [{ x: 5, y: 20, width: 2, height: 2 }, null],
    [{ x: 20, y: 71, width: 2, height: 2 }, null],
    [{ x: 20, y: 72, width: 2, height: 2 }, null],
    [{ x: 20, y: 145, width: 2, height: 2 }, null],
    [{ x: 20, y: 146, width: 2, height: 2 }, null],
    [{ x: 72, y: 200, width: 2, height: 2 }, null],
    [{ x: 246, y: 200, width: 2, height: 2 }, null],
  ] as const)('assigns boundary box %o to %s', (box, expected) => {
    expect(assignSemanticRegion(box, captureRegions)).toBe(expected);
  });

  it('rejects detections centered in a disabled semantic region', () => {
    const disabledDora: CaptureRegions = {
      ...captureRegions,
      dora_indicators: { ...captureRegions.dora_indicators, enabled: false },
    };

    expect(
      assignSemanticRegion({ x: 100, y: 90, width: 20, height: 20 }, disabledDora),
    ).toBeNull();
  });
});
