import type { TileKind } from '@/domain';
import {
  areFrameRecognitionDraftsEqual,
  RecognitionSemanticStabilizer,
} from '@/recognition/semantics/stabilizer';
import type {
  FrameObservationId,
  FrameRecognitionDraft,
  FrameRecognitionSnapshot,
} from '@/recognition/semantics/types';
import { describe, expect, it } from 'vitest';

describe('recognition semantic stabilization', () => {
  it('confirms after three consecutive equivalent eligible drafts while ignoring bbox jitter and frame-local ids', () => {
    const stabilizer = new RecognitionSemanticStabilizer();
    const draft = recognitionDraft('1m');

    expect(stabilizer.accept(snapshot(draft, 0.1, 'frame-a'))).toMatchObject({
      kind: 'stabilizing',
      consecutive: 1,
    });
    expect(stabilizer.accept(snapshot(draft, 0.13, 'frame-b'))).toMatchObject({
      kind: 'stabilizing',
      consecutive: 2,
    });
    expect(stabilizer.accept(snapshot(draft, 0.07, 'frame-c'))).toEqual({
      kind: 'confirmed',
      draft,
    });
  });

  it('restarts at one when semantic identity changes and clears the run on an ineligible frame', () => {
    const stabilizer = new RecognitionSemanticStabilizer();
    const first = recognitionDraft('1m');
    const changed = recognitionDraft('2m');

    expect(stabilizer.accept(snapshot(first, 0.1, 'a'))).toMatchObject({
      kind: 'stabilizing',
      consecutive: 1,
    });
    expect(stabilizer.accept(snapshot(first, 0.11, 'b'))).toMatchObject({
      kind: 'stabilizing',
      consecutive: 2,
    });
    expect(stabilizer.accept(snapshot(changed, 0.1, 'c'))).toMatchObject({
      kind: 'stabilizing',
      consecutive: 1,
    });

    expect(
      stabilizer.accept(snapshot(changed, 0.1, 'd', false)),
    ).toEqual({ kind: 'scanning' });
    expect(stabilizer.getState()).toEqual({ kind: 'scanning' });

    expect(stabilizer.accept(snapshot(changed, 0.1, 'e'))).toMatchObject({
      kind: 'stabilizing',
      consecutive: 1,
    });
  });

  it('keeps a confirmed result until reset and starts a fresh boundary afterward', () => {
    const stabilizer = new RecognitionSemanticStabilizer();
    const draft = recognitionDraft('3m');

    stabilizer.accept(snapshot(draft, 0.1, 'a'));
    stabilizer.accept(snapshot(draft, 0.1, 'b'));
    stabilizer.accept(snapshot(draft, 0.1, 'c'));

    expect(stabilizer.accept(snapshot(recognitionDraft('9s'), 0.1, 'd'))).toEqual({
      kind: 'confirmed',
      draft,
    });

    stabilizer.reset();
    expect(stabilizer.getState()).toEqual({ kind: 'scanning' });
    expect(stabilizer.accept(snapshot(draft, 0.1, 'e'))).toMatchObject({
      kind: 'stabilizing',
      consecutive: 1,
    });
  });

  it('semantic equality includes ordering, red identity, and meld reconstruction but not geometry', () => {
    const base = recognitionDraft('5m');
    const reordered: FrameRecognitionDraft = {
      ...base,
      completedHand: [...base.completedHand].reverse(),
    };
    const redChanged: FrameRecognitionDraft = {
      ...base,
      completedHand: base.completedHand.map((tile, index) =>
        index === 0 ? { ...tile, red: true } : tile,
      ),
    };

    expect(areFrameRecognitionDraftsEqual(base, recognitionDraft('5m'))).toBe(true);
    expect(areFrameRecognitionDraftsEqual(base, reordered)).toBe(false);
    expect(areFrameRecognitionDraftsEqual(base, redChanged)).toBe(false);
  });
});

function recognitionDraft(firstKind: TileKind): FrameRecognitionDraft {
  return {
    completedHand: [
      { kind: firstKind, red: false },
      { kind: '2m', red: false },
    ],
    doraIndicators: [{ kind: '3p', red: false }],
    meldGroups: [
      {
        kind: 'unresolved',
        tiles: [
          { kind: '1s', red: false },
          { kind: '4s', red: false },
          { kind: '7s', red: false },
        ],
      },
    ],
  };
}

function snapshot(
  draft: FrameRecognitionDraft,
  x: number,
  id: string,
  eligible = true,
): FrameRecognitionSnapshot {
  return {
    observations: [
      {
        id: id as FrameObservationId,
        region: 'completed-hand',
        bbox: { x, y: 0.7, width: 0.05, height: 0.1 },
        classification: {
          kind: 'tile',
          tile: draft.completedHand[0] ?? { kind: '1m', red: false },
        },
      },
    ],
    meldGroups: [],
    meldCommonAngleRadians: null,
    draft,
    commitEligibility: eligible
      ? { kind: 'eligible' }
      : { kind: 'ineligible', reason: 'insufficient-visible-tiles' },
  };
}
