import type { TileKind } from '@/domain';
import { buildFrameRecognitionSnapshot } from '@/recognition/semantics/frame-semantics';
import type {
  ClassifiedRecognitionCandidate,
  RecognitionRegion,
} from '@/recognition/semantics/types';
import { describe, expect, it } from 'vitest';

describe('recognition semantic observation and ordering', () => {
  it('keeps unresolved live observations while ordering recognized hand and dora tiles left-to-right', () => {
    const snapshot = buildFrameRecognitionSnapshot([
      candidate('hand-right', 'completed-hand', 0.6, 0.8, '2m'),
      invalidCandidate('hand-invalid', 'completed-hand', 0.4, 0.8),
      candidate('dora-right', 'dora-indicators', 0.5, 0.2, '9p'),
      candidate('hand-left', 'completed-hand', 0.2, 0.8, '1m'),
      candidate('dora-left', 'dora-indicators', 0.2, 0.2, '3s'),
    ]);

    expect(snapshot.observations).toHaveLength(5);
    expect(snapshot.observations.find((item) => item.id === 'hand-invalid')).toMatchObject({
      region: 'completed-hand',
      bbox: boxAt(0.4, 0.8),
      classification: { kind: 'invalid' },
    });
    expect(snapshot.draft.completedHand).toEqual([
      { kind: '1m', red: false },
      { kind: '2m', red: false },
    ]);
    expect(snapshot.draft.doraIndicators).toEqual([
      { kind: '3s', red: false },
      { kind: '9p', red: false },
    ]);
  });
});

describe('recognition meld grouping and reconstruction', () => {
  it.each([0, 22.5, -22.5])(
    'groups deterministic rows at %s degrees and orders groups top-to-bottom',
    (degrees) => {
      const snapshot = buildFrameRecognitionSnapshot([
        ...handCandidates(4),
        ...meldRow('top', 0.28, degrees, ['1m', '2m', '3m']),
        ...meldRow('bottom', 0.58, degrees, ['7p', '7p', '7p']),
      ]);

      expect(snapshot.meldGroups).toHaveLength(2);
      expect(snapshot.meldGroups[0]?.memberObservationIds).toEqual([
        'top-0',
        'top-1',
        'top-2',
      ]);
      expect(snapshot.meldGroups[0]?.interpretation.kind).toBe('chi');
      expect(snapshot.meldGroups[1]?.memberObservationIds).toEqual([
        'bottom-0',
        'bottom-1',
        'bottom-2',
      ]);
      expect(snapshot.meldGroups[1]?.interpretation.kind).toBe('pon');
    },
  );

  it('marks geometry beyond the accepted tilt as non-committable', () => {
    const snapshot = buildFrameRecognitionSnapshot([
      ...handCandidates(8),
      ...meldRow('steep', 0.4, 25, ['1m', '2m', '3m']),
    ]);

    expect(snapshot.meldGroups).toEqual([]);
    expect(snapshot.commitEligibility).toEqual({
      kind: 'ineligible',
      reason: 'unresolved-meld-geometry',
    });
  });

  it('reconstructs a two-visible-member same-base group as a concealed kan without fabricating hidden red', () => {
    const snapshot = buildFrameRecognitionSnapshot([
      candidate('kan-left', 'melds', 0.4, 0.45, '5m', true),
      candidate('kan-right', 'melds', 0.5, 0.45, '5m', false),
    ]);

    expect(snapshot.meldGroups).toHaveLength(1);
    expect(snapshot.meldGroups[0]).toEqual({
      memberObservationIds: ['kan-left', 'kan-right'],
      interpretation: {
        kind: 'concealed-kan',
        tiles: [
          { kind: '5m', red: false },
          { kind: '5m', red: true },
          { kind: '5m', red: false },
          { kind: '5m', red: false },
        ],
      },
    });
    expect(snapshot.observations).toHaveLength(2);
  });

  it('attaches open-kan semantics only when four visible identities make it unambiguous', () => {
    const snapshot = buildFrameRecognitionSnapshot([
      ...meldRow('open-kan', 0.45, 0, ['6p', '6p', '6p', '6p']),
    ]);

    expect(snapshot.meldGroups[0]?.interpretation).toEqual({
      kind: 'open-kan',
      tiles: [
        { kind: '6p', red: false },
        { kind: '6p', red: false },
        { kind: '6p', red: false },
        { kind: '6p', red: false },
      ],
    });
  });

  it('preserves a stable malformed meld identity composition as unresolved instead of applying scoring validity', () => {
    const snapshot = buildFrameRecognitionSnapshot([
      ...handCandidates(7),
      ...meldRow('malformed', 0.45, 0, ['1m', '4p', '7z']),
    ]);

    expect(snapshot.commitEligibility).toEqual({ kind: 'eligible' });
    expect(snapshot.meldGroups[0]?.interpretation).toEqual({
      kind: 'unresolved',
      tiles: [
        { kind: '1m', red: false },
        { kind: '4p', red: false },
        { kind: '7z', red: false },
      ],
    });
  });
});

describe('recognition capture eligibility', () => {
  it('counts the two visible members of each concealed kan rather than logical four-member expansion', () => {
    const eligible = buildFrameRecognitionSnapshot([
      ...handCandidates(2),
      ...concealedKanRows(4),
    ]);
    const insufficient = buildFrameRecognitionSnapshot([
      ...handCandidates(2),
      ...concealedKanRows(3),
    ]);

    expect(eligible.commitEligibility).toEqual({ kind: 'eligible' });
    expect(eligible.draft.meldGroups).toHaveLength(4);
    expect(
      eligible.draft.meldGroups.reduce((sum, group) => sum + group.tiles.length, 0),
    ).toBe(16);

    expect(insufficient.commitEligibility).toEqual({
      kind: 'ineligible',
      reason: 'insufficient-visible-tiles',
    });
  });

  it('requires at least two completed-hand observations even when the total visible minimum is met', () => {
    const snapshot = buildFrameRecognitionSnapshot([
      ...handCandidates(1),
      ...meldRow('row-a', 0.2, 0, ['1p', '2p', '3p']),
      ...meldRow('row-b', 0.4, 0, ['4p', '5p', '6p']),
      ...meldRow('row-c', 0.6, 0, ['7p', '8p', '9p']),
    ]);

    expect(snapshot.commitEligibility).toEqual({
      kind: 'ineligible',
      reason: 'insufficient-visible-tiles',
    });
  });

  it('excludes dora and invalid/background observations from the visible minimum', () => {
    const snapshot = buildFrameRecognitionSnapshot([
      ...handCandidates(2),
      ...concealedKanRows(3),
      candidate('dora-1', 'dora-indicators', 0.2, 0.1, '1p'),
      candidate('dora-2', 'dora-indicators', 0.3, 0.1, '2p'),
      invalidCandidate('invalid-meld', 'melds', 0.9, 0.9),
    ]);

    expect(snapshot.commitEligibility).toEqual({
      kind: 'ineligible',
      reason: 'insufficient-visible-tiles',
    });
  });
});

function concealedKanRows(count: number): ClassifiedRecognitionCandidate[] {
  return Array.from({ length: count }, (_, index) => {
    const y = 0.2 + index * 0.18;
    return [
      candidate(`kan-${index}-left`, 'melds', 0.75, y, '5s'),
      candidate(`kan-${index}-right`, 'melds', 0.84, y, '5s'),
    ];
  }).flat();
}

function handCandidates(count: number): ClassifiedRecognitionCandidate[] {
  return Array.from({ length: count }, (_, index) =>
    candidate(
      `hand-${index}`,
      'completed-hand',
      0.08 + index * 0.07,
      0.82,
      `${(index % 9) + 1}m` as TileKind,
    ),
  );
}

function meldRow(
  prefix: string,
  baseCenterY: number,
  degrees: number,
  kinds: readonly TileKind[],
): ClassifiedRecognitionCandidate[] {
  const tangent = Math.tan((degrees * Math.PI) / 180);
  const firstX = 0.74;

  return kinds.map((kind, index) => {
    const centerX = firstX + index * 0.075;
    const centerY = baseCenterY + tangent * (centerX - firstX);
    return candidate(`${prefix}-${index}`, 'melds', centerX, centerY, kind);
  });
}

function candidate(
  id: string,
  region: RecognitionRegion,
  centerX: number,
  centerY: number,
  kind: TileKind,
  red = false,
): ClassifiedRecognitionCandidate {
  return {
    id,
    region,
    bbox: boxAt(centerX, centerY),
    classification: {
      kind: 'tile',
      tile: { kind, red },
    },
  };
}

function invalidCandidate(
  id: string,
  region: RecognitionRegion,
  centerX: number,
  centerY: number,
): ClassifiedRecognitionCandidate {
  return {
    id,
    region,
    bbox: boxAt(centerX, centerY),
    classification: { kind: 'invalid' },
  };
}

function boxAt(centerX: number, centerY: number) {
  return {
    x: centerX - 0.025,
    y: centerY - 0.04,
    width: 0.05,
    height: 0.08,
  };
}
