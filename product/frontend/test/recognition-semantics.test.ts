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
  it.each([0, 22.5, -22.5, 45, -45])(
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
      expect(snapshot.meldCommonAngleRadians).toBeCloseTo(
        (degrees * Math.PI) / 180,
        6,
      );
    },
  );

  it('marks geometry beyond the accepted common tilt as non-committable', () => {
    const snapshot = buildFrameRecognitionSnapshot([
      ...handCandidates(8),
      ...meldRow('steep', 0.4, 50, ['1m', '2m', '3m']),
    ]);

    expect(snapshot.meldGroups).toEqual([]);
    expect(snapshot.commitEligibility).toEqual({
      kind: 'ineligible',
      reason: 'unresolved-meld-geometry',
    });
  });

  it('keeps a stable 3+2 meld partition under captured detector bbox-center jitter', () => {
    const snapshot = buildFrameRecognitionSnapshot([
      ...handCandidates(5),
      capturedMeldCandidate('nanodet:1055', 0.7758695144053913, 0.3991592205292671, 0.03685314459588783, 0.0854463617245128, '8m'),
      capturedMeldCandidate('nanodet:1059', 0.8069168413218071, 0.387080997691361, 0.035905241358343734, 0.08095318851186205, '9m'),
      capturedMeldCandidate('nanodet:1092', 0.7312824757391432, 0.44056500538946464, 0.05157752846196686, 0.06854383885798895, '7m'),
      capturedMeldCandidate('nanodet:1217', 0.7914229242189214, 0.4841931244052497, 0.04139111841385075, 0.07414596020985928, '1p'),
      capturedMeldCandidate('nanodet:1254', 0.7616863105844022, 0.4975748466154283, 0.03809097486246159, 0.07408504452926272, '1p'),
    ]);

    expect(snapshot.commitEligibility).toEqual({ kind: 'eligible' });
    expect(snapshot.meldGroups).toHaveLength(2);
    expect(snapshot.meldGroups[0]?.memberObservationIds).toEqual([
      'nanodet:1092',
      'nanodet:1055',
      'nanodet:1059',
    ]);
    expect(snapshot.meldGroups[0]?.interpretation.kind).toBe('chi');
    expect(snapshot.meldGroups[1]?.memberObservationIds).toEqual([
      'nanodet:1254',
      'nanodet:1217',
    ]);
    expect(snapshot.meldGroups[1]?.interpretation.kind).toBe('concealed-kan');
  });

  it('exposes a high common angle for the tilted iPhone capture without rejecting its stable 3+2 partition', () => {
    const snapshot = buildFrameRecognitionSnapshot([
      ...handCandidates(5),
      capturedMeldCandidate('tilted:nanodet:937', 0.7948620883955927, 0.34139502548512746, 0.039692818013710905, 0.09014629073494253, '8m'),
      capturedMeldCandidate('tilted:nanodet:941', 0.8287682451179188, 0.32070154662352623, 0.044132490685432096, 0.09126750199340482, '9m'),
      capturedMeldCandidate('tilted:nanodet:1014', 0.7483078707143881, 0.3898091279866965, 0.05688012403802012, 0.08166095004663365, '7m'),
      capturedMeldCandidate('tilted:nanodet:1100', 0.8200795173881215, 0.4262436826538486, 0.043178549798158085, 0.09032898924292493, '1p'),
      capturedMeldCandidate('tilted:nanodet:1176', 0.786132419038734, 0.4560145205994171, 0.04336905902487448, 0.07668033302148929, '1p'),
    ]);

    expect(snapshot.commitEligibility).toEqual({ kind: 'eligible' });
    expect(snapshot.meldGroups).toHaveLength(2);
    expect(snapshot.meldGroups[0]?.memberObservationIds).toEqual([
      'tilted:nanodet:1014',
      'tilted:nanodet:937',
      'tilted:nanodet:941',
    ]);
    expect(snapshot.meldGroups[1]?.memberObservationIds).toEqual([
      'tilted:nanodet:1176',
      'tilted:nanodet:1100',
    ]);
    expect(Math.abs(snapshot.meldCommonAngleRadians ?? 0)).toBeGreaterThan(
      (30 * Math.PI) / 180,
    );
    expect(Math.abs(snapshot.meldCommonAngleRadians ?? 0)).toBeLessThanOrEqual(
      (45 * Math.PI) / 180,
    );
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

    expect(snapshot.meldGroups).toHaveLength(1);
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

function capturedMeldCandidate(
  id: string,
  x: number,
  y: number,
  width: number,
  height: number,
  kind: TileKind,
): ClassifiedRecognitionCandidate {
  return {
    id,
    region: 'melds',
    bbox: { x, y, width, height },
    classification: {
      kind: 'tile',
      tile: { kind, red: false },
    },
  };
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
