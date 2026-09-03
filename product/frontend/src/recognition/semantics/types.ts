import type { TileIdentity } from '@/domain';

import type { TileClassification } from '../classifier/labels';

export const RECOGNITION_REGIONS = [
  'completed-hand',
  'dora-indicators',
  'melds',
] as const;

export type RecognitionRegion = (typeof RECOGNITION_REGIONS)[number];

export type FrameObservationId = string & {
  readonly __brand: 'FrameObservationId';
};

export interface NormalizedRect {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

export interface NormalizedOrientedRect {
  readonly cx: number;
  readonly cy: number;
  readonly width: number;
  readonly height: number;
  readonly angleDeg: number;
}

export interface ClassifiedRecognitionCandidate {
  readonly id: string;
  readonly region: RecognitionRegion;
  readonly bbox: NormalizedRect;
  readonly obb?: NormalizedOrientedRect;
  readonly classification: TileClassification;
}

export interface TileObservation {
  readonly id: FrameObservationId;
  readonly region: RecognitionRegion;
  readonly bbox: NormalizedRect;
  readonly obb?: NormalizedOrientedRect;
  readonly classification: TileClassification;
}

export type FrameMeldInterpretation =
  | {
      readonly kind: 'chi';
      readonly tiles: readonly [TileIdentity, TileIdentity, TileIdentity];
    }
  | {
      readonly kind: 'pon';
      readonly tiles: readonly [TileIdentity, TileIdentity, TileIdentity];
    }
  | {
      readonly kind: 'open-kan';
      readonly tiles: readonly [
        TileIdentity,
        TileIdentity,
        TileIdentity,
        TileIdentity,
      ];
    }
  | {
      readonly kind: 'concealed-kan';
      readonly tiles: readonly [
        TileIdentity,
        TileIdentity,
        TileIdentity,
        TileIdentity,
      ];
    }
  | {
      readonly kind: 'unresolved';
      readonly tiles: readonly TileIdentity[];
    };

export interface MeldGroupObservation {
  readonly memberObservationIds: readonly FrameObservationId[];
  readonly interpretation: FrameMeldInterpretation;
}

export interface FrameRecognitionDraft {
  readonly completedHand: readonly TileIdentity[];
  readonly doraIndicators: readonly TileIdentity[];
  readonly meldGroups: readonly FrameMeldInterpretation[];
}

export type FrameCommitEligibility =
  | { readonly kind: 'eligible' }
  | {
      readonly kind: 'ineligible';
      readonly reason:
        | 'insufficient-visible-tiles'
        | 'unresolved-meld-geometry';
    };

export interface FrameRecognitionSnapshot {
  readonly observations: readonly TileObservation[];
  readonly meldGroups: readonly MeldGroupObservation[];
  readonly meldCommonAngleRadians: number | null;
  readonly draft: FrameRecognitionDraft;
  readonly commitEligibility: FrameCommitEligibility;
}
