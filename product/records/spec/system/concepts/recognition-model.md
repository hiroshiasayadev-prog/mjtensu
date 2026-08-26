# Concept: Recognition model

- **id**: `spec:product.system.concepts.recognition_model`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.system`

## What this is

Semantic model separating current-frame camera observations from the recognition structure that is compared for stabilization and eventually committed to Application.

The separation prevents live detector geometry from becoming scoring state and prevents the UI from reimplementing recognition semantics.

## Recognition regions

```ts
export type RecognitionRegion =
  | 'completed-hand'
  | 'dora-indicators'
  | 'melds';
```

These correspond directly to the three visible semantic capture regions defined by the product recognition specification.

## Frame-local observation identity

Each retained detector candidate in one evaluated frame receives a frame-local observation identity.

```ts
export type FrameObservationId = string & {
  readonly __brand: 'FrameObservationId';
};
```

The identifier is valid only for relating data inside one frame snapshot, such as associating a meld group with the member bounding boxes used by the overlay.
It is not stable across frames and is not a committed tile-instance identity.

## Tile observation

```ts
export type TileClassification =
  | {
      readonly kind: 'tile';
      readonly tile: TileIdentity;
    }
  | {
      readonly kind: 'invalid';
    };

export interface TileObservation {
  readonly id: FrameObservationId;
  readonly region: RecognitionRegion;
  readonly bbox: NormalizedRect;
  readonly classification: TileClassification;
}
```

A `tile` classification represents the final base/red-five identity for that current detector candidate.
An `invalid` classification represents the 35-class base classifier's invalid/background outcome and does not become a tile in the recognition draft.

Model confidence and classifier probabilities are intentionally absent from this public semantic observation. Diagnostic telemetry may expose them through a separate debug-only boundary if needed.

## Meld-group observation

Spatial meld grouping is recognition logic, not UI logic.
The pipeline exposes which current-frame observations belong to one group and the current semantic interpretation.

```ts
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
```

The UI derives connector geometry from the member observations' bounding-box centers. Recognition owns membership and ordering; UI owns only how that relationship is drawn.

For concealed-kan reconstruction, the interpretation may contain four logical tile identities even though `memberObservationIds` contains only the two visible face-up observations.
No observation or bounding box is fabricated for hidden tiles.

## Frame recognition draft

The geometry-independent semantic result of one frame is:

```ts
export interface FrameRecognitionDraft {
  readonly completedHand: readonly TileIdentity[];
  readonly doraIndicators: readonly TileIdentity[];
  readonly meldGroups: readonly FrameMeldInterpretation[];
}
```

Ordering follows the product recognition contract:

- completed hand left to right;
- dora indicators left to right;
- meld groups top to bottom;
- visible members inside a meld group along that group's row direction.

The draft may be incomplete or scoring-invalid. Winning-shape validity and yaku existence are not recognition concerns.

## Frame snapshot

One frame evaluation exposes both live observations and the geometry-independent draft.

```ts
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
  readonly draft: FrameRecognitionDraft;
  readonly commitEligibility: FrameCommitEligibility;
}
```

`commitEligibility` answers only whether the frame may participate in temporal stabilization. It does not state that the draft is a legal winning hand.

## Semantic equality

Temporal equality compares `FrameRecognitionDraft`, not live observation geometry.

Equality includes:

- tile identities;
- completed-hand order;
- dora-indicator order;
- meld-group order;
- meld-group interpretation and logical tile identities.

Equality excludes:

- bounding-box coordinates;
- frame-local observation IDs;
- model confidence/probability;
- capture timestamp.

Therefore normal detector-box jitter does not reset stabilization when the semantic recognition result is unchanged.

## Commit materialization

A frame draft becomes a `RecognizedStructure` only after realtime stabilization confirms it.
At that boundary the system creates committed `TileInstanceId` values and converts frame meld interpretations into committed `RecognizedMeldGroup` values.

No frame-local observation identity survives as a committed tile-instance identity.
