# Concept: Canonical tile model

- **id**: `spec:product.system.concepts.canonical_tile_model`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.system`

## What this is

Library-independent tile and meld vocabulary shared by recognition, application, scoring, and UI.
The model distinguishes tile kind, red-five identity, one committed tile instance, permissive recognition meld drafts, and strict logical melds.

## Tile kind

The system uses the 34 ordinary riichi tile kinds as its canonical base kinds.
Numbered suits use `m`, `p`, and `s`; honors use `1z` through `7z`.

```ts
export type TileKind =
  | '1m' | '2m' | '3m' | '4m' | '5m' | '6m' | '7m' | '8m' | '9m'
  | '1p' | '2p' | '3p' | '4p' | '5p' | '6p' | '7p' | '8p' | '9p'
  | '1s' | '2s' | '3s' | '4s' | '5s' | '6s' | '7s' | '8s' | '9s'
  | '1z' | '2z' | '3z' | '4z' | '5z' | '6z' | '7z';
```

Honor mapping is:

| kind | tile |
|---|---|
| `1z` | East |
| `2z` | South |
| `3z` | West |
| `4z` | North |
| `5z` | White |
| `6z` | Green |
| `7z` | Red dragon |

UI presentation may render human-readable names or tile graphics instead of these internal codes.

## Red-five identity

Red fives are not separate base tile kinds. Red status is an attribute of a five.

```ts
export interface TileIdentity {
  readonly kind: TileKind;
  readonly red: boolean;
}
```

Invariant:

- `red === true` is valid only when `kind` is `5m`, `5p`, or `5s`.

This keeps the canonical base-kind space at 34 while preserving aka-dora identity explicitly.

## Tile instance

A committed scoring session must distinguish separate tile instances even when their tile identity is equal.

```ts
export type TileInstanceId = string & {
  readonly __brand: 'TileInstanceId';
};

export interface TileInstance {
  readonly id: TileInstanceId;
  readonly tile: TileIdentity;
}
```

`TileInstanceId` is an application-owned opaque identity. It exists so one physical/logical tile instance can remain selected as the winning tile through Conditions and Result correction.

It is not equivalent to the concrete scoring library's physical-tile identifier and must not encode or expose that library's internal tile numbering.

## Observation identity versus committed instance identity

Per-frame detector observations are ephemeral and use a separate frame-local observation identifier.
They must not be reused as committed `TileInstanceId` values.

Committed instance IDs are materialized only when a recognition draft becomes the committed `RecognizedStructure` for a scoring session.

## Recognition meld groups

Recognition must be able to preserve a spatially coherent meld-row draft even when the recognized tile identities do not yet form a legal scoring meld.
Therefore recognition uses a permissive tagged union containing an `unresolved` case.

```ts
export type RecognizedMeldGroup =
  | {
      readonly kind: 'chi';
      readonly tiles: readonly [TileInstance, TileInstance, TileInstance];
    }
  | {
      readonly kind: 'pon';
      readonly tiles: readonly [TileInstance, TileInstance, TileInstance];
    }
  | {
      readonly kind: 'open-kan';
      readonly tiles: readonly [
        TileInstance,
        TileInstance,
        TileInstance,
        TileInstance,
      ];
    }
  | {
      readonly kind: 'concealed-kan';
      readonly tiles: readonly [
        TileInstance,
        TileInstance,
        TileInstance,
        TileInstance,
      ];
    }
  | {
      readonly kind: 'unresolved';
      readonly tiles: readonly TileInstance[];
    };
```

An `unresolved` group is a recognition/correction draft and is not a valid scoring meld.

## Concealed-kan reconstruction

A physical concealed kan is expected to present two face-up matching base tiles between two face-down tiles.
Only the two face-up tiles produce detector observations.

Recognition reconstructs those two observations into one logical concealed-kan meld containing four logical tile instances before the committed `RecognizedStructure` is created.

```text
visible observations
    [5m] [5m]
        ↓
logical recognition structure
    concealed-kan [5m, 5m, 5m, 5m]
```

The two hidden members are reconstructed semantic members; they do not acquire fabricated camera bounding boxes.

Recognition must never infer an unseen red five. If a visible member is recognized as red, that red identity is preserved in the logical kan. Hidden reconstructed members use the ordinary-five identity.

## Strict scoring melds

The scoring boundary must not accept `unresolved` melds.
Before calculation, Conditions/Application must produce a strict logical meld structure containing only:

- chi with exactly three tile identities;
- pon with exactly three equal base tile identities;
- open kan with exactly four equal base tile identities;
- concealed kan with exactly four equal base tile identities.

The concrete scoring adapter maps that strict logical structure to the scoring library's completed-mentsu representation.

## Committed recognition structure

```ts
export interface RecognizedStructure {
  readonly completedHand: readonly TileInstance[];
  readonly doraIndicators: readonly TileInstance[];
  readonly meldGroups: readonly RecognizedMeldGroup[];
}
```

`RecognizedStructure` is a recognition draft suitable for Application and Conditions. It is not required to already represent a legal winning hand or to contain a yaku.

The winning tile is intentionally absent from this structure. Winning-tile selection belongs to the scoring session after recognition commitment.
