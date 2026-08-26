# Contract: Tile correction editor API

- **id**: `spec:product.system.contracts.correction_editor_api`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.system`

## What this is

Implementation-facing contract for the reusable semantic tile-correction editor used before initial calculation and after Result correction.

Correction state is deliberately more permissive than the canonical scoring-session structure. The user may temporarily create incomplete or malformed hand/meld structure while editing; only a validated completed winning structure may be committed back through the Application session boundary.

## Draft model

```ts
export type CorrectionMeldGroupId = string & {
  readonly __brand: 'CorrectionMeldGroupId';
};

export interface CorrectionDraft {
  readonly completedHand: readonly TileInstance[];
  readonly meldGroups: readonly CorrectionMeldGroupDraft[];
  readonly doraIndicators: readonly TileInstance[];
}

export interface CorrectionMeldGroupDraft {
  readonly id: CorrectionMeldGroupId;
  readonly tiles: readonly TileInstance[];

  // Meaningful only when the current member composition is a four-equal kan.
  readonly kanOpenness: 'open' | 'concealed' | null;
}
```

The draft does not store a separate `chi` or `pon` tag. Those semantics are derived from current member composition during validation/commit:

- valid three-tile same-suit sequence -> `chi`;
- three equal base tile kinds -> `pon`;
- four equal base tile kinds + `kanOpenness = 'open'` -> `open-kan`;
- four equal base tile kinds + `kanOpenness = 'concealed'` -> `concealed-kan`.

Keeping `chi`/`pon` out of the mutable draft avoids stale duplicate state when a tile identity changes.

A recognition-derived kan initializes `kanOpenness` from the committed recognition semantics. If editing first turns a group into a four-equal kan and no prior kan semantic exists, initialize it to `open`; the visible editor always exposes the current `明槓` / `暗槓` value so the user can flip it explicitly.

For a group whose current member composition is not a four-equal kan, `kanOpenness` is `null`.

## Tile-instance identity during correction

Correcting the identity of an existing tile does not replace the tile instance.

```text
{id: tile-123, tile: 3m}
        ↓ replace identity
{id: tile-123, tile: 4m}
```

The existing `TileInstanceId` is preserved because the user is correcting the semantic identity of the same observed/committed tile position.

Removing a tile destroys that instance. Adding a tile creates a new `TileInstanceId`.

This rule allows the current winning-tile selection to survive an identity correction when the selected physical/logical tile remains present. The scoring session falls back to its rightmost completed-hand default only when the selected instance is actually removed or moved out of the completed hand.

## Editing commands

```ts
export type CorrectionDestination =
  | {
      readonly kind: 'completed-hand';
    }
  | {
      readonly kind: 'dora-indicators';
    }
  | {
      readonly kind: 'meld';
      readonly groupId: CorrectionMeldGroupId;
    };

export type CorrectionCommand =
  | {
      readonly kind: 'add-tile';
      readonly destination: CorrectionDestination;
      readonly tile: TileIdentity;
      readonly index?: number;
    }
  | {
      readonly kind: 'replace-tile';
      readonly tileId: TileInstanceId;
      readonly tile: TileIdentity;
    }
  | {
      readonly kind: 'remove-tile';
      readonly tileId: TileInstanceId;
    }
  | {
      readonly kind: 'add-meld-group';
    }
  | {
      readonly kind: 'remove-meld-group';
      readonly groupId: CorrectionMeldGroupId;
    }
  | {
      readonly kind: 'move-tile';
      readonly tileId: TileInstanceId;
      readonly destination: CorrectionDestination;
      readonly index: number;
    }
  | {
      readonly kind: 'toggle-kan-openness';
      readonly groupId: CorrectionMeldGroupId;
    };
```

`move-tile` is the general semantic move/reorder operation. It can represent reordering within one region, moving between meld groups, or moving between completed hand and melds. The visible UI may expose only interaction patterns that are useful on smartphone; the command contract need not imply drag-and-drop.

`toggle-kan-openness` is valid only for a current four-equal kan candidate and flips between `open` and `concealed`.

## Editor service

```ts
export interface CorrectionEditorService {
  create(structure: RecognizedStructure): CorrectionDraft;

  update(
    draft: CorrectionDraft,
    command: CorrectionCommand,
  ): CorrectionDraft;

  validate(
    draft: CorrectionDraft,
  ): CorrectionValidation;

  commit(
    draft: CorrectionDraft,
  ): CorrectionCommit;
}
```

The service is synchronous and pure from the caller's perspective. The UI owns the current draft lifetime locally and replaces it with the returned draft after each command.

## Validation

```ts
export interface CorrectionValidation {
  readonly issues: readonly CorrectionIssue[];
  readonly canCommit: boolean;
}

export type CorrectionIssueTarget =
  | {
      readonly kind: 'completed-hand';
    }
  | {
      readonly kind: 'meld';
      readonly groupId: CorrectionMeldGroupId;
    }
  | {
      readonly kind: 'winning-structure';
    };

export type CorrectionIssue =
  | {
      readonly kind: 'completed-hand-count';
      readonly target: { readonly kind: 'completed-hand' };
    }
  | {
      readonly kind: 'invalid-meld';
      readonly target: {
        readonly kind: 'meld';
        readonly groupId: CorrectionMeldGroupId;
      };
    }
  | {
      readonly kind: 'not-winning-shape';
      readonly target: { readonly kind: 'winning-structure' };
    };
```

Issue targets are product-semantic locations used by the UI to place repair feedback. The editor UI maps issue kinds to visible copy; domain/application code does not return Japanese presentation strings.

Validation first checks editor-owned structure such as meld composition and completed-hand count, then delegates whole winning-shape determination to `ScoringService.validateWinningStructure()`.

Dora indicators do not participate in winning-shape validity. Lack of yaku and missing non-image Conditions do not create `CorrectionIssue` values.

## Commit

```ts
export type CorrectionCommit =
  | {
      readonly kind: 'invalid';
      readonly validation: CorrectionValidation;
    }
  | {
      readonly kind: 'valid';
      readonly structure: RecognizedStructure;
    };
```

`commit()` returns `valid` only when the same rules represented by `validate()` permit commit.

Successful commit materializes canonical meld semantics from the current draft composition and kan-openness value and returns one `RecognizedStructure`. The caller then applies that structure through the Application session `replace-structure` command.

An invalid draft never mutates canonical session state.

## Lifetime and state-container boundary

`CorrectionDraft` is transient editor state. It is not stored in the cross-page Zustand scoring-session store merely because the editor is reused by more than one page.

The owning correction surface creates a draft from the current canonical structure, keeps it as page/component-local state while editing, and either:

- discards it on cancel/unmount; or
- commits a validated `RecognizedStructure` into Application.

## Boundary

| concern | owner |
|---|---|
| Mutable correction draft, editing commands, validation locations, commit conversion | This contract. |
| Visible smartphone editing interaction | `spec:product.ui.components.tile_correction_editor`. |
| Whole-hand winning-shape determination | `spec:product.system.contracts.scoring_api`. |
| Canonical tile and meld vocabulary | `spec:product.system.concepts.canonical_tile_model`. |
| Canonical session replacement and winning-tile preservation | `spec:product.system.contracts.application_session_api`. |
| Fine-grained detector/bbox editing | Outside correction. |
