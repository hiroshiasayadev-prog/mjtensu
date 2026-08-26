# Contract: Application scoring-session API

- **id**: `spec:product.system.contracts.application_session_api`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.system`

## What this is

Implementation-facing contract for one active scoring session after recognition has committed a `RecognizedStructure`.

The Application module owns session transitions and scoring orchestration. UI code issues semantic commands and consumes resulting state; it must not reconstruct session invariants or call the concrete scoring library.

## Session state

```ts
export interface ScoringSessionState {
  readonly structure: RecognizedStructure;
  readonly winningTileId: TileInstanceId;
  readonly conditions: ScoringConditionsDraft;
  readonly ruleProfile: ScoringRuleProfile;
  readonly latestResult: ScoringCalculation | null;
}
```

The absence of an active scoring session is represented outside this type. Live Recognition state does not create an empty or partially initialized `ScoringSessionState`.

A session is created only from a committed recognition result. The product-level session-creation rule initializes `winningTileId` from the completed-hand ordering; later user selection may replace that value normally.

`latestResult` is present only when it describes the exact current structure, winning-tile selection, conditions, and rule profile.

## Session commands

```ts
export type ScoringSessionCommand =
  | {
      readonly kind: 'select-winning-tile';
      readonly tileId: TileInstanceId;
    }
  | {
      readonly kind: 'replace-structure';
      readonly structure: RecognizedStructure;
    }
  | {
      readonly kind: 'replace-conditions';
      readonly conditions: ScoringConditionsDraft;
    }
  | {
      readonly kind: 'replace-rule-profile';
      readonly ruleProfile: ScoringRuleProfile;
    };
```

These commands are session-level semantic replacements, not low-level editor operations.

Tile insertion/removal/reordering, tile-identity replacement, dora editing, and meld regrouping belong to the correction/editor responsibility. Once an editor accepts a corrected semantic structure, Application receives that result through `replace-structure`.

This prevents the session API from becoming a second tile-editor command language.

## Service boundary

```ts
export interface ScoringSessionService {
  create(
    structure: RecognizedStructure,
    ruleProfile: ScoringRuleProfile,
  ): ScoringSessionState;

  update(
    state: ScoringSessionState,
    command: ScoringSessionCommand,
  ): ScoringSessionState;

  preview(
    state: ScoringSessionState,
  ): ScoringPreview;

  calculate(
    state: ScoringSessionState,
  ): ScoringSessionCalculation;
}

export interface ScoringSessionCalculation {
  readonly state: ScoringSessionState;
  readonly result: ScoringCalculation;
}
```

The service is synchronous because the current scoring boundary is synchronous.

## Creation

`create()`:

- accepts only a committed `RecognizedStructure`;
- creates a new independent scoring session;
- applies the product-defined initial winning-tile selection from the completed-hand ordering;
- initializes conditions from `INITIAL_SCORING_CONDITIONS`;
- installs the supplied rule profile;
- initializes `latestResult` to `null`;
- does not calculate automatically merely because a session was created.

```ts
export const INITIAL_SCORING_CONDITIONS: ScoringConditionsDraft = {
  winMethod: 'tsumo',
  roundWind: 'east',
  seatWind: 'east',
  riichi: 'none',
  ippatsu: false,
  rinshan: false,
  chankan: false,
  haitei: false,
  houtei: false,
  tenhou: false,
  chiihou: false,
};
```

The caller does not supply alternate initial conditions for a newly recognized hand. Tsumo, East round, and East seat are the product convenience defaults and remain user-editable on Conditions; situational booleans start off.
Recognition commit eligibility guarantees at least two valid completed-hand observations, so a normal committed recognition result provides a completed-hand tile from which the initial selection can be created.

## Update behavior

### Select winning tile

`select-winning-tile`:

- accepts a tile instance belonging to `state.structure.completedHand`;
- replaces only `winningTileId`;
- invalidates `latestResult`.

Selection is not tied to the tile's position after initialization. Any completed-hand tile instance may be selected by the user.

### Replace structure

`replace-structure`:

- replaces the current semantic recognized/corrected structure;
- preserves the existing `winningTileId` when that tile instance still belongs to the replacement completed hand, including when correction changed only that instance's tile identity;
- otherwise applies the product-defined completed-hand default selection to the replacement structure;
- preserves conditions and rule profile;
- invalidates `latestResult`.

Correction identity replacement preserves `TileInstanceId`; removing the selected instance or moving it out of the completed hand does not. Fine-grained correction identity/lifetime rules are owned by `spec:product.system.contracts.correction_editor_api`.

A structure committed into an active scoring session must contain at least one completed-hand tile so the non-null `winningTileId` invariant can be maintained. A correction UI may use its own transient draft while editing; such a draft is not `ScoringSessionState` until committed through this boundary.

### Replace conditions

`replace-conditions`:

- preserves structure, winning-tile selection, and rule profile;
- normalizes the supplied condition draft through `spec:product.system.contracts.scoring_condition_policy`;
- stores the normalized condition draft;
- invalidates `latestResult`.

Application and UI use the same condition policy so an impossible dependent value cannot remain stored while its control is shown unavailable.

### Replace rule profile

`replace-rule-profile`:

- preserves structure, winning-tile selection, and conditions;
- replaces the scoring rule profile;
- invalidates `latestResult`.

The current product supplies `DEFAULT_RULE_PROFILE`; the command exists so later rule-profile selection does not require changing the session API.

## Preview orchestration

`preview(state)` constructs the scoring draft from current session state and delegates to the scoring boundary:

```text
ScoringSessionState
      ↓
ScoringDraft
      ↓
ScoringService.preview(draft, state.ruleProfile)
      ↓
ScoringPreview
```

Application must not implement its own winning-shape, yaku, fu, or point logic to produce preview feedback.

## Calculation orchestration

`calculate(state)` is valid only when the current session can be converted to the strict scoring input required by `spec:product.system.contracts.scoring_api`.

The operation:

1. constructs the strict `ScoringInput` from current session state;
2. calls `ScoringService.calculate(input, state.ruleProfile)`;
3. returns the `ScoringCalculation`;
4. returns a new session state whose `latestResult` is that exact calculation.

```text
ScoringSessionState
      ↓ strict input construction
ScoringInput
      ↓
ScoringService.calculate(input, ruleProfile)
      ↓
ScoringCalculation
      ↓
new ScoringSessionState(latestResult = calculation)
```

A failed calculation must not fabricate or install a result.

## Result invalidation

Every score-relevant session mutation invalidates the prior result. The update operation must therefore set `latestResult` to `null` when any of the following changes:

- recognized/corrected structure;
- winning-tile selection;
- scoring conditions;
- rule profile.

UI code must not own separate result-staleness logic.

## Session lifetime

Starting a new Recognition attempt discards the active scoring session as a whole. The Application-level owner may therefore represent flow state conceptually as:

```ts
type ActiveScoringSession = ScoringSessionState | null;
```

`null` means no scoring session exists. It is not an incomplete scoring session.

The exact router/store representation of this outer state is implementation-owned.

## UI access rule

UI may:

- read `ScoringSessionState` for presentation;
- issue `ScoringSessionCommand` values;
- request `preview()` and `calculate()` through the Application boundary.

UI must not:

- mutate session fields independently;
- decide whether a previous result remains valid after a state change;
- call PaiForge or another concrete scoring implementation directly;
- rebuild scoring validation separately from `ScoringService`.

## Test seams

The contract must permit:

- session-transition tests with a fake `ScoringService`;
- verification that each score-relevant command invalidates `latestResult`;
- verification that winning-tile selection is preserved across structure replacement when its tile instance survives;
- verification that a replacement structure chooses a valid default when the previous selected instance is removed;
- preview/calculation orchestration tests without UI rendering or recognition/model execution.

## Boundary

| concern | owner |
|---|---|
| Active scoring-session state and semantic transitions | Application / this contract. |
| Recognition commitment | Recognition contracts. |
| Draft/strict scoring semantics and rule execution | `spec:product.system.contracts.scoring_api`. |
| Fine-grained tile/meld correction editing | UI/application correction responsibility, not this session command API. |
| Navigation and visible interaction | Product UI specs. |
| Concrete store/reducer/framework implementation | Implementation. |
