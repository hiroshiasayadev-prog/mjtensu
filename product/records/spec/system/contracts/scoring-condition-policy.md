# Contract: Scoring condition policy

- **id**: `spec:product.system.contracts.scoring_condition_policy`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.system`

## What this is

Shared condition-dependency policy used by Application when updating condition state and by UI when deciding which condition controls are available.

The purpose is to keep stored condition values and their visible controls synchronized without duplicating dependency rules in pages/components.
This policy owns only dependencies that can be decided from `ScoringConditionsDraft`. Structure-dependent legality, such as whether riichi is compatible with the corrected meld structure, remains a scoring-input concern.

## Public boundary

```ts
export interface ScoringConditionAvailability {
  readonly ippatsu: boolean;
  readonly rinshan: boolean;
  readonly chankan: boolean;
  readonly haitei: boolean;
  readonly houtei: boolean;
  readonly tenhou: boolean;
  readonly chiihou: boolean;
}

export interface ScoringConditionPolicy {
  normalize(
    conditions: ScoringConditionsDraft,
  ): ScoringConditionsDraft;

  availability(
    conditions: ScoringConditionsDraft,
  ): ScoringConditionAvailability;
}
```

The same dependency policy must drive both normalization and control availability.
UI code must not maintain a separate copy of these rules.

## Interaction rule

A dependent/situational condition never changes its own prerequisite select values on the user's behalf.

For example:

- selecting Tenhou does not force Win method to Tsumo or Seat wind to East;
- selecting Chiihou does not force Win method to Tsumo or choose a non-East seat;
- selecting Ippatsu does not force Riichi on.

Instead, the dependent control is selectable only after its prerequisites are already satisfied.
If the user later changes an ordinary prerequisite so that a selected dependent condition becomes impossible, normalization turns that dependent condition off immediately.

This keeps the direction of automatic state changes predictable:

```text
ordinary/select condition
        ↓ determines
situational availability
        ↓
impossible selected situational value is cleared
```

## Availability and conflict table

| condition | selectable when | mutually incompatible state |
|---|---|---|
| Ippatsu | Riichi or Double riichi is selected. | Tenhou, Chiihou. |
| Rinshan kaihou | Win method is Tsumo. | Haitei, Tenhou, Chiihou. |
| Chankan | Win method is Ron. | Houtei, Tenhou, Chiihou. |
| Haitei | Win method is Tsumo. | Rinshan kaihou, Tenhou, Chiihou. |
| Houtei | Win method is Ron. | Chankan, Tenhou, Chiihou. |
| Tenhou | Win method is Tsumo, Seat wind is East, Riichi is None, and every other situational condition is off. | Ippatsu, Rinshan, Chankan, Haitei, Houtei, Chiihou, Riichi/Double riichi. |
| Chiihou | Win method is Tsumo, Seat wind is explicitly non-East, Riichi is None, and every other situational condition is off. | Ippatsu, Rinshan, Chankan, Haitei, Houtei, Tenhou, Riichi/Double riichi. |

`seatWind = null` is not a non-East seat for Chiihou availability.

The table intentionally does not claim that every pair of situational yaku is mutually exclusive. Only combinations whose winning circumstances cannot describe the same win are blocked here.

## Normalization behavior

`normalize()` is deterministic and idempotent.
It applies the following rules until the returned draft satisfies all of them:

1. Win-method dependencies:
   - `winMethod = 'ron'` clears Rinshan, Haitei, Tenhou, and Chiihou.
   - `winMethod = 'tsumo'` clears Chankan and Houtei.
2. Riichi dependency:
   - `riichi = 'none'` clears Ippatsu.
   - Riichi or Double riichi clears Tenhou and Chiihou.
3. Seat dependency:
   - East seat clears Chiihou.
   - an explicitly non-East seat clears Tenhou.
   - an absent seat wind clears both Tenhou and Chiihou.
4. Same-method mutually exclusive outcomes:
   - if Rinshan and Haitei are both true in an externally supplied draft, clear both;
   - if Chankan and Houtei are both true in an externally supplied draft, clear both.
5. Initial-draw yakuman isolation:
   - Tenhou survives only when all Tenhou prerequisites in the table are satisfied and Chiihou plus every other situational condition is off;
   - Chiihou survives only when all Chiihou prerequisites in the table are satisfied and Tenhou plus every other situational condition is off;
   - if an externally supplied draft has both Tenhou and Chiihou true, clear both.

Clearing both sides of a contradictory same-level pair is the defensive normalization rule for malformed whole-draft input. Ordinary UI interaction should not create those states because `availability()` disables the conflicting control before it can be selected.

Normalization is an Application/UI state-coherence rule, not the scoring engine's validation layer.

## Availability behavior

`availability()` reflects the same rule table against the normalized current draft.
An unavailable boolean condition is represented visually as off and disabled/unselectable.

In particular:

- Ippatsu is available only while Riichi or Double riichi is selected and neither Tenhou nor Chiihou is selected;
- Rinshan is available only for Tsumo when Haitei, Tenhou, and Chiihou are off;
- Haitei is available only for Tsumo when Rinshan, Tenhou, and Chiihou are off;
- Chankan is available only for Ron when Houtei is off;
- Houtei is available only for Ron when Chankan is off;
- Tenhou is available only in the exact prerequisite state from the table;
- Chiihou is available only in the exact prerequisite state from the table.

A currently selected condition remains operable so the user can turn it off; the policy must not produce a state where a selected condition is displayed as unavailable while remaining true.

The UI may hide an unavailable secondary condition instead of disabling it when that is clearer, but it must not display an unavailable condition as selected.

## Structure-dependent consistency remains downstream

This policy cannot determine facts not present in `ScoringConditionsDraft`.
The scoring-input/scoring boundary therefore remains responsible for structure-dependent consistency such as:

- riichi/double-riichi requiring a closed hand;
- Tenhou/Chiihou requiring a closed, no-meld winning structure;
- Rinshan requiring a kan in the winner's logical meld structure.

The UI may later use structure-aware feedback to explain such failures, but it must not duplicate the scoring solver or concrete scoring-library behavior.

## Scoring boundary remains defensive

`ScoringService` does not silently normalize contradictory input.
The scoring boundary continues to return `invalid-input` for contradictory drafts and to reject contradictory strict input defensively.

This creates two distinct responsibilities:

```text
ScoringConditionPolicy
  -> keep ordinary Application/UI interaction coherent

ScoringService
  -> validate the supplied scoring request defensively
```

The condition policy must not implement winning-shape, yaku, fu, or point logic.

## Boundary

| concern | owner |
|---|---|
| Condition dependency normalization and availability | This contract. |
| Stored scoring-session conditions | Application scoring-session contract. |
| Visible condition controls | UI condition-controls concept. |
| Structure-dependent and defensive scoring-input validation | Scoring API/input contracts. |
