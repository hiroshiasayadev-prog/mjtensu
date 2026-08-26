# Concept: Condition controls

- **id**: `spec:product.ui.components.condition_controls`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.ui.components`

## What this is

Reusable visible responsibility for entering non-image scoring conditions while keeping one internally consistent scoring context.

## Primary controls

The ordinary visible set contains:

- Ron / Tsumo;
- round wind;
- seat wind;
- riichi state: none / riichi / double riichi;
- ippatsu when applicable.

## Secondary situational controls

Less-common conditions may be grouped behind one secondary disclosure:

- Rinshan kaihou;
- Chankan;
- Haitei;
- Houtei;
- Tenhou;
- Chiihou.

Nagashi mangan is not part of this control set.

## Consistency behavior

Controls must not intentionally create two independent sources of truth for the same scoring fact.
In particular:

- dealer status is derived from seat wind East;
- ippatsu is unavailable without riichi/double riichi;
- rinshan and haitei require tsumo-compatible context;
- chankan and houtei require ron-compatible context;
- tenhou requires dealer-compatible context;
- chiihou requires non-dealer-compatible context;
- mutually exclusive situational outcomes cannot remain simultaneously selected.

When one control change makes another selected condition impossible, the dependent condition is cleared immediately and its unavailable control is shown off and disabled/unselectable (or hidden when that is clearer for a secondary condition).

The UI derives availability from `spec:product.system.contracts.scoring_condition_policy` and must not maintain an independent copy of the dependency rules. Application condition updates use the same policy normalization, so stored values and visible control state remain synchronized.

At minimum:

- selecting Ron clears and disables Rinshan and Haitei;
- selecting Tsumo clears and disables Chankan and Houtei;
- selecting Riichi = None clears and disables Ippatsu;
- an East seat clears and disables Chiihou;
- a non-East seat clears and disables Tenhou.

The scoring boundary still validates contradictory input defensively rather than silently normalizing it.

## Edit focus

A page may request that one control be initially emphasized when entered through a shortcut.
For example, the Result `親` / `子` status action may open Conditions with seat wind emphasized.
This does not create a separate dealer-state control.

## Non-goals

- House-rule configuration.
- Full game-state entry.
- Kyoku/honba/riichi-stick settlement.
- Concrete form-library implementation.

## Boundary

| concern | owner |
|---|---|
| Visible condition-control semantics | This concept. |
| Scoring validity | `spec:product.scoring.input`. |
| Page placement | `spec:product.ui.pages.conditions`. |
| Session state and recalculation | `spec:product.application.scoring_session`. |
