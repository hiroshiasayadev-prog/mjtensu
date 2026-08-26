# Contract: Screen flow

- **id**: `spec:product.ui.screen_flow`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.ui`

## What this is

Navigation contract for the first scoring flow.
Application owns scoring-session state; this contract owns which visible surface follows each accepted user or recognition outcome.

## Primary flow

```text
Top
  |
  | 判定する
  v
Recognition
  |
  | stable recognition auto-confirmed
  v
Conditions
  |
  | calculate
  v
Result
```

There is no mandatory standalone recognition-confirmation page in the primary path.
Conditions provides the first ordinary post-recognition view of the recognized hand, pre-calculation recognition correction, winning-tile selection, and scoring-condition entry.

## Result recovery paths

```text
Result
  +-- 認識結果を修正 --> Recognition correction
  |                         +-- corrected preview ready ----> recalculate -> Result
  |                         +-- no-yaku / input repair ----> Conditions -> recalculate -> Result
  +-- 条件を修正 -----> Conditions ---------------------------> recalculate -> Result
  +-- もう一度判定 ---> Recognition (new session)
```

- Recognition correction preserves the current non-image conditions when the corrected structure is installed.
- If the corrected structure remains calculation-ready with yaku, it is recalculated immediately and replaces Result.
- If a confirmed corrected structure produces `no-yaku`, `incomplete`, or `invalid-input`, the active corrected session continues to Conditions instead of restoring the stale pre-correction Result.
- In the `no-yaku` case Conditions shows `役なし`, allowing condition-derived yaku such as riichi to be supplied or changed.
- Condition correction preserves recognized structure and winning-tile selection.
- New recognition discards the current scoring session before live recognition starts.
- A failed recalculation must not return to Result with stale point information shown as current.

## Help flow

```text
Top <--> Help
```

Help explains the capture layout and ordinary operation without being part of the required scoring path.

## Recognition transition

Recognition leaves the live camera surface after `spec:product.recognition.runtime_recognition` commits a stable recognized structure. The structure does not need to have already passed winning-shape or yaku validation.
The UI must not require a shutter or an additional `OK` confirmation after stability succeeds.

If recognition is not yet acceptable, the user remains on Recognition with live feedback.

## Conditions transition

Conditions lets the user correct the recognized structure and exposes calculation only when required scoring input is complete and internally consistent.
The initial winning-tile selection is the rightmost completed-hand tile supplied by Application and can be changed before calculation.

When Conditions was opened from Result as a correction surface, successful calculation returns to Result rather than starting a new session.

## Back navigation

Back behavior must not silently create a second scoring session or apply stale results.
At minimum:

- Help -> Top returns without creating a scoring session.
- Recognition -> Top may abandon the current recognition attempt.
- Automatic Recognition -> Conditions navigation replaces the transient Recognition history entry. Normal back navigation from initial Conditions returns to Top rather than implicitly reopening the completed camera run; starting Recognition again is an explicit new-recognition action.
- Conditions reached directly from Result for condition correction may cancel edits and return to the unchanged Result state. Conditions reached after a confirmed recognition correction must not restore the stale pre-correction Result.
- Recognition correction may cancel edits and return to the unchanged Result state.

Concrete browser-history mechanics are implementation-owned as long as these semantic outcomes are preserved.

## Non-goals

- URL structure and router implementation.
- Browser-history API choice.
- Animation and transition styling.
- Application state internals.

## Boundary

| concern | owner |
|---|---|
| Visible screen destinations | This contract. |
| State preserved/discarded by correction and restart | `spec:product.application.scoring_session`. |
| Page content and layout | `spec:product.ui.pages`. |
| Recognition acceptance | `spec:product.recognition.runtime_recognition`. |
