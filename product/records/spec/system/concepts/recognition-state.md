# Concept: Realtime recognition state

- **id**: `spec:product.system.concepts.recognition_state`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.system`

## What this is

State model for shutterless realtime recognition and automatic confirmation.
The state machine separates runtime initialization, open-ended scanning, semantic stabilization, and one committed result.

## States

```text
INITIALIZING
    ↓ runtime ready
SCANNING
    ↓ eligible candidate
STABILIZING
    ↓ same semantic draft for 3 consecutive evaluations
CONFIRMED
```

### INITIALIZING

Recognition runtime/model dependencies are not yet ready to evaluate frames.
No stabilization state exists.

### SCANNING

Frames are evaluated continuously, but there is currently no eligible stabilization run.
Live observations may still be emitted for UI feedback.

### STABILIZING

The recognizer has an eligible semantic draft and is counting consecutive equivalent evaluations.
The count is internal state and is not required as user-visible progress.

### CONFIRMED

Exactly one stable `RecognizedStructure` has been materialized and delivered to Application.
The live recognition run no longer accepts another result unless a new run/reset is started.

## Commit-eligibility gate

A frame may start or continue stabilization only when all of the following hold:

1. at least `10` valid non-dora tile observations are present after detector duplicate suppression and tile classification;
2. at least `2` of those valid observations are in the `completed-hand` region;
3. meld-region observations, when present, can be assigned to stable spatial meld groups.

The visible-tile count includes valid tile observations in:

- `completed-hand`;
- `melds`.

It excludes:

- `dora-indicators`;
- invalid/background classifier outcomes;
- detector duplicates removed by duplicate suppression;
- detections outside the semantic capture regions or in composite padding/separators.

The count is based on actual visible recognized observations, not logical expansion. A concealed kan contributes the two visible face-up observations to this gate even though the recognition draft reconstructs a four-tile logical kan.

The combined thresholds permit the minimum visible non-dora case of a hand containing four concealed kans: two completed-hand observations plus two visible observations for each of four concealed kans.

## What eligibility does not validate

Commit eligibility must not validate:

- winning-hand shape;
- yaku existence;
- fu or points;
- whether a spatially reconstructed three/four-tile meld currently forms a legal scoring meld;
- the selected winning tile.

These remain Conditions/Application/scoring concerns.

## Stabilization transition rules

For each completed frame evaluation:

- an ineligible snapshot clears the current stabilization run and returns to `SCANNING`;
- an eligible snapshot whose semantic draft differs from the current candidate starts a new run at consecutive count `1`;
- an eligible snapshot semantically equal to the current candidate increments the consecutive count;
- count `3` confirms the candidate and transitions to `CONFIRMED`;
- semantic equality is defined by `spec:product.system.concepts.recognition_model` and excludes bounding-box jitter.

No fixed startup delay is required before recognition begins. Stabilization itself provides the temporal evidence needed for auto-confirmation.

## Scheduling and backpressure

While the Recognition page is active:

- target recognition-request cadence is one request every `100 ms`;
- at most one frame evaluation may own acceptance at a time;
- if the previous evaluation is still running at the next cadence point, stale work must not be queued merely to preserve cadence;
- the next accepted evaluation should use the latest available frame.

This intentionally prefers fresh camera state over processing every captured frame.

## Reset semantics

Reset clears:

- the current candidate draft;
- consecutive stabilization count;
- any previously confirmed result owned by the current live run.

Reset does not itself redefine model/runtime initialization state. Runtime lifecycle is owned by the recognition API contract.

## Failure boundary

A model/runtime execution failure is not an ineligible recognition frame and must not be silently converted into `SCANNING`.
Runtime failures are surfaced through the recognition error boundary so infrastructure failure cannot masquerade as ordinary camera content.
