# MLDB v2 Telemetry Design Guide

Telemetry is the backend-neutral observational stream used to understand Training/Evaluation behavior while work runs or after it finishes.

It is deliberately **not** canonical experiment truth. Canonical Training Result, Evaluation Result, Model, Study Result, and artifact references remain authoritative.

Formal contract: `../records/spec/common/telemetry.md`.

## 1. Reporter contract

Protocol code receives a `TelemetryReporter` through its callable context and may report scalar observations with:

    report_scalar(group=<str>, series=<str>, value=<int|float>, step=<int>)

Validation is strict:

- `group`: exact non-empty string;
- `series`: exact non-empty string;
- `value`: exact finite `int` or `float`; booleans are invalid;
- `step`: exact non-negative `int`; booleans are invalid;
- no coercion.

A malformed report is a Protocol contract violation. A valid event is accepted before any backend delivery attempt.

Protocol code must never import ClearML or another backend SDK to publish telemetry.

## 2. Required observability

A successful Train Protocol must establish at least one accepted scalar event that is materially useful for judging optimization progress, training quality, or another Protocol-relevant property.

A successful Evaluation must also establish at least one useful scalar event. Validated numeric `EvaluationCandidate.metrics` are projected automatically by the common execution harness, so an Evaluation Protocol does not need to re-report its terminal metrics manually. It may still emit intermediate/sweep telemetry when useful.

Artifact-only Evaluation is valid only when the Protocol explicitly emits at least one useful scalar observation.

The generic MLDB layer intentionally does not prescribe `loss`, `accuracy`, epoch cadence, batch cadence, or a universal step meaning.

## 3. Telemetry review template

When creating or semantically changing a Train/Evaluation Protocol, answer these questions before sealing it:

| Review item | Question |
|---|---|
| Decision purpose | What experiment/debugging decision will this observation support? |
| Group | What stable presentation group should related series share? |
| Series | What exact quantity is being observed? |
| Value semantics | Raw value, mean, weighted mean, rate, score, etc.? |
| Cadence | When is one point emitted: batch, epoch, validation pass, sweep coordinate, other? |
| Step semantics | What does step 0/1/N mean for this Protocol? |
| Cost | Does observing it add validation work, GPU sync, CPU transfer, or excessive volume? |
| Success usefulness | Would the emitted point(s) actually help judge the run, or only satisfy the minimum count? |
| Canonical relation | Is the value telemetry-only, or also a terminal Evaluation metric? |

Write the chosen semantics into the Protocol review/task evidence or nearby maintainable documentation. Do not rely on the chart title alone to explain them.
## 4. Selection heuristics

Prefer a small set of observations that explain the optimization/model-selection process. More charts are not automatically better.

Good defaults when they match the actual Protocol:

- report at a natural completed unit such as an epoch or validation pass rather than every batch;
- aggregate detached numeric values so telemetry does not retain computation graphs;
- avoid a GPU synchronization for every batch just to draw a chart;
- expose the quantity the Protocol actually uses to judge progress or select a best state;
- keep presentation values intuitive even when internal selection transforms them (for example, report positive loss even if a lexicographic key stores negative loss for maximization).

Do not invent generic semantics such as `step = epoch` unless that is the reviewed meaning for this Protocol.

## 5. Proven Protocol-specific examples

### Tile classifier Train Protocol v4

The classifier training loop has a meaningful per-epoch optimization loss and no per-epoch validation selection. Its reviewed telemetry is:

- group: `optimization`
- series: `cross_entropy_loss`
- value: sample-weighted mean of per-batch mean cross entropy over the completed epoch
- cadence: once per completed epoch
- step: 1-based completed epoch (`1..N`)

Loss accumulation is detached and only the completed epoch aggregate becomes a Python float. No learning-rate series is emitted because it was not required for the intended experiment decisions.

### Rotated FCOS Train Protocol v3

The detector already performs validation after each completed training epoch and uses a lexicographic model-selection key. Its reviewed telemetry is:

- group: `validation`
- series: `f1`, `recall`, `mean_iou`, `loss`
- cadence: once per series after each completed validation epoch
- step: 1-based completed epoch
- `loss`: positive mean validation loss, even though the model-selection tuple stores negative loss internally for maximization

Early stopping produces no phantom future steps; only completed validation epochs are reported.

These are examples of good Protocol-specific decisions, not reserved names.

## 6. Evaluation automatic projection

After an Evaluation Protocol returns an `EvaluationCandidate`, the runtime validates its declared metrics first. Only validated numeric metrics are then accepted as telemetry.

The current deterministic terminal-summary mapping is:

- group = exact Evaluation stage name;
- series = exact metric declaration key;
- value = validated numeric metric value;
- step = `0`.

Do not loosen candidate metric validation just to make a chart appear.

## 7. Backend delivery semantics

Accepted and delivered are different states. Once a valid event is accepted, a ClearML logger/network/reporting failure is best-effort observational loss and must not turn an otherwise valid stage into failure.

ClearML currently maps `group -> title`, `series -> series`, `value -> value`, and `step -> iteration`. Protocols must not depend on that backend mapping.

## 8. Versioning rule

Changing the scientific meaning, aggregation, cadence, or step semantics of telemetry changes the reviewed behavior of a Train/Evaluation Protocol. If the Protocol is sealed, create a new version ID rather than silently rewriting the historical definition.

## 9. Pre-seal telemetry checklist

Before sealing a Protocol, confirm: at least one useful successful-run scalar exists; group/series names are stable; value aggregation is defined; cadence and step meaning are explicit; telemetry cost is acceptable; no backend SDK is imported; reporter failures are not swallowed; and terminal canonical metrics are not confused with observational charts.