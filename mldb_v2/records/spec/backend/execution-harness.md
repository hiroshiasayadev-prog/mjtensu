# Contract: Backend execution harness

- **id**: `spec:mldb.v2.backend.execution_harness`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.backend`
- **contract_class**: `execution`

## Purpose

Execution backends schedule processes; they do not define MLDB domain invocation. Every backend
adapter starts the same MLDB execution harness for one immutable planned stage coordinate.

The harness is not a queue, worker registry, scheduler, lease system, or retry controller. It is the
backend-neutral process entrypoint that turns one Plan coordinate into one candidate terminal
outcome.

## Common preflight

The harness receives one `spec:mldb.v2.backend.stage_input`, checks out the Plan's pinned Git commit
or equivalent verified source snapshot, resolves the exact reusable definitions named by that input,
verifies companion/manifest/runtime-dependency integrity, and materializes required object-store
inputs into execution-local paths.
## Training execution

For training, the harness materializes the training Corpus, loads the selected Architecture and
Train Protocol companions, calls Architecture `build()` once to create the fresh context model,
constructs the exact `TrainContext` including a backend-neutral telemetry reporter, calls
`train(context)`, verifies that at least one valid scalar telemetry event was accepted, validates the
returned module against a fresh Architecture instance, serializes canonical weights, uploads them to
object storage, and returns the terminal backend candidate shape.

## Evaluation execution

For evaluation, the harness materializes the evaluation Corpus, resolves/downloads/verifies Model
lineage and canonical weights, builds a fresh Architecture module, strictly loads learned state,
loads the Evaluation Protocol companion, constructs the exact `EvaluationContext` including a
backend-neutral telemetry reporter, calls `evaluate(context)`, validates candidate metrics/artifacts,
automatically projects validated numeric metrics into telemetry, verifies that at least one valid
scalar telemetry event was accepted, publishes declared artifact candidates, and returns the terminal
backend candidate shape.

Candidate success is not canonical success. Collection/result acceptance validates lineage and
formal contracts before writing Training/Evaluation Results.

Telemetry acceptance and backend delivery are separate. Invalid telemetry calls are Protocol
contract failures. Failure of the backend/UI telemetry destination after a valid event is accepted is
non-canonical and MUST NOT turn an otherwise valid stage into failure. The harness/backend adapter
therefore treats telemetry delivery as best-effort projection.
