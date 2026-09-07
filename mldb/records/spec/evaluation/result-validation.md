# Reference: Evaluation result validation

- **id**: `spec:mldb.evaluation.result_validation`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.evaluation`

## What this is

Defines generic validation and materialization rules applied to `EvaluationResult` after one Evaluation Protocol invocation.

The runtime validates only the formal contract declared by the selected Evaluation Protocol and registered artifact schemas. Model-family-specific correctness remains inside protocol code.

## Scalar metric validation

Every metric declared by `outputs.metrics` must have exactly one concrete Run outcome: a valid returned scalar or an explicit unavailable-output report.

| condition | result |
|---|---|
| Returned metric key is declared by `outputs.metrics` | Continue validation. |
| Returned metric key is undeclared | Fail the Evaluation Run. |
| Returned value is finite `int` or `float` | Accept the scalar value. |
| Returned value is boolean | Fail the Evaluation Run. |
| Returned value is string, list, mapping, or another non-numeric structure | Fail the Evaluation Run. |
| Returned value is NaN or positive/negative infinity | Fail the Evaluation Run. |
| Declared metric is explicitly reported unavailable and is not returned | Record the unavailable output and make the Run partial if no failure condition applies. |
| Declared metric is neither returned nor explicitly reported unavailable | Fail the Evaluation Run for silent omission. |
| Metric is both returned and reported unavailable | Fail the Evaluation Run. |

Accepted scalar metrics are stored directly under `EvaluationRun.result.metrics`.
Explicitly unavailable metrics are not stored as scalar placeholders such as `null` or NaN.

When the sealed Evaluation Protocol declares no metrics, a complete or partial Evaluation Run records an empty `result.metrics` mapping.

MLDB v1 does not assign universal meaning to a metric key across different Evaluation Protocols.

## Structured artifact validation

Every returned or explicitly unavailable artifact key is matched against `outputs.artifacts` from the sealed Evaluation Protocol.

| condition | result |
|---|---|
| Returned artifact key is undeclared | Fail the Evaluation Run. |
| Unavailable artifact reference is undeclared | Fail the Evaluation Run. |
| Declared required artifact is neither returned nor reported unavailable | Fail the Evaluation Run. |
| Declared required artifact is explicitly reported unavailable | Fail the Evaluation Run. |
| Declared optional artifact is omitted without an unavailable report | Accept omission without making the Run partial. |
| Declared optional artifact is explicitly reported unavailable | Record the unavailable output and make the Run partial if no failure condition applies. |
| Artifact is both returned and reported unavailable | Fail the Evaluation Run. |
| Returned path does not identify a file beneath the Evaluation Run `work/` directory | Treat the artifact as invalid. |
| Returned file does not match the declared formal format | Treat the artifact as invalid. |
| Registered generic schema validation fails | Treat the artifact as invalid. |
| Required artifact is invalid | Fail the Evaluation Run. |
| Optional artifact is invalid | Omit it from formal results, record a validation issue, and make the Run partial if no failure condition applies. |

A file's existence beneath `work/` does not make it a formal artifact. It must be explicitly returned through `EvaluationResult.artifacts` under a declared key.

## Formal artifact import

For each accepted structured artifact, generic MLDB tooling performs:

1. validate the returned key against the sealed protocol declaration;
2. verify the candidate file resides beneath the Run `work/` directory;
3. validate the declared format;
4. apply the registered schema contract when generic validation exists for that schema;
5. copy or move the accepted bytes into the Evaluation Run `artifacts/` directory;
6. calculate SHA-256 and byte size;
7. record canonical Run-relative path, format, schema, SHA-256, and byte size under `result.artifacts`.

After terminal completion, accepted formal artifact bytes and their recorded integrity metadata are immutable.

## Required and optional behavior

| declaration | omitted silently | explicitly unavailable | returned but invalid | returned and valid |
|---|---|---|---|---|
| metric declaration | Run fails. | `completed_partial` if no failure condition applies. | Run fails. | Accept metric. |
| artifact `required: true` | Run fails. | Run fails. | Run fails. | Import and record. |
| artifact `required: false` | Valid omission; complete status remains possible. | `completed_partial` if no failure condition applies. | Omit, record `validation_issues`, and use `completed_partial` if no failure condition applies. | Import and record. |

An optional artifact validation problem must not discard otherwise valid scalar metrics or required artifacts.

## Completion selection

| validated outcome | terminal status |
|---|---|
| All declared metrics returned valid, all required artifacts valid, and no returned optional artifact rejected | `completed`. |
| At least one declared metric is explicitly unavailable, or one optional artifact is explicitly unavailable or rejected, with every completion-critical contract otherwise valid | `completed_partial`. |
| Any execution or completion-critical contract failure occurs | `failed`. |

Failure takes precedence over partial completion.
A `completed_partial` Run must retain at least one accepted formal metric or accepted formal artifact. If no trustworthy formal output remains, the Evaluation Run fails rather than using `completed_partial`.

## Working-file boundary

Protocol code may create any diagnostics it needs beneath `work/`, including formats not registered as formal MLDB artifacts.

Such files remain working material unless the sealed Evaluation Protocol declares them and `EvaluationResult` returns them for formal validation.

New artifact families require an explicit versioned schema contract before generic MLDB tooling treats them as standardized formal structured outputs.

## Failure isolation

A formal-result validation failure or partial-completion condition affects only the Evaluation Run being finalized.

Generic validation must not cancel unrelated Evaluation Runs, Models, Training Runs, or independent Study branches.

## Boundary

| concern | owner |
|---|---|
| Declared metric and artifact keys | `spec:mldb.evaluation.evaluation_protocol_format`. |
| `EvaluationResult` return shape | `spec:mldb.evaluation.evaluate_interface`. |
| Persisted accepted result metadata | `spec:mldb.evaluation.evaluation_run_format`. |
| Run state resulting from validation | `spec:mldb.evaluation.evaluation_run_lifecycle`. |
| Concrete structured artifact content schemas | `spec:mldb.evaluation.artifacts`. |
| Protocol-specific metric formula correctness | Evaluation Protocol implementation. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.evaluation` | Parent evaluation overview. |
| `spec:mldb.evaluation.evaluate_interface` | Supplies candidate formal results. |
| `spec:mldb.evaluation.evaluation_protocol_format` | Declares the formal result contract. |
| `spec:mldb.evaluation.evaluation_run_format` | Records accepted formal results. |
