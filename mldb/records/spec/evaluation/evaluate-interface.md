# Contract: Evaluation callable interface

- **id**: `spec:mldb.evaluation.evaluate_interface`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.evaluation`
- **contract_class**: `interface`

## What this is

Defines the common executable boundary used to invoke one resolved Evaluation Protocol.

The interface standardizes context and returned formal-result candidates without standardizing model-family-specific preprocessing, inference, matching, or metric algorithms.

## Request

Evaluation Protocol v1 exposes:

```python
def evaluate(context: EvaluationContext) -> EvaluationResult:
    ...
```

Conceptually, the runtime supplies:

```python
@dataclass(frozen=True)
class EvaluationContext:
    task: TaskHandle
    corpus: CorpusHandle
    model: ModelHandle
    parameters: Mapping[str, Any]
    work_dir: Path
```

| context field | contract |
|---|---|
| `task` | Resolved Task shared by the selected Model, Corpus, and Evaluation Protocol. |
| `corpus` | Resolved immutable evaluation Corpus. |
| `model` | Resolved learned Model. The handle can load the learned module through Model lineage. |
| `parameters` | Complete resolved mapping of the Evaluation Protocol's public parameters. |
| `work_dir` | Evaluation Run working directory available to protocol-owned temporary and candidate result files. |

Before Evaluation Run allocation, launch preflight must resolve Model, Corpus, and Evaluation Protocol, verify sealed/static integrity requirements and Task compatibility, and resolve complete public parameters. After allocation, the executor loads the executable implementation and constructs this context.

Evaluation Run v1 has no universal evaluation seed field. A stochastic protocol that needs caller-controlled seed must publish `seed` as an ordinary public parameter.

## Response

A successful protocol call returns conceptually:

```python
@dataclass(frozen=True)
class UnavailableOutput:
    output: str
    type: str
    message: str

@dataclass(frozen=True)
class EvaluationResult:
    metrics: Mapping[str, int | float]
    artifacts: Mapping[str, Path]
    unavailable_outputs: Sequence[UnavailableOutput]
```

| response field | contract |
|---|---|
| `metrics` | Candidate formal scalar metrics keyed by Evaluation Protocol declaration. |
| `artifacts` | Candidate formal structured artifact paths keyed by Evaluation Protocol declaration. |
| `unavailable_outputs` | Explicit reports for declared formal outputs that the protocol could not legitimately compute or produce for this Run. May be empty. |

Each unavailable-output report contains:

| field | contract |
|---|---|
| `output` | Exactly one declared output reference using `metrics.<key>` or `artifacts.<key>`. |
| `type` | Non-empty short machine-readable reason string. MLDB v1 defines no universal reason enum. |
| `message` | Concise human-readable explanation. |

An output must not be both returned and reported unavailable.
A declared metric must be either returned with a valid scalar value or explicitly reported unavailable.
A required structured artifact must not be reported unavailable without failing the Evaluation Run.

Returned artifact paths must refer to files beneath `context.work_dir`.

Files written beneath `work_dir` but omitted from `EvaluationResult.artifacts` remain protocol-owned working files and are not formal results.

The runtime validates `EvaluationResult` after return. Returning from `evaluate()` does not by itself complete the Evaluation Run.

## Errors

| condition | result |
|---|---|
| Model, Corpus, or Evaluation Protocol resolution fails during preflight | Reject the launch request; do not allocate an Evaluation Run. |
| Required static compatibility or integrity validation fails during preflight | Reject the launch request; do not allocate an Evaluation Run. |
| Public parameter resolution fails during preflight | Reject the launch request; do not allocate an Evaluation Run. |
| Sibling Python implementation cannot load after Run allocation | Fail the affected Evaluation Run. |
| Declared `evaluate` entrypoint is missing or not callable after Run allocation | Fail the affected Evaluation Run. |
| `evaluate(context)` raises | Fail the affected Evaluation Run. |
| Return value does not satisfy the EvaluationResult boundary | Fail the affected Evaluation Run. |
| `unavailable_outputs` references an undeclared output or conflicts with a returned output | Fail the affected Evaluation Run. |
| Returned formal result violates metric or artifact contracts | Apply evaluation result-validation rules. |

One invocation failure must not implicitly cancel unrelated Runs.

## Rules

- Protocol code owns evaluation-specific preprocessing, inference, decoding, matching, thresholding, aggregation, and metric calculation.
- Generic runtime must not branch on model family to implement protocol-specific evaluation algorithms.
- Protocol code must consume published values from `context.parameters` rather than substitute hidden defaults for those keys.
- Protocol code may load the learned module through `context.model` rather than hard-code Training Run or Architecture paths.
- Protocol code may write arbitrary diagnostics beneath `work_dir`.
- Protocol code must return only files intended to enter formal result validation under `EvaluationResult.artifacts`.
- Protocol code must report legitimate declared-output unavailability explicitly rather than returning NaN or silently omitting a declared metric.
- Generic runtime owns validation, partial/completed status selection, import, hashing, and immutable recording of accepted formal results.

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.evaluation` | Parent evaluation overview. |
| `spec:mldb.evaluation.evaluation_protocol_format` | Declares the entrypoint, parameters, metrics, and artifact keys. |
| `spec:mldb.evaluation.result_validation` | Validates the returned EvaluationResult. |
| `spec:mldb.model.identity` | Defines learned Model loading lineage. |
