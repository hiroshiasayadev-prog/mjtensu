# Contract: Evaluation callable interface

- **id**: `spec:mldb.v2.evaluation.evaluate_interface`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.evaluation`
- **contract_class**: `interface`

## Entrypoint

Evaluation Protocol v1 companion modules expose exactly:

```python
def evaluate(context: EvaluationContext) -> EvaluationCandidate:
    ...
```

Generic MLDB resolves and loads the canonical Model before invocation. Protocol code does not resolve
MLDB files, backend Tasks, object-store credentials, or canonical result paths itself.

## Context shape

```python
@dataclass(frozen=True)
class MaterializedCorpus:
    definition: Corpus
    root: Path

@dataclass(frozen=True)
class LoadedModel:
    definition: Model
    training_result: TrainingResult
    architecture: Architecture
    module: torch.nn.Module

@dataclass(frozen=True)
class EvaluationContext:
    task: Task
    corpus: MaterializedCorpus
    model: LoadedModel
    parameters: Mapping[str, PublicParameterValue]
    telemetry: TelemetryReporter
    work_dir: Path
```
`Task`, `Corpus`, `Model`, `TrainingResult`, and `Architecture` mean immutable parsed canonical values;
concrete Python classes are frozen by Skeleton. `corpus.root` matches the sealed Corpus manifest.
`model.module` is a fresh Architecture module with canonical learned state loaded strictly according
to `spec:mldb.v2.training.canonical_weights`. `parameters` is the complete resolved Evaluation
Protocol mapping. `telemetry` is the backend-neutral reporter from
`spec:mldb.v2.common.telemetry`. `work_dir` is an execution-local writable directory.

Backend IDs, queue names, credentials, raw S3 clients, and ClearML objects are not context fields.

## Candidate response

```python
@dataclass(frozen=True)
class EvaluationCandidate:
    metrics: Mapping[str, int | float]
    artifacts: Mapping[str, Path]
```

`metrics` and `artifacts` keys are declaration keys from the selected Evaluation Protocol. Metric
values are finite scalars matching the declared `integer|number` type; boolean is invalid. Artifact
paths are regular files beneath `work_dir`. An output MUST NOT appear under an undeclared key.

Required declared outputs must be present. Optional declared outputs may be absent; v2 has no partial
success state and no separate unavailable-output value. Files under `work_dir` but absent from
`artifacts` are non-canonical working files.

Protocol code owns preprocessing, inference, decoding, matching, thresholding, aggregation, and metric
algorithms. Generic MLDB owns artifact publication/integrity, declaration validation, and promotion
into a canonical Evaluation Result. Returning from `evaluate()` alone does not establish success.
Validated numeric candidate metrics are automatically projected by the execution harness into
telemetry; Protocol code may report additional intermediate or sweep observations. A successful
evaluation MUST establish at least one valid scalar telemetry point.
