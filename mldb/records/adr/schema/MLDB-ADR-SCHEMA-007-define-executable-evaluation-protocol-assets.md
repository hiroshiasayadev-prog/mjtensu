# MLDB-ADR-SCHEMA-007: Define executable Evaluation Protocol assets

- **status**: accepted
- **date**: 2026-09-03
- **depends_on**: MLDB-ADR-SCHEMA-001, MLDB-ADR-SCHEMA-002, MLDB-ADR-SCHEMA-006
- **supersedes**:
- **migrated_to_spec**:

## Context

MLDB needs a reusable way to evaluate learned Models under repeatable conditions while supporting automated experiment sweeps and queued execution.

Training-time validation used by a Train Protocol to select its learned result is part of training behavior. Post-training evaluation has a different responsibility: it measures an already-created Model on a selected Corpus so Models can be compared under common conditions, fed into visualization systems such as MLflow or a project-owned dashboard, and later used by promotion or release decisions.

The project needs substantially different evaluation logic for categorical tile classifiers and object detectors. A universal MLDB evaluator would therefore have the same problem as a universal trainer: model-family-specific logic would accumulate in generic infrastructure.

At the same time, Evaluation Protocol outputs cannot be completely unconstrained. Queue-driven sweeps may generate hundreds of Evaluation Runs, and downstream tooling must be able to compare scalar metrics and consume structured outputs without knowing arbitrary protocol-specific filenames or undocumented table layouts.

Evaluation Protocol therefore needs a small executable boundary, an explicit public parameter interface, declared scalar metrics, and declared structured artifacts with versioned schemas.

## Decision

Introduce `EvaluationProtocol` as a versioned executable MLDB asset that defines how a compatible learned Model is evaluated for one Task.

An Evaluation Protocol consists of a YAML metadata record and a sibling Python implementation file.

MLDB v1 standardizes the entrypoint:

```python
def evaluate(context: EvaluationContext) -> EvaluationResult:
    ...
```

The Evaluation Protocol performs all task- and model-family-specific evaluation logic. Generic MLDB tooling resolves the Model and Corpus, supplies published parameter values, invokes the protocol, validates the declared result contract, stores scalar metrics, and imports declared structured artifacts into MLDB-owned Evaluation Run artifacts.

### Physical placement and basename contract

Evaluation Protocol assets live under:

```text
mldb_data/
  evaluation_protocols/
```

Each Evaluation Protocol occupies two sibling files:

```text
mldb_data/evaluation_protocols/<evaluation-protocol-id>.yaml
mldb_data/evaluation_protocols/<evaluation-protocol-id>.py
```

Example:

```text
mldb_data/evaluation_protocols/tile-classifier-standard-eval-v1.yaml
mldb_data/evaluation_protocols/tile-classifier-standard-eval-v1.py
```

The basename is the immutable Evaluation Protocol ID. The Python path is derived from this rule and is not repeated in metadata.

### ID and lifecycle

Every Evaluation Protocol ID must end with:

```text
-v<positive-integer>
```

Examples:

```text
tile-classifier-standard-eval-v1
tile-classifier-angle-robustness-v1
rotated-detector-holdout-eval-v1
```

Evaluation Protocol status is one of:

```text
draft
sealed
```

A `draft` protocol may be edited freely while being implemented and tested.

A `sealed` protocol is immutable as an executable evaluation definition and must never return to `draft`.

Any change after sealing that can alter evaluation results, parameter meaning, output metric meaning, or structured artifact schema requires a new Evaluation Protocol revision.

### Metadata format

Evaluation Protocol metadata uses:

```text
mjtensu.mldb/evaluation-protocol/v1
```

Example:

```yaml
schema: mjtensu.mldb/evaluation-protocol/v1

id: tile-classifier-standard-eval-v1
status: sealed

task: tile-shape-classification-35-v1

name: Standard tile classifier evaluation

description: >
  Evaluate a learned tile classifier on a selected compatible Corpus
  using the protocol-defined preprocessing and prediction procedure.

implementation:
  entrypoint: evaluate
  sha256: 0123456789abcdef...

parameters:
  batch_size:
    default: 4096

outputs:
  metrics:
    accuracy:
      type: number
      description: Overall categorical accuracy.
    cross_entropy:
      type: number
      description: Mean categorical cross entropy.

  artifacts:
    predictions:
      format: jsonl
      schema: mjtensu.mldb/eval-artifact/categorical-predictions/v1
      required: true

    confusion_matrix:
      format: csv
      schema: mjtensu.mldb/eval-artifact/confusion-matrix/v1
      required: false
```

The required fields are:

- `schema`;
- `id`;
- `status`;
- `task`;
- `name`;
- `description`;
- `implementation.entrypoint`;
- `parameters`;
- `outputs.metrics`;
- `outputs.artifacts`.

`implementation.sha256` is required when `status: sealed` and may be omitted while `status: draft`.

`parameters`, `outputs.metrics`, and `outputs.artifacts` are mappings and may be empty.

### Public parameter interface

Evaluation Protocol parameters follow the same automation principle as Train Protocol parameters.

Each key under `parameters` is explicitly public and must contain a `default` value:

```yaml
parameters:
  batch_size:
    default: 4096
  confidence_threshold:
    default: 0.25
```

For one concrete Evaluation Run, generic MLDB tooling starts from all protocol defaults, replaces only caller-supplied values for published keys, rejects unknown keys, records the complete resolved mapping, and passes it through `EvaluationContext.parameters`.

An empty `parameters: {}` mapping means the protocol exposes no run-varying parameters.

Evaluation logic that is not intentionally sweepable remains fixed in the sealed Python implementation. An Evaluation Run must not contain an unrestricted override mapping.

### EvaluationContext contract

The v1 context provides at least:

```text
task
corpus
model
parameters
work_dir
```

Conceptually:

```python
@dataclass(frozen=True)
class EvaluationContext:
    task: TaskHandle
    corpus: CorpusHandle
    model: ModelHandle
    parameters: Mapping[str, Any]
    work_dir: Path
```

The Model handle must provide access to the learned PyTorch module represented by the Model, conceptually:

```python
module = context.model.load()
```

Model loading resolves the Model's completed Training Run, its selected Architecture, and its canonical learned weights according to MLDB-ADR-SCHEMA-006.

The Corpus handle provides the immutable evaluation Corpus and metadata. The Task handle provides the common semantic prediction contract.

`work_dir` is protocol-owned execution space. The protocol may place temporary files, detailed diagnostics, visualizations, or candidate result files there. Files in `work_dir` are not automatically formal Evaluation Run outputs.

### EvaluationResult contract

A successful call returns:

```python
@dataclass(frozen=True)
class EvaluationResult:
    metrics: Mapping[str, int | float]
    artifacts: Mapping[str, Path]
```

`metrics` contains only scalar numeric result values.

`artifacts` maps declared structured-artifact keys to files written beneath `EvaluationContext.work_dir`.

The concrete implementation type may be defined later by MLDB runtime code; the semantic contract is fixed here.

### Scalar metric contract

Every metric returned by the protocol must be declared under `outputs.metrics`.

Metric keys not declared by the sealed protocol are invalid formal results.

For Evaluation Protocol v1, every declared metric uses:

```yaml
type: number
```

A returned metric value must be a finite numeric scalar. Boolean values, strings, lists, mappings, NaN, and positive or negative infinity are invalid as formal metrics.

Metrics are intended for direct comparison, sorting, filtering, dashboards, MLflow metric logging, and automated study summaries.

The metric name and description are protocol-local definitions. MLDB v1 does not introduce a separate universal `MetricDefinition` entity. Two protocols that use the same metric key are not assumed to have identical semantics unless their protocol definitions establish the same conditions.

Structured information such as confusion matrices, per-sample predictions, curves, distributions, failure cases, or per-class tables must not be encoded into scalar metrics.

### Structured artifact declaration

Every formal structured artifact must be declared under `outputs.artifacts` with:

- `format`;
- `schema`;
- `required`.

Evaluation Protocol v1 supports formal structured formats:

```text
jsonl
csv
json
```

A protocol may write any other file type under `work_dir`, including images or framework-specific diagnostics, but such files are not formal structured EvaluationResult artifacts unless a later MLDB decision adds their format and schema contract.

The key returned in `EvaluationResult.artifacts` must match a key declared by the protocol. Unknown artifact keys are invalid formal results.

If `required: true`, the protocol must return that artifact on successful evaluation. If `required: false`, it may be omitted.

### MLDB-owned artifact import

The Evaluation Protocol writes candidate structured outputs into `work_dir` and returns their paths.

Generic MLDB tooling then:

1. verifies that every returned artifact key is declared;
2. verifies that required artifacts are present;
3. verifies the declared format;
4. validates the artifact against its registered schema when generic validation is defined;
5. copies or moves the validated bytes into the Evaluation Run's MLDB-owned `artifacts/` directory;
6. records the canonical relative path, format, schema ID, SHA-256, and byte size.

This separates protocol-owned working files from formal, immutable Evaluation Run results.

A debug CSV or image written to `work_dir` does not become part of the formal result merely because it exists.

### Initial standard structured schemas

Evaluation Protocol v1 defines two initial reusable structured schemas.

#### `mjtensu.mldb/eval-artifact/categorical-predictions/v1`

Format:

```text
jsonl
```

Each non-empty line is one JSON object representing one evaluated sample.

Required fields:

```text
sample_id   string
 target     string
 prediction string
```

`sample_id` is the Corpus-local sample identifier.

`target` and `prediction` use the categorical Task label strings.

Additional fields are permitted so a protocol may include information such as confidence or task-specific diagnostics without changing the base schema. Such additional fields have no generic MLDB semantics.

#### `mjtensu.mldb/eval-artifact/confusion-matrix/v1`

Format:

```text
csv
```

The CSV uses long form with required columns:

```text
target,prediction,count
```

Each row records one target/prediction pair and its non-negative integer count.

`target` and `prediction` use Task label strings.

The combination of `target` and `prediction` must be unique within the table.

Additional columns are permitted but have no generic MLDB semantics.

New structured result schemas may be added by later MLDB design decisions as real evaluation needs appear. In particular, detector predictions, PR-curve tables, calibration data, and other specialized diagnostics should receive explicit versioned schemas before being promoted from protocol `work_dir` files to formal structured artifacts.

### Output validation responsibility

Generic MLDB tooling validates the parts of the result contract it understands.

At minimum it should be able to validate:

- returned metric keys against `outputs.metrics`;
- finite scalar metric values;
- returned artifact keys against `outputs.artifacts`;
- required artifact presence;
- supported artifact format;
- known standard artifact schemas;
- artifact file existence beneath the Evaluation Run work directory.

Protocol-specific correctness that cannot be usefully represented by generic schema remains the responsibility of the sealed Evaluation Protocol implementation.

### Project-owned evaluation behavior is self-contained

A sealed Evaluation Protocol must be self-contained with respect to project-owned result-affecting evaluation logic.

Project-owned code that determines preprocessing, inference procedure, matching, thresholding, metric calculation, aggregation, or formal artifact generation must live in the sibling `<evaluation-protocol-id>.py` file itself rather than being imported from another mutable project-local implementation module.

The protocol may import non-evaluation MLDB infrastructure such as context/result types, asset resolution, metadata access, filesystem utilities, and logging transport. It may also import third-party libraries. Exact third-party versions are execution-environment facts recorded by the later Evaluation Run.

### Training validation boundary

Evaluation Protocol is not the definition of training-time validation used solely to select a checkpoint.

If a Train Protocol performs validation during training to select its returned learned state, that validation behavior remains part of the Train Protocol and is frozen by the Train Protocol implementation hash.

Evaluation Protocol applies to an already-created Model and exists so Models can be measured independently and repeatedly under explicit post-training conditions.

### Task, Corpus, and Model relationships

An Evaluation Protocol references exactly one Task.

It does not permanently reference one Model or one Corpus.

The concrete Model and Corpus are supplied by the Evaluation Run so the same sealed Evaluation Protocol can evaluate many Models on the same holdout Corpus or the same Model on multiple compatible Corpora.

Before execution, generic tooling must verify that:

- the selected Corpus references the Evaluation Protocol Task;
- the selected Model resolves to an Architecture whose Task equals the Evaluation Protocol Task;
- referenced assets and their integrity records are valid.

### Revision triggers

After sealing, a new Evaluation Protocol revision is required for executable or semantic changes including, but not limited to:

- preprocessing or input transformation changes;
- model inference or decoding changes;
- matching or threshold changes;
- metric formulas or aggregation changes;
- metric key additions, removals, or semantic changes;
- public parameter key additions or removals;
- public parameter default changes;
- formal artifact key additions, removals, or requiredness changes;
- formal artifact format or schema changes;
- substantive changes to `evaluate()` or its result-affecting project-owned logic.

Draft changes do not require revision churn while the protocol remains `draft`.

## Rationale

The executable boundary keeps MLDB automation uniform without forcing classification and detection evaluation into one generic algorithm.

An explicit public parameter interface makes queue-driven sweeps practical while retaining a clear authority boundary: only protocol-published values may vary between Runs.

Restricting formal metrics to finite numeric scalars gives downstream comparison and visualization systems a predictable surface. MLflow, a custom dashboard, CLI summaries, SQL indexes, or future study tooling can all consume the same result values without understanding protocol internals.

Structured artifacts remain available for deeper analysis without turning the scalar metric map into an arbitrary nested document. Requiring declared artifact keys, formats, and versioned schemas prevents large automated experiment sets from degenerating into undocumented per-protocol output files.

Using `work_dir` as the unconstrained diagnostic space preserves flexibility. Only files explicitly returned under the sealed output contract become MLDB-owned formal artifacts.

Starting with a small number of real standard schemas avoids prematurely designing a universal evaluation-artifact ontology. Additional schemas can be introduced when concrete classifier, detector, robustness, or calibration workflows require them.

## Rejected alternatives

### Allow arbitrary metric keys and values

This would let protocols return nested mappings, arrays, strings, or changing metric names, making automated comparison and visualization unreliable.

Formal metrics are therefore declared in the sealed protocol and limited to finite numeric scalars.

### Treat every file in the evaluation directory as a result artifact

Protocols often produce debugging output, temporary files, framework logs, plots, or large intermediate data. Automatically registering all such files would make the formal result unstable and noisy.

Only declared files returned through `EvaluationResult.artifacts` become formal artifacts.

### Define one universal structured output table

Classification predictions, confusion matrices, detector boxes, PR curves, calibration tables, and failure-case diagnostics have materially different structures. A single universal table would become either too weak or excessively sparse.

MLDB uses explicit versioned schemas per structured artifact family instead.

### Introduce a separate MetricDefinition entity in v1

Metric meaning depends not only on a familiar name such as `precision` or `accuracy` but also on protocol-specific inference, matching, thresholds, preprocessing, and aggregation conditions.

The Evaluation Protocol already freezes those semantics. A separate metric ontology is deferred until a concrete cross-protocol reuse requirement appears.

### Bind Evaluation Protocol to one Model or Corpus

That would prevent straightforward comparison of multiple Models under one evaluation definition and would duplicate protocols whenever a holdout Corpus changes.

Model and Corpus are execution inputs supplied by Evaluation Run.

## Consequences

Future Evaluation Protocol tooling should be able to:

- discover sibling `.yaml` and `.py` assets under `mldb_data/evaluation_protocols/`;
- validate terminal `-vN` IDs and `draft` / `sealed` lifecycle;
- resolve and validate the referenced Task;
- verify sealed implementation hashes;
- resolve protocol defaults plus caller-supplied public parameter values;
- reject unknown parameter keys;
- construct `EvaluationContext` with Task, Corpus, Model, resolved parameters, and work directory;
- invoke `evaluate(context)`;
- validate returned scalar metrics against the declared metric keys;
- reject non-finite or non-numeric formal metric values;
- validate returned structured artifact keys, requiredness, format, and known schemas;
- import valid formal artifacts into MLDB-owned Evaluation Run storage and hash them;
- leave undeclared diagnostic files as protocol-owned work files;
- expose the scalar metric result to optional sinks such as MLflow without making those sinks the source of truth.

A later Evaluation Run entity can record the exact Model, Corpus, Evaluation Protocol, resolved parameter mapping, environment, scalar metrics, and immutable structured-artifact metadata for one concrete execution.

## Evidence

The current project already evaluates categorical classifiers using overall and per-condition accuracies, class confusions, and angle sweeps. These workflows naturally separate a small number of scalar comparison metrics from larger per-sample or confusion diagnostics.

The detector workflows likewise compute scalar precision, recall, F1, IoU, and angle-error summaries while also producing richer prediction/matching detail. This demonstrates why scalar metrics should be standardized at the generic result boundary while detailed structured outputs use explicit schemas rather than one universal result object.

The project's repeated architecture, augmentation, and robustness comparisons would benefit directly from a protocol-defined parameter surface and queue-friendly scalar result contract, while optional external visualization systems can consume the same immutable Evaluation Run outputs.