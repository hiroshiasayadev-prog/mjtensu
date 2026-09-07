# MLDB-ADR-SCHEMA-008: Define immutable Evaluation Run records

- **status**: accepted
- **date**: 2026-09-04
- **depends_on**: MLDB-ADR-SCHEMA-002, MLDB-ADR-SCHEMA-006, MLDB-ADR-SCHEMA-007
- **supersedes**:
- **migrated_to_spec**:

## Context

Evaluation Protocol defines a reusable post-training evaluation procedure, but it does not identify one concrete execution against one learned Model and one Corpus.

MLDB needs an immutable record for each such execution so automated studies and queued workers can evaluate many Models under repeatable conditions while retaining the exact Model, Corpus, Evaluation Protocol, resolved public parameter values, execution environment, scalar metrics, and formal structured artifacts.

Evaluation Protocol deliberately allows arbitrary temporary and diagnostic files under its work directory, but formal outputs are restricted to declared scalar metrics and declared schema-versioned structured artifacts. Evaluation Run is therefore the natural place where generic MLDB tooling validates those outputs, imports accepted structured artifacts into MLDB-owned storage, records their hashes, and exposes scalar results to optional sinks such as MLflow or a project-owned visualization system.

Large automated studies also require failure isolation. One broken evaluation or one malformed optional diagnostic artifact must not halt unrelated queued work. Evaluation Run status therefore describes one execution only. Orchestration may continue independent jobs even when another Evaluation Run fails.

Evaluation Run is an execution-history entity. It is not revised in place. Repeating the same evaluation conditions creates another Evaluation Run unless a later orchestration layer chooses to reuse an existing completed Run as a cache optimization.

## Decision

Introduce `EvaluationRun` as the immutable MLDB record for one concrete invocation of one sealed Evaluation Protocol against one Model and one immutable Corpus.

An Evaluation Run records:

- the selected Model;
- the selected Corpus;
- the selected Evaluation Protocol;
- the complete resolved public parameter mapping;
- execution lifecycle and timing;
- best-effort execution environment metadata;
- validated scalar metrics;
- accepted formal structured artifacts;
- local validation issues that did not invalidate the entire Run, when applicable.

Evaluation Run does not define evaluation behavior. The referenced Evaluation Protocol remains authoritative for preprocessing, inference, metric semantics, matching, thresholds, aggregation, and formal output declarations.

### Physical placement

Evaluation Runs live under:

```text
mldb_data/
  evaluation_runs/
```

Each Evaluation Run occupies one directory named by its Run ID:

```text
mldb_data/evaluation_runs/<evaluation-run-id>/
  run.yaml
  work/
  artifacts/
```

The Evaluation Protocol receives `work/` through `EvaluationContext.work_dir` and may write arbitrary temporary or diagnostic files there.

`artifacts/` is MLDB-owned formal result storage. Only structured artifacts returned by `EvaluationResult`, declared by the sealed Evaluation Protocol, and accepted by generic MLDB validation are copied or moved there.

Example:

```text
mldb_data/evaluation_runs/ev-20260904-001/
  run.yaml
  work/
    temporary-predictions.jsonl
    debug-summary.txt
  artifacts/
    predictions.jsonl
    confusion-matrix.csv
```

### Evaluation Run ID

Evaluation Run IDs are event identifiers rather than versioned definition IDs.

MLDB v1 uses:

```text
ev-YYYYMMDD-NNN
```

where `YYYYMMDD` is the local calendar date on which the Evaluation Run is allocated and `NNN` is a zero-padded per-date sequence beginning at `001`.

Examples:

```text
ev-20260904-001
ev-20260904-002
ev-20260905-001
```

Evaluation Run IDs do not use the terminal `-vN` revision grammar.

An ID is immutable once allocated and must be unique within `mldb_data/evaluation_runs/`.

### Lifecycle

Evaluation Run status is one of:

```text
running
completed
failed
cancelled
```

A Run is created with `status: running` before `EvaluationProtocol.evaluate(context)` is invoked.

It must eventually transition to exactly one terminal state:

```text
completed
failed
cancelled
```

A Run may become `completed` only after generic MLDB tooling has validated the formal scalar metric result and all required structured artifacts, imported accepted formal artifacts into `artifacts/`, and recorded immutable result metadata.

A Run becomes `failed` when the evaluation itself fails or when a contract violation prevents the formal result from being trusted. Examples include:

- an exception from `evaluate(context)`;
- an undeclared formal metric key;
- a non-numeric or non-finite formal metric value;
- a missing required structured artifact;
- a required structured artifact that fails format or schema validation;
- a returned Model/Corpus/Protocol compatibility failure detected before or during execution.

A malformed artifact declared with `required: false` does not by itself fail the Evaluation Run. Generic MLDB tooling rejects that artifact, leaves it out of `result.artifacts`, records a local validation issue, and may still complete the Run if all required outputs and scalar metrics are valid.

Terminal Evaluation Runs are immutable and must never return to `running`.

Retrying a failed or cancelled evaluation creates a new Evaluation Run ID.

### Failure isolation and queue behavior

Evaluation Run lifecycle is local to one execution.

A failed or cancelled Evaluation Run must not implicitly cancel unrelated Training Runs, Models, Evaluation Runs, or queued jobs.

A later queue, Study, or DAG orchestration layer should continue jobs that do not depend on the failed Run. Jobs that explicitly depend on a successful result from the failed Run may be skipped, blocked, or marked accordingly by that orchestration layer, but that dependency behavior is not encoded by changing unrelated Evaluation Run statuses.

This permits large parameter sweeps to finish as far as possible even when individual evaluations fail.

### Metadata format

Evaluation Run metadata uses:

```text
mjtensu.mldb/evaluation-run/v1
```

A completed example is:

```yaml
schema: mjtensu.mldb/evaluation-run/v1

id: ev-20260904-001
status: completed

model: mdl-20260903-001
corpus: gray35-final-holdout-v1
evaluation_protocol: tile-classifier-standard-eval-v1

study:
  run: sr-20260904-001
  trial: trial-0001
  stage: final-holdout

parameters:
  batch_size: 4096

execution:
  started_at: 2026-09-04T10:00:00+09:00
  finished_at: 2026-09-04T10:00:14+09:00

environment:
  host: bugratserver
  python: 3.12.0
  torch: 2.8.0
  cuda: "12.8"
  device: NVIDIA GeForce RTX 4090

result:
  metrics:
    accuracy: 0.9976
    cross_entropy: 0.0121

  artifacts:
    predictions:
      path: artifacts/predictions.jsonl
      format: jsonl
      schema: mjtensu.mldb/eval-artifact/categorical-predictions/v1
      sha256: 0123456789abcdef...
      bytes: 123456

    confusion_matrix:
      path: artifacts/confusion-matrix.csv
      format: csv
      schema: mjtensu.mldb/eval-artifact/confusion-matrix/v1
      sha256: fedcba9876543210...
      bytes: 5432
```

A completed Run with one rejected optional artifact may additionally contain:

```yaml
validation_issues:
  - output: artifacts.confusion_matrix
    type: schema-validation-failed
    message: Duplicate target/prediction pair; optional artifact was omitted.
```

A failed example is:

```yaml
schema: mjtensu.mldb/evaluation-run/v1

id: ev-20260904-002
status: failed

model: mdl-20260903-002
corpus: gray35-final-holdout-v1
evaluation_protocol: tile-classifier-standard-eval-v1

parameters:
  batch_size: 4096

execution:
  started_at: 2026-09-04T10:05:00+09:00
  finished_at: 2026-09-04T10:05:12+09:00

failure:
  type: EvaluationOutputValidationError
  message: Required artifact 'predictions' was not returned.
```

### Required fields

Every Evaluation Run record must contain:

- `schema`;
- `id`;
- `status`;
- `model`;
- `corpus`;
- `evaluation_protocol`;
- `parameters`;
- `execution.started_at`.

A terminal Evaluation Run additionally requires:

- `execution.finished_at`.

A `completed` Evaluation Run additionally requires:

- `result.metrics`;
- `result.artifacts`.

Both result mappings may be empty only when the sealed Evaluation Protocol declares no metrics and no formal artifacts respectively.

A `failed` Evaluation Run should record `failure.type` and `failure.message` when the failure can be represented safely as concise text.

`environment` is optional and free-form.

### Study lineage

An Evaluation Run created from a Study Run may additionally contain:

```yaml
study:
  run: sr-20260904-001
  trial: trial-0001
  stage: final-holdout
```

`study.run` references the Study Run, `study.trial` identifies the Study-local training trial whose Model is being evaluated, and `study.stage` identifies the evaluation stage declared in the Study plan.

The mapping is optional because Evaluation Runs may also be launched directly outside a Study.

When present, all three fields are required and must correspond to one planned evaluation entry in the referenced Study Run plan. The Evaluation Run's Model must resolve to the Model created by that trial's completed Training Run, and its Corpus, Evaluation Protocol, and fully resolved parameters must match the planned evaluation stage.

Study lineage is immutable execution provenance and does not create dependencies between sibling evaluation stages beyond the shared requirement for the trial's Model.

`validation_issues` is optional. It records non-fatal formal-output validation problems such as rejection of an optional artifact. Each issue should identify the affected output, a short machine-readable type, and a concise message.

### Input relationships

Each Evaluation Run references exactly one:

- Model;
- Corpus;
- Evaluation Protocol.

Evaluation Run does not repeat Task.

Before execution, generic MLDB tooling must verify that:

- the referenced Model exists and resolves to one completed Training Run;
- the Model's Architecture Task equals the Evaluation Protocol Task;
- the selected Corpus Task equals the Evaluation Protocol Task;
- the selected Corpus artifact integrity is valid;
- the referenced Evaluation Protocol is `sealed` and its implementation hash is valid;
- the Model's canonical learned-weight artifact and sealed Architecture implementation hashes validate.

The referenced Model and Corpus are execution inputs rather than properties of the Evaluation Protocol itself.

### Resolved Evaluation Protocol parameters

Evaluation Run records the complete resolved values of the public parameter interface declared by its referenced Evaluation Protocol.

Generic MLDB tooling:

1. starts from every `EvaluationProtocol.parameters.<key>.default`;
2. replaces only caller-supplied values for published keys;
3. rejects unknown parameter keys;
4. records the complete resolved mapping under `parameters` before invocation;
5. passes the exact same mapping through `EvaluationContext.parameters`.

This permits queue-driven parameter sweeps while preventing Evaluation Run from becoming an unrestricted evaluation override system.

Evaluation Run v1 has no special universal random-seed field. If a particular evaluation intentionally uses randomness, that Evaluation Protocol should expose a `seed` parameter explicitly.

### Invocation sequence

The generic execution flow is conceptually:

```text
allocate Evaluation Run ID
  -> create run.yaml with status=running
  -> create work/ and artifacts/
  -> resolve Model / Corpus / Evaluation Protocol
  -> verify Task compatibility and asset integrity
  -> resolve and record complete Evaluation Protocol parameters
  -> construct EvaluationContext
  -> invoke EvaluationProtocol.evaluate(context)
  -> receive EvaluationResult or failure
  -> validate scalar metric contract
  -> validate each returned structured artifact
       required invalid/missing -> Run failure
       optional invalid         -> omit artifact + record validation issue
  -> import accepted formal artifacts into artifacts/
  -> hash and record accepted artifacts
  -> finalize run.yaml as completed / failed / cancelled
  -> freeze terminal Run
```

The launcher may also publish completed scalar metrics and formal artifact references to MLflow or another visualization/indexing system, but such sinks are not authoritative MLDB storage.

### Scalar metric handling

`result.metrics` stores exactly the formal scalar metrics returned by the Evaluation Protocol and accepted under the protocol's `outputs.metrics` contract.

Generic MLDB tooling must reject:

- undeclared metric keys;
- booleans;
- strings;
- lists or mappings;
- NaN;
- positive or negative infinity.

An invalid formal scalar metric makes the Evaluation Run `failed` because scalar metrics are part of the primary machine-comparable result surface.

The metric mapping is stored directly in `run.yaml` so queue orchestration, study summaries, CLI reports, MLflow synchronization, or future project-owned dashboards can consume metrics without parsing protocol-specific artifact files.

### Structured artifact handling

The Evaluation Protocol writes candidate files under `work/` and returns declared artifact keys mapped to those paths.

For each returned formal artifact, generic MLDB tooling validates:

- the artifact key exists in the sealed Evaluation Protocol declaration;
- the file exists beneath the Run work directory;
- the declared format matches;
- a known registered schema validates when generic validation is defined.

For a required artifact, missing or invalid output fails the Evaluation Run.

For an optional artifact, missing output is valid. If an optional artifact is returned but fails validation, that artifact is rejected without failing the whole Run. The rejection is recorded in `validation_issues`.

Accepted artifacts are copied or moved into the Evaluation Run's `artifacts/` directory and recorded under `result.artifacts` with:

- canonical relative `path`;
- `format`;
- schema identifier;
- SHA-256;
- byte size.

Files remaining only in `work/` are not formal Evaluation Run outputs.

### Environment capture

Evaluation Run records execution environment information on a best-effort basis.

Useful fields may include host, operating system, Python version, PyTorch version, CUDA version, accelerator, third-party framework versions, or protocol-specific runtime information.

MLDB v1 does not require a complete environment lockfile. The mapping exists to make execution differences visible when debugging unexpected evaluation results.

### Immutability

Evaluation Run is a historical event and receives no `-vN` revision.

While `status: running`, the record may be updated as execution facts become known.

After reaching `completed`, `failed`, or `cancelled`, its selected inputs, resolved parameters, timing, result metrics, accepted artifact identities, validation issues, and failure metadata are immutable.

Re-running an identical evaluation creates a new Evaluation Run rather than editing or revising the previous one.

### Repeated identical evaluations and caching

Evaluation Run itself does not deduplicate identical executions.

The same Model, Corpus, Evaluation Protocol, and resolved parameters may therefore produce multiple Evaluation Runs when explicitly executed multiple times.

A later Study or queue layer may calculate an input fingerprint and reuse an existing completed Evaluation Run instead of scheduling another execution. Such cache/reuse policy belongs to orchestration and does not change Evaluation Run identity semantics.

### Training validation boundary

Evaluation Run records post-training evaluation of an already-created Model.

Validation performed inside Train Protocol solely to select the learned state returned by training remains part of Training Run and Train Protocol behavior. It is not retrospectively represented as an Evaluation Run unless the same completed Model is separately evaluated through an Evaluation Protocol.

### No downstream entity creation

Unlike a completed Training Run, which automatically produces one Model identity, a completed Evaluation Run does not automatically create another MLDB domain entity.

Its immutable result is the Evaluation Run itself.

Later Promotion or Release logic may consume Evaluation Runs and their scalar metrics when making selection decisions, but those decisions do not mutate the Evaluation Run.

## Rationale

Evaluation Run mirrors Training Run's execution-history model while reflecting the different nature of evaluation output.

Recording Model, Corpus, Evaluation Protocol, and complete resolved parameters gives each result exact lineage and makes queue-driven comparisons reproducible without relying on command-line reconstruction or directory naming conventions.

Restricting primary metrics to finite numeric scalars produces a stable comparison surface that can be indexed directly by MLflow, a future SQL catalog, CLI summaries, or project-owned visualization without making any of those systems authoritative.

Formal structured artifacts are validated and hash-identified so deeper diagnostics remain machine-readable and reproducible. Keeping arbitrary protocol files in `work/` avoids forcing every debug image, temporary table, or framework log into the formal result contract.

Treating optional artifact validation failures as local omissions provides useful resilience. A malformed optional confusion matrix should not discard otherwise valid primary accuracy metrics. Conversely, missing required outputs or invalid primary scalar metrics invalidate that Run because its promised result contract was not fulfilled.

Failure isolation at the Run level is essential for parameter sweeps. One broken evaluation should produce one failed historical record while independent jobs continue, rather than collapsing the entire experiment queue.

Keeping deduplication and DAG behavior outside the Run schema preserves a simple distinction: Evaluation Run records what actually happened; orchestration decides what should execute next.

## Rejected alternatives

### Abort the entire experiment queue when one Evaluation Run fails

Large studies can contain many independent Model/evaluation combinations. One implementation error, corrupt optional diagnostic, or device-specific failure should not erase useful results from unrelated jobs.

Failure is therefore local to the affected Evaluation Run. Later orchestration controls dependency-aware continuation.

### Fail a Run for every invalid optional artifact

This would make secondary diagnostics as important as the primary scalar result and required outputs. A malformed optional confusion matrix could discard otherwise valid accuracy or detector metrics.

Optional formal artifacts are therefore rejected individually and recorded as validation issues without necessarily failing the Run.

### Accept invalid required artifacts and merely warn

A required artifact is part of the sealed Evaluation Protocol's promised formal output. Silently completing without it would make ostensibly comparable Evaluation Runs have different result contracts.

Missing or invalid required artifacts therefore fail the Run.

### Store all outputs directly in MLflow

MLflow is useful for visualization and experiment comparison, but making it authoritative would couple MLDB history to one external service and complicate reconstruction if that service is reset or replaced.

Evaluation Run YAML and hashed formal artifacts remain the source of truth; external visualization is a derived sink.

### Put detailed structured results directly inside run.yaml

Per-sample predictions, confusion matrices, detector boxes, curves, and other tables can be large and have specialized schemas.

Evaluation Run stores only scalar metrics and artifact metadata in YAML. Structured data stays in versioned formal artifact files.

### Give Evaluation Run a revision lifecycle

An evaluation is an event, not an authored definition. Re-execution should create another historical event rather than another revision of the previous result.

Evaluation Run therefore uses event IDs and terminal immutability.

## Consequences

Future Evaluation Run tooling should be able to:

- allocate unique `ev-YYYYMMDD-NNN` IDs;
- create `run.yaml`, `work/`, and `artifacts/` before invocation;
- resolve and validate Model, Corpus, and sealed Evaluation Protocol assets;
- verify Task compatibility and relevant artifact/implementation hashes;
- resolve Evaluation Protocol defaults plus caller-supplied published parameters;
- reject unknown parameter keys;
- pass the resolved mapping through `EvaluationContext.parameters`;
- invoke `evaluate(context)`;
- treat evaluation exceptions as failure of that Run only;
- validate formal scalar metric keys and finite numeric values;
- validate required structured artifacts and fail the affected Run when they are absent or invalid;
- validate optional structured artifacts independently, omit invalid optional artifacts, and record non-fatal validation issues;
- import accepted formal artifacts into `artifacts/`;
- calculate and persist artifact SHA-256 and byte size;
- capture best-effort execution environment metadata;
- finalize and freeze `completed`, `failed`, or `cancelled` Runs;
- retain failed and cancelled Run directories as historical evidence;
- allow unrelated queued work to continue after one Evaluation Run fails;
- expose completed scalar metrics and formal artifact references to optional sinks such as MLflow without changing MLDB authority.

A later Study or queue/DAG design can use these isolated immutable execution records to expand parameter sweeps, schedule dependent evaluations after Model creation, continue independent branches after failures, cache existing completed evaluations, and aggregate scalar results for model comparison.

## Evidence

The current project routinely compares multiple classifier architectures, augmentation conditions, and robustness settings. These comparisons naturally form independent evaluation jobs where one failed condition should not invalidate measurements from all other conditions.

Current classifier evaluation produces small scalar summaries such as accuracy together with larger per-class or confusion diagnostics. This supports treating scalar metrics as the primary comparable result while allowing optional diagnostic artifacts to fail independently when they are not required by the protocol.

Current detector evaluation similarly produces scalar precision, recall, F1, IoU, and angle-error summaries while richer prediction and matching detail can be represented as structured artifacts. Queue-driven detector experiments benefit from the same Run-local failure boundary: one malformed output or failed model evaluation should be recorded and surfaced without stopping unrelated Models or parameter combinations.
