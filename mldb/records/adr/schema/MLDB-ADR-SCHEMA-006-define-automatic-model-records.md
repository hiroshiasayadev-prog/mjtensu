# MLDB-ADR-SCHEMA-006: Define automatic Model records

- **status**: accepted
- **date**: 2026-09-03
- **depends_on**: MLDB-ADR-SCHEMA-003, MLDB-ADR-SCHEMA-005
- **supersedes**:
- **migrated_to_spec**:

## Context

MLDB-ADR-SCHEMA-005 establishes that every successfully completed Training Run produces exactly one canonical learned-weight artifact. The Train Protocol returns the trained `torch.nn.Module`, and generic MLDB tooling validates it against the selected Architecture, serializes its learned CPU `state_dict` to `artifacts/weights.pt`, and records the artifact format, SHA-256, and byte size in the immutable Training Run.

Downstream activities such as evaluation, export, comparison, promotion, and release need a stable identity for the learned model itself rather than treating a Training Run ID as the public model identity.

At the same time, Model must not duplicate the learned-weight artifact or repeat metadata already captured by Training Run. The completed Training Run already identifies the Architecture, Corpus, Train Protocol, seed, execution environment, and canonical weight artifact. Copying those facts into a second Model record would create redundant sources of truth.

Model also must not imply quality or selection. A Training Run that completes successfully may produce a poor model, but it is still a valid learned model that can be evaluated and compared. Decisions such as "best", "candidate", "production", or "released" belong to later evaluation, promotion, and release concepts.

Because each successful Training Run already has exactly one canonical learned result, no manual Model-registration step is necessary.

## Decision

Introduce `Model` as an automatically generated immutable MLDB entity representing the canonical learned result of exactly one completed Training Run.

For MLDB v1:

- every `completed` Training Run has exactly one Model;
- `running`, `failed`, and `cancelled` Training Runs have no Model;
- Model creation is automatic as part of successful Training Run finalization;
- Model has no draft/sealed lifecycle;
- Model is immutable from creation;
- Model does not own or copy learned-weight bytes;
- Model does not carry quality, promotion, release, or deployment status.

### Physical placement

Model records live outside the Brewprint Design Records tree under:

```text
mldb_data/
  models/
```

Each Model is represented by exactly one YAML file:

```text
mldb_data/models/<model-id>.yaml
```

Example:

```text
mldb_data/
  training_runs/
    tr-20260903-001/
      run.yaml
      work/
      artifacts/
        weights.pt

  models/
    mdl-20260903-001.yaml
```

Model has no sibling `.pt` file. The canonical learned-weight bytes remain in the Training Run that created them.

### Model ID

Model IDs are immutable object identifiers rather than revision identifiers. They do not use the terminal `-vN` grammar used by versioned definition assets such as Architecture and Train Protocol.

For Training Run v1 IDs of the form:

```text
tr-YYYYMMDD-NNN
```

the corresponding Model ID is derived deterministically by replacing the `tr-` prefix with `mdl-`:

```text
tr-20260903-001
        ->
mdl-20260903-001
```

The date and sequence suffix must match exactly.

This establishes a strict one-to-one identity relationship:

```text
tr-YYYYMMDD-NNN <-> mdl-YYYYMMDD-NNN
```

A Model ID is never manually chosen and is never reused for another Training Run.

If a future Training Run schema adopts a different Run ID grammar, a later Model schema decision may define the corresponding derivation rule for those new Runs without changing existing Model IDs.

### Metadata format

Model metadata is YAML and uses schema identifier:

```text
mjtensu.mldb/model/v1
```

The complete v1 record is intentionally minimal:

```yaml
schema: mjtensu.mldb/model/v1
id: mdl-20260903-001
training_run: tr-20260903-001
```

The required fields are exactly:

- `schema`;
- `id`;
- `training_run`.

MLDB Model v1 does not define `name`, `description`, `status`, `notes`, `architecture`, `task`, `corpus`, `train_protocol`, `seed`, `artifact`, `metrics`, or deployment fields.

Those facts either belong to the referenced Training Run and its upstream assets or to later downstream entities.

### Training Run relationship

`training_run` must reference one existing Training Run with `status: completed`.

The referenced Training Run must contain a valid canonical result under:

```text
result.weights
```

including the Training Run v1 fields:

```text
format
path
sha256
bytes
```

The Model ID and Training Run ID must satisfy the deterministic prefix mapping. For example:

```text
model.id:           mdl-20260903-001
model.training_run: tr-20260903-001
```

is valid, while:

```text
model.id:           mdl-20260903-001
model.training_run: tr-20260903-002
```

is invalid.

A completed Training Run must not be referenced by more than one Model, and one Model must not reference more than one Training Run.

### Model identity and learned weights

Model identity is the stable MLDB identity assigned to the canonical learned state produced by its Training Run.

The learned parameter bytes are not duplicated under `mldb_data/models/`. A Model resolves its learned artifact through:

```text
Model.training_run
  -> TrainingRun.result.weights.path
```

The Architecture required to instantiate those weights is resolved through:

```text
Model.training_run
  -> TrainingRun.architecture
  -> Architecture.build()
```

Conceptually, generic Model loading is:

```python
model_record = load_model(model_id)
run = load_training_run(model_record.training_run)
architecture = load_architecture(run.architecture)

module = architecture.build()
state_dict = load_canonical_weights(run.result.weights)
module.load_state_dict(state_dict, strict=True)
```

Before use, tooling may verify the Training Run weight SHA-256 and the sealed Architecture implementation SHA-256.

Model YAML does not repeat the Architecture ID or weight SHA-256 because the referenced immutable Training Run is the authoritative source for both.

### Automatic creation

Model creation is part of successful Training Run finalization.

Conceptually:

```text
TrainProtocol.train(context)
  -> trained torch.nn.Module
  -> MLDB validates Architecture compatibility
  -> MLDB serializes artifacts/weights.pt
  -> MLDB records result.weights hash and bytes
  -> Training Run reaches completed
  -> MLDB ensures corresponding Model YAML exists
```

The logical invariant after successful finalization is:

```text
completed Training Run <=> exactly one corresponding Model
```

Filesystem operations are not assumed to provide a database transaction. Because Model identity and contents are completely deterministic from the completed Training Run, Model creation must be idempotent. If a process interruption leaves a completed Run without its Model YAML, MLDB reconciliation tooling may regenerate the exact missing Model record without mutating the terminal Training Run.

A Model file whose deterministic contents disagree with its completed Training Run is invalid and must not be silently rewritten to represent another Run.

### Immutability

A Model is immutable from the moment it is created.

There is no `draft` or `sealed` state because Model is not an authored reusable definition. It is a derived identity for an already immutable learned result.

A different training execution, even with the same Corpus, Architecture, Train Protocol, and seed, produces a different Training Run and therefore a different Model.

A Model is never revised in place. There is no Model `-vN` lifecycle.

### No quality or promotion semantics

Automatic Model creation does not mean the learned model is good, accepted, production-ready, or preferred.

For example:

```text
tr-20260903-001 -> mdl-20260903-001
tr-20260903-002 -> mdl-20260903-002
tr-20260903-003 -> mdl-20260903-003
```

all three Models exist if all three Runs completed, regardless of their later evaluation results.

Evaluation records may compare them. A later Promotion or Release entity may identify which Model is preferred for a particular purpose.

Model itself records only learned-model identity.

### Responsibility boundary

Model owns:

- a stable immutable Model ID;
- the one-to-one reference to its completed Training Run.

Model does not own:

- Task semantics;
- Corpus identity or data;
- Architecture definition;
- Train Protocol definition;
- random seed;
- execution environment;
- training logs or framework checkpoints;
- canonical weight serialization or artifact metadata;
- evaluation metrics;
- benchmark results;
- exported ONNX or other runtime formats;
- promotion, release, production, or deployment status.

Task, Corpus, Architecture, Train Protocol, seed, environment, and canonical weight artifact remain reachable through the immutable Training Run lineage rather than being duplicated into Model YAML.

### Downstream references

Downstream entities that operate on a learned model independently of the act of training should reference `Model` rather than `TrainingRun`.

Examples include future:

- Evaluation Run;
- Export Run;
- Benchmark Run where the subject is a learned Model;
- Promotion;
- Release.

Training Run remains the execution-history entity. Model is the learned-object identity exposed to downstream lifecycle operations.

### Imported or externally obtained learned models

Model v1 defines only Models produced automatically from completed MLDB Training Runs.

A pre-existing external checkpoint, vendor model, or legacy learned artifact must not be represented by manually inventing a Model v1 record that points to no Training Run. If the project needs to register such learned models directly, a later import/provenance decision should define that path explicitly.

## Rationale

Automatic one-to-one Model creation removes an unnecessary manual registration step and guarantees that every successful training result has a stable identity available for evaluation and export.

Keeping Model YAML minimal avoids duplicating immutable lineage already recorded by Training Run. In particular, Architecture, Corpus, Train Protocol, seed, and weight hash each remain authoritative in exactly one place.

Keeping the weight bytes inside the Training Run avoids physical duplication and prevents two paths from appearing to be independent authoritative copies of the same learned state.

The deterministic `tr-...` to `mdl-...` mapping makes the relationship immediately visible to humans while remaining trivial for tooling to validate and reconstruct.

Separating Model identity from promotion semantics avoids biasing the inventory toward only successful experiments. Poor but valid learned models remain available for evaluation, regression analysis, and historical comparison, while later Promotion or Release records can represent selection decisions explicitly.

Because Model content is deterministic, crash recovery is simple: a missing Model YAML for a completed Run can be recreated exactly without changing the historical Run.

## Rejected alternatives

### Manually register selected Training Runs as Models

This would make Model creation dependent on a human promotion step and would cause otherwise valid learned results to lack stable Model identities until someone selected them.

Selection and quality judgment are separate lifecycle concerns. Every completed Training Run therefore receives a Model automatically.

### Copy `weights.pt` into the Model directory

The completed Training Run already owns one canonical, hash-identified learned-state artifact. Copying the same bytes to `mldb_data/models/` would duplicate storage and create two paths whose authority would need to be reconciled.

Model references the immutable Training Run result instead.

### Repeat Architecture, Corpus, Train Protocol, seed, and artifact metadata in Model YAML

All of these facts are already reachable through the referenced immutable Training Run. Repeating them would create redundant fields and require consistency checks without adding identity information.

Model v1 therefore stores only the explicit Training Run reference.

### Give Model a draft/sealed lifecycle

Model is produced from a completed immutable Run rather than authored and iterated like Architecture or Train Protocol. There is no meaningful draft stage.

Model is immutable immediately on creation.

### Give Model a `-vN` revision suffix

A Model is a learned result object, not a revisionable definition. Another learned state comes from another Training Run and receives another deterministic Model ID.

Human-facing generations such as "production v2" belong to later Promotion or Release concepts rather than Model identity.

### Store quality or production status on Model

A completed training execution does not imply quality or operational acceptance. Putting fields such as `best`, `candidate`, or `production` on Model would mix learned-object identity with mutable selection policy.

Evaluation, Promotion, and Release own those decisions instead.

## Consequences

Future Model tooling should be able to:

- derive `mdl-YYYYMMDD-NNN` from a completed `tr-YYYYMMDD-NNN`;
- automatically create `mldb_data/models/<model-id>.yaml` during successful Training Run finalization;
- perform creation idempotently so a missing deterministic Model record can be repaired after interruption;
- validate that Model YAML contains the required schema, ID, and Training Run reference;
- validate exact Model/Training Run ID suffix correspondence;
- require the referenced Training Run to be `completed`;
- require a valid canonical `result.weights` record and artifact hash;
- reject Models for running, failed, or cancelled Runs;
- reject multiple Models for one completed Run;
- resolve Architecture and learned weights through the referenced Training Run;
- load the Architecture through `build()` and the canonical learned state through strict `state_dict` loading;
- reject mutation of existing Model records.

Downstream MLDB interfaces can use Model ID as the standard subject identifier for learned-model evaluation, export, comparison, promotion, and release while retaining complete training lineage through one immutable reference.

## Evidence

MLDB-ADR-SCHEMA-005 already defines one canonical `pytorch-state-dict` artifact for each completed Training Run and records its immutable path, SHA-256, and byte size. This provides a unique learned result that Model can identify without introducing another weight file.

MLDB-ADR-SCHEMA-003 defines one sealed Architecture as a deterministic `build() -> torch.nn.Module` construction boundary. Training Run records the exact Architecture ID and validates the learned state against a fresh Architecture instance before canonical serialization, so a Model can reconstruct its executable PyTorch module by combining the referenced Run's Architecture and canonical state dict.

The current project produces many training attempts and framework-specific checkpoints. Treating only the single canonical result of a completed Run as one Model avoids turning periodic or resume checkpoints into separate learned-model identities while preserving every successful training result for downstream evaluation.
