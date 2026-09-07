# MLDB-ADR-SCHEMA-001: Define versioned Task records for ML asset management

- **status**: accepted
- **date**: 2026-09-03
- **depends_on**:
- **supersedes**:
- **migrated_to_spec**:

## Context

The recognition project has accumulated multiple classifier architectures, corpora, training conditions, checkpoints, exported models, evaluation scripts, and benchmark results across several investigations.
Those assets are currently reproducible only by reconstructing information from experiment runners, result files, investigation records, local artifact directories, and Git history.

The project is entering a phase where a comparatively stable corpus is reused while model architecture, training recipe, evaluation method, and deployment constraints are varied repeatedly.
That requires explicit, versioned identities for ML assets and their relationships rather than relying on directory names or one-off experiment documentation.

A new records-only app namespace, `mldb`, will hold the project-owned ML asset model and its versioned records under the Brewprint design-record placement model.
The initial entity is `Task`, because every corpus, architecture, training protocol, and evaluation protocol must ultimately state which prediction problem it serves.

Task must describe the semantic prediction problem only.
It must not absorb dataset construction, pixel representation, training procedure, model architecture, evaluation results, or deployment details that belong to later MLDB entities.

## Decision

Introduce a versioned `Task` entity under MLDB.

A Task defines:

- the semantic input unit presented to the prediction problem;
- the problem type;
- the ordered prediction target vocabulary when the task is categorical;
- task-specific target semantics;
- explicit in-scope and out-of-scope behavior.

Task records are machine-readable YAML files and are intended to be validated by a future schema/validation tool.
Markdown documentation may be generated from or link to these records, but YAML is the authoritative Task record format.

### Storage

Task records will live under:

```text
mldb_data/tasks/
```

A Task record filename should match its immutable Task identifier:

```text
<task-id>.yaml
```

Example:

```text
mldb_data/tasks/tile-shape-classification-35-v1.yaml
```

### Task record shape

The initial Task record contract is:

```yaml
schema: mjtensu.mldb/task/v1

id: tile-shape-classification-35-v1
name: Base tile shape classification
problem_type: multiclass-classification

description: >
  Classify one visible Japanese mahjong tile instance into the canonical
  base-tile label set used by the recognition pipeline.

input:
  semantic_unit: single-tile-image

target:
  type: categorical
  labels:
    - 1m
    - 2m
    - 3m
    - 4m
    - 5m
    - 6m
    - 7m
    - 8m
    - 9m
    - 1p
    - 2p
    - 3p
    - 4p
    - 5p
    - 6p
    - 7p
    - 8p
    - 9p
    - 1s
    - 2s
    - 3s
    - 4s
    - 5s
    - 6s
    - 7s
    - 8s
    - 9s
    - east
    - south
    - west
    - north
    - white
    - green
    - red
    - invalid

semantics:
  red_five_handling: classify-as-base-five
  invalid: >
    Input does not represent one valid classifiable tile instance.

scope:
  includes:
    - visible Japanese mahjong tiles
  excludes:
    - red-five discrimination
    - tile localization
    - meld structure interpretation
```

The concrete example above documents the intended first Task, but this ADR establishes the Task record contract rather than creating every future Task instance.

### Required fields

A Task record must contain:

- `schema`;
- `id`;
- `name`;
- `problem_type`;
- `description`;
- `input`;
- `target`;
- `semantics`;
- `scope`.

For categorical tasks, `target.labels` is required, must contain unique non-empty values, and its ordering is normative.
The array index is the canonical class index for that Task version unless a future Task schema explicitly defines another mapping.

### Task responsibility boundary

Task records define semantic meaning only.
They must not define:

- corpus or dataset identity;
- train/validation/test split membership;
- source image paths;
- pixel resolution;
- channel count or grayscale/RGB representation;
- normalization;
- preprocessing implementation;
- augmentation;
- optimizer or learning rate;
- epoch count or seed;
- neural-network architecture;
- checkpoint or model artifact identity;
- evaluation cases or metric results;
- runtime, latency, model size, ONNX, or deployment behavior.

Those concerns will be assigned to later MLDB entities such as Corpus, Architecture, Train Protocol, Model, Evaluation Protocol, and related run/artifact records.

A Task may state the semantic unit `single-tile-image` without requiring a specific tensor encoding.
For example, grayscale `64 x 64` and RGB `96 x 96` classifiers may implement the same Task if they predict the same semantic target contract.

### Target semantics

Task-level semantics include decisions that change what the prediction means rather than how examples are represented or learned.

For the base-tile classification example, mapping red fives to the corresponding base `5m`, `5p`, or `5s` target is a Task semantic because it changes the target meaning.
The number of red-five examples or how those examples are sampled belongs to Corpus or Train Protocol instead.

Likewise, the semantic meaning of `invalid` belongs to Task, while the quantity, source, and distribution of invalid examples belong to Corpus.

### Relationships

Task is a referenced root entity.
Later MLDB records may refer to a Task by immutable `id`, for example:

```yaml
task: tile-shape-classification-35-v1
```

Task records must not maintain reverse lists of referencing corpora, architectures, protocols, models, or runs.
Adding a new dependent asset therefore does not mutate the Task.

### Identity and versioning

A Task identifier is immutable once another MLDB record references it.

A new Task version or new Task identity is required when the semantic prediction contract changes, including:

- changing the semantic input unit;
- changing the prediction target type;
- adding or removing a categorical label;
- reordering categorical labels when array order defines class index;
- changing the meaning of a label;
- changing the meaning of `invalid` or another task-level outcome;
- changing red-five handling or another target-mapping rule;
- materially changing what is in or out of scope.

Non-semantic editorial corrections, such as spelling or wording that does not change the contract, do not require a new Task identity.

Task identifiers should be descriptive and include a human-visible revision suffix such as `-v1`.
The schema version (`mjtensu.mldb/task/v1`) and Task revision are separate concepts: schema version describes the YAML structure, while Task revision describes the semantic prediction contract.

## Rationale

Separating Task from the rest of the ML lifecycle provides a stable root for comparisons across different corpora, architectures, and training recipes.
A model that changes from Plain CNN to MobileNet remains comparable when both reference the same Task, even if their tensor representations and learned feature structures differ.

Keeping Task intentionally thin prevents implementation details from turning semantic-equivalent models into artificial new tasks.
It also makes future validation straightforward: a validator can verify Task identity, required fields, label uniqueness, label ordering, and downstream references without understanding training code.

YAML is selected because these records must be readable in code review, diff cleanly in Git, remain easy to generate, and support machine validation.
The project does not need a heavier metadata service to define the semantic contract itself.

Ordered labels are normative because classifier output position is part of the current base-classifier integration contract.
Treating a label-order change as a semantic revision prevents two nominally identical 35-class classifiers from silently disagreeing about class indices.

## Rejected alternatives

### Store Task as Markdown only

Markdown is convenient for explanation but too permissive as the authoritative format for dependency validation and automated inventory generation.
The project intends to add lightweight guides and validation scripts once entity relationships and schemas are established.
Machine-readable YAML is a better source of truth for that workflow.

### Put tensor shape and preprocessing in Task

A fixed tensor contract would make Task describe one implementation representation rather than one prediction problem.
That would incorrectly split semantically identical grayscale, RGB, or differently sized models into separate tasks.
Representation details belong to Corpus, Architecture, or another later entity depending on their responsibility.

### Put labels in a separate LabelSpace entity immediately

A reusable LabelSpace entity could normalize shared vocabularies across many tasks, but the current project has not demonstrated enough cross-task reuse to justify another first-class entity.
The ordered label contract will remain embedded in Task v1.
It may be extracted later if multiple independent tasks need to share the same versioned label space.

### Maintain reverse references from Task

Reverse references would require modifying the Task every time a corpus, architecture, or protocol is added.
MLDB relationships should be expressed by dependent records referencing immutable upstream identities instead.

## Consequences

`mldb` is the root-level records-only app namespace for ML asset-management design records, which live under `mldb/records/`. MLDB-owned machine-readable data is separate from Brewprint Design Records and lives under the repository-level `mldb_data/` root.

Task YAML records live under `mldb_data/tasks/` and conform to `mjtensu.mldb/task/v1`.
Later ADRs will define the responsibilities and relationships of additional entities one at a time rather than attempting to freeze the full MLDB model up front.

Future guide and validation tooling can enforce this ADR without requiring a large implementation.
At minimum, a validator should eventually be able to verify:

- supported Task schema version;
- Task ID and filename consistency;
- required fields;
- unique ordered categorical labels;
- valid downstream Task references;
- immutability expectations through Git review rather than hidden runtime mutation.

MLflow or another experiment viewer may later mirror Task IDs and related metadata, but such tools do not own the Task definition.
The repository-managed MLDB record remains the semantic source of truth.
