# Reference: MLDB repository layout

- **id**: `spec:mldb.repository.layout`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.repository`

## What this is

Defines the canonical repository placement for current MLDB entity kinds and executable-asset tests.

The layout makes typed entity resolution deterministic from entity kind and ID. File contents and lifecycle rules belong to entity-specific specifications.

## Placement reference

| entity kind | canonical location | physical form |
|---|---|---|
| Task | `mldb_data/tasks/<task-id>.yaml` | One YAML file. |
| Corpus | `mldb_data/corpora/<corpus-id>.yaml`, `.sqlite`, `.py` | Three sibling files sharing one basename. |
| Architecture | `mldb_data/architectures/<architecture-id>.yaml`, `.py` | Two sibling files sharing one basename. |
| Train Protocol | `mldb_data/train_protocols/<train-protocol-id>.yaml`, `.py` | Two sibling files sharing one basename. |
| Training Run | `mldb_data/training_runs/<training-run-id>/` | Runtime-generated Run directory containing `run.yaml`, `work/`, and `artifacts/`; a completed Run owns canonical learned weights at `artifacts/weights.pt`. |
| Model | `mldb_data/models/<model-id>.yaml` | Runtime-generated YAML identity record only; learned bytes remain in the originating completed Training Run. |
| Evaluation Protocol | `mldb_data/evaluation_protocols/<evaluation-protocol-id>.yaml`, `.py` | Two sibling files sharing one basename. |
| Evaluation Run | `mldb_data/evaluation_runs/<evaluation-run-id>/` | One Run directory containing `run.yaml`, `work/`, and `artifacts/`. |
| Study | `mldb_data/studies/<study-id>.yaml` | One YAML file. |
| Study Run | `mldb_data/study_runs/<study-run-id>/` | One Run directory containing `run.yaml` and `plan.jsonl`. |

## Executable-asset test placement

| executable asset kind | canonical pytest directory |
|---|---|
| Architecture | `mldb_tests/architectures/<architecture-id>/` |
| Train Protocol | `mldb_tests/train_protocols/<train-protocol-id>/` |
| Evaluation Protocol | `mldb_tests/evaluation_protocols/<evaluation-protocol-id>/` |

Test files beneath an asset test directory use pytest discovery names such as `test_*.py`.

Corpus builder tests are not assigned a mandatory per-asset directory by the current contract.

## Rules

- `mldb_data/` and `mldb_tests/` are repository-level roots.
- `mldb_data/` must remain separate from `mldb/records/`.
- Entity kind determines the containing directory before entity ID is resolved.
- A single-file entity uses its entity ID as the filename basename.
- A sibling-file entity uses exactly one entity ID basename for every required sibling.
- A Run directory uses the Run ID as the directory name. `run.yaml` is not renamed to the Run ID.
- Training Run `work/` is execution-owned working space. Training Run `artifacts/` is MLDB-owned formal result storage.
- A completed Training Run owns the canonical learned artifact at `artifacts/weights.pt`.
- Successful Training Run finalization automatically creates the corresponding Model YAML identity record.
- Model YAML does not duplicate `weights.pt`; Model resolves learned bytes through its originating completed Training Run.
- Evaluation Run `work/` is execution-owned working space. Evaluation Run `artifacts/` is MLDB-owned formal result storage.
- Study Run `plan.jsonl` is stored beside `run.yaml` in the Study Run directory.
- Runtime resolution must not scan other entity-kind directories to compensate for a missing canonical location.
- Asset metadata must not require an additional path field when the corresponding sibling or Run location is fully derived by this layout.
- Runtime implementation source and generic runtime tests are outside this placement reference.
- Queue, Worker cache, temporary workspace, and external visualization storage are non-canonical orchestration or tooling storage. V1 Queue placement is defined separately as `.local/mldb/queue.sqlite` by `spec:mldb.orchestration.queue_storage_format`.

## Boundary

| concern | owner |
|---|---|
| Entity kind and ID to canonical path mapping | This reference. |
| YAML field schemas | Entity-specific format specs. |
| Corpus SQLite table schema | Corpus data-format specs. |
| Run directory lifecycle and mutation rules | Training, evaluation, and Study Run lifecycle specs. |
| Canonical weight and evaluation artifact semantics | Training and evaluation specs. |
| Executable asset pytest sealing requirement | Verification topic. |
| Runtime typed-reference resolution | `spec:mldb.runtime.asset_resolution`. |
| Queue and Worker-cache ownership | `spec:mldb.orchestration`; durable Queue lifecycle is `spec:mldb.orchestration.queue_lifecycle`; concrete Queue placement/schema is `spec:mldb.orchestration.queue_storage_format`. |
| Brewprint Design Record placement under `mldb/records/` | Brewprint Design Records repository-layout contract. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.repository` | Parent repository overview. |
| `spec:mldb.runtime.asset_resolution` | Uses this layout to resolve typed entity references. |
| `spec:mldb.orchestration` | Keeps Queue and Worker-local cache storage outside canonical MLDB entity placement. |
| `spec:mldb.orchestration.queue_lifecycle` | Defines the Controller-local SQLite Queue as durable operational, non-canonical storage. |
| `spec:mldb.orchestration.queue_storage_format` | Defines `.local/mldb/queue.sqlite` and the concrete v1 Queue schema. |
