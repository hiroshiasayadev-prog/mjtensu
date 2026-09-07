# MLDB-ADR-SCHEMA-011: Define executable asset pytest layout

- **status**: accepted
- **date**: 2026-09-04
- **depends_on**: MLDB-ADR-SCHEMA-003, MLDB-ADR-SCHEMA-004, MLDB-ADR-SCHEMA-007
- **supersedes**:
- **migrated_to_spec**:

## Context

MLDB executable assets are intentionally small and self-contained: Architecture owns one executable model definition, Train Protocol owns one executable training procedure, and Evaluation Protocol owns one executable evaluation procedure. Their YAML and sibling Python files are resolved and validated by MLDB runtime tooling.

Runtime validation can enforce structural contracts such as ID and basename equality, referenced-Task compatibility, implementation hashes, entrypoint presence, public parameter resolution, return types, result schemas, and Run lifecycle rules. It cannot establish that a human-authored executable asset actually behaves as intended under representative inputs.

The project therefore needs pytest coverage for executable assets, but pytest itself must not become a runtime dependency. A Training Run or Evaluation Run should not require pytest merely to execute a previously sealed asset.

The remaining practical problem is test discovery and ownership. If tests may live at arbitrary paths or if each asset records a custom test path, tooling must maintain additional path metadata and humans must search the repository to determine which tests belong to an asset. Directly importing an asset Python file from tests would also create a second resolution path separate from the MLDB runtime resolver.

A deterministic convention can avoid both problems.

## Decision

Define a repository-level `mldb_tests/` tree for tests owned by individual MLDB executable assets.

Asset test placement is derived mechanically from asset kind and asset ID. No test path is stored in asset YAML.

### Asset test placement

The v1 layout is:

```text
mldb_tests/
  architectures/
    <architecture-id>/
      test_*.py

  train_protocols/
    <train-protocol-id>/
      test_*.py

  evaluation_protocols/
    <evaluation-protocol-id>/
      test_*.py
```

Examples:

```text
mldb_tests/architectures/plain-cnn-v1/test_forward.py
mldb_tests/train_protocols/tile-classifier-adamw-cosine-v1/test_contract.py
mldb_tests/evaluation_protocols/tile-classifier-standard-eval-v1/test_metrics.py
```

The executable asset directories use the exact MLDB asset ID, including hyphens and terminal revision suffixes.

The mapping is therefore:

```text
Architecture <id>
  -> mldb_tests/architectures/<id>/

TrainProtocol <id>
  -> mldb_tests/train_protocols/<id>/

EvaluationProtocol <id>
  -> mldb_tests/evaluation_protocols/<id>/
```

MLDB tooling does not search arbitrary repository locations for asset tests.

### Executable assets requiring asset tests

The following executable asset kinds require asset-specific pytest coverage before they may be sealed:

```text
Architecture
TrainProtocol
EvaluationProtocol
```

Task, Model, Training Run, Evaluation Run, Study, and Study Run are not executable assets and do not receive per-record pytest directories under this convention. Their generic invariants are tested by the MLDB runtime test suite and enforced by runtime validation.

Corpus v1 is also excluded from the mandatory asset-test rule. A Corpus is identified by its immutable materialized artifact and hash. Its sibling builder Python file records how the materialized Corpus was produced but does not guarantee reproduction from mutable upstream sources. Builder-specific tests may still be added when useful; making them mandatory is deferred until a concrete need appears.

### Sealing requirement

An Architecture, Train Protocol, or Evaluation Protocol may be sealed only after both runtime validation and its derived asset pytest directory succeed.

Conceptually:

```text
resolve draft asset
  -> runtime/schema/integrity validation
  -> run pytest for mldb_tests/<kind>/<asset-id>/
  -> pytest exits successfully
  -> record implementation SHA-256
  -> set status: sealed
```

The asset test directory must exist and pytest must collect and pass at least one test. An empty or missing directory does not satisfy the sealing requirement.

The canonical invocation shape is conceptually:

```text
python -m pytest mldb_tests/<kind>/<asset-id>/
```

The concrete CLI command that orchestrates validation and sealing may be defined by implementation work later. Pytest execution belongs to development/sealing tooling, not to ordinary MLDB Run execution.

### Runtime does not depend on pytest

The MLDB runtime remains responsible for enforcing contracts during ordinary use, including as applicable:

- YAML loading;
- asset ID and basename validation;
- sibling Python resolution;
- sealed implementation hash verification;
- Task and other asset compatibility checks;
- public parameter resolution and unknown-key rejection;
- Architecture construction;
- Train Protocol and Evaluation Protocol invocation;
- Run lifecycle transitions;
- canonical artifact validation and hashing.

These runtime checks are not skipped merely because asset pytest passed at sealing time.

Pytest instead verifies that the runtime enforcement behaves correctly and that executable assets satisfy behavior that cannot be proven from metadata alone.

A production or worker environment that only consumes already-sealed assets therefore does not need pytest installed unless that environment is also being used for validation or sealing.

### Asset tests resolve assets through MLDB runtime

Asset-specific tests must use the MLDB asset resolver/registry rather than importing the sibling asset implementation through a second project-specific path convention.

Conceptually:

```python
architecture = registry.architecture("plain-cnn-v1")
model = architecture.build()
```

rather than:

```python
from some_project_module import PlainTileClassifier
```

or manually loading:

```text
mldb_data/architectures/plain-cnn-v1.py
```

through test-owned import logic.

The exact runtime API may differ from the conceptual example, but there must be one shared asset-resolution mechanism used by tests, CLI tooling, Study workers, and concrete Run execution.

This ensures asset tests also exercise the same YAML-to-Python resolution path used in real execution.

Tests may of course inspect or exercise the resolved executable object after resolution.

### Scope of asset tests

Asset tests should test behavior owned by that asset rather than repeat all generic MLDB validator tests.

Typical Architecture tests include:

- `build()` returns the expected `torch.nn.Module` family;
- representative forward inputs satisfy the declared interface;
- output dimensions or structures satisfy the Task contract;
- architecture-specific invariants such as parameter-count bounds or normalization choices.

Typical Train Protocol tests include:

- a lightweight or fake execution consumes `TrainContext.parameters` correctly;
- the protocol returns the selected trained `torch.nn.Module`;
- protocol-specific loss, augmentation, checkpoint-selection, or external-adapter logic behaves as intended where those behaviors can be tested cheaply;
- the returned learned state remains compatible with the selected Architecture.

Typical Evaluation Protocol tests include:

- metric calculations for small deterministic fixtures;
- formal metric keys and meanings expected by the protocol;
- generation of declared structured artifacts;
- protocol-specific thresholding, matching, or aggregation logic.

Asset tests should avoid expensive full-corpus or full-training executions when a small deterministic fixture can verify the same executable contract.

### MLDB runtime tests are separate

Tests for MLDB itself are distinct from tests owned by individual assets.

Runtime tests verify generic behavior such as:

- registry and resolver behavior;
- schema and ID validation;
- hash verification;
- parameter resolution;
- Training Run state transitions and canonical weight handling;
- automatic Model creation;
- Evaluation Run output validation and failure isolation;
- Study grid expansion and Study Run planning;
- queue/DAG behavior when such implementation is added.

These tests belong to the MLDB implementation's own test tree rather than `mldb_tests/<kind>/<asset-id>/`.

This ADR does not require a specific source-code location for the future MLDB runtime package. Its implementation tests should remain colocated with or conventionally attached to that runtime package according to the repository's Python project layout.

### Test code is verification, not asset identity

Asset pytest files are not part of the executable asset identity and are not included in `implementation.sha256`.

The implementation hash continues to cover the sibling executable `.py` file defined by the Architecture, Train Protocol, or Evaluation Protocol ADR.

Tests may be strengthened or corrected without creating a new asset revision because changing a test does not change the asset's executable behavior. If a later test reveals that a sealed asset is defective, the sealed implementation itself remains immutable; fixing the executable defect requires a new asset revision under the existing lifecycle rules.

## Rationale

A derived test location makes ownership obvious and machine-resolvable. Given only asset kind and asset ID, a human or tool can identify exactly which pytest directory belongs to that asset without reading additional metadata.

Keeping test paths out of YAML avoids redundant state and eliminates path-drift problems when repository structure is refactored within the established convention.

Requiring asset tests only for executable definitions matches the actual risk boundary. Static records are primarily protected by runtime/schema validation, while Python-defined model, training, and evaluation behavior benefits from executable examples and regression tests.

Keeping pytest outside the runtime dependency graph preserves a small production execution surface. The runtime enforces contracts; pytest verifies both the enforcement mechanism and behavior of human-authored executable assets.

Resolving assets through the runtime in tests prevents the test suite from creating a parallel import mechanism that could succeed while real MLDB execution fails. It also naturally tests basename resolution, YAML metadata loading, implementation loading, and the public runtime handle before asset-specific assertions begin.

Using small deterministic fixtures keeps the sealing gate fast enough to run routinely. Full GPU training and large-corpus evaluation are experiment workloads, not appropriate prerequisites for every code edit or sealing operation.

## Rejected alternatives

### Store an arbitrary test path in each asset YAML

A field such as `tests.path` would duplicate information that can be derived from asset kind and ID. It would also introduce stale path metadata and allow otherwise identical assets to use inconsistent test layouts.

MLDB instead uses one repository convention.

### Place asset tests next to each `.py` implementation

Files such as `mldb_data/architectures/test_plain_cnn_v1.py` would mix executable data assets with development verification code and make sibling-basename discovery noisier.

A dedicated `mldb_tests/` tree keeps registered data assets and test code distinct while preserving a deterministic mapping.

### Let pytest discover all repository tests during every seal

Running the complete repository suite for a one-asset sealing operation would make local iteration increasingly expensive and would obscure which tests are required for the asset being sealed.

Sealing runs only the derived asset directory. CI may additionally run all MLDB runtime tests and all asset tests.

### Import asset implementations directly in pytest

Direct imports or test-owned `importlib` rules would duplicate the runtime resolver and could allow tests to pass even when actual MLDB resolution is broken.

Asset tests therefore resolve through the MLDB runtime.

### Make pytest part of ordinary Run execution

Re-running pytest before each Training Run or Evaluation Run would add unnecessary dependencies and latency while duplicating validation already performed at sealing and runtime boundaries.

Pytest is a development and sealing tool. Runtime integrity checks remain active during execution.

## Consequences

MLDB implementation work must provide one reusable asset-resolution API that can be consumed by CLI tooling, Run execution, Study workers, and pytest.

Sealing tooling for Architecture, Train Protocol, and Evaluation Protocol must derive the corresponding `mldb_tests/` path, execute that test directory, and refuse sealing when tests are missing, uncollected, or failing.

Repository CI can cheaply separate two responsibilities:

```text
MLDB runtime tests
  -> generic infrastructure correctness

mldb_tests/
  -> all registered executable asset tests
```

When an individual executable asset is edited while still `draft`, developers can run only its derived test directory. When MLDB runtime behavior changes, the runtime test suite should be run and CI may additionally run all asset tests to detect integration regressions.

No new metadata field is added to Architecture, Train Protocol, or Evaluation Protocol YAML for test discovery.

The future implementation should keep the core runtime free of pytest imports. Any command that runs tests as part of sealing should live in development/CLI tooling layered on top of the runtime.

## Evidence

The existing repository already uses lightweight Python tests under `tools/recognition/tests/` for model construction, geometry, dataset building, experiment helpers, and training-related code. Representative tests use tiny tensors, temporary directories, and synthetic SQLite fixtures rather than full production workloads.

For example, `tools/recognition/tests/test_rotated_fcos_nano.py` verifies model output shapes, normalization choices, target generation, geometry, and parameter-count constraints with small deterministic tensors. `tools/recognition/tests/test_build_tile_classifier_dataset.py` verifies dataset contracts using temporary SQLite databases and synthetic images. These patterns demonstrate that meaningful executable-contract coverage can remain fast and local.

The current MLDB design makes the same strategy especially effective: generic infrastructure owns ID, hash, lineage, parameter, lifecycle, and artifact rules, while asset pytest only needs to verify behavior specific to each executable Architecture or Protocol.
