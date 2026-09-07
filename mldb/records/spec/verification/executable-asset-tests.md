# Reference: MLDB executable asset verification

- **id**: `spec:mldb.verification.executable_asset_tests`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.verification`

## What this is

Defines the pytest verification required before an executable MLDB asset may be sealed.

This contract covers Architecture, Train Protocol, and Evaluation Protocol asset tests. Generic MLDB runtime tests are separate and ordinary execution of already-sealed assets does not require pytest.

## Verification matrix

| asset kind | asset-specific pytest required before sealing | canonical test directory |
|---|---:|---|
| Architecture | yes | `mldb_tests/architectures/<architecture-id>/` |
| Train Protocol | yes | `mldb_tests/train_protocols/<train-protocol-id>/` |
| Evaluation Protocol | yes | `mldb_tests/evaluation_protocols/<evaluation-protocol-id>/` |
| Task | no | none |
| Corpus | no mandatory v1 asset-test directory | none |
| Training Run | no | none |
| Model | no | none |
| Evaluation Run | no | none |
| Study | no | none |
| Study Run | no | none |

Corpus builder-specific tests may exist when useful, but Corpus v1 does not require them for registration.

## Asset test placement

The test location is derived only from executable asset kind and exact asset ID.

```text
Architecture <id>
  -> mldb_tests/architectures/<id>/

Train Protocol <id>
  -> mldb_tests/train_protocols/<id>/

Evaluation Protocol <id>
  -> mldb_tests/evaluation_protocols/<id>/
```

The asset test directory uses the exact MLDB asset ID, including hyphens and terminal revision suffix.

No executable asset YAML stores a custom test path.

MLDB sealing tooling must not search arbitrary repository locations to discover asset-owned tests.

## Sealing gate

An executable asset may transition from `draft` to `sealed` only after generic runtime validation and its derived pytest directory both succeed.

Conceptually:

```text
resolve draft asset
  -> runtime/schema/compatibility validation applicable before sealing
  -> run pytest for mldb_tests/<kind>/<asset-id>/
  -> require successful pytest exit
  -> require at least one collected and passed test
  -> calculate implementation SHA-256
  -> persist sealed metadata and implementation hash
```

The pytest invocation shape is:

```text
python -m pytest mldb_tests/<kind>/<asset-id>/
```

The public sealing operation belongs to `spec:mldb.api.controller`. A concrete CLI command may be a thin adapter over that operation.

| condition | sealing result |
|---|---|
| Derived test directory is missing | Refuse sealing. |
| Pytest collects zero tests | Refuse sealing. |
| Any collected test fails or errors | Refuse sealing. |
| Runtime validation fails | Refuse sealing. |
| Tests pass but implementation hash cannot be established | Refuse sealing. |
| Runtime validation succeeds, at least one asset test passes, and implementation hash succeeds | Asset may be sealed. |

## Resolver rule

Asset-specific tests must resolve the target asset through the same MLDB runtime resolution and executable-loading boundary used by production execution tooling.

Conceptually:

```python
architecture = registry.architecture("plain-cnn-v1")
model = architecture.build()
```

Tests must not establish a second asset-loading convention by directly importing or manually loading the sibling implementation path.

This rule applies to Architecture, Train Protocol, and Evaluation Protocol tests.

After resolution, a test may inspect and exercise the returned executable object normally.

## Test ownership

Asset-specific tests verify behavior owned by that executable asset.

Typical Architecture coverage includes:

- `build()` returns a compatible `torch.nn.Module`;
- representative inputs satisfy the declared interface;
- output structure satisfies Task-facing expectations;
- Architecture-specific invariants such as parameter count or normalization behavior.

Typical Train Protocol coverage includes:

- published parameters are consumed through `TrainContext.parameters`;
- lightweight execution returns the protocol-selected trained module;
- protocol-specific loss, augmentation, checkpoint-selection, or external-adapter behavior works on small fixtures;
- returned state remains compatible with the selected Architecture.

Typical Evaluation Protocol coverage includes:

- metric calculations on small deterministic fixtures;
- expected formal metric keys and semantics;
- declared artifact generation;
- protocol-specific matching, thresholding, or aggregation behavior.

Asset tests should prefer small deterministic fixtures over full-corpus, long-running, or GPU-heavy experiment workloads when the smaller fixture verifies the same behavior.

## Runtime test boundary

Generic MLDB runtime tests are separate from `mldb_tests/<kind>/<asset-id>/`.

Runtime tests own generic behavior such as:

- typed registry and asset resolution;
- schema, ID, basename, and hash validation;
- public parameter resolution;
- Training Run lifecycle and canonical-weight handling;
- automatic Model creation;
- Evaluation Run result validation and failure isolation;
- Study expansion and Study Run plan validation;
- queue or DAG behavior when such infrastructure is later introduced.

The concrete source location of the MLDB runtime test suite is an implementation-layout decision rather than part of this contract.

## Pytest dependency boundary

Pytest is development and sealing tooling.

- Ordinary Training Run execution of a previously sealed Train Protocol must not invoke pytest.
- Ordinary Evaluation Run execution of a previously sealed Evaluation Protocol must not invoke pytest.
- Model loading must not invoke pytest.
- Study execution must not invoke asset pytest merely because it consumes sealed assets.
- Runtime schema, compatibility, integrity, lifecycle, and output checks remain active during ordinary execution.
- A worker environment that only consumes sealed assets does not require pytest unless it also performs validation or sealing.

Passing pytest does not waive any runtime contract check.

## Test identity boundary

Asset pytest files are verification code, not executable asset identity.

They are not included in `implementation.sha256`.

Tests may be strengthened or corrected without creating a new executable asset revision because test changes do not change the asset implementation itself.

If a later test exposes a defect in a sealed executable asset, the sealed implementation remains immutable. Correcting the executable behavior requires a new Architecture, Train Protocol, or Evaluation Protocol revision under that asset's lifecycle rules.

## Boundary

| concern | owner |
|---|---|
| Canonical `mldb_tests/` physical paths | `spec:mldb.repository.layout`. |
| Asset ID, lifecycle, and implementation hash fields | Entity-specific Architecture, Train Protocol, and Evaluation Protocol format specs. |
| Typed runtime asset resolution | `spec:mldb.runtime.asset_resolution`. |
| Architecture callable behavior | `spec:mldb.catalog.architecture_build`. |
| Train Protocol callable behavior | `spec:mldb.training.train_interface`. |
| Evaluation Protocol callable behavior | `spec:mldb.evaluation.evaluate_interface`. |
| Generic runtime test source-tree placement | Implementation. |
| Public sealing operation | `spec:mldb.api.controller`. |
| Concrete CLI syntax and pytest subprocess implementation | Implementation. |
| CI policy beyond the per-asset sealing gate | Repository automation policy. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.verification` | Parent verification Index. |
| `spec:mldb.api.controller` | Exposes the public sealing operation that invokes this verification gate. |
| `spec:mldb.repository.layout` | Defines the deterministic asset-test directory mapping. |
| `spec:mldb.runtime.asset_resolution` | Defines the shared resolution path asset tests must use. |
| `spec:mldb.catalog.architecture_build` | Architecture behavior under verification. |
| `spec:mldb.training.train_interface` | Train Protocol behavior under verification. |
| `spec:mldb.evaluation.evaluate_interface` | Evaluation Protocol behavior under verification. |
