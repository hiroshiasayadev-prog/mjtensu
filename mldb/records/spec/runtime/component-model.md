# Concept: MLDB runtime component model

- **id**: `spec:mldb.runtime.component_model`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.runtime`

## What this is

Defines the logical components of the MLDB runtime and the responsibility boundary between them.

The component model does not prescribe Python package names, classes, constructors, or dependency-injection mechanics.

## Concept model

| component | owns | does not own |
|---|---|---|
| repository access | Read and write registered MLDB definitions, Run records, and MLDB-owned artifacts through canonical repository locations. | Entity semantics, executable behavior, queue transport, or model-family logic. |
| asset resolver | Resolve an MLDB ID through canonical repository placement to validated metadata and referenced upstream assets. Produce resolver-owned handles whose provenance and resource paths are canonical. | Training, evaluation, Study expansion, Worker-local materialization, or executable asset business logic. |
| executable loader | Load the Python implementation selected by a runtime handle and expose its declared entrypoint after integrity checks. For resolver output this is the canonical sibling; distributed execution may use an integrity-verified Worker-local copy of Controller-selected bytes. | Interpretation of model-family-specific code, repository authority, Worker cache policy, or pytest execution. |
| training executor | Create and finalize one Training Run, invoke one resolved Train Protocol, materialize canonical learned weights, and create the resulting Model on success. | Train Protocol internals, Study grid expansion, or Evaluation Run behavior. |
| evaluation executor | Create and finalize one Evaluation Run, invoke one resolved Evaluation Protocol, validate formal results, and materialize accepted result artifacts. | Evaluation Protocol internals, training behavior, or cross-Run cancellation. |
| study materializer | Resolve one sealed Study and materialize one immutable Study Run plan from its exclusive training-grid or existing-Model source plus common evaluation stages. | Worker claiming, queue leases, retry timers, or execution of model-family logic. |

The listed components are logical runtime responsibility units. One implementation object may implement more than one unit when the responsibility boundaries remain observable and testable.

Distributed orchestration may split one logical execution component across process boundaries. Controller owns canonical asset resolution, concrete Run allocation, canonical result acceptance, finalization, and canonical persistence, while Worker owns assigned immutable-byte verification/materialization, executable invocation, and execution-local candidate preparation according to `spec:mldb.orchestration`.

Worker materialization is not a second asset-resolution mechanism. Controller-selected validated metadata and exact immutable bytes may be re-expressed through the same runtime handle types for execution after Worker verifies the assigned integrity identities. Repository-provenance locations may be unavailable on Worker, while required execution-resource paths point to verified Worker-local copies. Those local paths never become canonical repository locations or sources of truth.

## Dependency direction

```text
repository access
      ^
      |
asset resolver <--------- study materializer
      ^
      |
executable loader
      ^
      |
      +-------------------+
      |                   |
training executor   evaluation executor
```

`asset resolver -> repository access` means the resolver depends on repository access.
`executable loader -> asset resolver` means executable loading starts from a resolved asset.
Training and evaluation executors depend on the executable loader and repository access for their Run results.
Study materialization depends on asset resolution and repository access, but not on training or evaluation execution.

The diagram shows logical dependency only. It does not require inheritance, one object per box, or the exact call graph shown.

## Rules

- Canonical ID-to-file lookup must use common asset resolution rather than being reimplemented by runtime execution components.
- A distributed Worker must not perform independent repository lookup for assigned assets; it consumes Controller-selected metadata and exact immutable bytes and verifies required integrity before constructing the same runtime execution handles.
- Runtime execution components must use the declared executable entrypoint resolved by the executable loader.
- The asset resolver must not execute training or evaluation code while resolving metadata.
- The executable loader must not interpret or rewrite project-owned Architecture, Train Protocol, or Evaluation Protocol behavior.
- Training Run failure must not mutate or cancel unrelated Runs.
- Evaluation Run failure must not mutate or cancel unrelated Runs.
- Study materialization must produce execution intent and lineage without owning queue-worker state.
- Queue transport and Worker coordination must remain replaceable without changing persistent MLDB entity semantics.
- In distributed execution, Worker-produced result files are candidates until Controller accepts and commits them according to `spec:mldb.orchestration`.
- Formal MLDB artifacts must be committed through the runtime component that owns the corresponding canonical execution result.

## Boundary

| concern | owner |
|---|---|
| Canonical repository paths and placement grammar | `spec:mldb.repository` topic. |
| Exact ID, YAML, SQLite, JSONL, and artifact formats | Entity-specific format specs. |
| Asset resolution and integrity procedure | `spec:mldb.runtime.asset_resolution` topic. |
| Public parameter resolution | `spec:mldb.runtime.public_parameters` topic. |
| Train Protocol callable interface | Training interface contract. |
| Evaluation Protocol callable interface | Evaluation interface contract. |
| Study Model-source expansion semantics | `spec:mldb.study`. |
| Executable asset pytest placement and invocation | Verification topic. |
| Concrete Python modules and classes | Implementation. |
| Controller, Queue, Worker, claim, lease, heartbeat, and retry responsibility boundary | `spec:mldb.orchestration`. |
| Exact Queue persistence schema and Worker API wire contract | Future orchestration contract. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.runtime` | Parent runtime overview and end-to-end flow. |
| `spec:mldb.orchestration` | Splits logical runtime execution responsibilities across Controller, Queue, and Worker. |
