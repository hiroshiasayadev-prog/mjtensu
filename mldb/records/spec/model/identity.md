# Concept: MLDB Model identity

- **id**: `spec:mldb.model.identity`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.model`

## What this is

Defines how one completed Training Run produces one immutable Model identity and how that Model resolves its learned state.

Model identity is deterministic and derived. It is not manually registered and does not imply quality or deployment selection.

## Concept model

| concept | contract |
|---|---|
| source execution | Exactly one completed Training Run. |
| learned bytes | The Training Run's canonical `artifacts/weights.pt`. |
| Model identity | Deterministically derived from the Training Run ID. |
| Model record | One minimal YAML record containing only schema, Model ID, and Training Run reference. |
| downstream subject | Evaluation and later learned-model operations reference Model. |

For Training Run v1 IDs, identity mapping is:

```text
tr-YYYYMMDD-NNN
        |
        v
mdl-YYYYMMDD-NNN
```

The date and sequence suffix are preserved exactly.

## Automatic creation

Successful training finalization establishes this invariant:

```text
completed Training Run <=> exactly one corresponding Model
```

Conceptually:

```text
Train Protocol returns selected trained module
  -> canonical weight validation and serialization
  -> Training Run result metadata recorded
  -> Training Run finalized completed
  -> corresponding Model YAML ensured
```

Model creation must be idempotent because filesystem writes are not assumed to form one database transaction.

If interruption leaves a valid completed Training Run without its deterministic Model YAML, reconciliation may create the missing exact Model record.

If an existing Model file disagrees with the deterministic relationship, tooling must reject the inconsistency rather than repurpose that Model ID.

## Learned-state resolution

A Model does not contain or copy learned bytes.

The learned module is resolved through immutable lineage:

```text
Model
  -> training_run
  -> TrainingRun.architecture
  -> Architecture.build()

Model
  -> training_run
  -> TrainingRun.result.weights
  -> artifacts/weights.pt
```

Conceptually, Model loading performs:

1. resolve Model;
2. resolve its completed Training Run;
3. verify canonical learned-weight integrity;
4. resolve the Training Run Architecture and its required integrity;
5. load `artifacts/weights.pt` with `torch.load(..., map_location="cpu", weights_only=True)`;
6. validate that the loaded value satisfies the canonical plain `Mapping[str, torch.Tensor]` contract;
7. construct a fresh module through the Architecture build interface;
8. apply the loaded state through `load_state_dict(..., strict=True)`;
9. return the learned module to the downstream operation.

The Model record does not repeat Architecture ID or weight hash because the immutable Training Run is authoritative for both.

## Rules

- Model ID is never manually selected.
- A completed Training Run must not have zero or multiple Models after successful reconciliation.
- A non-completed Training Run must not have a Model.
- Model has no revision suffix independent from its Training Run-derived ID.
- Model is immutable immediately after creation.
- Repeating otherwise identical training creates a new Training Run and therefore a new Model.
- Model creation does not mean the learned result is good, accepted, best, candidate, or production-ready.
- Model must not duplicate canonical learned-weight bytes.
- Downstream operations concerning a learned model should use Model ID rather than Training Run ID.
- External or legacy learned weights without a completed MLDB Training Run are outside Model v1.

## Boundary

| concern | owner |
|---|---|
| Model YAML fields | `spec:mldb.model.model_format`. |
| Training Run completion semantics | `spec:mldb.training.training_run_lifecycle`. |
| Canonical learned-weight serialization and integrity | `spec:mldb.training.canonical_weights`. |
| Architecture construction | `spec:mldb.catalog.architecture_build`. |
| Evaluation of a learned Model | `spec:mldb.evaluation`. |
| Model export | Future export topic. |
| Quality selection, promotion, release, or deployment | Future lifecycle topics. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.model` | Parent Model overview. |
| `spec:mldb.model.model_format` | Defines the minimal persisted identity record. |
| `spec:mldb.training.canonical_weights` | Owns the learned state represented by Model. |
| `spec:mldb.catalog.architecture_build` | Constructs the module that receives the learned state. |
