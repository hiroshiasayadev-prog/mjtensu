# Overview: MLDB Model

- **id**: `spec:mldb.model`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb`

## What this is

Defines the MLDB Model as the stable learned-object identity automatically produced from one completed Training Run.

This overview owns only the Model-level role and flow. Model YAML shape and one-to-one identity/loading semantics belong to focused child specifications.

## Current contract

A Model is not a second learned-weight artifact and is not a promotion record.

```text
completed Training Run
  |
  +--> artifacts/weights.pt
  |
  +--> automatic deterministic Model YAML
           |
           v
      downstream learned-model reference
```

The completed Training Run remains authoritative for Architecture, Corpus, Train Protocol, seed, execution environment, and canonical learned-weight metadata.

The Model provides one stable immutable ID that downstream operations can reference without treating the Training Run itself as the learned-object identity.

## Rules

- Every completed Training Run has exactly one corresponding Model.
- Running, failed, and cancelled Training Runs have no Model.
- Model creation is automatic during successful Training Run finalization.
- Model is immutable from creation and has no `draft` / `sealed` lifecycle.
- Model does not own or copy `weights.pt`.
- Model does not record quality, promotion, release, or deployment status.
- Downstream operations acting on a learned model should reference Model rather than Training Run.

## Non-goals

- Define canonical learned-weight serialization.
- Define training execution lifecycle.
- Define Evaluation Run behavior.
- Define ONNX or other export artifacts.
- Define production selection, promotion, release, or deployment state.
- Define imported or externally obtained learned models in v1.

## Topics

| title | kind | ref | summary |
|---|---|---|---|
| Model format | Contract | `spec:mldb.model.model_format` | Minimal Model YAML schema and validation rules. |
| Model identity | Concept | `spec:mldb.model.identity` | Deterministic Training Run to Model mapping, automatic creation, immutability, and learned-state loading lineage. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.training` | Produces the completed Training Run and canonical learned weights represented by Model. |
| `spec:mldb.training.canonical_weights` | Owns the learned bytes resolved through Model lineage. |
