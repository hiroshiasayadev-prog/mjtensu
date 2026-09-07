# Overview: MLDB training

- **id**: `spec:mldb.training`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb`

## What this is

Defines the MLDB training boundary from one reusable Train Protocol and concrete training inputs to one terminal Training Run and its canonical learned state.

This overview owns the training flow and topic routing. Train Protocol metadata, callable semantics, Run metadata, lifecycle, and learned-weight serialization belong to focused child specs.

## Current contract

One training execution combines exactly one Corpus, one Architecture, one Train Protocol, one seed, and one resolved public-parameter mapping.

```text
Corpus + Architecture + Train Protocol
              |
              +--> seed
              +--> caller parameters
              |
              v
      launch preflight
  resolve / compatibility /
 integrity / seed / parameters
              |
              v
      create Training Run
          status=running
              |
              v
         TrainContext
              |
              v
          train(context)
              |
              v
       trained nn.Module
              |
              v
 strict Architecture-state check
              |
              v
 artifacts/weights.pt
              |
              v
    Training Run completed
              |
              v
        automatic Model
```

A launch request rejected during preflight creates no Training Run. After a Training Run is allocated, a failed or cancelled execution remains a terminal Training Run and produces no Model.

## Responsibility boundaries

| concern | owner |
|---|---|
| Reusable training identity, Task binding, implementation hash, and public parameter declaration | Train Protocol format. |
| `TrainContext` and `train(context)` callable boundary | Train interface. |
| Concrete Run fields and Study lineage fields | Training Run format. |
| `running` to terminal state transitions and immutability | Training Run lifecycle. |
| Canonical `artifacts/weights.pt` validation, serialization, and integrity metadata | Canonical weights. |
| Common public-parameter resolution | `spec:mldb.runtime.public_parameters`. |
| Architecture construction | `spec:mldb.catalog.architecture_build`. |
| Model identity created after successful completion | `spec:mldb.model`. |
| Queue transport and worker state | Outside the current training contract. |

## Non-goals

- Define one universal training loop.
- Define optimizer, loss, augmentation, checkpoint-selection, or framework internals for all protocols.
- Define Study grid expansion.
- Define post-training evaluation.
- Define exported ONNX or deployment artifacts.
- Define the concrete Python package or CLI command that launches training.

## Topics

| title | kind | ref | summary |
|---|---|---|---|
| Train Protocol format | Contract | `spec:mldb.training.train_protocol_format` | Versioned Train Protocol YAML, lifecycle, public parameter declaration, and implementation integrity. |
| Train interface | Contract | `spec:mldb.training.train_interface` | `TrainContext` request and trained `torch.nn.Module` response contract. |
| Training Run format | Contract | `spec:mldb.training.training_run_format` | Concrete Training Run YAML fields, references, execution facts, and result metadata. |
| Training Run lifecycle | Reference | `spec:mldb.training.training_run_lifecycle` | Run state transitions, retry semantics, terminal immutability, and failure isolation. |
| Canonical learned weights | Reference | `spec:mldb.training.canonical_weights` | Runtime-owned state-dict validation, serialization, path, and integrity rules. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb` | Parent MLDB overview. |
| `spec:mldb.runtime.public_parameters` | Defines shared public-parameter resolution. |
| `spec:mldb.catalog.architecture_build` | Defines fresh Architecture construction used by training. |
| `spec:mldb.repository.layout` | Defines Train Protocol and Training Run placement. |
