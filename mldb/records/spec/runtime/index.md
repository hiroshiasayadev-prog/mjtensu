# Overview: MLDB runtime

- **id**: `spec:mldb.runtime`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb`

## What this is

The MLDB runtime resolves registered MLDB assets, launches concrete training and evaluation work, materializes Study execution plans, and records resulting immutable execution state whether execution occurs locally or through a separate Worker.

This overview owns only the runtime-level responsibility model and end-to-end flow. File formats, callable interfaces, lifecycle rules, parameter resolution, and artifact schemas belong to focused child specifications.

## Current contract

The runtime operates across four responsibility classes.

| class | runtime responsibility |
|---|---|
| definition resolution | Resolve registered reusable definitions and verify the generic compatibility and integrity rules required before execution. |
| execution launch | After launch preflight succeeds, create concrete Run state, invoke or dispatch the selected executable definition through its common interface, and finalize that Run independently. |
| result materialization | Persist MLDB-owned canonical results only after the applicable result contract succeeds. |
| orchestration expansion | Materialize a sealed Study into a concrete Study Run plan whose trials can be executed independently according to their dependencies. |

The runtime does not own the model-family-specific implementation inside Architecture, Train Protocol, or Evaluation Protocol code.

## Runtime flow

```text
sealed Study execution request
                  |
                  v
       Study validation
 asset resolution / compatibility /
 integrity / parameters / Model source
                  |
                  v
          Study Run + plan
                  |
          +-------+-------+
          |               |
          v               v
 training-derived     existing Model
      trial               trial
          |               |
          v               |
    Training Run           |
          |               |
          v               |
     Train Protocol        |
          |               |
          v               |
         Model <-----------+
          |
          v
   Evaluation Run(s)
          |
          v
 Evaluation Protocol
          |
          v
 validated formal results
```

A failed Training Run or Evaluation Run finalizes only that Run. Independent Runs remain executable.

A dependent operation may proceed only when its required upstream result exists. A training-derived Study trial waits for its completed Training Run and Model before evaluation; an existing-Model Study trial begins from the immutable Model selected by the Study.

## Responsibility boundaries

| concern | runtime-level owner | detailed contract owner |
|---|---|---|
| Registered asset discovery and reference resolution | MLDB runtime | Asset-resolution topic. |
| Physical `mldb_data` and `mldb_tests` placement | MLDB repository contract | Repository topic. |
| YAML and SQLite field formats | Entity-specific format contracts | Catalog, training, model, evaluation, and study topics. |
| Public parameter default resolution and override validation | Common runtime rule | Public-parameters topic. |
| `build()`, `train()`, and `evaluate()` signatures | Executable asset interfaces | Entity-specific interface contracts. |
| Training Run execution and canonical learned weights | Training runtime | Training topic. |
| Model identity | Model contract | Model topic. |
| Evaluation result validation and formal artifacts | Evaluation runtime | Evaluation topic. |
| Study Model-source expansion and Study Run planning | Study runtime | Study topic. |
| Durable Queue lifecycle, Worker claim, lease, retry state, and reconciliation | Orchestration layer | `spec:mldb.orchestration.queue_lifecycle`. |
| Pytest invocation for executable asset verification | Verification tooling, not execution runtime | Verification topic. |

## Non-goals

- Define exact YAML fields for any MLDB entity.
- Define Python package or class names used by the implementation.
- Define exact SQLite Queue schema, Worker API payloads, or distributed transport.
- Define model-family-specific training, inference, decoding, loss, or metric algorithms.
- Define MLflow synchronization behavior.
- Define export, promotion, release, or deployment workflows not yet specified by MLDB.

## Boundary

The runtime is the common enforcement and execution boundary for MLDB records.

Top-level queued execution begins from a sealed Study under `spec:mldb.orchestration`. Study validation completes before Study Run allocation and Queue insertion. Concrete Training Run or Evaluation Run preflight still occurs before that child Run is allocated; a child preflight rejection does not create the child Run.

Reusable executable assets own their project-specific learned or evaluation behavior. The runtime owns generic resolution, compatibility checks, Run creation after successful preflight, formal result handling, and lifecycle enforcement.

Working files remain execution-local until a dedicated result contract promotes them into an MLDB-owned formal artifact.

Distributed execution must preserve the same runtime semantics. `spec:mldb.orchestration` assigns canonical Run allocation, finalization, and artifact commit to Controller while Worker owns compute-side invocation and candidate result preparation.

## Topics

| title | kind | ref | summary |
|---|---|---|---|
| Runtime component model | Concept | `spec:mldb.runtime.component_model` | Logical runtime components, responsibility boundaries, and dependency direction. |
| Asset resolution | Concept | `spec:mldb.runtime.asset_resolution` | Typed entity lookup, metadata identity, upstream references, integrity checks, and resolved handles. |
| Public parameter resolution | Reference | `spec:mldb.runtime.public_parameters` | Shared default resolution, caller override validation, resolved mapping, and seed boundary. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb` | Parent MLDB specification overview. |
| `spec:mldb.api` | Exposes runtime validation, sealing, Study execution, and reads through the public Controller application boundary. |
| `spec:mldb.orchestration` | Defines how runtime responsibilities split across Controller, Queue, and Worker for distributed execution. |
