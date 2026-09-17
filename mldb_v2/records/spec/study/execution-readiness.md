# Contract: Planned-stage semantic readiness

- **id**: `spec:mldb.v2.study.execution_readiness`
- **status**: draft
- **date**: 2026-09-17
- **parent**: `spec:mldb.v2.study`
- **contract_class**: `lifecycle`

## Purpose

MLDB defines when a planned stage is semantically eligible to execute. The execution backend may encode the same dependencies in a native Pipeline/DAG and owns physical scheduling, but it must not release a stage before the corresponding MLDB semantic gate is satisfied.

This boundary is not a queue, retry scheduler, worker selector, or resource scheduler.

## Readiness

For a training-source trial, the training stage is initially eligible. Its Evaluation coordinates are blocked until the Training candidate has been formally accepted, the canonical Training Result is `completed`, and the deterministic Model lineage exists.

For an existing-Model trial, Evaluation coordinates are initially eligible after the referenced Model/Training Result lineage and required artifacts pass normal MLDB validation.

Evaluation coordinates are siblings: failure of one does not semantically block another unless a future Study contract explicitly declares such a dependency.

## Terminal upstream handling

If training becomes canonically `failed`, every still-unreleased dependent Evaluation coordinate becomes `skipped: upstream_failed`. If training becomes canonically `cancelled`, they become `skipped: upstream_cancelled`.

When Study Result status is `cancelling`, no new semantic gate may open. The backend is asked to cancel the Pipeline/active child work; never-started downstream stages remain unreleased and are closed according to the canonical cancellation reconciliation rules.

## Backend realization

A backend-native Pipeline may predeclare all Plan nodes for UI/DAG purposes. Predeclaration is not execution readiness. The adapter must prevent a gated node from being queued/executed until MLDB confirms the gate.

For training-produced Models, the runtime `StageInput` for dependent Evaluation must be materialized only after the accepted canonical Training Result/Model exists. Backend controller callbacks/hooks may request that materialization from generic MLDB lifecycle services; they must not synthesize Model lineage from backend Task outputs alone.

The backend may independently choose queue order, worker, retry timing, resource placement, and physical concurrency among semantically eligible sibling stages.
