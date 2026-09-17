# Contract: Execution backend port

- **id**: `spec:mldb.v2.backend.backend_port`
- **status**: draft
- **date**: 2026-09-17
- **parent**: `spec:mldb.v2.backend`
- **contract_class**: `port`

## Purpose

The backend port separates MLDB experiment semantics from operational execution mechanics. MLDB supplies one immutable Study Plan plus one canonical Study Result identity; the backend realizes that execution using its native orchestration model.

MLDB owns what the Study means. The backend owns how the already-defined work is scheduled and kept alive.

## Required capabilities

A backend adapter supports these semantic capabilities:

- idempotently create or recover one backend Study execution for one exact Study Result;
- preserve the Plan's trial/stage identities and semantic dependencies when mapping them to backend work;
- execute every admitted training/evaluation step through `spec:mldb.v2.backend.execution_harness`;
- expose per-stage active/terminal observations and terminal candidates conforming to `spec:mldb.v2.backend.candidate_outcome`;
- release downstream work only after MLDB semantic gates required by `spec:mldb.v2.study.execution_readiness` are satisfied;
- request cancellation of the backend Study execution and its active child work.

## Operational ownership

The backend MAY use its native pipeline/controller, queue, worker registry, retry, heartbeat, resource scheduling, and cancellation mechanisms. Generic MLDB must not duplicate those mechanisms merely to drive one backend.

Backend retry creates operational attempts of the same MLDB logical stage. Retry policy and timing are backend concerns, but collected attempt provenance must remain attributable to the exact Study Result + trial + stage coordinate.

Automatic backend cache/reuse is disabled by default for formal execution. Reuse of a previous backend Task must not silently satisfy a fresh MLDB Study Result unless a future explicit MLDB contract authorizes that behavior.

## Semantic gates

Backend dependency scheduling does not grant semantic authority. In particular, an Evaluation that depends on newly-trained weights MUST NOT start until the Training candidate has passed MLDB formal acceptance and the canonical Model lineage exists.

The backend may encode Plan dependencies in a native DAG, but release of a semantically gated downstream step occurs only after MLDB acceptance confirms the required canonical predecessor.

## Optional observational capabilities

A backend MAY expose Study/pipeline UI navigation, child Task logs, richer telemetry, summary plots/artifacts, or backend execution URLs for read-only queries. Lack of those extras does not alter canonical MLDB semantics.

## Rules

- Backend execution identity is opaque provenance, not canonical identity.
- Backend status never directly writes formal MLDB results.
- Backend `completed` is insufficient for canonical completion; result acceptance still runs.
- The port consumes compiled/canonical values, never model-family-specific runner configuration.
- Backend telemetry/logs/UI are projections only unless accepted by a formal MLDB result contract.
- Canonical Study Result mutation remains MLDB-owned.
