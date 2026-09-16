# Overview: MLDB v2

- **id**: `spec:mldb.v2`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `root`

## What this is

MLDB v2 is the repository-owned protocol for defining ML experiment assets, compiling deterministic
experiment plans, preserving learned-model lineage, and accepting formal execution results while
delegating scheduling and live execution to replaceable backends.

MLDB v2 deliberately keeps the semantic strengths of MLDB v1 and removes queue/worker
infrastructure from the core.

## System boundary

```text
Git: mldb_data/
  canonical definitions, plans, formal history/results
          |
          v
MLDB v2 protocol + application: mldb_v2/src/
  resolve -> validate -> plan -> drive readiness -> backend -> collect/accept -> close
          |
          v
Execution backend
  ClearML initially: Tasks, Queues, Agents, retries, logs, UI
```          |
          +----------------------+
          v                      v
S3-compatible object store    backend database
large immutable bytes         operational state / projection
```

Git and content-verified object storage are sufficient to preserve formal MLDB history. Backend
state is reconstructible projection/provenance, not canonical truth.

## Core concepts

| concept | responsibility |
|---|---|
| Namespace | Coarse continuing experiment concept and backend comparison group. |
| Task | Semantic prediction problem. |
| Corpus | Immutable materialized sample collection fixed by manifest hash. |
| Architecture | Versioned unweighted model structure. |
| Train Protocol | Versioned reusable training procedure. |
| Evaluation Protocol | Versioned evaluation procedure and formal output declaration. |
| Study | Human-authored comparison intent. |
| Study Plan | Immutable fully expanded execution contract. |
| Training Result | Terminal formal outcome of one planned training stage. |
| Model | Canonical learned identity produced by completed training. |
| Evaluation Result | Terminal formal outcome and accepted metrics/artifacts. |
| Study Result | Canonical execution-history aggregate for one submitted plan. |

## Explicit non-goals
MLDB v2 does not own backend queues, workers/agents, GPU scheduling, leases, heartbeat, live logs,
retry algorithms, experiment UI, or a generic artifact byte store.

## Topics

| ref | responsibility |
|---|---|
| `spec:mldb.v2.architecture` | System responsibility and component boundaries. |
| `spec:mldb.v2.common` | Shared identity, public-parameter, diagnostic, and telemetry contracts. |
| `spec:mldb.v2.repository` | Namespace-first repository placement and typed resolution. |
| `spec:mldb.v2.catalog` | Namespace, Task, Corpus, and Architecture definitions. |
| `spec:mldb.v2.training` | Train Protocol, terminal Training Result, and Model identity. |
| `spec:mldb.v2.evaluation` | Evaluation Protocol, formal metrics/artifacts, and validation. |
| `spec:mldb.v2.study` | Study intent, deterministic expansion, immutable plans, and readiness. |
| `spec:mldb.v2.results` | Canonical formal execution history. |
| `spec:mldb.v2.storage` | Object references, corpus manifests, and integrity. |
| `spec:mldb.v2.backend` | Backend-independent execution port and ClearML mapping. |
| `spec:mldb.v2.api` | Public application operations and resumable Study driver. |
| `spec:mldb.v2.verification` | Definition verification/sealing and result acceptance. |
| `spec:mldb.v2.cli` | Installed user command over the application API. |
