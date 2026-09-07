# Overview: MLDB

- **id**: `spec:mldb`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `root`

## What this is

MLDB is the project-owned system for defining machine-learning assets, recording concrete executions, and preserving model lineage and formal evaluation results.

This overview is the navigation root for MLDB specifications. Detailed contracts belong to focused child topics.

## Current contract

MLDB separates reusable definitions from concrete execution records.

| class | examples | responsibility |
|---|---|---|
| reusable definition | Task, Corpus, Architecture, Train Protocol, Evaluation Protocol, Study | Defines stable inputs, executable behavior, or experiment intent. |
| execution record | Training Run, Evaluation Run, Study Run | Records one concrete execution and its immutable result state. |
| learned identity | Model | Identifies the canonical learned result of one completed Training Run. |
| formal artifact | canonical weights, validated evaluation artifacts, Study Run plan | Preserves immutable result bytes and their integrity metadata. |
| working file | protocol checkpoints, logs, temporary predictions, diagnostics | Supports one execution without becoming a formal MLDB result by presence alone. |

## Non-goals

- MLDB does not define model-training algorithms for every model family.
- MLDB does not make MLflow or another visualization system the source of truth.
- MLDB does not make queue or Worker operational state authoritative experiment history; exact queue storage, lease algorithms, and Worker transport belong to focused orchestration contracts.
- MLDB does not define production recognition-runtime behavior.

## Topics

| title | kind | ref | summary |
|---|---|---|---|
| MLDB runtime | Overview | `spec:mldb.runtime` | Runtime responsibilities, execution flow, and boundaries between definitions, executions, formal artifacts, and working files. |
| MLDB repository | Overview | `spec:mldb.repository` | Repository-level MLDB data and asset-test roots plus canonical entity placement. |
| MLDB catalog | Index | `spec:mldb.catalog` | Task, Corpus, and Architecture contracts used as reusable inputs before concrete execution. |
| MLDB training | Overview | `spec:mldb.training` | Train Protocol execution, Training Run records and lifecycle, and canonical learned-weight materialization. |
| MLDB Model | Overview | `spec:mldb.model` | Automatic learned-model identity derived one-to-one from completed Training Runs. |
| MLDB evaluation | Overview | `spec:mldb.evaluation` | Evaluation Protocol execution, Evaluation Run history, formal metrics, and validated structured artifacts. |
| MLDB Study | Overview | `spec:mldb.study` | Declarative Model selection through training grids or existing Models, immutable Study Run plans, common evaluation stages, and child lineage. |
| MLDB application API | Overview | `spec:mldb.api` | Public Controller operations for validation, sealing, Study execution/status/cancellation, and entity reads. |
| MLDB orchestration | Overview | `spec:mldb.orchestration` | Controller, Queue, and Worker responsibility boundaries for queued and distributed execution. |
| MLDB verification | Index | `spec:mldb.verification` | Executable-asset pytest ownership and sealing verification rules. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.runtime` | Runtime overview and common execution responsibilities. |
| `spec:mldb.repository` | Canonical repository placement for MLDB data and executable-asset tests. |
| `spec:mldb.catalog` | Reusable Task, Corpus, and Architecture definitions. |
| `spec:mldb.training` | Reusable training procedure and concrete Training Run execution contracts. |
| `spec:mldb.model` | Learned-model identity and loading lineage over completed Training Runs. |
| `spec:mldb.evaluation` | Post-training Model evaluation and formal result contracts. |
| `spec:mldb.study` | Study Model-source definitions and immutable Study Run materialization. |
| `spec:mldb.api` | Public Controller application boundary for callers that operate MLDB. |
| `spec:mldb.orchestration` | Control-plane, operational queue, and compute-worker responsibility boundaries. |
| `spec:mldb.verification` | Executable-asset verification and sealing gate. |
