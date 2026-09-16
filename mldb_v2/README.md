# MLDB v2

MLDB v2 is the repository-owned experiment protocol used to define, execute, observe, and compare ML experiments in `mjtensu`.

The goal is not to develop MLDB for its own sake. The normal success condition is: define the experiment, run it on the real GPU backend, persist canonical results, compare them, and move on to the next ML question.

## Start here

Choose the document by what you are trying to do:

- Run an existing Study or restore the execution environment: [`docs/OPERATIONS.md`](docs/OPERATIONS.md)
- Create or change an experiment: [`docs/EXPERIMENT_AUTHORING.md`](docs/EXPERIMENT_AUTHORING.md)
- Decide which training/evaluation values should be observable: [`docs/TELEMETRY.md`](docs/TELEMETRY.md)
- Diagnose a failed plan/run/backend execution: [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)
- AI/agent operating rules for v2 work: [`AGENTS.md`](AGENTS.md)

Formal contracts remain under `records/spec/`. These docs explain how to use those contracts; they do not replace them.

## Supported user entrypoint

From the repository root, use `mldb.cmd` on Windows. It selects `.venv\Scripts\python.exe` when present and invokes `python -m mldb_v2.src.cli`.

Common commands include `mldb.cmd doctor`, `validate`, `verify`, `seal`, `plan`, `run`, `resume`, `status`, `watch`, and `logs`.

Do not execute internal implementation files directly as the normal experiment workflow.

## Authority model

Canonical experiment truth lives in repository-owned MLDB records under namespace-first `mldb_data/<namespace>/...` paths. ClearML Tasks, logs, status, and Charts are operational projections.

A ClearML reporting failure must not rewrite a valid canonical Training/Evaluation/Study result. Conversely, a green ClearML Task is not a substitute for canonical result acceptance.

## Source execution rule

Formal Study Plans pin one Git commit. Every required v2 source input must match that commit, and the commit must be reachable by the backend execution environment. Unrelated working-tree files may be dirty.

Never work around `source_not_pinned` by shipping an ad-hoc working-tree patch to the worker. Commit only the required experiment source, push it to a backend-reachable ref, then plan again.

## Current validated path

As of 2026-09-15, actual ClearML execution on worker `bugrat-gpu0` / queue `default` has completed on an NVIDIA GeForce RTX 3090 for both the rotated detector and tile classifier, including scalar Charts and canonical result acceptance.

The exact W010 closure evidence lives in `records/tasks/MLDB-V2-TASK-010-05-verify-clearml-telemetry-closure.md`. Treat that as historical evidence, not as a hard-coded execution ID for future Studies.

## Repository map

- `src/`: MLDB v2 implementation
- `skeleton/`: frozen/public contract mirrors used by conformance work
- `records/spec/`: formal contracts
- `records/tasks/` and `records/work-items/`: implementation history, not user manuals
- `docs/`: human/agent operational manuals
- `../mldb_data/<namespace>/`: reusable definitions plus canonical plans/results/models
- `../mldb_tests/`: executable-definition verification assets

For v1 historical/SSH-worker material, see `../mldb/`. Do not mix the v1 SSH execution path into v2 ClearML operation.