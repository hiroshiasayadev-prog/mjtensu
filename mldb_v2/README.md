# MLDB v2

MLDB v2 is the repository-owned experiment protocol used to define, execute, observe, and compare ML experiments in `mjtensu`.

The goal is not to develop MLDB for its own sake. The normal success condition is: define the experiment, run it on the real GPU backend, persist canonical results, compare them, and move on to the next ML question.

## Start here

Choose the document by what you are trying to do:

- Run an existing Study or restore the execution environment: [`docs/OPERATIONS.md`](docs/OPERATIONS.md)
- Create or change an experiment: [`docs/EXPERIMENT_AUTHORING.md`](docs/EXPERIMENT_AUTHORING.md)
- Decide which training/evaluation values should be observable: [`docs/TELEMETRY.md`](docs/TELEMETRY.md)
- Understand the approved StudyResult -> ClearML Pipeline execution mapping: [`docs/CLEARML_PIPELINES.md`](docs/CLEARML_PIPELINES.md)
- Understand the global Python runtime registry and worker rollout: [`docs/RUNTIME_REGISTRY.md`](docs/RUNTIME_REGISTRY.md)
- Diagnose a failed plan/run/backend execution: [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)
- AI/agent operating rules for v2 work: [`AGENTS.md`](AGENTS.md)

Formal contracts remain under `records/spec/`. These docs explain how to use those contracts; they do not replace them.

## Supported user entrypoint

From the repository root, use `mldb.cmd` on Windows. It selects `.venv\Scripts\python.exe` when present and invokes `python -m mldb_v2.src.cli`.

Common commands include `mldb.cmd doctor`, `validate`, `verify`, `seal`, `plan`, `run`, `resume`, `retry-stage`, `rerun`, `status`, `watch`, and `logs`.

Do not execute internal implementation files directly as the normal experiment workflow.

## Authority model

Canonical experiment truth lives in repository-owned MLDB records under namespace-first `mldb_data/<namespace>/...` paths. ClearML Pipeline Runs, child Tasks, logs, status, Charts, and summary projections are operational state/UI.

The approved W011 mapping is one MLDB Study Result -> one ClearML Pipeline Run, with planned training/evaluation work represented as child Pipeline Tasks. ClearML owns physical scheduling/retry/liveness; MLDB retains semantic gates and canonical result acceptance.

A ClearML reporting failure must not rewrite a valid canonical Training/Evaluation/Study result. Conversely, a green ClearML Pipeline/Task is not a substitute for canonical result acceptance.

The W011 Pipeline mapping is implemented. New Study Results also pin one immutable global runtime-registry version at run creation. That version is propagated to every stage and retained across retry/resume/rerun; package versions are not authored into Study, Architecture, Train Protocol, or Evaluation Protocol definitions. See `docs/CLEARML_PIPELINES.md` and `docs/RUNTIME_REGISTRY.md`.

## Source execution rule

Formal Study Plans pin one Git commit. Every required v2 source input must match that commit, and the commit must be reachable by the backend execution environment. Unrelated working-tree files may be dirty.

Source pinning and runtime pinning are separate. The Plan pins repository source; the Study Result pins `runtime_registry_version`. A fresh run resolves registry `latest` exactly once when the Study Result is created. Individual stages must never resolve `latest` again.

Never work around `source_not_pinned` by shipping an ad-hoc working-tree patch to the worker. Commit only the required experiment source, push it to a backend-reachable ref, then plan again.

## Current validated path

As of 2026-10-02, the global runtime registry is live at v2 with 67 direct pins / 86 resolved packages. v2 extends the initial worker baseline with `opencv-python-headless==4.11.0.86` so the recognition functional-video render runtime is represented by the global Python snapshot.

Registry enforcement is active on both `precision5820-gpu3060` and `old-gpu3090`. The old RTX 3090 worker keeps its `default` and `recognition-functional` queues and its Chrome/ffmpeg render image while re-executing MLDB through the same reusable managed-venv mechanism. `old-cpu`, `old-iphone`, and `dev-wsl-gpu3060` are not yet migrated. Earlier RTX 3090 telemetry/result-acceptance evidence remains valid historical evidence in `records/tasks/MLDB-V2-TASK-010-05-verify-clearml-telemetry-closure.md`.

## Repository map

- `src/`: MLDB v2 implementation
- `skeleton/`: frozen/public contract mirrors used by conformance work
- `records/spec/`: formal contracts
- `records/tasks/` and `records/work-items/`: implementation history, not user manuals
- `docs/`: human/agent operational manuals
- `../mldb_data/<namespace>/`: reusable definitions plus canonical plans/results/models
- `../mldb_tests/`: executable-definition verification assets

For v1 historical/SSH-worker material, see `../mldb/`. Do not mix the v1 SSH execution path into v2 ClearML operation.