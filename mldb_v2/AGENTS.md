# MLDB v2 Agent Operating Notes

This file is the short entrypoint for AI/agent work on MLDB v2. It does not duplicate the full manuals.

## Before doing v2 work

1. Read repository-root `AGENTS.md`.
2. Read `../mldb/AGENTS.md` for the top-level principle that MLDB exists to run ML, not to become the project goal.
3. Read `README.md` and the relevant document under `docs/`.
4. Check actual filesystem and Git state; do not use old chat history as the source of truth.

## Normal behavior

- Prefer the public `mldb` CLI / Application boundaries over internal helpers.
- Reuse existing Task, Corpus, Architecture, Train Protocol, Evaluation Protocol, and Study definitions when their semantics already fit.
- Use a Study parameter matrix for value sweeps instead of cloning Protocols for every value.
- Treat sealed reusable definitions as immutable. Semantic changes require a new version ID.
- For Train/Evaluation Protocol changes, explicitly review telemetry semantics using `docs/TELEMETRY.md`.
- Execute real experiments when the user asked for an experiment; do not stop after planning unless asked to.

## Core-change rule

Do not change `mldb_v2/src/` merely to improve abstractions, style, scheduling, or future flexibility. Change core only when a concrete requested experiment cannot run through the existing surface and the blocker is reproduced.

## Git/source-pinning safety

- `source_not_pinned` is a protection boundary, not an inconvenience to bypass.
- Formal execution uses the current planned Git commit or an equivalent verified snapshot; uncommitted patches are not backend source.
- Unrelated working-tree dirt is allowed. Stage only the exact experiment/source closure required for the Study.
- Never use `git add .` / `git add -A` in a dirty repository without explicit user authorization and a reviewed allowlist.
- Never rewrite canonical result files to repair an execution.

## Authority

- Canonical MLDB results/models are experiment truth.
- ClearML status, logs, parameters, and telemetry are operational projections.
- S3/object bytes are formal artifacts referenced and validated by canonical records.
- Telemetry delivery failure after a valid event is accepted must not change canonical stage success.

## Secrets

Never write credentials, API keys, MinIO passwords, or session tokens into Markdown, definitions, task records, source files, or chat output. Documentation may name environment variables and non-secret endpoints only.

## Manuals

- `docs/OPERATIONS.md`: environment, ClearML/S3/GPU, commands, source pinning, normal execution
- `docs/EXPERIMENT_AUTHORING.md`: how to decide which entity to create/change and how to author/verify/seal it
- `docs/TELEMETRY.md`: how to choose useful Protocol-specific observations
- `docs/TROUBLESHOOTING.md`: symptom-first recovery runbook

When a manual disagrees with executable code or a formal spec, stop and reconcile the documentation; do not silently invent behavior.