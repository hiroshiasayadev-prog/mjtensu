# MLDB-V2-TASK-008-01: Implement CLI public request mirrors and parser normalization

- **status**: completed
- **date**: 2026-09-13
- **work_item**: MLDB-V2-WORK-008
- **task_type**: implementation
- **depends_on**: [MLDB-V2-WORK-001]
- **outputs**: exact `src/cli` public value/request mirrors, argv parser/normalizer, focused parser tests

## Start gate clarification — 2026-09-13
This Task starts immediately. W007 completion and concrete Application implementation are not required.

## Exclusive ownership
- `mldb_v2/src/cli/types.py`
- `mldb_v2/src/cli/requests.py`
- `mldb_v2/src/cli/parser.py`
- `mldb_v2/tests/test_cli_parser.py`

## Work
- Mirror frozen `cli/types.py` and `cli/requests.py` exactly under `src`, replacing only skeleton imports with src imports.
- Parse every frozen command/resource/kind spelling and normalize argv into the exact `CliCommandRequest` shapes.
- Normalize `--json` to `presentation.format=json`, positive limits, shared selectors, `--failed`, `-f`, backend options, and explicit `--all`.
- Reject unsupported selector/resource combinations, ambiguous positional inputs, invalid kind/ID forms, and implicit multi-target mutation. Never infer kind via filesystem/path scanning.
- Do not import repository/backend/ClearML or implement Application semantics/rendering.

## Verification
Focused parser/value-shape tests including exact Skeleton mirror checks. No broad regression; no commit/stage/push.

## Completion evidence - 2026-09-13
- Added exact runtime mirrors `src/cli/types.py` and `src/cli/requests.py`; only `mldb_v2.skeleton.*` imports are replaced by `mldb_v2.src.*`.
- Added `src/cli/parser.py` covering all frozen commands/resources/kinds and normalized selectors/output/backend/log options without repository/backend/ClearML access.
- Explicit bulk sealing requires `--all --namespace`; exact sealing requires explicit kind + typed ID. Aggregate `definitions` exact lookup is rejected because no kind may be inferred from an ID.
- Focused verification: `.\.venv\Scripts\python.exe -m pytest mldb_v2\tests\test_cli_parser.py -q` -> `36 passed in 0.31s`.
- `py_compile` passed for the three CLI runtime files and the focused test. No broad `mldb_v2/tests` run was performed.
