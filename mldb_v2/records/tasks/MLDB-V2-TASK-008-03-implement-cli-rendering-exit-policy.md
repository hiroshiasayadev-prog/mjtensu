# MLDB-V2-TASK-008-03: Implement CLI rendering and exit policy

- **status**: completed
- **date**: 2026-09-13
- **work_item**: MLDB-V2-WORK-008
- **task_type**: implementation
- **depends_on**: []
- **outputs**: deterministic table/wide/json/yaml rendering, stdout/stderr and exit-code policy, focused tests

## Start gate clarification — 2026-09-13
This Task starts immediately and requires neither W007 completion nor T008-01. Test renderers with representative frozen API-shaped values and ApplicationError values.

## Exclusive ownership
- `mldb_v2/src/cli/rendering.py`
- `mldb_v2/src/cli/exit_status.py`
- `mldb_v2/tests/test_cli_rendering.py`

## Work
- Render table/wide for humans and deterministic JSON/YAML for machines without changing API semantic fields.
- JSON list remains array, single value remains object; structured stdout has no banners, ANSI, spinners, or diagnostic decoration.
- Keep deterministic ordering without parsing table headings back into data.
- Define non-zero exit behavior for ApplicationError and failing validate/verify reports while preserving stable public error code/message semantics on stderr.
- Do not parse argv, call ApplicationInterface, access repository/backend/ClearML, or invent command semantics.

## Verification
Focused rendering/golden-value tests for list/object/report/error values, determinism, stdout/stderr separation, and no decoration in structured modes. No broad regression; no commit/stage/push.

## Evidence — 2026-09-13
- Implemented presentation-only `table`, `wide`, deterministic `json`, and deterministic `yaml` rendering without Application/repository/backend access.
- Structured JSON/YAML preserve list/object shape and API field names; stdout contains no banner, ANSI, spinner, or progress decoration.
- Application errors preserve frozen `code` / `message` on stderr and return non-zero; validate/verify reports return non-zero for any invalid selected item or repository issue.
- PyYAML 6.0.3 is available in the repository `.venv`; real `safe_dump` / `safe_load` round-trip coverage verifies YAML object/list semantics and determinism.
- Focused verification only; full suite intentionally not run. No commit/stage/push.
