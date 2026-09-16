# MLDB-V2-TASK-001-01: Implement common runtime values

- **status**: completed
- **date**: 2026-09-09
- **work_item**: MLDB-V2-WORK-001
- **task_type**: implementation
- **depends_on**: []
- **outputs**: `mldb_v2/src/common/**`, focused tests

## Goal
Implement the frozen common identity, public-parameter, and diagnostic runtime behavior needed by later repository/storage/domain code.

## Work
- Mirror the public shapes/signatures in `mldb_v2/skeleton/common/` without changing them.
- Implement only identity grammar that is actually fixed by common Specifications: Namespace/local kebab-case segments, typed `<namespace>/<local-id>` references, `trial-NNNN`, and `eval-NNNN`; kind-specific version/result/execution-key rules remain with their owning format contracts.
- Implement recursive JSON-compatible public-parameter validation, finite-number/type-sensitive equality, declaration/default/constraint resolution, and seed integer validation where common contracts require it.
- Implement the common canonical JSON digest-form encoding helper from the identity Specification; entity-specific field exclusion/ID construction stays with the owning format.
- Implement Diagnostic validation including lowercase-snake-case code and 4096-byte UTF-8 message bound.
- Keep YAML/file persistence, domain, backend, and filesystem behavior outside this Task.

## Done condition
Common values can be validated/resolved deterministically with no classifier/detector/backend assumptions and with tests covering bool-vs-int/float distinctions, invalid numeric values, malformed IDs, arrays/maps, defaults, and diagnostics.

## Verification
Run focused tests plus syntax/type/import checks; confirm no `mldb.src`, ClearML, repository scanning, or `tools/` dependency is introduced.

## Evidence
- 2026-09-09 integrated review: `84 passed` across common identity/parameter/diagnostic tests.
- `mldb_v2/src/common/*.py` compiles successfully.
- No `mldb.src`, ClearML, torch, filesystem, subprocess, boto, or S3 dependency is present in common runtime code.
- `git diff --check` passes for the Task scope.
