# MLDB-V2-TASK-002-01: Implement core Catalog definition validation

- **status**: completed
- **date**: 2026-09-11
- **work_item**: MLDB-V2-WORK-002
- **task_type**: implementation
- **depends_on**: [MLDB-V2-WORK-001]
- **outputs**: `mldb_v2/src/catalog/namespace.py`, `task.py`, `corpus.py`, focused tests

## Goal
Implement frozen Namespace, Task, and Corpus runtime shapes plus typed parsing/semantic validation using W001 exact repository resolution and common value validators.

## Work
- Mirror the frozen Catalog Skeleton shapes without runtime imports from `mldb_v2.skeleton`.
- Validate exact schemas/required field types, ID/path/version rules, lifecycle values, typed Task references, and JSON-compatible open metadata; reject unknown top-level fields only where the owning frozen format explicitly requires it, and enforce Namespace backend-metadata exclusion without inventing a broader Namespace key policy.
- Enforce Task categorical-label ordering/uniqueness semantics without inventing a universal target vocabulary.
- Enforce Corpus storage/manifest filename, representation/split, optional-builder, and draft-vs-sealed structural rules from the Corpus format.
- Keep manifest byte verification, builder hashing, object-store reads, executable loading, Study validation, sealing mutation, and backend behavior outside this Task.

## Done condition
Namespace/Task/Corpus canonical documents resolve into frozen runtime values or fail with deterministic validation errors; domain-specific open metadata is preserved rather than normalized or specialized.

## Verification
Add focused valid/invalid fixtures including current classifier/rotated-detector examples; verify format-specific unknown top-level rejection where specified, version/reference/lifecycle boundaries, no model-family branches, no v1/Skeleton runtime imports, py_compile/import, and `git diff --check`.

## Evidence
- 2026-09-11 coordinator cross-review: focused `test_catalog_core_definitions.py` **65 passed**.
- Full `mldb_v2/tests` regression with T002-01 implementation present: **461 passed**.
- `namespace.py`, `task.py`, `corpus.py`, and `_core_definition_validation.py` py_compile/import smoke: PASS.
- Current tile-classifier and rotated-fcos Namespace/Task/Corpus examples validate through exact W001 resolution.
- No `mldb_v2.skeleton`, mldb-v1 runtime, ClearML, boto/minio, `tools/`, or model-family-specific dependency in T002-01 src scope.
- Parallel T002-02 scope was not modified during coordinator closure.
