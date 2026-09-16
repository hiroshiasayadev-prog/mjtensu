# MLDB-V2-TASK-002-05: Implement Study validation and entity verification composition

- **status**: completed
- **date**: 2026-09-11
- **work_item**: MLDB-V2-WORK-002
- **task_type**: implementation
- **depends_on**: [MLDB-V2-TASK-002-01, MLDB-V2-TASK-002-02, MLDB-V2-TASK-002-03, MLDB-V2-TASK-002-04]
- **outputs**: `mldb_v2/src/study/study.py`, `mldb_v2/src/verification/definition_lifecycle.py` validation/verification composition, focused tests

## Goal
Implement frozen Study definition validation and compose entity-level `validate` / `verify` behavior for every reusable definition without compiling a StudyPlan.

## Work
- Mirror the frozen Study runtime shape and validate exact model-source/evaluation-stage structure, versioned identity, stage IDs, axes, seed types/uniqueness, and type-sensitive duplicate values.
- Resolve referenced Task/Corpus/Architecture/TrainProtocol/EvaluationProtocol definitions through W001/W002 boundaries; enforce sealed status and same-Task compatibility where the Study seal gate requires it.
- For existing-Model Study sources, read Model -> completed TrainingResult lineage only to the extent required by `study-format` to establish canonical lineage and Task compatibility; do not own result acceptance or generated-record persistence.
- Compose `DefinitionValidator` from T002-01/T002-02/Study validators and `DefinitionVerifier` from validation plus T002-03 integrity, T002-04 asset tests, and Corpus manifest/object-byte verification required by `definition_lifecycle`.
- Corpus verification uses injected W001 object-byte access/transport boundaries; this Task does not implement the concrete S3 adapter owned by W004 T004-02.
- Keep grid expansion, source-commit pin collection, Plan digest/creation, companion Train/Evaluate execution, backend access, and sealing mutation outside this Task.

## Done condition
`validate` remains schema/semantic validation, `verify` adds the frozen external integrity/reference/test gates, and Study verification establishes complete seal-time planning compatibility without producing trials, coordinates, or a Plan.

## Verification
Cover all definition kinds, cross-namespace refs, sealed/unsealed/incompatible refs, Study train/existing-model variants, parameter-axis/default constraints, Corpus manifest/object mismatch, executable verifier composition, public shape conformance, and no W003 planning behavior.
## Completion evidence - 2026-09-12
- Coordinator completed the interrupted final verification after the manifest-gate ordering repair; no additional implementation change was required.
- Focused Study validation: **38 passed**; focused definition lifecycle: **31 passed**. W002 T002-01..05 regression: **306 passed, 1 skipped**; full `mldb_v2/tests`: **702 passed, 1 skipped**.
- Synthetic training-Study compatibility, existing-Model lineage, and draft/sealed Corpus manifest/object-byte verification all pass in the focused lifecycle suite; public Study and definition-lifecycle shapes match the frozen Skeleton.
- Current canonical Studies both validate locally and intentionally fail full verification with `referenced_definition_not_sealed` because referenced reusable definitions remain draft.
- Changed src/tests py_compile PASS (6 files); public/runtime import smoke PASS (4 modules). Forbidden dependency/downstream-ownership scan is clean: no mldb v1, Skeleton runtime import, ClearML, boto/minio, `tools/`, StudyPlan/grid expansion, or canonical weight/model runtime implementation.
- Actual canonical listing remains **18 items / 0 issues** before/after verification; canonical-domain `__pycache__` remains **0**, confirming verification does not pollute `mldb_data`.
- `git diff --check -- mldb_v2` PASS. Existing tracked-flat deletion + namespace-first untracked migration state is unchanged; no commit or Git mutation was performed.
