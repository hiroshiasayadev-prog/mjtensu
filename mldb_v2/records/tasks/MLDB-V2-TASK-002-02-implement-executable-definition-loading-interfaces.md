# MLDB-V2-TASK-002-02: Implement executable definition parsing and callable loading

- **status**: completed
- **date**: 2026-09-11
- **work_item**: MLDB-V2-WORK-002
- **task_type**: implementation
- **depends_on**: [MLDB-V2-WORK-001]
- **outputs**: executable definition/interface runtime modules, the frozen public `verification/executable_integrity.py` shape needed by their `ExecutableSource` annotations, plus focused loader tests

## Goal
Implement Architecture, Train Protocol, and Evaluation Protocol frozen runtime shapes, semantic validation, and the single same-basename companion-loading/interface boundary reused by later verification and execution.

## Work
- Implement `src/catalog/architecture.py` and `architecture_build.py`, `src/training/train_protocol.py` and `train_interface.py`, and `src/evaluation/evaluation_protocol.py` and `evaluate_interface.py` from the frozen Skeleton. Also establish the frozen public shapes in `src/verification/executable_integrity.py` so executable-definition TypedDicts can reference the exact public `ExecutableSource`; T002-03 later implements verification behavior behind those shapes.
- Validate exact schema/top-level fields, ID/path/version/lifecycle rules, Task refs, public parameter declarations, Architecture framework/entrypoint/interface/structure, and Evaluation metric/artifact declarations.
- Load only the exact same-basename canonical companion for executable definitions; do not introduce `tools/`, helper-script, sibling-scan, or alternate import conventions.
- Verify frozen callable presence/signature/interface. Do not execute Train/Evaluation behavior during definition parsing; Architecture build verification follows its explicit callable contract without model-family branches.
- Keep source hashing, asset pytest, Study compatibility, sealing mutation, runtime StageInput execution, canonical weights, and Model loading outside this Task.

## Done condition
All three executable definition kinds parse generically and expose one reusable companion/callable boundary matching the frozen interfaces; current classifier/detector/custom-evaluation definitions require no family-specific core logic.

## Verification
Cover malformed schemas/entrypoints/parameters/output declarations, companion missing/wrong basename/non-callable/signature failures, current example loading, public shape conformance, py_compile/import, and forbidden v1/Skeleton-runtime/`tools/` dependencies.

## Coordinator cross-review — 2026-09-11
- Parallel implementation is incomplete and remains `planned`.
- Present: Architecture parser/build loader, TrainProtocol parser/TrainContext boundary, shared executable helper, and a partial EvaluationProtocol module.
- Missing: complete EvaluationProtocol parser, `src/evaluation/evaluate_interface.py`, and `tests/test_executable_definition_loading.py`.
- Cross-review also requires the callable loader to resolve the canonical definition before companion loading, and final public shapes must match the frozen Skeleton rather than expose the private `_ExecutableSource` type in public TypedDict annotations.
- T002-01 independently passed coordinator review and is `completed`; do not modify its files while finishing T002-02.

## Completion evidence ? 2026-09-12
- Coordinator verified the completed continuation against the frozen Architecture/Train/Evaluation and executable-integrity shapes.
- Focused `test_executable_definition_loading.py`: **96 passed**; combined T002-01/T002-02: **161 passed**; full `mldb_v2/tests`: **557 passed**.
- All changed implementation/test modules py_compile; 7 public/runtime modules import successfully; forbidden v1/Skeleton-runtime/ClearML/boto/minio/`tools/` dependency scan is clean.
- All current classifier/rotated-fcos Architecture, TrainProtocol, and EvaluationProtocol companions load through the canonical-definition-first exact same-basename boundary; Train/Evaluation callables are not invoked by loading.
- `EvaluationProtocol`/`evaluate_interface.py` are complete, executable definitions use the frozen public `ExecutableSource`, and `src/verification/executable_integrity.py` contains public shape only; integrity behavior remains T002-03.
- `git diff --check -- mldb_v2` PASS; repository remains in the pre-existing `?? mldb_v2/` untracked-tree state. No commit created.

## Coordinator integration repair — 2026-09-12
- Cross-task smoke found that importing canonical executable companions created `__pycache__/` under `mldb_data/<namespace>/<domain>/`, which violates W001 canonical listing/layout by introducing nested domain directories.
- The shared companion loader now suppresses bytecode-cache writes only while executing the exact companion import and restores the prior interpreter setting afterward. Generated cache directories were removed; no canonical YAML or companion source was changed.
- Added a regression proving companion loading does not create `__pycache__` in the canonical domain. Fresh T002-02 focused: **97 passed**; full `mldb_v2/tests`: **633 passed, 1 skipped**.
- Fresh actual companion smoke loaded all 10 current executables, left **0** canonical-domain `__pycache__` directories, and W001 actual listing returned **18 items / 0 issues**. `git diff --check -- mldb_v2` PASS. Task remains `completed`.
