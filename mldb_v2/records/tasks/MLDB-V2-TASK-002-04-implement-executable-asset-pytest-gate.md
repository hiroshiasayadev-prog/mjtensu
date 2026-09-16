# MLDB-V2-TASK-002-04: Implement executable asset pytest gate

- **status**: completed
- **date**: 2026-09-11
- **work_item**: MLDB-V2-WORK-002
- **task_type**: implementation
- **depends_on**: [MLDB-V2-TASK-002-02, MLDB-V2-TASK-002-03]
- **outputs**: `mldb_v2/src/verification/executable_asset_tests.py`, focused tests

## Goal
Implement the frozen derived asset-test gate for Architecture, Train Protocol, and Evaluation Protocol definitions.

## Work
- Mirror `ExecutableAssetTestRequest`, `PytestRunResult`, `PytestRunner`, result shape, and verifier public boundary.
- Derive the canonical test directory only from kind/id under `mldb_tests/<namespace>/<domain>/<local-id>/`; definitions never provide arbitrary test paths.
- Reject missing directory, zero collected tests, any failed/error test, runner failure, and inability to establish prerequisite executable integrity; at least one test must pass.
- Provide the concrete private/default pytest runner needed by entity verification while preserving the injected `PytestRunner` protocol for focused tests.
- Asset tests must resolve/load the target through the T002-02 typed loader boundary; do not create a second direct-import convention.
- Keep Corpus builder tests optional, normal sealed-asset execution free of pytest, and all backend/runtime execution outside this Task.

## Done condition
Executable reusable definitions have one deterministic pytest verification gate with derived locations, explicit no-test/failure behavior, and no YAML-stored test path.

## Verification
Use fake runner and bounded real pytest fixtures for missing/zero/pass/fail/error cases; verify exact derived paths, no direct companion import bypass, public shape conformance, py_compile/import, and `git diff --check`.
## Completion evidence - 2026-09-12
- Mirrored the frozen public `verification/executable_asset_tests.py` shape exactly under `src`; no Skeleton runtime import or public extension was added.
- Added private `_RepositoryExecutableAssetTestVerifier` with exact namespace-first path derivation, traversal/symlink containment checks, T002-03 integrity prerequisite short-circuiting, strict `PytestRunResult` validation, and deterministic bounded diagnostics.
- Added private `_SubprocessPytestRunner` / `_default_pytest_runner`; pytest runs only the exact derived directory in a fresh subprocess and reports counts through an internal pytest hook/JSON result rather than parsing CLI summary text.
- Gate semantics reject missing/non-directory paths, zero collected, no passed tests (including skipped-only), any failures/errors, runner exceptions, malformed/impossible counts, and integrity failure.
- Focused `test_executable_asset_tests.py`: **32 passed, 1 skipped**; the skip is the portable Windows symlink-escape test when directory symlink creation is unavailable.
- W002 T002-01/02/03/04 regression: **236 passed, 1 skipped**. Full `mldb_v2/tests`: **632 passed, 1 skipped**.
- Actual `mldb_tests/` smoke: all 10 current Architecture/TrainProtocol/EvaluationProtocol definitions pass executable integrity but reject at the asset gate with `asset_test_directory_missing`; the repository currently has legacy flat `mldb_tests/<domain>/...` directories, not the required `mldb_tests/<namespace>/<domain>/...`. No canonical test assets were invented or modified.
- Changed Python py_compile/import smoke PASS; public-shape mirror test PASS; fake-runner and bounded real-runner pass/fail/collection-error/zero/skipped-only coverage PASS.
- Forbidden dependency/direct-companion-import scan CLEAN; no mldb v1, `mldb_v2.skeleton`, ClearML, boto/minio, `tools` implementation dependency, or new `importlib.spec_from_file_location` convention. `git diff --check -- mldb_v2` PASS; no commit or Git mutation performed.
