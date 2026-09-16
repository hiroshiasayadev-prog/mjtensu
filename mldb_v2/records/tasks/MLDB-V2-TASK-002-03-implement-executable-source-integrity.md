# MLDB-V2-TASK-002-03: Implement executable and builder source integrity

- **status**: completed
- **date**: 2026-09-11
- **work_item**: MLDB-V2-WORK-002
- **task_type**: implementation
- **depends_on**: [MLDB-V2-TASK-002-01, MLDB-V2-TASK-002-02]
- **outputs**: verification behavior/private helpers added behind the frozen `mldb_v2/src/verification/executable_integrity.py` public shape established by T002-02, plus focused tests

## Goal
Implement the frozen executable-definition and Corpus-builder integrity verification boundary using W001 Git/source and repository primitives.

## Work
- Reuse the already-established frozen `ExecutableSource`, request/result, and `ExecutableIntegrityVerifier` public shapes without changing them or importing Skeleton at runtime; this Task adds the concrete verification behavior/private helpers behind that boundary.
- Hash the exact same-basename executable companion and compare sealed `implementation.sha256`; verify Corpus builder sibling/hash when present.
- Validate declared project-source entries as sorted unique repository-relative regular files with exact lowercase SHA-256; reject absolute paths, directories, globs, `tools/`, unsafe traversal, and hash mismatch.
- Verify declared project-owned result-affecting source bytes through the selected repository working/source boundary without mutating Git.
- Provide only private digest/evidence helpers needed later by sealing to populate derived sealed hashes; public verification remains read-only `valid + diagnostics`.
- Keep source-import graph discovery, environment fingerprinting, pytest execution, Study planning, object materialization, and sealing writes outside this Task.

## Done condition
Executable/Corpus-builder integrity can deterministically prove or reject the exact sibling and declared source set required by the frozen contracts, with no backend or model-family assumptions.

## Verification
Cover sibling/source hash match/mismatch, missing/unowned/unsafe/unsorted/duplicate source entries, optional Corpus builder cases, no Git mutation, public shape conformance, py_compile/import, and dependency scans.

## Completion evidence — 2026-09-12
- Implemented private `_RepositoryExecutableIntegrityVerifier` in `mldb_v2/src/verification/_executable_integrity.py`; frozen public `verification/executable_integrity.py` remains unchanged.
- Executable definitions verify exact same-basename companion regular files and exact-byte SHA-256. Draft definitions may omit recorded hashes; present draft hashes and all sealed hashes are compared without mutating canonical YAML.
- Declared project sources are verified only for their exact repository-relative files; missing/non-regular/unreadable/hash-mismatch cases produce deterministic diagnostics. No import-graph discovery, whole-repository fingerprinting, or Git mutation was added.
- Corpus builder verification enforces the exact same-basename ownership rule, verifies builder bytes without executing them, and exposes private sealing evidence using the same digest primitive as verification.
- Current Architecture 5 / Train Protocol 2 / Evaluation Protocol 3 draft examples verify `valid=True`; the two current Corpus examples without builders also verify `valid=True`.
- Focused `test_executable_integrity.py`: **43 passed**. T002-01/02/03 combined reported **204 passed**. Coordinator fresh full `mldb_v2/tests`: **600 passed**.
- Coordinator fresh `git diff --check -- mldb_v2`: PASS; `git status --short -- mldb_v2` remains the pre-existing `?? mldb_v2/` state. No commit or Git mutation was performed.
- py_compile/import, Frozen public-shape mirror, and forbidden dependency scans were reported PASS/CLEAN; no blocker remains in T002-03 scope.
