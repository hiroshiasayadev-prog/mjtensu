# MLDB-V2-TASK-002-07: Repair executable source declaration completeness

- **status**: completed
- **date**: 2026-09-12
- **work_item**: MLDB-V2-WORK-002
- **task_type**: focused implementation repair
- **depends_on**: [MLDB-V2-TASK-002-03, MLDB-V2-TASK-002-06]
- **outputs**: executable-integrity completeness verification and focused regression tests

## Goal
Close W002 closure finding `W002-CL01`: an executable definition that imports result-affecting project-owned source without declaring it under `implementation.sources` is non-conforming, but the current verifier accepts such a definition as valid.

## Work
- Enforce the frozen `spec:mldb.v2.verification.executable_integrity` source-declaration completeness rule without changing public Skeleton shapes or canonical schemas.
- Reuse the existing T002-03 sibling/declared-source hash verifier and keep sealing evidence semantics unchanged.
- At minimum, statically resolvable project-owned imports reachable from the executable companion must not be silently omitted from `implementation.sources`; third-party, stdlib, and non-behavioral MLDB infrastructure remain excluded by the frozen contract.
- Do not invent a project-wide fingerprint, environment fingerprint, `tools/` executable boundary, or W003 source-commit behavior.
- If the frozen distinction between result-affecting and non-result-affecting project source cannot be implemented soundly without an additional contract decision, report that exact blocker instead of weakening verification or silently over-restricting all imports.

## Done condition
The focused closure probe with a same-basename executable companion importing an undeclared project-owned behavior module is rejected, declared exact source/hash passes, existing current examples remain valid, and W002 full regression remains green.

## Verification
Cover direct and relevant transitive project-owned imports, declared/missing/mismatched sources, stdlib/third-party/MLDB-infrastructure exclusions, no canonical mutation, no Git mutation, public-shape conformance, and full W002 regression.

## Completion evidence — 2026-09-12
- Added private static import-graph inspection in `mldb_v2/src/verification/_source_imports.py` and integrated it behind the existing T002-03 verifier; public `executable_integrity.py`, Skeleton, and canonical schemas remain unchanged.
- Direct and transitive statically-resolvable repository-owned runtime imports must be present in `implementation.sources`; missing declarations now produce `source_declaration_missing`. Repository-owned `tools/` imports are rejected as `source_import_forbidden`.
- Standard-library / installed third-party imports are ignored because they do not resolve to repository source files. `mldb_v2/src/**` infrastructure imports are explicitly excluded, and `TYPE_CHECKING` / constant-false import branches are not treated as runtime behavior sources.
- Package `__init__.py`, namespace-package `from ... import submodule`, absolute imports, and relative transitive imports are covered without importing the inspected modules. Declared-source hashing and sealing evidence remain the existing T002-03 implementation.
- Focused executable-integrity regression: **51 passed**. W002 T002-01..07 regression: **342 passed, 1 skipped**. Full `mldb_v2/tests`: **738 passed, 1 skipped**.
- Current canonical executable examples remain **10/10 valid** with no declared project sources required; actual canonical listing remains **18 items / 0 issues** and canonical-domain `__pycache__` remains **0**.
- Changed Python py_compile/import PASS; frozen public executable-integrity shape comparison PASS; forbidden dependency scan CLEAN; `git diff --check -- mldb_v2` PASS. No Git/canonical mutation or commit was performed.
