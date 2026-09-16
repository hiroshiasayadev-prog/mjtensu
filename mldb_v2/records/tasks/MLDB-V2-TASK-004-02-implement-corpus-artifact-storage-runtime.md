# MLDB-V2-TASK-004-02: Implement Corpus/artifact storage runtime

- **status**: completed
- **date**: 2026-09-12
- **work_item**: MLDB-V2-WORK-004
- **task_type**: implementation
- **depends_on**: [MLDB-V2-WORK-001, MLDB-V2-WORK-002]
- **outputs**: Corpus/artifact materialization and configured S3-compatible object-byte transport runtime, focused tests

## Goal
Implement sealed Corpus materialization, immutable formal-artifact publication/integrity helpers, and the production configured S3-compatible `_ObjectByteTransport` required by the execution harness.

## Work
- Reuse W001 ArtifactRef/Corpus manifest/object-byte boundaries and W002 sealed Corpus verification; do not redefine canonical storage values.
- Materialize exactly the sealed Corpus manifest object set into an execution-local root with verified bytes and no builder execution.
- Publish candidate artifact bytes immutably, returning exact URI/bytes/SHA-256 metadata and preserving domain-added immutable format/schema metadata at higher boundaries.
- Provide at least one production configured S3-compatible transport implementing W001 read/publish semantics; endpoint/credentials remain operational configuration and never canonical.
- Immutable publish must fail rather than replace different existing bytes; fake/in-memory transport is test-only and does not satisfy Task completion.
- Keep canonical weights/Model loading, Train/Evaluation invocation, StageInput orchestration, formal Result persistence, and backend scheduling outside this Task.

## Done condition
The runtime can materialize verified sealed Corpus bytes and publish/read immutable artifact bytes through both focused fake transport and a production configured S3-compatible transport without leaking credentials/config into canonical values.

## Verification
Cover manifest-driven materialization, missing/hash/size failures, immutable publication/idempotent-same-byte behavior, overwrite rejection, path safety, production transport configuration boundary, no builder execution, py_compile/import, dependency scan, and full regression.


## Implementation evidence / closure gate  E2026-09-13
- Corpus materialization, candidate artifact publication, and lazy boto3-style production S3 transport code are implemented with focused verification complete.
- T004-02 focused tests reported **18 passed, 1 skipped**; W001 dependent smoke **146 passed**; changed-module py_compile/import and genericity/dependency scans PASS.
- Coordinator Phase-A join reran T004-01 plus T004-02 focused tests in the shared working tree: **40 passed, 1 skipped**.
- The remaining blocker is operational only: the repository has no dependency declaration mechanism and current `.venv` has no boto3, so real production dependency activation/client-configuration smoke has not run.
- This operational activation is a **W004 closure gate**, not a code dependency for T004-03/T004-04 implementation. The storage runtime surface is frozen for read-only consumption by Phase B.
- Task status remains non-completed until production boto3 activation/configuration smoke is satisfied. No dependency file or environment package was silently added.
## Closure gate recheck  E2026-09-13
- Coordinator rechecked the active `.venv`: `boto3` is still absent.
- No AWS/S3/MinIO endpoint, profile, region, or credential environment is currently configured; only presence/absence was inspected, no secret values were read or recorded.
- Repository search found no usable real MinIO/S3 runtime configuration outside mocked test configuration.
- Therefore the production transport code remains verified only through focused injected-client tests; the required real activation/configuration smoke cannot yet run.
- T004-02 remains non-completed for this single operational reason. No silent `pip install` or new dependency-management file was introduced.


## Production S3 activation evidence — 2026-09-13
- Production endpoint: `https://mldb-s3.thebugrat.dev`, backed by MinIO Community Edition (`GNU AGPLv3`) running on the server objectstore.
- Production activation used an ephemeral local virtual environment with `boto3`; no repository dependency file or project `.venv` was modified.
- Actual `mldb_v2.src.storage.s3_transport.S3ObjectByteTransport` / `_create_s3_transport` path was exercised against the live endpoint using operational credentials from ignored local `.env` configuration.
- Live smoke passed: bucket creation, immutable PUT, verified GET, idempotent same-byte replay, rejection of different-byte overwrite, and cleanup.
- Smoke result: **`W004_S3_SMOKE=PASS`**.
- The former AIStor license blocker is removed; T004-02 completion condition is satisfied.
