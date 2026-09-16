# MLDB-V2-TASK-001-04: Implement artifact, Git, and storage primitives

- **status**: completed
- **date**: 2026-09-09
- **work_item**: MLDB-V2-WORK-001
- **task_type**: implementation
- **depends_on**: [MLDB-V2-TASK-001-01]
- **outputs**: `mldb_v2/src/storage/**` and focused internal Git/source helpers required by later Work Items

## Goal
Implement reusable integrity/materialization primitives for ArtifactRef, Corpus manifests, committed source reads, and S3-compatible object bytes without leaking runtime credentials into canonical values.

## Work
- Validate ArtifactRef URI/bytes/SHA values and CorpusManifestEntry rows exactly.
- Implement manifest parsing/digest/count verification and safe relative-path handling.
- Implement internal object-byte read/write/verify operations needed later for Corpus materialization and result artifacts.
- Implement committed Git source-byte lookup/status checks required by source pinning and executable-integrity verification.
- Keep Plan compilation, sealing decisions, model loading, and backend behavior outside this Task.

## Done condition
Later verification/runtime code has one reusable byte-integrity/source-snapshot boundary; canonical values contain logical URI/hash/size only and never credentials or presigned URLs.

## Verification
Test digest/size mismatches, manifest ordering/path failures, immutable logical URI handling, committed-vs-working-tree source reads, and configured storage failure propagation without backend-specific canonical fields.

## Evidence
- 2026-09-09 integrated review: 56 focused ArtifactRef/manifest/object-byte/Git snapshot tests PASS.
- Review repaired base ArtifactRef validation so domain-specific immutable metadata extensions such as canonical weight `format` and evaluation artifact `schema` are accepted while the three required base fields remain mandatory.
- Git helpers read exact committed bytes and selected-source cleanliness only; unrelated dirty files do not fail source pinning.
- Object storage is exposed through a configured immutable byte-transport boundary; no concrete S3 client dependency or credentials are embedded in canonical values.

## Final adversarial T005 reopen — 2026-09-10
- W001-RV09 remains open: base ArtifactRef validation permits forbidden canonical extension fields such as `password`, `access_key`, `presigned_url`, and backend-local `cache_path`.
- W001-RV10 remains open: Corpus manifest JSONL parsing uses default `json.loads`, so duplicate JSON members are silently last-value-wins before entry validation.
- T004 is reopened for these two narrow canonical storage-integrity repairs. Domain-specific immutable metadata extensions must remain supported, and object-byte/Git/source behavior must not regress.
- T005/W001 remain `planned` until independent re-verification closes the findings.

## Focused RV09/RV10 repair closure — 2026-09-11
- **W001-RV09 closed by implementation**: ArtifactRef still accepts immutable descriptive extensions, including nested canonical JSON metadata, but recursively rejects explicitly forbidden credential/presigned/cache field categories. Field matching normalizes case and separators only; general descriptive names such as `key`, `path`, `url`, `metadata`, `format`, `schema`, `media_type`, `compression`, and `logical_name` remain valid.
- **W001-RV10 closed by implementation**: Corpus manifest JSONL rows use strict object-pair decoding so duplicate members are rejected before entry validation; non-finite Python JSON constants are rejected at the same boundary. Exact input bytes remain the manifest digest source and no YAML fallback exists.
- Focused ArtifactRef/Corpus unit tests: **61 passed**. Focused set including read-only W001 integrated probes: **78 passed**.
- Storage/source regression: **77 passed**. Full `mldb_v2/tests`: **341 passed**.
- Storage/source Python compile: **5 files compiled**. Dependency scan: **0 boto3/botocore/minio imports**.
- Concrete configured S3 transport remains owned by W004 T004-02; no transport implementation or canonical credential/config shape was added here.
- T005/W001 status remains unchanged pending independent re-verification.

## Coordinator storage hardening after RV09/RV10 repair ? 2026-09-11
- Adversarial review found two equivalent bypass shapes not covered by field-name-only RV09 tests: a presigned HTTP URL stored under a generic nested `url` key, and an absolute backend-local cache path stored as `metadata.cache.path`.
- ArtifactRef validation now recursively rejects recognized presigned-query URL values and local absolute path values when nested under cache context, while benign generic URLs/relative logical paths remain accepted. Existing explicit forbidden-field rejection and domain metadata extensions remain intact.
- Focused ArtifactRef/Corpus/integrated set after this coordinator hardening: **81 passed**; full `mldb_v2/tests`: **344 passed**; `git diff --check -- mldb_v2` PASS.
- No boto/botocore/minio dependency or concrete S3 transport was introduced. T004 remains `completed`; T005/W001 remain `planned` for independent verification.

## Final T005 W001-RV14 reopen — 2026-09-11
- W001-RV14 remains open: ArtifactRef recursive operational-metadata validation rejects absolute local paths under cache context but still accepts the same Windows/Unix absolute path under generic nested backend context (`metadata.backend.path`).
- The repair must cover backend/runtime/cache operational context without banning every `path` field or benign relative/documentation metadata. Existing presigned-URL, explicit credential-field, URI, manifest, object-byte, and Git/source behavior must remain unchanged.
- T004 is reopened for this narrow ArtifactRef metadata-context repair. T005/W001 remain `planned`; concrete S3 transport ownership stays with W004 T004-02.

## Focused W001-RV14 repair closure — 2026-09-11
- **W001-RV14 closed by implementation**: ArtifactRef recursive metadata validation treats exact normalized backend/runtime/cache context categories as operational and rejects local absolute paths anywhere below those contexts, including deeper nested output/path structures. Clear separator/case variants such as `backend_runtime`, `runtime_backend`, and `local_cache` normalize into the same explicit operational categories; descriptive substring names such as `backend_notes` and `runtime_format` do not.
- Existing `_looks_like_local_absolute_path()` semantics are retained for Windows drive paths, UNC paths, Unix absolute paths, and `file://` local paths. Relative values such as `runtime/output.bin` and `./relative/output.bin` remain valid even inside operational context.
- Non-overblocking evidence: generic/descriptive relative paths, benign documentation URLs, immutable `format`/`schema`/`media_type`/`logical_name` metadata, and nested descriptive absolute values outside operational context remain accepted. Existing credential-field, nested-credential, direct/generic-key presigned URL, logical S3 URI, byte-count/SHA, and domain-extension validation remains green.
- RV14-focused ArtifactRef unit plus unchanged integrated RV14 probe: **49 passed**. Requested focused ArtifactRef + read-only W001 integrated set: **69 passed**.
- Explicit storage/source regression (`test_storage_*.py` + `test_source_*.py`): **92 passed**. Full `mldb_v2/tests`: **396 passed**.
- Storage/source Python compile: **5 files compiled**. Dependency scan: **0 boto3/botocore/minio Python references**; no concrete S3 adapter indicator in storage/source.
- `git diff --check -- mldb_v2`: PASS. T005/W001 remain unchanged for independent closure verification. No commit created.
