# MLDB-V2-TASK-001-05: Verify foundation and repository runtime

- **status**: completed
- **date**: 2026-09-09
- **work_item**: MLDB-V2-WORK-001
- **task_type**: verification
- **depends_on**: [MLDB-V2-TASK-001-02, MLDB-V2-TASK-001-03, MLDB-V2-TASK-001-04]
- **outputs**: focused conformance evidence/tests for W001

## Goal
Verify the integrated W001 implementation against common/storage/repository Specifications and frozen Skeleton before downstream Work Items depend on it.

## Work
- Run focused unit/integration tests across common values, exact resolution, listing, writes, coordination, Git/source access, and ArtifactRef/manifest integrity.
- Check public src shapes against Skeleton and scan for forbidden v1/ClearML/queue/lease/heartbeat dependencies.
- Exercise namespace-first `mldb_data/` examples plus malformed/legacy-flat fixtures.
- Verify repository writes never auto-commit Git and lock scope remains short/canonical-only.
- Record exact failures rather than repairing unrelated downstream code inside this verification Task.

## Done condition
W001 receives PASS with no unresolved common/repository/storage contract mismatch, or an explicit NEEDS REVISION list tied to the owning T01-T04 implementation.

## Verification
Run full W001 test set, syntax/type/import checks, `git diff --check`, and report exact changed files/test counts/results.

## Review focus added by integration coordinator
- Verify that repository YAML handling does not silently reinterpret unsupported/tagged/aliased YAML as ordinary strings. Current private loader is dependency-free and must be judged against the frozen YAML/public-parameter contracts rather than assumed conformant because current examples parse.
- Recheck actual UUID4 execution-key enforcement and StudyResult terminal-closure/status consistency repaired during T003 integration review.
- Recheck base ArtifactRef acceptance of domain-specific immutable metadata extensions repaired during T004 integration review.
- Decide explicitly whether W001 is complete with the configured immutable object-byte transport port and no concrete S3 client, or whether a named follow-up implementation Task is required before downstream runtime work.

## Evidence

### Verdict: NEEDS REVISION

Independent review was performed against the frozen common/repository/storage/source/result contracts and Skeleton. No `src/` implementation was repaired in this verification Task. A verification-only test file was added at `mldb_v2/tests/test_w001_integrated_verification.py` to preserve the blocking probes below.

#### Finding W001-V01 — CRITICAL — owner T002
- **violated Spec**: `spec:mldb.v2.common.public_parameters`; `spec:mldb.v2.repository.resolution`; canonical files are YAML and tagged public-parameter values are invalid.
- **exact file/function**: `mldb_v2/src/repository/_yaml.py` — `_load_yaml`, `_parse_plain_scalar`, `_parse_sequence`.
- **finding**: the dependency-free private YAML subset silently reinterprets YAML syntax as different ordinary values. `value: !!str 1` becomes the string `"!!str 1"` instead of being rejected; anchor/alias syntax such as `base: &base value` / `copy: *base` becomes literal strings instead of alias semantics or a clear rejection; a valid plain scalar such as `- s3://bucket/path/to/object` is interpreted as a mapping because `_parse_sequence` treats any colon-containing item as an inline mapping.
- **why it matters**: exact resolution/listing can accept canonical YAML with semantics different from the authored YAML, and tagged public-parameter values lose the information required to reject them. Current JSON-form examples parsing successfully is therefore insufficient evidence.
- **required correction**: T002 must either use a YAML parser/configuration that implements the frozen YAML semantics while rejecting tagged public-parameter values, or explicitly reject unsupported YAML syntax before it can be reinterpreted. Do not preserve the current silent subset semantics as a new undocumented format decision.

#### Finding W001-V02 — HIGH — owner T003
- **violated Spec**: `spec:mldb.v2.results.study_result_format` — `submitted`/`cancelling` are non-terminal states; all-completed closes `completed`; a `cancelling` Study closes `cancelled` once every planned stage is terminal; otherwise terminal non-completed closure is `completed_with_failures` unless global failure closes `failed`.
- **exact file/function**: `mldb_v2/src/repository/canonical_writes.py` — `_validate_study_result_record` and its use from `replace_nonterminal_study_result`.
- **finding**: the validator checks only that terminal top-level statuses contain no `pending` slot. It does not enforce the converse. A StudyResult with every stage terminal is accepted with `status: submitted`, and a fully terminal cancelling StudyResult is accepted with `status: cancelling`.
- **why it matters**: canonical history can persist a logically closed Study as non-terminal, so later progression/query code cannot rely on the frozen status/closure invariant.
- **required correction**: T003 must reject non-terminal top-level status when no planned stage remains `pending`, and enforce the frozen closure/status mapping including cancellation precedence. Terminal-slot immutability must remain unchanged.

#### Finding W001-V03 — HIGH — owner T003
- **violated Spec**: `spec:mldb.v2.repository.canonical_writes` requires complete validated content at the canonical write boundary; `spec:mldb.v2.architecture.component_model` states that canonical writers validate complete objects before persistence.
- **exact file/function**: `mldb_v2/src/repository/canonical_writes.py` — `CanonicalRepositoryWriter.create_immutable` / `_create_idempotent`.
- **finding**: immutable creation validates only JSON compatibility plus matching `id`. It does not validate the requested kind's schema or complete frozen format. For example, a `study_plan` write with only `schema` + `id`, or even a Model schema at a StudyPlan path, is accepted and persisted.
- **why it matters**: the canonical writer can create invalid immutable history that later callers cannot repair because immutable identity conflicts correctly prevent overwrite. The current focused T003 test itself uses an intentionally incomplete pseudo-Plan and therefore does not prove the frozen writer contract.
- **required correction**: T003 must validate the complete immutable document for the requested canonical kind before any create/replay decision, including schema-kind and required format invariants. Keep domain execution/scheduler behavior out of the writer; this is persistence-boundary validation only.
- **contract resolution — 2026-09-10**: the frozen repository contract now uses dependency inversion rather than repository-owned copies of domain validators. `CanonicalRepositoryWriter` requires a non-optional `CanonicalRecordValidator`; `create_immutable` passes `(kind, entity_id, document)` to it, and an existing record must pass the same validation before exact equality can count as replay. The repository retains only generic kind/ID/path/atomicity/idempotence/conflict mechanics. The validator/composition layer owns StudyPlan/TrainingResult/Model/EvaluationResult format semantics, including wrong-schema, completeness, derived/cross-field, and status-dependent payload checks. Child Result/Model validation explicitly must not require the parent StudyResult slot to have been updated, preserving child-before-parent recovery semantics. Dedicated StudyResult create/transition validation is outside this Finding 3 repair and remains Finding 2/T003 work.
- **implementation status**: `mldb_v2/src/repository/canonical_writes.py` was intentionally not changed by this contract repair and remains non-conformant with the repaired Skeleton until the T003 implementation repair is applied.
- **ADR decision**: no new ADR. The explicit validator port is a direct clarification of the already-frozen `component_model` rule that canonical writers validate complete objects plus the existing architecture dependency direction; it does not introduce a new product/runtime architecture choice.

#### Finding W001-V04 — HIGH — owner T004 / follow-up owner missing
- **violated boundary**: `spec:mldb.v2.storage` defines S3-compatible object storage as the initial transport/storage class; `MLDB-V2-WORK-009` requires an actual ClearML + S3-compatible E2E smoke in T009-04.
- **exact file/function**: `mldb_v2/src/storage/object_bytes.py` — `_ObjectByteTransport` / `_ObjectByteAccess`; Work Item graph W001-W009.
- **finding**: W001 currently provides only an injected transport Protocol and fake-transport tests. No concrete boto3/minio/S3-compatible transport implementation exists, and no downstream implementation Task explicitly owns creating one. W004-T02 owns materialization/publication behavior but does not explicitly own the concrete configured S3 transport; W009-T04 is verification only.
- **why it matters**: the graph currently reaches an E2E requirement that cannot be satisfied without an unowned implementation step. "A concrete adapter will be added later" is not an executable owner assignment.
- **required correction**: assign the concrete configured S3-compatible transport to an exact implementation Task before W009-T04. The natural existing owner is `MLDB-V2-TASK-004-02` if its responsibility is explicitly extended to include the adapter; otherwise add a named implementation Task and dependency. Until that owner exists, this review selects decision **B**, not A.
- **planning resolution — 2026-09-10**: `MLDB-V2-WORK-004` now explicitly assigns the configured concrete S3-compatible object-byte transport adapter to `MLDB-V2-TASK-004-02`, and W004 completion explicitly rejects fake/in-memory transport as sufficient. T004 remains the backend-neutral byte/integrity boundary; concrete transport implementation is therefore a named downstream responsibility before the actual W009 ClearML+S3 smoke.

### Conforming areas observed by independent source/test review
- Common ID/public-parameter/canonical-JSON/Diagnostic implementation matches the frozen shapes and preserves bool/int/float distinctions; no domain-specific result-ID grammar was moved into Common.
- Exact resolution is namespace-first, typed fixed-path, read-only, with no legacy-flat fallback, sibling scan, executable import, backend access, or object materialization. Representative `tile-classifier` and `rotated-fcos` canonical YAML paths have matching namespace-first IDs/schemas.
- Broad listing is fixed-domain/non-recursive and reports malformed namespaces/YAML, schema/ID mismatches, orphan/disallowed Python, manifest placement issues, unknown direct domains, and nested domain directories rather than silently hiding them.
- UUID4 enforcement in T003 is actual `uuid.UUID(...).version == 4` plus RFC4122 variant validation; `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa` is rejected.
- ArtifactRef keeps required `uri`/`bytes`/`sha256` integrity while accepting JSON-compatible immutable domain extensions such as `format` and `schema`; logical S3 URIs reject credential/query leakage.
- Corpus manifest validation covers exact UTF-8 JSONL bytes, required fields, cross-platform safe relative paths, lexical order, uniqueness, exact digest, and optional expected count.
- Git snapshot helpers read exact committed blobs, reject unsafe paths/missing commits, compare only selected working-tree sources, and do not stage/commit/push.
- Object-byte access separates configured transport from canonical ArtifactRef, verifies size/SHA after read, uses immutable publication, and materializes with create-only semantics.
- Mutation coordination uses OS-backed process locking outside `mldb_data`, serializes same keys, permits independent different keys, and does not implement queue/lease/heartbeat/scheduler/backend polling behavior.
- Public src names/fields inspected in W001 match the frozen Skeleton; no mldb-v1, ClearML, boto/minio, classifier/detector-specific, or `tools/` dependency was found in the reviewed W001 src modules.

### Actual-example note
Both `mldb_data/tile-classifier` and `mldb_data/rotated-fcos` currently contain generated `__pycache__/` directories inside executable domain directories. The broad-listing implementation will correctly report these nested directories as structural issues rather than silently hiding them. This verification Task did not modify `mldb_data`.

### Command-execution evidence limitation
The connected filesystem allowed full file review/editing, but the installed Remote Desktop Commander had no connected device in this session, so the required fresh `pytest`, `py_compile`, import-smoke, `git diff --check`, and `git status --short` commands could not be rerun here. Existing T001-T004 records report the prior 177-test integrated baseline, but this T005 verdict does not treat that prior green count as closing the findings above. The newly added verification-only tests are intentionally expected to expose W001-V01, W001-V02, and W001-V03 until their owning Tasks are revised.

## Independent re-verification — 2026-09-10

### Verdict: NEEDS REVISION

Fresh review was repeated from Specification/Skeleton through src/tests and actual `mldb_data`; no src, Spec, Skeleton, or canonical example was repaired.

Previous-finding closure:
- **W001-V01 / T002**: original tagged/anchor/alias and colon-containing plain-scalar probes are repaired, and current canonical examples parse. **Not fully closed**: block scalar YAML is still silently reinterpreted (RV01 below).
- **W001-V02 / T003**: UUID4 enforcement, submitted/cancelling terminal closure, completed/completed_with_failures/global-failure basics, cancellation transition, and existing-Model `training: null` are repaired. **Not fully closed**: failed closure still accepts cancellation-only skip semantics (RV02 below).
- **W001-V03 / T003**: **closed**. `CanonicalRecordValidator` is required/non-optional; proposed and existing records are validated before create/replay success; failures preserve bytes and are not reclassified; exact replay/conflict behavior remains generic; no immutable domain validators were copied into repository code and StudyResult remains dedicated.
- **W001-V04 / downstream owner gap**: **closed by planning**. W004 explicitly assigns the configured concrete S3-compatible transport adapter to `MLDB-V2-TASK-004-02`, W004 completion rejects fake/in-memory-only transport, and W009 depends on W004 before the actual ClearML+S3 smoke.

#### Finding W001-RV01 — CRITICAL — owner T002
- **violated Spec**: `spec:mldb.v2.common.public_parameters`; `spec:mldb.v2.repository.resolution` / canonical YAML semantics.
- **exact file/function**: `mldb_v2/src/repository/_yaml.py:42` — `_prepare_lines` block-scalar handling.
- **finding**: accepted `|` / `>` block scalars lose YAML clip semantics. `value: |\n  a\n  b\n` becomes `"a\nb"` instead of `"a\nb\n"`; folded form similarly drops the terminal LF. This is silent value reinterpretation, not rejection of unsupported syntax.
- **required correction**: implement block-scalar fold/chomp semantics correctly, or reject block scalar syntax until supported. Do not preserve the current altered string semantics as a private YAML subset.
#### Finding W001-RV02 — HIGH — owner T003
- **violated Spec**: `spec:mldb.v2.results.study_result_format` closure/status mapping and cancellation precedence.
- **exact file/function**: `mldb_v2/src/repository/canonical_writes.py:233` — `_validate_study_result_record` (with transition admitted by `_validate_study_result_transition`).
- **finding**: a `submitted -> failed` replacement is accepted when one slot is `skipped: global_failure` and another is `skipped: study_cancelled`. `study_cancelled` is cancellation closure semantics, but `cancelling` may close only as `cancelled`; it cannot be encoded inside a `failed` StudyResult merely because another global-failure skip exists.
- **required correction**: enforce the reverse status/disposition closure invariant so `failed` cannot contain cancellation-only `study_cancelled` semantics. Preserve valid already-terminal ordinary child dispositions and the dedicated `cancelling -> cancelled` path.

Verification-only probes were added to `mldb_v2/tests/test_w001_integrated_verification.py`; both fail on current src and no src repair was made.

Fresh command evidence after adding the probes:
- focused W001/T005 command: **2 failed, 40 passed**; failures are exactly RV01 and RV02.
- full `mldb_v2/tests`: **2 failed, 210 passed**; failures are exactly RV01 and RV02.
- before adding the new independent probes, the requested baseline reproduced exactly: focused **40 passed**, full **210 passed**.
- W001 src py_compile/import smoke: **14 compiled, 14 imported**.
- actual `CanonicalRepositoryListing("mldb_data").list_entities()`: **18 items, 0 issues**; Namespaces are `rotated-fcos`, `tile-classifier`.
- actual example scan found no block scalar usage under those two namespaces, so RV01 does not require modifying current canonical examples.
- public repository method parameter shapes match frozen Skeleton for writer/resolver/listing/coordinator, including required keyword-only `record_validator`.
- forbidden runtime import/dependency scan found no mldb v1, `mldb_v2.skeleton`, ClearML, boto/botocore/minio, or `tools/` imports in reviewed W001 src. Queue/lease/heartbeat/scheduler/backend-polling scan found no implementation (only `_release_os_lock` substring noise).
- `git diff --check -- mldb_v2`: PASS. `git status --short -- mldb_v2`: `?? mldb_v2/` (pre-existing repository state; not treated as a finding).

T005 remains `planned`; `MLDB-V2-WORK-001` remains `planned` because unresolved RV01/RV02 block PASS.


## Final independent re-verification — 2026-09-10

### Verdict: NEEDS REVISION

A fresh independent W001 review was repeated from frozen Specifications and Skeleton through current
src/tests and the actual canonical examples. No `src/`, Specification, Skeleton, `mldb_data/`, or
downstream implementation was repaired by this Task.

Previous finding closure was re-established independently:
- **W001-V01 / T002: closed.** Tags, anchors, aliases are rejected; colon-containing plain scalars
  such as S3/HTTPS URIs remain scalar values. The earlier V01 probes pass.
- **W001-V02 / T003: closed.** Non-terminal closure, completed/completed-with-failures/global-failure
  basics, cancellation closure, existing-Model `training: null`, and actual UUID4/RFC variant checks pass.
- **W001-V03 / T003: closed.** `CanonicalRecordValidator` is required and non-optional; proposed and
  existing records are validated before create/replay success and failures preserve canonical bytes.
- **W001-V04: closed by downstream ownership.** W004 T004-02 explicitly owns the configured concrete
  S3-compatible transport and W004 cannot complete with fake/in-memory transport only; W009 depends on W004.
- **W001-RV01 / T002: closed.** Unsupported block scalar nodes, including chomp/indent variants and
  sequence nodes, fail closed while ordinary `|`/`>` characters in strings remain valid.
- **W001-RV02 / T003: closed.** `failed` rejects any `study_cancelled` skip while preserving valid
  global-failure closure, ordinary terminal child dispositions, and the `cancelling -> cancelled` path.

#### Finding W001-RV03 — CRITICAL — owner T002
- **violated Spec**: `spec:mldb.v2.common.public_parameters`; `spec:mldb.v2.repository.resolution`;
  `spec:mldb.v2.repository.listing` malformed-candidate/fail-closed semantics.
- **exact file/function**: `mldb_v2/src/repository/_yaml.py:127` `_reject_unsupported_node_indicator`;
  `:176` `_parse_sequence`, especially `:199` `item.update(extra)`.
- **reproduction**: `_load_yaml("value: @reserved\n")` and the equivalent backtick-leading scalar
  are accepted as ordinary strings although those node-start indicators are unsupported YAML syntax.
  `values:\n  - key: first\n    key: second\n` is accepted as `{"values":[{"key":"second"}]}`;
  the duplicate key is silently overwritten although duplicate block/flow mappings are rejected.
- **required correction**: extend the explicit subset to reject unsupported reserved node indicators
  before scalar reinterpretation, and reject duplicate keys when an inline sequence mapping is merged
  with following mapping lines. Do not add a character-wide grep that rejects valid quoted content.

#### Finding W001-RV04 — HIGH — owner T003
- **violated Spec**: `spec:mldb.v2.results.study_result_format` non-terminal closure invariant.
- **exact file/function**: `mldb_v2/src/repository/canonical_writes.py:320-325`
  `_validate_study_result_record`.
- **reproduction**: an otherwise valid initial `status: submitted` StudyResult with `trials: []` is
  accepted and persisted by `create_study_result`; an existing-Model trial with `training: null` and
  no Evaluations is likewise accepted. There is no pending planned work.
- **required correction**: `submitted` and `cancelling` must require at least one pending planned stage
  even when the collected disposition list is empty. Preserve valid `training: null` when planned
  Evaluation slots exist.

#### Finding W001-RV05 — HIGH — owner T003
- **violated Spec**: `spec:mldb.v2.results.study_result_format` requires `created_at` to be RFC3339 UTC
  using `Z`.
- **exact file/function**: `mldb_v2/src/repository/canonical_writes.py:220` `_validate_created_at`.
- **reproduction**: `created_at: 2026-W37-3T09:00:00Z` is accepted; Python's broad ISO parser also
  accepts reduced forms such as `2026-09-09T09Z`. These are not the frozen RFC3339 timestamp shape.
- **required correction**: validate the exact RFC3339 UTC-with-`Z` lexical form before/while parsing;
  reject ISO-8601 extensions or reduced time forms not admitted by RFC3339.

#### Finding W001-RV06 — HIGH — owner T003
- **violated Spec**: `spec:mldb.v2.results.study_result_format` identity rule and
  `spec:mldb.v2.study.plan_format` Study-namespace Plan identity/path rule.
- **exact file/function**: `mldb_v2/src/repository/canonical_writes.py:258-267`
  `_validate_study_result_record`.
- **reproduction**: StudyResult `demo/run-<uuid4>` is accepted with `study: other/example-study`;
  an independently probed `plan: other/example-plan` is also accepted. The validator checks only
  typed-reference grammar and never ties the StudyResult namespace back to its source Study/Plan.
- **required correction**: enforce the frozen StudyResult namespace identity against the referenced
  Study and Plan namespace before persistence, without introducing sibling scans or backend lookup.

Fresh command evidence:
- Requested focused W001 command before adding new independent probes: **91 passed**.
- Full `mldb_v2/tests` before adding new probes: **231 passed**.
- After preserving RV03-RV06 as verification-only probes, focused W001: **4 failed, 91 passed**;
  full `mldb_v2/tests`: **4 failed, 231 passed**. The four failures are exactly RV03-RV06.
- Common/storage/source focused verification after adding probes: **140 passed**.
- W001 src py_compile/import smoke: **14 compiled, 14 imported**.
- Public Skeleton -> src structural method/field shape comparison: **PASS**.
- Forbidden import scan: no `mldb` v1, `mldb_v2.skeleton`, ClearML, boto/botocore/minio, or `tools/`
  imports in W001 src. Queue/lease/heartbeat/scheduler/backend-polling implementation scan is clean;
  the only `lease` substring is `_release_os_lock`.
- Actual `CanonicalRepositoryListing("mldb_data").list_entities()`: **18 items / 0 issues** across
  `rotated-fcos` and `tile-classifier`; all current canonical YAML examples parse under the loader.
- StudyResult listing fixture verified namespace/study/status/lower-upper-created-at/limit filters and
  deterministic namespace/id ordering.
- Final `git diff --check -- mldb_v2` after this Evidence update: **PASS**.
- Final `git status --short -- mldb_v2`: `?? mldb_v2/`, the pre-existing repository state.
  Whole-working-tree cleanliness is not required.

T001-T004 remain recorded `completed`; this verification Task does not edit their statuses under the
independence rule. T005 remains `planned`, and W001 remains `planned` because RV03-RV06 block downstream
dependency approval. No commit was created.


## Final adversarial independent verification after coordinator YAML hardening — 2026-09-10

### Verdict: NEEDS REVISION

The coordinator baseline was reproduced before adding new probes: T002/YAML focused **99 passed**,
combined W001 focused **154 passed**, and full `mldb_v2/tests` **294 passed**. Frozen Specifications,
Skeleton, current src/tests, and actual `tile-classifier` / `rotated-fcos` canonical examples were then
reviewed independently again. No `src/`, Specification, Skeleton, `mldb_data/`, T001-T004 implementation,
or downstream implementation was repaired by this verification.

Previous findings **W001-V01-V04 and W001-RV01-RV06 are closed** on current source behavior. In
particular, the hardened YAML loader preserves/rejects the previously problematic tag/anchor/alias,
block-scalar, reserved-indicator, duplicate-key, nested sequence mapping, flow-colon, alternate
numeric, raw-tab, string-key-only, JSON duplicate-key, and JSON non-finite classes without reopening
the earlier valid-input regressions.

#### Finding W001-RV07 — HIGH — owner T003
- **violated Spec**: `spec:mldb.v2.repository.canonical_writes` repository-generic canonical
  namespace/path safety; `spec:mldb.v2.repository.resolution` Namespace metadata consistency.
- **exact file/function**: `mldb_v2/src/repository/canonical_writes.py` — `_canonical_path`, reached
  from `CanonicalRepositoryWriter.create_immutable` / StudyResult write paths.
- **minimal reproduction**: create `mldb_data/demo/namespace.yaml` with Namespace schema but `id: other`,
  then create immutable `demo/example-plan`; the writer succeeds because it checks file existence only.
- **semantic impact**: the canonical writer can create immutable history beneath a directory that the
  resolver/listing layer does not recognize as that Namespace, breaking the repository's own canonical
  path invariant at the persistence boundary.
- **required correction**: before any canonical write, validate the target Namespace metadata schema
  and exact ID/directory match using repository-generic logic; do not add sibling scans or Catalog-domain validation.

#### Finding W001-RV08 — HIGH — owner T003
- **violated Spec**: `spec:mldb.v2.common.identity`; `spec:mldb.v2.results.study_result_format`.
- **exact file/function**: `mldb_v2/src/repository/canonical_writes.py` — `_validate_study_result_record`.
- **minimal reproduction**: records containing the 10,000th trial or evaluation are accepted as
  `trial-10000` / `eval-10000`; Common correctly defines and validates only exact `trial-NNNN` /
  `eval-NNNN` four-digit identities.
- **semantic impact**: StudyResult canonical history can contain child coordinates outside the frozen
  common ID grammar, so typed identity and deterministic child-result derivation disagree.
- **required correction**: validate StudyResult trial/evaluation IDs through the frozen Common ID
  grammar (or equivalently reject expansion beyond 9999) while retaining contiguous authored order.

#### Finding W001-RV09 — HIGH — owner T004
- **violated Spec**: `spec:mldb.v2.storage.artifact_reference`.
- **exact file/function**: `mldb_v2/src/storage/artifact_reference.py` — `_validate_artifact_ref`.
- **minimal reproduction**: a valid ArtifactRef remains accepted after adding `password`, `access_key`,
  `presigned_url`, or backend-local `cache_path` metadata.
- **semantic impact**: the base storage validator can authorize canonical ArtifactRef values that
  directly embed secret/runtime transport state forbidden by the frozen storage contract.
- **required correction**: make ArtifactRef validation reject forbidden credentials, presigned URLs,
  access-key/password material, and backend-local cache/configuration metadata while preserving valid
  domain-specific immutable format/media/schema extensions.

#### Finding W001-RV10 — HIGH — owner T004
- **violated Spec**: `spec:mldb.v2.storage.corpus_manifest` exact canonical entry semantics and digest commitment.
- **exact file/function**: `mldb_v2/src/storage/corpus_manifest.py` — `_parse_corpus_manifest`, at the
  per-line `json.loads` boundary.
- **minimal reproduction**: JSONL rows with duplicate `path`, `bytes`, or `sha256` members are accepted
  with Python JSON last-member-wins semantics, e.g. `{"path":"a","path":"b",...}` becomes path `b`.
- **semantic impact**: authored manifest membership/integrity data can be silently reinterpreted before
  validation even though the exact manifest bytes are the canonical digest commitment.
- **required correction**: reject duplicate JSON object members during row decoding before entry
  validation; do not normalize or choose one duplicate value.

Verification-only probes for RV07-RV10 were added to
`mldb_v2/tests/test_w001_integrated_verification.py`. Current src fails all four probes; no production
implementation was changed.

Fresh command evidence after preserving the new probes:
- requested T002/YAML focused set: **4 failed, 99 passed**;
- requested combined W001 focused set: **4 failed, 154 passed**;
- full `mldb_v2/tests`: **4 failed, 294 passed**;
- the four failures are exactly RV07, RV08, RV09, and RV10; all pre-existing tests remain green.
- all **14** W001 src Python files py_compile and all **14** import successfully.
- actual `CanonicalRepositoryListing("mldb_data").list_entities()`: **18 items / 0 issues**;
  all 18 current `tile-classifier` / `rotated-fcos` canonical YAML files parse as mappings with the
  expected namespace-first schema/id structure.
- public repository Resolver/Listing/Writer/Coordinator and `CanonicalRecordValidator` parameter
  names, positional/keyword-only shape, requiredness, and defaults match the frozen Skeleton.
- forbidden import scan: zero mldb-v1, `mldb_v2.skeleton`, ClearML, boto3/botocore/minio, or `tools/`
  imports; queue/heartbeat/scheduler/backend-poll/retry-wait implementation scan is zero.
- W004 still explicitly assigns the production S3-compatible adapter to T004-02 and W009 depends on
  W004 before actual ClearML+S3 smoke, so original V04 remains closed by named downstream ownership.
- final `git diff --check -- mldb_v2`: **PASS**.
- final `git status --short -- mldb_v2`: `?? mldb_v2/`, the pre-existing untracked-tree state; whole-tree
  cleanliness is not required.
- coordinator/root temporary scripts `.agent_yaml_patch.py` and `.local/finish_yaml_hardening.py` are
  absent; this verification's `.local/t005` probe directory was removed after use.

Recorded statuses remain: T001 `completed`, T002 `completed`, T003 `completed`, T004 `completed`,
T005 `planned`; W001 remains `planned`. Under the verification write-scope, T003/T004 implementation
or status was not changed even though RV07/RV08 are assigned to T003 and RV09/RV10 to T004.
Downstream implementation MUST NOT treat W001 as a frozen foundation until these four blocking findings
are repaired and independently re-verified. No commit was created.

## Final independent adversarial re-verification — 2026-09-11

### Verdict: NEEDS REVISION

Frozen Specs/Skeleton, W001/T001-T005 records, current W001 src/tests, W004 S3 ownership,
and all 18 actual canonical YAML examples were re-read independently. No `src/`, Specification,
Skeleton, `mldb_data/`, T001-T004 implementation, or downstream implementation was changed.

Previous findings are closed on current source behavior:
- **W001-V01-V04: closed.** YAML tag/anchor/alias and colon-scalar behavior, StudyResult closure,
  immutable validator ownership, and concrete S3 downstream ownership remain repaired.
- **W001-RV01-RV06: closed.** Block scalars fail closed; reserved indicators/duplicate sequence
  mappings and the systematic parser hardening remain repaired; StudyResult cancellation, zero-topology,
  RFC3339-Z, and namespace invariants remain repaired.
- **W001-RV07-RV10: closed.** Canonical writes validate Namespace metadata; StudyResult trial/eval IDs
  pass Common four-digit grammar; ArtifactRef forbidden metadata is recursively rejected; manifest
  duplicate JSON members/non-finite constants fail closed.
- Latest coordinator ArtifactRef hardening is also valid: generic nested AWS-style presigned URLs and
  absolute backend/cache paths are rejected, while benign documentation URLs and descriptive metadata
  remain accepted.

#### Finding W001-RV11 — CRITICAL — owner T002
- **violated Spec**: `spec:mldb.v2.common.public_parameters` value preservation and the repository
  YAML fail-closed/semantic-preservation boundary used by resolution/listing.
- **exact file/function**: `mldb_v2/src/repository/_yaml.py` — `_prepare_lines`, `_parse_scalar`, and
  related generic `.strip()`/`.isspace()` whitespace handling.
- **reproduction**: `_load_yaml("value: foo\u00a0\n")` returns `{"value": "foo"}` instead of
  preserving U+00A0 as scalar content or rejecting the construct. Leading NBSP in a mapping key is
  likewise stripped (`"\u00a0key" -> "key"`), and NBSP inside a flow collection is treated as syntax
  whitespace. U+2003 reproduces the same trailing-scalar loss.
- **semantic impact**: authored UTF-8 string/key content can be silently normalized into a different
  canonical value. This is the same forbidden class as earlier YAML silent reinterpretation findings.
- **required correction**: distinguish YAML syntax separation whitespace from Unicode scalar content;
  preserve valid non-separation Unicode characters or explicitly fail closed before normalization.
  Do not globally ban such characters inside quoted strings merely to make this probe green.

#### Finding W001-RV12 — HIGH — owner T002
- **violated Spec**: `spec:mldb.v2.repository.layout` and `spec:mldb.v2.repository.listing` structural
  diagnostics; final T005 inventory requirement that invalid files are not silently ignored.
- **exact file/function**: `mldb_v2/src/repository/listing.py` — `_namespace_layout_issues` and
  `_list_kind`.
- **reproduction**: in a valid Namespace, create `tasks/garbage.txt` and direct child
  `unexpected.txt`; `CanonicalRepositoryListing(...).list_entities()` returns the Namespace with
  **0 issues**. Unknown non-directory files at Namespace root and unknown file types inside fixed
  domains fall through without a diagnostic.
- **semantic impact**: broad validation can report a clean canonical tree while invalid/unrecognized
  files exist inside that tree, contradicting the listing boundary's malformed-structure visibility.
- **required correction**: report unexpected files in canonical Namespace roots and fixed domain
  directories unless they are one of the exact canonical/companion forms allowed by layout. Preserve
  the explicit v1-flat exclusion and do not recurse into arbitrary structures.

Verification-only probes for RV11/RV12 were added to
`mldb_v2/tests/test_w001_integrated_verification.py`; both fail on current src. No production code was
repaired or weakened.
Fresh evidence:
- before the new independent probes, requested T003 focused: **94 passed**; storage focused:
  **81 passed**; combined focused: **244 passed**; full `mldb_v2/tests`: **344 passed**.
- after preserving RV11/RV12 probes, T003 focused: **2 failed, 94 passed**; storage focused:
  **2 failed, 81 passed**; combined focused: **2 failed, 244 passed**; full: **2 failed, 344 passed**.
  The two failures are exactly RV11 and RV12.
- all **14** W001 src Python modules py_compile and import successfully.
- actual `CanonicalRepositoryListing("mldb_data").list_entities()`: **18 items / 0 issues**;
  all **18** actual canonical YAML files parse as mappings.
- fresh ArtifactRef adversarial probes reject case/separator credential variants, nested runtime
  password, generic-key AWS presigned URL, and Unix/Windows backend-cache absolute paths; benign
  documentation URL plus `logical_name`/`format`/`schema`/`media_type`/`compression` is accepted.
- public Writer/Resolver/Listing/Coordinator signatures match frozen Skeleton parameter shape across
  **8** checked methods. Forbidden import scan is empty for mldb-v1, `mldb_v2.skeleton`, ClearML,
  boto3/botocore/minio, and `tools/`; hidden heartbeat/scheduler/backend-poll/retry-wait terms are empty.
- W004 T004-02 still explicitly owns configured concrete S3-compatible transport and W004 completion
  rejects fake/in-memory-only transport.
- `git diff --check -- mldb_v2`: **PASS**. `git status --short -- mldb_v2` remains `?? mldb_v2/`,
  the pre-existing untracked-tree state.

Statuses intentionally remain: T001 `completed`, T002 `completed`, T003 `completed`, T004 `completed`,
T005 `planned`; W001 `planned`. Under the independence rule this verification does not reopen/edit
implementation Task status and does not repair T002. W001 MUST NOT be used as a downstream frozen
dependency until RV11/RV12 are repaired and independently re-verified. No commit was created.

## Final independent adversarial closure verification after RV11/RV12 repair — 2026-09-11

### Verdict: NEEDS REVISION

The coordinator repair baseline was reproduced before adding any new verification-only probes:
requested repository/YAML/integrated focused set **133 passed**, all explicit
`test_repository_*.py` regression **191 passed**, and full `mldb_v2/tests` **374 passed**.
Frozen Specs/Skeleton, W001/T001-T005 records, current W001 src/tests, W004/W009 S3 ownership,
and all current canonical examples were then independently reviewed again. No production `src/`,
Specification, Skeleton, `mldb_data/`, T001-T004 status, or downstream implementation was changed.

Previous findings are closed on current behavior:
- **W001-V01-V04: closed.** The original YAML semantic-reinterpretation, StudyResult closure,
  immutable validator ownership, and concrete S3 ownership gaps remain repaired.
- **W001-RV01-RV10: closed.** Block scalars/reserved indicators/duplicate mappings/flow ambiguity/
  alternate numerics/raw tabs/non-string keys/strict JSON, StudyResult zero-topology/RFC3339/
  namespace/four-digit sequence identity, canonical Namespace write validation, ArtifactRef secret
  and AWS-style presigned URL rejection, and strict manifest JSON all remain repaired.
- **W001-RV11: closed.** NBSP U+00A0 and EM SPACE U+2003 are preserved as scalar/key content,
  ASCII U+0020 alone drives comment/mapping separation and flow whitespace, CRLF/LF are explicit,
  U+0085/U+2028/U+2029 fail closed, and `_yaml.py` has no generic `.strip()`/`.lstrip()`/`.rstrip()` /
  `.isspace()` / `.splitlines()` syntax dependency.
- **W001-RV12: closed.** Namespace-root and fixed-domain unexpected files are diagnosed. Fresh probes
  for an extensionless file, hidden `.foo`, uppercase `foo.YAML`, backup `foo.yaml~`, and swap
  `.foo.swp` all produce `repository_unexpected_domain_file`. Symlink creation was not available on
  the Windows test host (`WinError 1314`), so no symlink fixture was forced; source review confirms
  non-recursive fixed-domain handling. Specific Python/manifest/nested/domain-file diagnostics remain
  distinct and legacy flat v1 root directories remain excluded from v2 Namespace inventory.

YAML encoding/control review also confirmed that mid-key/mid-scalar U+FEFF is preserved rather than
silently deleted; U+0000/U+0001/U+001F/U+007F and form-feed/vertical-tab are either preserved as
scalar content or rejected by the quoted/scalar boundary, never removed/truncated or treated as ASCII
syntax whitespace. ASCII comment/mapping behavior remains correct and NBSP around flow separators is
preserved as scalar content rather than normalized to ASCII-space syntax.

#### Finding W001-RV13 — CRITICAL — owner T002
- **violated Spec**: `spec:mldb.v2.common.public_parameters` value-preservation boundary together with
  `spec:mldb.v2.repository.resolution` / `spec:mldb.v2.repository.listing` fail-closed canonical YAML
  semantics; final T005 BOM rule requires a document-leading U+FEFF to be either intentionally
  supported as the UTF-8 BOM or rejected.
- **exact file/function**: `mldb_v2/src/repository/_yaml.py` — `_load_yaml`, `_prepare_lines`, and
  `_parse_mapping_key`.
- **minimal reproduction**: `_load_yaml("\ufeffschema: mjtensu.mldb-v2/namespace/v1\nid: demo\n")`
  succeeds as `{"\ufeffschema": "mjtensu.mldb-v2/namespace/v1", "id": "demo"}`. It neither parses
  the BOM-bearing document as the intended `schema` key nor fails closed.
- **semantic impact**: the document-start encoding marker is silently reinterpreted as canonical key
  content, changing the authored mapping. Common schema-first records are likely rejected later for a
  missing `schema`, but the YAML boundary itself is non-semantic and a different first-field ordering
  can preserve the altered key into a resolved document instead of exposing the encoding boundary.
- **required correction**: either recognize U+FEFF only at absolute document start as a supported
  UTF-8 BOM and parse the remaining document identically, or reject a document-leading U+FEFF before
  mapping parsing. Preserve U+FEFF occurring inside ordinary scalar/key content; do not globally strip it.

#### Finding W001-RV14 — HIGH — owner T004
- **violated Spec**: `spec:mldb.v2.storage.artifact_reference` operational-metadata exclusion and the
  final W001 ArtifactRef requirement that backend/cache local absolute paths cannot enter canonical values.
- **exact file/function**: `mldb_v2/src/storage/artifact_reference.py` —
  `_validate_no_forbidden_artifact_metadata` / `_looks_like_local_absolute_path`.
- **minimal reproduction**: a valid ArtifactRef remains accepted after adding either
  `metadata: {backend: {path: "C:/runtime/object.bin"}}` or
  `metadata: {backend: {path: "/srv/runtime/object.bin"}}`. Equivalent absolute paths under a
  `cache` context are correctly rejected.
- **semantic impact**: generic nested metadata can still persist machine/backend-local runtime path
  state in an otherwise canonical ArtifactRef, bypassing the operational-state exclusion by changing
  only the parent field name from cache to backend.
- **required correction**: reject local absolute path values when they occur in backend/runtime/cache
  operational context, including generic nested `path` keys, while continuing to allow benign
  descriptive relative paths and documentation URLs. Do not ban every field named `path` or every URL.

Verification-only probes for RV13/RV14 were added to
`mldb_v2/tests/test_w001_integrated_verification.py`; both fail on current src and no production code
was repaired. Fresh evidence after preserving them:
- requested focused resolution/listing/integrated set: **2 failed, 133 passed**; failures are exactly
  RV13 and RV14.
- repository regression (`test_repository_canonical_writes.py`, `test_repository_listing.py`,
  `test_repository_mutation_coordination.py`, `test_repository_resolution.py`): **191 passed**.
- full `mldb_v2/tests`: **2 failed, 374 passed**; failures are exactly RV13 and RV14.
- all **14** W001 source modules py_compile successfully and all **14** import successfully.
- actual `CanonicalRepositoryListing("mldb_data").list_entities()`: **18 items / 0 issues**;
  all **18/18** current `tile-classifier` / `rotated-fcos` canonical YAML documents parse as mappings.
- frozen repository public parameter shape: **9** Writer/Validator/Resolver/Listing/Coordinator methods
  checked and PASS; common/storage public TypedDict fields checked for Diagnostic,
  PublicParameterDeclaration, ArtifactRef, and CorpusManifestEntry and PASS.
- forbidden dependency scan is empty for mldb-v1 runtime imports, `mldb_v2.skeleton` runtime imports,
  ClearML, boto3/botocore/minio, `tools/`, and detector/classifier specialization. Hidden queue,
  heartbeat, scheduler, backend-poll, and retry-wait term scans are empty in W001 source.
- ArtifactRef still rejects nested cache absolute paths and generic-key AWS-style presigned URLs while
  accepting a benign documentation URL; RV14 is specifically the nested backend-path bypass above.
- W004 still explicitly assigns configured concrete S3-compatible transport implementation to
  `MLDB-V2-TASK-004-02`, fake/in-memory transport cannot complete W004, and W009 depends on W004
  before the actual ClearML+S3 smoke. W001 itself therefore still requires no boto/minio dependency.
- `git diff --check -- mldb_v2`: **PASS**. `git status --short -- mldb_v2` remains `?? mldb_v2/`,
  the pre-existing untracked-tree state.

Statuses remain intentionally unchanged under the independence rule: T001 `completed`, T002
`completed`, T003 `completed`, T004 `completed`, T005 `planned`; W001 `planned`. W001 MUST NOT yet be
used as a downstream frozen dependency. No commit was created.


## Final W001 closure verification ? 2026-09-11

### Verdict: PASS

Independent closure verification was repeated against the frozen W001 Specs/Skeleton, T001-T005 records,
current W001 src/tests, actual canonical repository examples, and downstream S3 ownership. No production
`src/`, Specification, Skeleton, `mldb_data/`, T001-T004 implementation/status, or downstream implementation
was changed by this verification.

- **W001-V01-V04: closed.** Original YAML semantic preservation, StudyResult closure, immutable validator
  ownership, and concrete S3 downstream-ownership findings remain repaired/resolved.
- **W001-RV01-RV14: closed.** All previously recorded YAML, listing, canonical-write/StudyResult,
  ArtifactRef, and Corpus-manifest repair classes were rechecked against current behavior with no regression.
- **RV13 BOM boundary:** exactly one U+FEFF at absolute document index 0 is consumed for both block YAML and
  strict JSON; double leading BOM fails closed; non-leading/mid-key/mid-scalar/quoted U+FEFF remains content.
- **RV14 ArtifactRef operational paths:** absolute Windows/Unix/UNC/`file://` local paths are rejected under
  explicit normalized backend/runtime/cache operational contexts, including deeper nesting. Relative operational
  paths and descriptive absolute paths/benign documentation URLs outside those contexts remain accepted.
- Fresh T002 resolution/listing/integrated verification: **143 passed**.
- Fresh T004 ArtifactRef/integrated verification: **69 passed**.
- Explicit `test_repository_*.py` regression: **199 passed**.
- Explicit `test_storage_*.py` + `test_source_*.py` regression: **92 passed**.
- Full `mldb_v2/tests`: **396 passed**.
- All W001 src Python modules: **14/14 py_compile PASS**, **14/14 import PASS**.
- Actual canonical inventory: **18 items / 0 issues**; actual canonical YAML: **18/18 parse PASS**.
- Frozen public shape: **9/9 public method parameter/default/requiredness checks PASS**; common/storage public
  TypedDict field/requiredness and public enum-value checks PASS.
- Forbidden dependency/specialization scan: no mldb-v1 runtime imports, `mldb_v2.skeleton` runtime imports,
  ClearML, boto3/botocore/minio, `tools/`, classifier/detector specialization, queue/lease/heartbeat/scheduler,
  backend-polling, or retry-wait implementation in W001 src.
- Concrete S3 ownership remains downstream: W004 T004-02 explicitly owns the configured concrete S3-compatible
  object-byte transport; W004 cannot complete on fake/in-memory transport alone; W009 depends on W004 before
  the actual ClearML+S3 E2E smoke. W001 therefore does not require a concrete boto/minio adapter.
- `git diff --check -- mldb_v2`: PASS before closure-record edits. `git status --short -- mldb_v2` remains
  `?? mldb_v2/`, the pre-existing untracked-tree state.
- No verification-only test was added or changed in this final closure pass. No commit was created.

T001, T002, T003, T004, and T005 are now `completed`; W001 is `completed`.

MLDB-V2-WORK-001 may now be used as a downstream dependency.
