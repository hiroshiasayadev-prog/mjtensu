# MLDB-V2-TASK-001-02: Implement repository resolution and listing

- **status**: completed
- **date**: 2026-09-09
- **work_item**: MLDB-V2-WORK-001
- **task_type**: implementation
- **depends_on**: [MLDB-V2-TASK-001-01]
- **outputs**: `mldb_v2/src/repository/resolution.py`, `listing.py`, focused tests

## Goal
Implement namespace-first exact canonical resolution and deterministic broad inventory/listing without fallback scans or kind guessing.

## Work
- Implement `kind + exact id -> fixed canonical path` using repository layout rules.
- Discover v2 Namespaces only from direct `mldb_data/` children containing valid `namespace.yaml`.
- Enumerate only fixed canonical domain directories and deterministic basename ordering.
- Report malformed Namespace/domain/canonical candidates, orphan companions, invalid basenames, and ID/path/schema mismatches as structural diagnostics.
- Keep companion import, pytest, backend queries, object materialization, and lifecycle mutation out of listing/resolution.

## Done condition
Exact resolution and broad listing are separate, deterministic, read-only implementations; legacy flat v1 directories are never returned as v2 entities.

## Verification
Test exact success/not-found/mismatch cases, aggregate ordering, malformed candidates, legacy flat-layout exclusion, and absence of arbitrary recursive filesystem search.

## Evidence
- 2026-09-09 integrated review: 21 focused repository resolution/listing tests PASS.
- Exact resolution remains typed/fixed-path with no legacy-flat fallback or recursive kind guessing.
- Broad inventory preserves deterministic namespace/kind/entity order and reports malformed structural candidates.
- Full W001 suite passes after integration review; `git diff --check` and py_compile pass.

## T005 re-verification reopen — 2026-09-10
- W001-RV01 remains open: _yaml.py::_prepare_lines accepts |/> block scalars but drops YAML clip semantics. T002 is reopened until block scalars are either parsed with correct fold/chomp semantics or rejected before reinterpretation.

## T005 W001-RV01 block-scalar repair — 2026-09-10
- _yaml.py now fail-closes unsupported YAML block scalar nodes (| / > including chomping and indentation indicators) instead of accepting them with altered fold/chomp semantics.
- Ordinary quoted/plain strings containing | or > remain valid; earlier tag/anchor/alias rejection and colon-containing plain-scalar handling remain intact.
- Fresh coordinator verification after the RV01/RV02 repairs: combined W001 focused set **91 passed**; full mldb_v2/tests **231 passed**; all 14 mldb_v2/src Python files py_compile; actual repository listing **18 items / 0 issues**; git diff --check -- mldb_v2 PASS.
- T002 is returned to completed. T005/W001 remain owned by independent re-verification.

## Final T005 re-verification reopen — 2026-09-10
- W001-RV03 remains open. The private YAML loader still accepts reserved-leading plain scalars such as `@reserved` / `` `reserved`` instead of failing closed, and a duplicate key inside a sequence inline mapping is silently overwritten by `item.update()`.
- T002 is reopened for a narrow parser safety repair only. Existing tag/anchor/alias, colon-scalar, and block-scalar repairs must remain intact.
- T005/W001 remain `planned` until another independent verification closes the finding.

## T005 W001-RV03 YAML fail-closed completion repair - 2026-09-10
- `_yaml.py` now rejects unsupported node-leading YAML indicators before plain-scalar reinterpretation, including reserved `@` / backtick starts, while preserving quoted values, flow collections, signed numeric scalars, valid `-`/`?`/`:` plain starts, and indicator characters occurring inside ordinary strings.
- Sequence inline mappings no longer silently overwrite a decoded duplicate key when continuation mapping entries are merged. Quoted/unquoted keys are decoded before the duplicate check; nonduplicate continuation mappings remain supported.
- Existing tag/anchor/alias, block-scalar, colon-containing URI/plain-scalar, flow-mapping duplicate, and top-level block duplicate behavior remains green. The T005 RV03 verification probe was not weakened or removed.
- Fresh focused verification: **72 passed** across repository resolution, listing, and W001 integrated verification. Full `mldb_v2/tests`: **267 passed**.
- All **14** `mldb_v2/src` Python files py_compile; actual canonical listing **18 items / 0 issues**; `git diff --check -- mldb_v2` PASS. `git status --short -- mldb_v2` remains `?? mldb_v2/`, the pre-existing repository state.
- T002 is returned to completed. T005/W001 remain unchanged for independent verification. No commit was created.


## Coordinator adversarial reopen after RV03 repair ? 2026-09-10
- Fresh focused/full regression is green after RV03-RV06 repair, but a pre-T005 adversarial parser probe found additional silent reinterpretation paths in `_yaml.py`.
- A sequence inline mapping with a nested mapping value (`- key:` followed by a deeper `nested: value`) is flattened to sibling keys instead of being rejected or parsed with YAML semantics.
- A flow-sequence implicit pair (`[key: value]`) is accepted as the string `"key: value"`; malformed flow mapping content such as `{a: b: c}` is likewise accepted as a string value instead of failing closed.
- Unquoted scalar forms with alternate YAML numeric syntax such as `0x10`, `1_000`, and `1_000.5` are accepted as strings, losing authored scalar semantics.
- Literal tab indentation is expanded to spaces before parsing and therefore accepted instead of rejected; unquoted scalar mapping keys such as `1`, `true`, and `null` are coerced into string keys, defeating the string-key-only value-domain check.
- The initial `json.loads` fast path also needs duplicate-key/non-finite fail-closed review because Python JSON decoding otherwise silently overwrites duplicate object keys and accepts non-standard constants.
- T002 is reopened for one systematic parser-boundary hardening pass before another independent T005. No src was changed by this coordinator note.

## Coordinator YAML hardening completion — 2026-09-10
- `_yaml.py` now fail-closes JSON duplicate keys and non-finite constants/overflow in the JSON fast path; those semantic rejections do not fall through to alternate YAML interpretation.
- Raw literal tabs are rejected instead of expanded. Unquoted mapping keys are decoded using scalar semantics and non-string keys are rejected consistently in block and flow mappings; quoted equivalents remain valid strings.
- Sequence inline mappings now preserve a genuinely nested value (`- key:` followed by a deeper block) while retaining sibling continuation mappings and duplicate-key rejection.
- Unsupported flow implicit pairs / extra mapping-separator forms no longer become plain strings; URI/plain colon scalars without YAML mapping-separator syntax remain supported.
- Unsupported alternate numeric-looking forms (`0x`, `0o`, `0b`, numeric underscores) fail closed rather than becoming strings. Existing decimal integer/float/scientific parsing and ordinary underscore-bearing strings remain intact.
- Added focused regression coverage for nested sequence mappings, flow-colon ambiguity, alternate numerics, literal tabs, string-only mapping keys, JSON duplicate keys, and JSON non-finite values.
- Fresh focused resolution/listing/W001 verification: **99 passed**. Combined W001 repository verification: **154 passed**. Full `mldb_v2/tests`: **294 passed**.
- All **14** `mldb_v2/src` Python files py_compile; actual canonical listing remains **18 items / 0 issues**; `git diff --check -- mldb_v2` PASS.
- Temporary `.agent_yaml_patch.py` and coordinator `.local/finish_yaml_hardening.py` were removed. T002 is returned to `completed`; T005/W001 remain unchanged for independent verification. No commit was created.

## Final T005 re-verification reopen — 2026-09-11
- W001-RV11 remains open: `_yaml.py` uses Python Unicode whitespace semantics (`strip`/`isspace`) at YAML syntax boundaries, so NBSP/U+00A0 and similar non-ASCII whitespace can be silently removed or treated as separator whitespace instead of preserved or rejected.
- W001-RV12 remains open: broad listing silently ignores unknown regular files at Namespace root and unknown file types inside fixed canonical domain directories, allowing structurally invalid canonical trees to report clean.
- T002 is reopened for these two narrow repository parser/listing repairs. Existing V01/RV01/RV03 and systematic YAML hardening must not regress.
- T005/W001 remain `planned` until independent re-verification closes RV11/RV12. No Spec/Skeleton or canonical data change is required by this reopen note.

## T005 W001-RV11/RV12 completion repair — 2026-09-11
- `_yaml.py` now treats only ASCII space U+0020 as syntax separation/indentation whitespace in the private subset. Generic Unicode `strip`/`isspace`/`splitlines` semantics were removed from YAML syntax decisions; CRLF/LF remain supported and unsupported Unicode line separators fail closed.
- NBSP U+00A0 and EM SPACE U+2003 are preserved as plain/quoted/key string content. NBSP does not become a comment separator before `#` or a block mapping separator after `:`.
- `listing.py` now reports unexpected regular files at valid Namespace roots with `repository_unexpected_namespace_entry`, and unexpected noncanonical files inside fixed domains with `repository_unexpected_domain_file`. Existing specific Python/manifest/domain/subdirectory diagnostics remain authoritative without generic duplicate reports; legacy flat v1 root entries remain excluded from v2 inventory.
- Fresh focused verification (`test_repository_resolution.py`, `test_repository_listing.py`, `test_w001_integrated_verification.py`): **133 passed**. Repository regression: **191 passed**. Full `mldb_v2/tests`: **374 passed**.
- All **6** repository source Python modules py_compile. Actual `CanonicalRepositoryListing("mldb_data").list_entities()` remains **18 items / 0 issues**; all **18** actual canonical YAML documents parse successfully.
- `git diff --check -- mldb_v2` PASS. `git status --short -- mldb_v2` remains `?? mldb_v2/`, the pre-existing repository state. T005/W001 were not modified. No commit was created.
- T002 is returned to `completed`; no parser or listing contract ambiguity remains blocking this focused repair.

## Final T005 W001-RV13 reopen — 2026-09-11
- W001-RV13 remains open: `_load_yaml` accepts a document-leading U+FEFF BOM as part of the first mapping key, e.g. `schema` becomes `\ufeffschema`, instead of intentionally consuming the BOM or failing closed.
- Mid-key/mid-scalar U+FEFF preservation is already correct; the repair must be limited to the absolute document-start BOM boundary and must not globally strip U+FEFF.
- T002 is reopened for this narrow YAML encoding-boundary repair. T005/W001 remain `planned`; Spec/Skeleton/canonical data are unchanged.
## T005 W001-RV13 document-leading UTF-8 BOM boundary repair — 2026-09-11
- BOM policy: support exactly one U+FEFF only at absolute document index 0 as the UTF-8 BOM. `_load_yaml` consumes that single marker before either the strict JSON fast path or block-YAML path; a double leading BOM fails closed instead of being repeatedly stripped.
- Repair boundary is parser entry only. `_prepare_lines`, mapping-key parsing, scalar parsing, ASCII-space semantics, and the previously closed Unicode/YAML hardening paths were not broadened or redesigned.
- Non-leading and mid-document U+FEFF remains ordinary string content: mid-key, mid-scalar, and both quoted forms are preserved. A U+FEFF after ASCII leading spaces is not treated as a document BOM and is not normalized away.
- Fresh focused verification (`test_repository_resolution.py`, `test_repository_listing.py`, `test_w001_integrated_verification.py`): **143 passed**. Existing RV13 integrated probe is green without modification.
- Explicit `test_repository_*.py` regression using concrete file enumeration: **199 passed** across 4 files. Full `mldb_v2/tests`: **396 passed**.
- All **6** `mldb_v2/src/repository` Python modules py_compile successfully. Actual `CanonicalRepositoryListing("mldb_data").list_entities()` remains **18 items / 0 issues**; all **18/18** actual canonical YAML documents parse successfully.
- `git diff --check -- mldb_v2` PASS. `git status --short -- mldb_v2` remains `?? mldb_v2/`, the pre-existing repository state. T005/W001 were not modified. No commit was created.
- T002 is returned to `completed`; no blocker remains for W001-RV13.
