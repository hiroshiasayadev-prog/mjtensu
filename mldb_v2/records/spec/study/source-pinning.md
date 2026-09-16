# Contract: Study source pinning

- **id**: `spec:mldb.v2.study.source_pinning`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.study`
- **contract_class**: `validation`

## Formal-plan cleanliness scope

Formal Study planning does not require the entire repository working tree to be clean. It requires
all canonical v2 inputs consumed by the plan to match the selected Git commit exactly.

The required clean set includes `mldb_v2/src/`, the referenced Namespace/Study/Task/Corpus/
Architecture/Protocol YAML, same-basename executable sibling `.py` files, declared same-namespace
`lib/` helper sources, Corpus manifests/builders, and referenced canonical Model or result records.
Unrelated application files, Brewprint records/tests, and unrelated namespaces may be dirty.
## Execution rule

The Study Plan records one source Git commit. Backend execution checks out that commit, or consumes
an equivalent verified repository snapshot, and MUST NOT include the caller's uncommitted patch.
Therefore unrelated application work may continue in the local working tree without changing the
formal experiment source.

If a referenced v2 canonical input differs from the selected commit, formal planning fails rather
than hashing the dirty file into an ad hoc snapshot. Submission revalidates the persisted Plan and
its pinned source identity before backend ownership begins.
