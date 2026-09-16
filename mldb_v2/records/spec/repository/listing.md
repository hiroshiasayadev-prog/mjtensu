# Contract: Canonical inventory and listing

- **id**: `spec:mldb.v2.repository.listing`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.repository`
- **contract_class**: `repository`

## Purpose

The repository layer provides deterministic broad discovery so application/CLI callers never need to
implement their own filesystem search to enumerate MLDB v2 objects.

## Namespace inventory

V2 Namespace discovery examines direct children of `mldb_data/` only. A child is a Namespace
candidate only when it contains `namespace.yaml`. Valid Namespace inventory is ordered by Namespace
ID.

Legacy flat v1 domain directories are not returned as v2 Namespaces.

## Entity inventory

Within a Namespace, entity discovery examines only the fixed domain directories from
`spec:mldb.v2.repository.layout`. It does not recurse into arbitrary folders or infer kind from file
contents.

For each kind, canonical YAML basenames are enumerated lexicographically and validated for
ID/path/schema consistency before becoming listable canonical entities.
## Structural diagnostics

Inventory also reports repository-structure issues needed by broad validation, including malformed
Namespace metadata, unknown direct domain names, orphan executable companions, invalid basenames,
and canonical YAML that fails ID/path/schema validation.

An invalid candidate is reported as a validation issue; it is not silently omitted in a way that
would make `mldb validate` appear clean.

## Ordering and side effects

Cross-Namespace aggregate order is Namespace ID, canonical kind order, then typed entity ID.
Listing is read-only: it does not import companion Python, run tests, query a backend, materialize
Corpus bytes, or mutate lifecycle state.

`spec:mldb.v2.api.query_interface` and broad validation/verification consume this inventory instead
of implementing independent filesystem traversal.
