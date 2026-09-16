# Contract: Typed canonical resolution

- **id**: `spec:mldb.v2.repository.resolution`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.repository`
- **contract_class**: `repository`

## Lookup

Resolution input is an entity kind plus exact typed ID `<namespace>/<local-id>`. The canonical path
is derived only from `spec:mldb.v2.repository.layout`.

Resolution MUST:

1. validate the ID grammar;
2. require `mldb_data/<namespace>/namespace.yaml` with matching Namespace ID;
3. derive the fixed domain path for the requested kind;
4. read exactly the expected canonical file;
5. validate schema kind and ID/path consistency before returning the parsed value.

Cross-namespace references use the same rule. A missing exact path is `not_found`; resolution does
not scan sibling namespaces/domains or guess a similarly named entity.

Existing v1 flat `mldb_data/<domain>/...` paths are never fallback candidates for v2 resolution.
## Side-effect boundary

Canonical resolution is read-only. It does not import executable companions, run pytest, download
Corpus/object bytes, query ClearML, or mutate lifecycle state.

Callers that require executable or materialized integrity apply the focused verification/runtime
contract after typed resolution.
