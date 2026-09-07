# Concept: MLDB asset resolution

- **id**: `spec:mldb.runtime.asset_resolution`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.runtime`

## What this is

Defines how the MLDB runtime turns a typed entity reference into a resolved runtime handle.

Resolution owns deterministic lookup, metadata identity checks, direct upstream-reference resolution, and applicable integrity checks. Resolution does not import executable Python code or decide training, evaluation, or Study execution policy.

## Concept model

| concept | contract |
|---|---|
| typed reference | The caller or containing field determines the expected MLDB entity kind. The resolver does not search all entity directories for a matching string ID. |
| canonical lookup | Entity kind and ID determine one canonical repository location through the repository-placement contract. |
| metadata validation | The resolved metadata must satisfy the entity-specific format contract and schema identifier for the expected kind. |
| identity agreement | Requested ID, metadata ID, and the entity-defined canonical file or directory identity must agree. |
| upstream reference | Only fields defined by an entity contract as MLDB references are traversed. Arbitrary string values are never inferred as references. |
| resolver-produced handle | Runtime representation returned by canonical typed resolution. It exposes validated metadata, canonical repository provenance, and canonical execution-resource locations without embedding model-family behavior. |
| Worker-materialized handle | The same runtime handle shape constructed outside canonical resolution from Controller-selected validated metadata plus exact immutable bytes after required integrity verification and Worker-local materialization. Missing repository provenance is not replaced with synthetic paths, and local execution paths are not canonical locations. |
| executable loading | Loading the Python implementation selected by the handle is a separate executable-loader responsibility. Resolver output selects the canonical sibling; Worker materialization may select the verified local execution copy of the same assigned bytes. |
| execution readiness | Training, evaluation, Study materialization, and verification may impose additional status or compatibility requirements after resolution. |

Resolution is directional. A downstream entity resolves only the upstream references it explicitly declares.

Canonical typed resolution remains the only ID-to-repository lookup mechanism. Distributed Worker materialization is downstream of Controller resolution and assignment: it does not resolve an ID, discover a repository location, or establish a second source of truth.

```text
requested kind + ID
        |
        v
canonical repository location
        |
        v
entity metadata
        |
        +--> schema and identity validation
        |
        +--> declared upstream references
        |          |
        |          v
        |      recursive resolution
        |
        +--> applicable integrity validation
        |
        v
resolved handle
```

## Rules

- Resolution must use the canonical repository mapping for the expected entity kind.
- Every resolver-produced handle must retain canonical repository provenance and canonical resource paths required by that entity; optional implementation-level provenance fields do not weaken this resolver guarantee.
- Resolution must reject missing metadata at the canonical location.
- Resolution must reject a metadata schema that does not belong to the expected entity kind.
- Resolution must reject disagreement between the requested ID and the entity's recorded canonical identity.
- Resolution must follow only references declared by the target entity's format contract.
- Resolution must not discover relationships by scanning for reverse references.
- Resolution must not use fuzzy ID matching, aliases, or filename guessing unless a later contract explicitly adds them.
- Resolution must not import or execute Architecture, Train Protocol, or Evaluation Protocol Python implementations.
- A `draft` executable asset may resolve successfully when its format is valid.
- A caller that requires a sealed executable asset must reject a resolved `draft` asset before execution.
- A sealed executable asset must pass its declared implementation-integrity check before runtime execution or sealing-dependent verification consumes it.
- An immutable materialized artifact must pass its entity-defined integrity check before a runtime operation consumes the artifact.
- A distributed Worker may construct the existing runtime handle types only from metadata/identity fixed by the Controller assignment and execution bytes fixed by its immutable descriptors. Every required execution byte object must be verified against the supplied integrity identity before its Worker-local path is eligible for runtime consumption.
- Worker-local materialization paths are execution locations only. They must not be represented as canonical repository locations, and Worker must not synthesize metadata files merely to populate repository-provenance paths.
- Corpus builder source is part of canonical Corpus authoring/materialization provenance, not a Training/Evaluation execution resource. Distributed Worker execution need not receive or materialize the Corpus builder.
- Task, interface, Corpus, Model, and Protocol compatibility beyond reference identity belongs to the execution contract that combines those entities.
- The asset resolver itself must not create or mutate Training Runs, Evaluation Runs, Study Runs, Models, or formal artifacts. An execution component that already created a Run may record resolution failure through that Run's lifecycle contract.

## Boundary

| concern | owner |
|---|---|
| Physical location grammar for each MLDB entity kind | Repository topic. |
| Exact YAML, SQLite, JSONL, and Run-record fields | Entity-specific format specs. |
| Terminal `-vN` grammar and lifecycle fields | Entity-specific format or lifecycle specs. |
| Sibling Python import and entrypoint exposure | Executable loader and entity-specific callable interface specs. |
| Public parameter default and override resolution | `spec:mldb.runtime.public_parameters` topic. |
| Training Task/Corpus/Architecture/Protocol compatibility | Training topic. |
| Evaluation Task/Corpus/Model/Protocol compatibility | Evaluation topic. |
| Study grid validation and materialization | Study topic. |
| Pytest discovery and invocation | Verification topic. |
| Concrete resolver classes, caches, and exception types | Implementation. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.runtime` | Parent runtime overview. |
| `spec:mldb.runtime.component_model` | Defines the asset resolver and executable loader as separate runtime responsibilities. |
