# Contract: Runtime Model format

- **id**: `spec:mldb.v2.catalog.runtime_model_format`
- **status**: draft
- **date**: 2026-10-01
- **parent**: `spec:mldb.v2.catalog`
- **contract_class**: `format`

## Meaning

Runtime Model is an immutable catalog identity for an already-deployable model artifact whose historical training lineage is unavailable or intentionally is not represented as an MLDB Training Result.

It exists so Evaluation Protocols can select such artifacts by typed ID without inventing Training Result provenance or accepting raw URI, SHA, or preprocessing parameters from users.

Runtime Model is not a replacement for a normal learned `Model`. If canonical MLDB Training Result lineage exists, the learned Model remains authoritative.

## Shape

Schema is `mjtensu.mldb-v2/runtime-model/v1`. Canonical files live at `mldb_data/<namespace>/runtime_models/<local-id>.yaml`. Local ID is versioned (`-vN`).

Required fields are `schema`, `id`, non-empty `name`, `description`, `role`, `format`, non-empty `runtime_spec`, and `artifact`. Optional `provenance` may describe historical source evidence but MUST NOT fabricate canonical Training Result lineage.

`role` is one of `detector`, `tile-classifier`, or `red-five-classifier`. v1 `format` is exactly `onnx`.

`artifact` is an immutable ArtifactRef with logical S3 URI, byte count, and lowercase SHA-256. Operational credentials, presigned URLs, and backend-local paths are forbidden.

`runtime_spec` names the browser/deployment preprocessing and output contract required by the artifact. Evaluation implementations MUST fail closed when a selected runtime spec is unsupported.

## Loading

Generic MLDB resolves Runtime Model IDs referenced by Evaluation Protocol `model_parameters`, reads the artifact through backend-neutral object access, verifies byte count and SHA-256, and exposes `LoadedRuntimeModel(definition, artifact)` through `EvaluationContext.models`.

Protocol code does not receive S3 credentials or resolve the catalog entry itself.

A Runtime Model entry and its ArtifactRef are source-pinned inputs to execution. Changing artifact identity, role, runtime spec, or result-affecting provenance such as required preprocessing creates a new versioned Runtime Model ID.
