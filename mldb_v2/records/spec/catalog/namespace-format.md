# Contract: Namespace format

- **id**: `spec:mldb.v2.catalog.namespace_format`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.catalog`
- **contract_class**: `format`

## Meaning

A Namespace represents one continuing experiment concept. It is intentionally coarser than a Study.

Keep experiments in the same Namespace when it remains useful to compare their architectures,
training conditions, and results together. Create a new Namespace when the experiment concept
changes materially enough that such comparison becomes misleading or cluttered.

A Namespace is not a per-Study folder.

## YAML

`mldb_data/<namespace>/namespace.yaml` uses:

```yaml
schema: mjtensu.mldb-v2/namespace/v1
id: rotated-fcos
name: Rotated FCOS
description: >
  Mahjong rotated-object-detection experiments based on the Rotated FCOS family.
```

Required fields are `schema`, `id`, `name`, and `description`.

`id` is exactly one lowercase kebab-case segment and MUST match the directory name. Namespace IDs
are stable once referenced.

Backend metadata is forbidden in Namespace YAML. The ClearML Project mapping is derived by the
ClearML adapter.
