# Contract: Corpus format

- **id**: `spec:mldb.v2.catalog.corpus_format`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.catalog`
- **contract_class**: `format`

## Meaning

Corpus identifies one frozen materialized sample collection for one Task. Materialized bytes, not a
mutable annotation/source database, are the reproducibility boundary.

## YAML

Schema is `mjtensu.mldb-v2/corpus/v1`.

Required fields are:

| field | contract |
|---|---|
| `schema` | Exact schema identifier. |
| `id` | Full versioned `<namespace>/<local-id>` identity. |
| `status` | `draft` or `sealed`. |
| `task` | Typed Task reference. |
| `description` | Human-readable summary. |
| `storage.root_uri` | Non-empty immutable logical S3-compatible root URI. |
| `manifest.file` | Exact same-basename `<local-id>.manifest.jsonl`. |
| `representation` | JSON-compatible mapping with required non-empty `kind`. |
| `splits` | Mapping from non-empty split name to non-negative integer sample count. |

For `sealed`, `manifest.sha256` and `manifest.entries` are additionally required.
`manifest.sha256` is 64 lowercase hex and `manifest.entries` is a non-negative integer describing
exact manifest row count. The sealed manifest contract is `spec:mldb.v2.storage.corpus_manifest`.

## Optional builder companion

Corpus MAY contain:

```yaml
builder:
  entrypoint: build
  sha256: <64 lowercase hex>   # required when sealed
  parameters: {}               # JSON-compatible mapping
```

Presence of `builder` permits same-basename `<local-id>.py`; absence forbids a Corpus `.py` sibling.
The builder is provenance/materialization behavior and is not authoritative over sealed Corpus bytes.
Generic MLDB v2 does not execute the builder during Study execution.

## Identity and validation

Local ID ends in `-v<positive-integer>`. A sealed Corpus requires resolvable sealed Task, valid
manifest hash/count, valid object byte/hash verification, and valid builder hash when applicable.
Replacing any materialized byte, changing representation meaning, split membership/count meaning, or
Task binding requires a new Corpus revision.

Unknown top-level keys are invalid in schema v1. Representation may contain domain-specific
JSON-compatible keys beyond `kind`; those are part of Corpus meaning and become immutable when sealed.
