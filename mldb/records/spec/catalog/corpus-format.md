# Contract: Corpus format

- **id**: `spec:mldb.catalog.corpus_format`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.catalog`
- **contract_class**: `format`

## What this is

Defines the YAML contract for one immutable MLDB Corpus.

A Corpus is the frozen materialized sample collection consumed by later training or evaluation. Mutable annotation stores and external source installations remain outside the mandatory MLDB reproducibility boundary.

## Current contract

Corpus YAML uses schema identifier:

```text
mjtensu.mldb/corpus/v1
```

The required fields are:

| field | contract |
|---|---|
| `schema` | Exact Corpus metadata schema identifier. |
| `id` | Immutable Corpus identity and sibling basename. |
| `task` | Referenced Task ID. |
| `artifact.format` | Materialized artifact format. Corpus v1 uses `sqlite`. |
| `artifact.sha256` | SHA-256 of the sibling materialized artifact. |
| `data.schema` | Concrete schema identifier for the data stored inside the artifact. |
| `data.table` | Canonical sample table name. |
| `representation` | Description of the materialized input representation. |
| `builder.parameters` | Effective builder parameter mapping. |
| `splits` | Split inventory and sample counts. |

Optional fields are:

| field | contract |
|---|---|
| `description` | Human-readable Corpus description. |
| `artifact.bytes` | Byte size of the materialized artifact. |
| `statistics` | Descriptive statistics derived from the Corpus. |
| `origin` | Human-oriented upstream source summary. |

## Rules

- Corpus v1 uses one `.yaml`, one `.sqlite`, and one `.py` sibling with the same Corpus ID basename.
- `artifact.format` is `sqlite` for Corpus v1.
- `artifact.sha256` identifies the exact immutable SQLite bytes.
- `artifact.bytes`, when present, describes the same immutable artifact.
- `task` references exactly one Task.
- `data.schema` selects the concrete physical-data contract inside the SQLite artifact.
- `data.table` names the canonical sample table used by the selected concrete data schema.
- `representation` describes materialized sample representation, not Task semantics or training-time augmentation.
- `builder.parameters` records effective materialization parameters that are not recoverable from builder source alone.
- Builder source plus recorded parameters explains materialization but does not guarantee byte-for-byte regeneration from mutable upstream sources.
- Split names are Corpus-defined non-empty strings. MLDB v1 does not define a global split-name enum.
- YAML split counts must match the canonical sample table.
- `statistics` and `origin` do not replace the immutable artifact hash.
- Registered Corpus artifact bytes must not be overwritten in place.
- Any material artifact change requires a new Corpus identity or revision.
- A metadata change that changes immutable artifact meaning also requires a new Corpus identity or revision.
- Editorial `description` or `origin` corrections may keep the same Corpus ID when artifact meaning is unchanged.

## Validation rules

| condition | result |
|---|---|
| `schema` is not `mjtensu.mldb/corpus/v1` | Invalid Corpus metadata. |
| Required field is absent | Invalid Corpus metadata. |
| Sibling `.yaml`, `.sqlite`, or `.py` basename disagrees with `id` | Invalid Corpus asset. |
| Referenced Task does not resolve | Invalid Corpus asset. |
| `artifact.format` is not `sqlite` | Unsupported Corpus v1 artifact. |
| Calculated SQLite SHA-256 differs from `artifact.sha256` | Invalid Corpus integrity. |
| `artifact.bytes` is present and differs from the actual byte size | Invalid Corpus metadata. |
| `data.table` does not exist | Invalid Corpus artifact. |
| YAML split names or counts disagree with the concrete sample table | Invalid Corpus metadata/artifact pair. |

Validation of concrete sample columns and target/class-index agreement belongs to the selected `data.schema` contract.

## Boundary

| concern | owner |
|---|---|
| Corpus sibling placement | `spec:mldb.repository.layout`. |
| Task semantics | `spec:mldb.catalog.task_format`. |
| Image-classification SQLite sample contract | `spec:mldb.catalog.image_classification_corpus`. |
| Builder implementation details | Corpus sibling `.py` implementation. |
| How splits are consumed during training or evaluation | Training and evaluation topics. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.catalog` | Parent catalog Index. |
| `spec:mldb.catalog.task_format` | Defines the referenced semantic Task. |
| `spec:mldb.repository.layout` | Defines Corpus sibling placement. |
| `spec:mldb.runtime.asset_resolution` | Resolves the Corpus and verifies applicable integrity before consumption. |
