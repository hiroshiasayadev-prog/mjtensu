# Contract: Image-classification Corpus format

- **id**: `spec:mldb.catalog.image_classification_corpus`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.catalog`
- **contract_class**: `format`

## What this is

Defines the concrete SQLite data contract selected by:

```text
mjtensu.mldb/image-classification-corpus/v1
```

The contract standardizes only the minimum sample fields needed by generic MLDB classification tooling. Builder-specific SQLite columns remain open.

## Current contract

The Corpus metadata field `data.table` names the canonical sample table.

That table must expose at least:

| column | contract |
|---|---|
| `sample_id` | Corpus-local unique sample identifier. |
| `split` | Non-empty Corpus-defined split identifier. |
| `target` | Task target label for the sample. |
| `class_index` | Canonical Task class index for `target`. |
| payload column | Materialized image payload named by `representation.payload_column`. |

For this data schema, `representation` must provide enough information to interpret the image payload, including:

| field | contract |
|---|---|
| `kind` | Materialized representation kind. For this schema the value is `image`. |
| `dtype` | Stored image element dtype. |
| `shape` | Materialized image tensor shape. |
| `payload_column` | Name of the sample-table payload column. |

For the initial grayscale classifier Corpus, the payload is a SQLite BLOB representing the declared `uint8` image data.

## Rules

- `sample_id` must be unique within the Corpus.
- `sample_id` does not define global identity across different Corpora.
- `split` must be non-empty.
- `target` must be a valid label from the referenced categorical Task.
- `class_index` must equal the zero-based position of `target` in the Task's normative ordered labels.
- The payload column named by `representation.payload_column` must exist.
- Payload storage must be compatible with the declared representation.
- The sample table may contain arbitrary additional columns.
- Generic MLDB validation must tolerate additional columns that this schema does not define.
- Additional columns may preserve provenance, acquisition conditions, source IDs, geometry, or other builder-specific facts.
- The Corpus YAML does not enumerate all additional SQLite columns.
- No universal optional metadata or JSON-tag vocabulary is required by this schema.

## Validation rules

| condition | result |
|---|---|
| `data.schema` is not `mjtensu.mldb/image-classification-corpus/v1` | This contract does not apply. |
| Canonical sample table is missing | Invalid Corpus artifact. |
| Required core column is missing | Invalid Corpus artifact. |
| `sample_id` is duplicated | Invalid Corpus artifact. |
| `split` is empty | Invalid Corpus artifact. |
| `target` is not in the referenced categorical Task | Invalid Corpus artifact. |
| `class_index` disagrees with Task label order | Invalid Corpus artifact. |
| Declared payload column is missing | Invalid Corpus artifact. |
| Payload storage is incompatible with the declared representation | Invalid Corpus artifact. |
| Unknown additional column exists | Valid; ignore generically. |

The outer Corpus metadata contract owns artifact hashing and YAML split-summary agreement.

## Boundary

| concern | owner |
|---|---|
| Outer Corpus YAML and artifact identity | `spec:mldb.catalog.corpus_format`. |
| Categorical label meaning and ordering | `spec:mldb.catalog.task_format`. |
| Builder-specific optional SQLite columns | Corpus builder and SQLite schema itself. |
| Training-time preprocessing or augmentation | Train Protocol. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.catalog` | Parent catalog Index. |
| `spec:mldb.catalog.corpus_format` | Selects this concrete data schema through `data.schema`. |
| `spec:mldb.catalog.task_format` | Defines target labels and canonical class indices. |
