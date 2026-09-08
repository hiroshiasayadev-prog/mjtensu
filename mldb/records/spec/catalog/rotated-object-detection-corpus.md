# Contract: Rotated-object-detection Corpus format

- **id**: `spec:mldb.catalog.rotated_object_detection_corpus`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.catalog`
- **contract_class**: `format`

## What this is

Defines the concrete SQLite data contract selected by:

```text
mjtensu.mldb/rotated-object-detection-corpus/v1
```

The artifact is self-contained for Training/Evaluation. Worker execution must not need
repository image directories or mutable annotation stores in addition to the immutable
Corpus SQLite bytes.

## Current contract

The table named by `data.table` must expose at least:

| column | contract |
|---|---|
| `sample_id` | Corpus-local unique sample identifier. |
| `split` | Non-empty Corpus-defined split identifier. |
| `annotations_json` | UTF-8 JSON array of labeled rotated rectangles. |
| payload column | Image bytes named by `representation.payload_column`. |

`representation` must provide:

| field | contract |
|---|---|
| `kind` | `image` |
| `dtype` | `uint8` |
| `shape` | Positive `[C,H,W]` dimensions. |
| `payload_column` | Non-empty SQLite BLOB column name. |

Each `annotations_json` value is a JSON array. Every item must be an object containing:

```json
{"label":"mahjong_tile","obb":[160.0,120.0,24.0,36.0,-12.5]}
```

`label` must exist in the referenced rotated-object-detection Task's ordered `labels`.
`obb` is `[center_x, center_y, width, height, angle_deg]`; all values are finite and
width/height are strictly positive.

## Rules

- `sample_id` is unique within the Corpus.
- `split` is non-empty for every row.
- Payload storage must be a non-NULL SQLite BLOB whose byte length equals the element
  count implied by the declared `uint8` shape.
- The referenced Task must use `target.type: rotated-object-detection`.
- Unknown additional columns are valid and ignored generically.
- YAML split counts must equal the canonical sample-table counts.

## Validation rules

| condition | result |
|---|---|
| Canonical sample table is missing | Invalid Corpus artifact. |
| Required core column is missing | Invalid Corpus artifact. |
| `sample_id` is duplicated | Invalid Corpus artifact. |
| `split` is empty | Invalid Corpus artifact. |
| Task target is not rotated-object-detection | Invalid Corpus/Task pairing. |
| Annotation JSON is malformed or not an array | Invalid Corpus artifact. |
| Annotation label is outside Task labels | Invalid Corpus artifact. |
| OBB is not five finite numbers or has non-positive size | Invalid Corpus artifact. |
| Payload byte size disagrees with representation | Invalid Corpus artifact. |
| Unknown additional column exists | Valid; ignore generically. |

The outer Corpus contract owns artifact SHA-256, byte-size integrity, sibling placement,
and YAML split-summary agreement.

## Boundary

| concern | owner |
|---|---|
| Semantic labels and OBB target meaning | Rotated-object-detection Task contract. |
| Outer immutable Corpus identity | Corpus format contract. |
| Model preprocessing/normalization | Train/Evaluation Protocols. |
| Dense detector output tensors | Architecture. |
