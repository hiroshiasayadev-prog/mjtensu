# MLDB-ADR-SCHEMA-002: Define immutable Corpus assets

- **status**: accepted
- **date**: 2026-09-03
- **depends_on**: MLDB-ADR-SCHEMA-001
- **supersedes**:
- **migrated_to_spec**:

## Context

MLDB needs a stable data boundary for comparing model architectures and training conditions.
The current recognition workflow does not begin from immutable raw sources: capture and annotation databases are working stores that may receive new captures, human corrections, detector refreshes, and schema changes. External corpora may likewise be imported through project-specific builders and transformed through crop extraction, resizing, resampling, relabeling, filtering, or split construction.

Requiring MLDB to reconstruct complete sample-level provenance through every mutable upstream source would make Corpus management depend on the internal schema and lifecycle of each annotation or import tool. It would also make sample identity unstable under ordinary materialization changes such as recropping or resampling.

The useful reproducibility boundary is therefore the completed training/evaluation corpus itself. Once admitted to MLDB, a Corpus should be a frozen data asset that downstream training and evaluation can reference exactly.

The existing tile-classifier SQLite artifacts already approximate this boundary. They contain materialized samples, labels, split membership, and additional source-specific columns. MLDB should standardize only the minimum contract needed to identify and consume such corpora rather than imposing a universal metadata ontology on every builder.

## Decision

Introduce `Corpus` as an immutable MLDB asset bound to exactly one Task.

A Corpus is the frozen, materialized sample collection used as an input to downstream training or evaluation. MLDB guarantees identity and integrity from the Corpus artifact downstream. MLDB does not require complete reconstruction of the mutable upstream annotation, capture, import, or preprocessing history that existed before the Corpus was materialized.

### Physical placement and basename contract

Corpus assets live outside the Brewprint Design Records tree under the repository-level MLDB data root:

```text
mldb_data/
  corpora/
```

One Corpus occupies three sibling files sharing exactly the same basename, which is the immutable Corpus ID:

```text
mldb_data/corpora/<corpus-id>.sqlite
mldb_data/corpora/<corpus-id>.yaml
mldb_data/corpora/<corpus-id>.py
```

For the initial Corpus contract:

- `<corpus-id>.sqlite` is the materialized Corpus artifact;
- `<corpus-id>.yaml` is the authoritative Corpus metadata record;
- `<corpus-id>.py` is the builder that materializes that Corpus from upstream sources.

Example:

```text
mldb_data/corpora/gray35-jp500-v3-jp189.sqlite
mldb_data/corpora/gray35-jp500-v3-jp189.yaml
mldb_data/corpora/gray35-jp500-v3-jp189.py
```

The basename relation is normative. The YAML does not repeat the builder path or artifact path because both are resolved from the Corpus ID and this placement rule.

A future non-SQLite Corpus format requires a later schema decision rather than silently changing the v1 basename/extension contract.

### Corpus metadata format

Corpus metadata is YAML and uses schema identifier `mjtensu.mldb/corpus/v1`.

The initial shape is:

```yaml
schema: mjtensu.mldb/corpus/v1

id: gray35-jp500-v3-jp189
task: tile-shape-classification-35-v1

description: >
  Grayscale 35-class tile-shape classification corpus combining
  imported and reviewed materialized tile crops.

artifact:
  format: sqlite
  sha256: 0123456789abcdef...
  bytes: 123456789

data:
  schema: mjtensu.mldb/image-classification-corpus/v1
  table: sample

representation:
  kind: image
  dtype: uint8
  shape: [1, 64, 64]
  payload_column: image_gray_u8

builder:
  parameters:
    seed: 42
    image_size: 64
    jp_train_per_class: 500
    jp_valid_per_class: 200
    manual_train_fraction: 0.8

splits:
  train:
    count: 19593
  manual_val:
    count: 450
  jp_val:
    count: 6800

statistics:
  train:
    mean: [0.6815832403977466]
    std: [0.2725553681973969]

origin:
  - kind: annotation-database
    description: Reviewed real-capture annotations.
  - kind: external-dataset
    description: Imported Mahjong tile corpus.
```

The following fields are required in `mjtensu.mldb/corpus/v1`:

- `schema`;
- `id`;
- `task`;
- `artifact.format`;
- `artifact.sha256`;
- `data.schema`;
- `data.table`;
- `representation`;
- `builder.parameters`;
- `splits`.

The following fields are optional:

- `description`;
- `artifact.bytes`;
- `statistics`;
- `origin`.

`artifact.format` is `sqlite` in Corpus v1. `artifact.sha256` is the SHA-256 digest of the sibling `<corpus-id>.sqlite` artifact and is authoritative for mutation detection.

`data.schema` identifies the concrete physical-data contract inside the artifact. It is separate from the outer Corpus metadata schema. For example, `mjtensu.mldb/image-classification-corpus/v1` defines the required table/column contract for an image-classification SQLite Corpus, while `mjtensu.mldb/corpus/v1` defines the YAML record surrounding that artifact.

`data.table` identifies the canonical sample table consumed by generic Corpus tooling.

`representation` describes the materialized model input stored by the Corpus. It is not Task semantics and is not a training-time augmentation recipe. For image-classification corpora it records enough information to interpret the payload, including the payload column, materialized shape, and dtype.

### Corpus identity and immutability

Each Corpus has an immutable `id` and references one immutable Task ID through `task`.

The Corpus ID must equal the basename of its `.sqlite`, `.yaml`, and `.py` siblings.

The artifact content hash is authoritative for detecting mutation. Once a Corpus is admitted to MLDB, its `.sqlite` artifact must not be overwritten in place. Any content change produces a new Corpus identity or revision and therefore a new basename.

Changing metadata that describes immutable artifact facts, such as Task binding, representation, split membership, or builder parameters, likewise requires a new Corpus identity when the change means the existing artifact no longer satisfies the record.

Editorial corrections to `description` or `origin` that do not change the meaning or identity of the artifact do not require a new Corpus ID.

### Concrete image-classification Corpus schema

The initial concrete data schema is `mjtensu.mldb/image-classification-corpus/v1`.

Its canonical sample table is identified by `data.table` and must expose at least these columns:

| column | responsibility |
|---|---|
| `sample_id` | Corpus-local unique sample identity. |
| `split` | Non-empty split membership identifier. |
| `target` | Task target label for the sample. |
| `class_index` | Canonical Task class index corresponding to `target`. |
| payload column | Materialized image payload named by `representation.payload_column`. |

The concrete SQL type of the payload column must be compatible with the declared `representation`. For the initial grayscale classifier corpus this is a BLOB containing the materialized `uint8` image representation.

`target` and `class_index` must agree with the referenced Task contract. For categorical Tasks, `class_index` is the zero-based index of `target` in the Task's normative ordered label list.

`sample_id` is unique only within the Corpus. MLDB does not define global sample identity across separately materialized corpora.

### Additional SQLite columns are intentionally open

A Corpus may contain any additional columns beyond the concrete schema's required columns.

MLDB does not define a universal optional-metadata schema, JSON tag ontology, provenance column set, or enumeration catalog for these columns. Generic MLDB validation must tolerate columns it does not understand.

Builders may add whatever typed columns are useful for that Corpus, including source image identifiers, capture IDs, source partitions, layout identifiers, lighting conditions, annotation angles, original dimensions, or future project-specific attributes. Downstream analysis may query those columns directly with SQL when present.

The YAML metadata does not enumerate every additional SQLite column. The SQLite schema itself is authoritative for builder-specific queryable columns.

### Provenance is a builder recommendation

When importing or transforming external or mutable source data, builders should preserve useful provenance in ordinary additional columns when doing so is practical and reliable.

This is a recommendation, not a Corpus validity requirement. MLDB does not prescribe which provenance fields must exist, how they are named, or whether they survive recropping, resampling, relabeling, source-database migration, or other transformations.

The purpose is pragmatic: when a source identifier, capture property, or acquisition condition is already available, retaining it can later enable SQL subset selection and failure analysis. MLDB does not require upstream systems such as annotation tools to conform to a Corpus metadata ontology.

### Builder contract

The sibling `<corpus-id>.py` file is the authoritative builder implementation for that Corpus.

The YAML does not store a builder filename because basename equality makes it derivable. `builder.parameters` records the effective materialization parameters that are not recoverable from the builder source alone.

The builder and its parameters explain how the frozen Corpus was produced. They do not guarantee byte-for-byte regeneration from mutable or unavailable upstream sources.

A Corpus builder should be self-contained enough that a maintainer can determine:

- which upstream inputs it expects;
- how samples are selected and materialized;
- how targets and splits are assigned;
- which optional query columns it retains;
- how the declared Corpus representation is produced.

### Splits

Split membership is part of the immutable Corpus artifact.

`split` values are Corpus-defined non-empty strings. MLDB does not prescribe a global split-name enum. The YAML `splits` mapping summarizes the split names present in the artifact and their sample counts.

The YAML split counts must match the canonical sample table. This duplication is intentional because split inventory is important summary metadata and is cheap to validate automatically.

Later Train Protocol and Evaluation Protocol entities decide how declared splits are consumed.

### Statistics and origin

`statistics` is optional descriptive metadata derived from the Corpus itself, such as channel mean/std or class-distribution summaries. When present, its calculation basis should be evident from its nesting or keys, for example `statistics.train.mean`.

`origin` is optional human-oriented information about upstream source families. It is not a machine-enforced provenance chain and does not claim reproducibility of mutable upstream sources.

Neither `statistics` nor `origin` replaces builder code or the immutable artifact hash.

### Reproducibility boundary

MLDB treats mutable annotation stores, capture databases, external dataset installations, and other pre-materialization working data as upstream systems outside the mandatory Corpus reproducibility boundary.

The boundary is:

```text
mutable or external source data
  -> mldb_data/corpora/<corpus-id>.py
  -> mldb_data/corpora/<corpus-id>.sqlite
  -> training / evaluation / model lifecycle
```

The sibling YAML records the identity and contract of that frozen artifact. From the immutable Corpus artifact downstream, MLDB records must be able to identify exactly which Corpus was used.

## Rationale

Freezing Corpus at the materialized dataset boundary gives downstream experiments a stable comparison surface without requiring MLDB to become a version-control system for every annotation or data-acquisition tool.

A narrow core schema makes automated validation and generic tooling practical. `sample_id`, `split`, `target`, and `class_index` provide a common query contract for classification corpora, while concrete payload columns preserve representation freedom.

Allowing arbitrary additional columns preserves the value already present in project-specific dataset builders. Existing classifier corpora can retain capture IDs, source partitions, angles, lighting metadata, and other useful fields without forcing those details into the generic MLDB ontology.

Recording the builder remains useful even when full source reproducibility is impossible. It identifies the transformation code and parameters that produced the Corpus without making an unrealistic claim that mutable annotation history can always be reconstructed.

Artifact hashing provides a much stronger and simpler integrity rule than attempting to derive identity from individual crops. Resampling, crop-boundary changes, or representation changes naturally produce a different Corpus artifact and therefore a different immutable Corpus identity.

## Rejected alternatives

### Require complete sample-level provenance

Complete provenance would require MLDB to version annotation-tool source databases, capture assets, crop identities, correction history, and all intermediate materialization steps. Ordinary recropping or resampling would make identity rules difficult to maintain.

The operational cost is disproportionate to the project's need. MLDB instead starts its reproducibility guarantee at the frozen Corpus boundary.

### Standardize all optional metadata columns

A fixed optional-metadata schema would require every source and builder to map different concepts into one shared vocabulary. It would also require MLDB to predict future diagnostic dimensions such as camera pose, annotation confidence, physical tile identity, or source-specific partitions.

Additional columns are therefore intentionally open.

### Require a universal JSON tag column

A JSON tag field would provide a generic escape hatch, but the current SQLite workflow already benefits from normal typed columns and direct SQL selection. Requiring all optional metadata to be nested inside JSON would add indirection without solving a demonstrated problem.

Builders may still use JSON columns where that is locally useful; MLDB does not require or forbid them.

### Define global sample identity across corpora

A global sample identity would be unstable when the same upstream source is recropped, resampled, recolored, or otherwise materialized differently. Corpus-local identity is sufficient for training and evaluation and avoids pretending that transformed samples have one universal semantic identity.

### Treat the mutable annotation database as the Corpus

The annotation database is an editing and acquisition system. Its content may change as annotations are corrected or captures are added. Using it directly as the MLDB Corpus would make past training inputs dependent on mutable upstream state.

MLDB therefore records a frozen materialized Corpus instead.

## Consequences

Future Corpus tooling needs only a small amount of generic behavior:

- validate that `<corpus-id>.sqlite`, `<corpus-id>.yaml`, and `<corpus-id>.py` exist as siblings under `mldb_data/corpora/` and share the YAML `id` basename;
- validate the required `mjtensu.mldb/corpus/v1` YAML fields and Task reference;
- validate the SQLite artifact SHA-256 and `artifact.bytes` when that optional value is present;
- validate `data.table` existence and the required columns for the declared `data.schema`;
- validate the declared representation and payload column;
- verify categorical `target` / `class_index` agreement with Task;
- verify Corpus-local `sample_id` uniqueness;
- verify YAML split names/counts against the SQLite sample table;
- ignore unknown additional SQLite columns unless the concrete Corpus schema explicitly requires them.

Existing classifier dataset builders can be adapted incrementally rather than rewritten around a new metadata ontology.

Builders should preserve useful source attributes as ordinary extra columns when doing so is cheap and reliable. Those columns remain available for later ad hoc SQL evaluation, but their absence does not invalidate a Corpus.

Registered Corpus artifacts become immutable. Development builders may still use temporary or overwriteable outputs before registration, but once an artifact is admitted as an MLDB Corpus, mutation requires a new Corpus identity or revision.

The later Train Protocol decision can now assume that its Corpus reference resolves to a frozen, Task-compatible data asset and can focus only on how that data is consumed during training.

## Evidence

The current capture annotation store at `.local/recognition/capture_dataset/dataset.sqlite` is explicitly mutable. It contains campaign and capture state, detector outputs, `draft` and `complete` annotations, review provenance, and detector-refresh state.

The current classifier builder `tools/recognition/build_tile_classifier_dataset.py` materializes separate SQLite experiment corpora with a `sample` table containing split membership, class identity, image payload, and many source-specific columns including source partition, source image and annotation IDs, capture/layout identifiers, lighting attributes, annotation angle, and original dimensions. Its current class-label column is named `base_label`, so existing artifacts require an explicit adaptation or migration before they satisfy the new `target` column contract.

That existing shape demonstrates both sides of this ADR: a small stable classification core can be standardized, while useful source metadata varies substantially and is better retained as builder-owned additional columns than promoted into a universal MLDB Corpus contract.
