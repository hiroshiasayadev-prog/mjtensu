# Contract: Study Plan format

- **id**: `spec:mldb.v2.study.plan_format`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.study`
- **contract_class**: `format`

## Meaning

Study Plan is the immutable fully expanded execution contract consumed by the application Study
driver and backend adapter. Schema is `mjtensu.mldb-v2/study-plan/v1`.

A Plan contains no backend Task IDs, queue state, retry state, or live progress. It contains only
committed source identity, exact canonical input pins, deterministic trial expansion, and complete
resolved public parameters.

## Top-level shape

```yaml
schema: mjtensu.mldb-v2/study-plan/v1
id: rotated-fcos/spatial-screen-v2-plan-0123456789abcdef
content_sha256: <64 lowercase hex>
study: rotated-fcos/spatial-screen-v2
source_commit: <full unabbreviated Git commit object id>
pins: []
trials: []
```
## Pin shape

Every source object needed to interpret or execute the Plan appears exactly once in `pins`:

```yaml
- kind: architecture
  id: rotated-fcos/stem-s1-v1
  yaml_sha256: <64 lowercase hex>
  companion_sha256: <64 lowercase hex or null>
  sources:
    - path: mldb_data/rotated-fcos/lib/backbone.py
      sha256: <64 lowercase hex>
  manifest_sha256: null
  manifest_entries: null
```

`kind` is one of `namespace`, `task`, `corpus`, `architecture`, `train_protocol`,
`evaluation_protocol`, `study`, `model`, or `training_result`. Namespace pins use their one-segment
Namespace ID; all other pins use typed canonical IDs.

`yaml_sha256` is always non-null and is SHA-256 of the exact canonical YAML bytes stored by Git at
`source_commit`; Namespace pins hash `namespace.yaml` while their `id` remains one segment.
`companion_sha256` is non-null only when the pinned entity owns a sibling executable/builder.
`sources` contains the sorted exact helper path/hash set declared by executable definitions; it is
empty when no namespace-private helpers are used. Corpus pins additionally carry their manifest
digest/count; non-Corpus pins use null.
## Trial shape

Training-source trial:

```yaml
- trial: trial-0001
  source:
    kind: training
    task: rotated-fcos/mahjong-obb-v1
    corpus: rotated-fcos/train-v3
    architecture: rotated-fcos/stem-s1-v1
    train_protocol: rotated-fcos/standard-v2
    parameters: {batch_size: 32, learning_rate: 0.001}
    seed: 42
  evaluations: []
```

Existing-Model trial:

```yaml
- trial: trial-0001
  source:
    kind: existing_model
    model: rotated-fcos/run-abcd-trial-0003-model
  evaluations: []
```

The two `source` variants are disjoint; fields from the other variant are invalid.
Each Evaluation coordinate is fully materialized:

```yaml
- coordinate: eval-0001
  stage: final-holdout
  task: rotated-fcos/mahjong-obb-v1
  corpus: rotated-fcos/human-obb-v3
  evaluation_protocol: rotated-fcos/human-obb-v2
  parameters: {iou_threshold: 0.5}
```

`parameters` mappings are complete resolved mappings, not sparse overrides. Trial and Evaluation
ordering is exactly `spec:mldb.v2.study.grid_expansion` and is semantically significant.

## Identity and canonical order

`pins` is ordered by kind in this fixed order: namespace, task, corpus, architecture,
train_protocol, evaluation_protocol, study, model, training_result; IDs are ascending Unicode
code-point order within one kind. Duplicate `(kind,id)` pairs are invalid.

The Plan ID is `<study-namespace>/<study-local-id>-plan-<sha256-prefix16>`. `content_sha256` is
computed by `spec:mldb.v2.common.identity` after excluding only `id` and `content_sha256`. There is
no compilation timestamp in canonical Plan content, so identical committed inputs always reproduce
byte-equivalent logical Plan content and the same ID.

The canonical path is `mldb_data/<study-namespace>/study_plans/<local-plan-id>.yaml`. A Plan is never
edited after creation. Formal planning and start both enforce `spec:mldb.v2.study.source_pinning`.
Readiness is derived only by `spec:mldb.v2.study.execution_readiness`.
