# PRODUCT-INV-RECOGNITION-022: Audit legacy COCO annotation semantics with the production tile classifier

- **status**: completed
- **date**: 2026-10-03
- **trigger**: The detector source corpus is collapsed to one `mahjong_tile` class for NanoDet, but the original COCO sources still carry per-tile semantics and `data/coco_mahjong` was suspected to contain wrong tile labels.
- **scope**: Inspect the original COCO category schemas and detector-source provenance; use the actual deployed tile classifiers as a non-authoritative semantic audit signal over every source GT bbox; produce review manifests/contact sheets; and define a guarded path for writing new corrected COCO JSON only after explicit human approval.
- **non_scope**: Automatically relabel source ground truth, mutate original annotation JSON, retrain a detector, or modify MLDB infrastructure.
- **source_refs**:
  - `product/frontend/src/recognition/production-pipeline.ts`
  - `PRODUCT-INV-RECOGNITION-005`
  - `PRODUCT-INV-RECOGNITION-006`
  - `PRODUCT-INV-RECOGNITION-007`
  - `PRODUCT-INV-RECOGNITION-014`
  - `tools/recognition/build_nanodet_single_class_coco_dataset.py`
  - `tools/recognition/build_tile_crop_dataset.py`
  - `tools/recognition/audit_coco_mahjong_annotations.py`

## Production classifier authority

The repository development model set is not the deployment authority for this audit. The actual production checkout under `/persist/srv-bugrat/mjtensu-product/src`, together with INV-014, binds the deployed base classifier to:

- `tile-c8-gray35-v3-jp189.onnx`
- SHA-256 `b8a8fa3ff6c6d1e944a7593fa0afc947e0cd2513fb79ca46e5f8fcd6e19c97d0`
- runtime contract `c8-tile-35-v1`
- labels: 34 base tile identities plus `invalid`

The deployed red-five specialist is:

- `red-five-c8-rgb-warmaug.onnx`
- SHA-256 `c2b780f682d84bf186db90290050f8b05016c3e8058de559eea679a28eeb80c6`
- runtime contract `c8-red-five-v1`

The audit checks those ONNX hashes and, when CUDA is available, uses the provenance-linked production checkpoints through the same ordinary-Tensor C8 export module used to create the deployed ONNX. Canonical export metadata records source-to-export parity and ONNX Runtime parity.

## Detector source provenance

`.local/recognition/nanodet_single_class_dataset/provenance.json` confirms that detector training is generated directly from the following original sources before all categories are collapsed to `mahjong_tile`:

| source | generated split | images | annotations |
|---|---|---:|---:|
| `coco_mahjong/train2017` | train | 1,709 | 11,181 |
| `coco_mahjong_jp_v2/train` | train | 12,144 | 1,207,281 |
| `coco_mahjong/val2017` | val | 427 | 3,352 |
| `coco_mahjong_jp_v2/valid` | val | 722 | 71,745 |
| `coco_mahjong_jp_v2/test` | test | 370 | 36,925 |

No separate corrected COCO annotation file was found under either source dataset. The audit therefore treats the five original JSON files as the source records and never rewrites them.

## Source category schemas

For `coco_mahjong`, the canonical mapping is:

- `circle_1..9 -> 1p..9p`
- `bamboo_1..9 -> 1s..9s`
- `character_1..9 -> 1m..9m`
- IDs 28..34 -> north, south, west, east, green, red, white

Both train and val category arrays contain duplicate ID 34 / `white`; the duplicate is semantically identical. The train JSON also defines odd IDs through 48 that duplicate existing semantic categories. Twenty-three train annotations and seven val annotations actually use those odd IDs. All 30 used odd-ID crops were visually reviewed and matched the duplicated semantic category, so the odd IDs are schema/taxonomy anomalies rather than label errors by themselves.

For `coco_mahjong_jp_v2`, all 75 category IDs are unique. ID 0 is the unused parent-like `mahjong-tiles` category. The remaining schema contains both:

- numeric names `0..36`, mapped with the existing visually verified `JP_NUMERIC_TILE_LABELS`; and
- explicit tile names, including `5mr/5pr/5sr`.

Thus the same 37 semantic tile identities appear in two source-category families. Correction output preserves the original numeric-vs-named family when selecting a replacement category ID.

## Audit method

For every source annotation the tool:

1. resolves the original image and validates its COCO dimensions;
2. extracts the GT crop with the detector-source policy: floor left/top, ceil right/bottom, clipped to the image;
3. reproduces production 64x64 classifier preprocessing: runtime-equivalent Lanczos resizing, median-border letterbox, runtime grayscale conversion, and deployed normalization;
4. runs the production base classifier in batches;
5. records top-1 label/confidence, top-2 margin, second label/confidence, bbox/crop diagnostics, and source SHA-256;
6. for jp_v2 only, runs the production red-five specialist when the base result is 5m/5p/5s.

Legacy `coco_mahjong` has no red-five semantic class, so red-five identity is intentionally not introduced into its mismatch decision.

The review buckets are explicit:

- **high-confidence mismatch**: label disagreement with confidence >= 0.90 and top-2 margin >= 0.25;
- **ambiguous mismatch**: all other label disagreements;
- **low-confidence match**: labels agree but confidence < 0.60 or margin < 0.10;
- **confident match**: remaining agreements.

These thresholds rank review candidates; they are not claims that the classifier is ground truth.

## Full-corpus results

| source split | checked | high-conf mismatch | ambiguous mismatch | low-conf match | confident match |
|---|---:|---:|---:|---:|---:|
| `coco_mahjong/train2017` | 11,181 | 505 | 3,850 | 881 | 5,945 |
| `coco_mahjong/val2017` | 3,352 | 145 | 1,172 | 240 | 1,795 |
| `jp_v2/train` | 1,207,281 | 55 | 122 | 60 | 1,207,044 |
| `jp_v2/valid` | 71,745 | 5 | 17 | 8 | 71,715 |
| `jp_v2/test` | 36,925 | 0 | 4 | 2 | 36,919 |
| **total** | **1,330,484** | **710** | **5,165** | **1,191** | **1,323,418** |

The combined total is useful for accounting only. The two source families have radically different classifier-domain behavior and must not be interpreted from one aggregate error rate.

### Legacy `coco_mahjong`

The production classifier has a strong domain shift on this older tile artwork. Common disagreement patterns include `white -> invalid`, `south -> green`, `west -> south`, `1m -> invalid`, and `9s -> 6s`. Contact-sheet review found many cases where the source label is visibly correct despite very high classifier confidence.

The same review also found clear source-label defects. Examples from `val2017` include:

- annotation 2602: source `3s`, crop visibly `6s`;
- annotation 842: source `1p`, crop visibly `4p`;
- annotation 2598: source `5m`, crop visibly `5s`;
- annotation 3213: source `8m`, crop visibly `8p`.

Therefore the suspected legacy annotation problem is real, but the 650 high-confidence mismatches cannot be bulk-approved. The classifier is useful here as a candidate generator, not as a relabeling oracle.

### `coco_mahjong_jp_v2`

Only 60 of 1,315,951 annotations entered the high-confidence mismatch bucket. Visual review of the leading candidates found repeated, obvious source errors, for example:

- train annotation 433696: source west, crop visibly north;
- train annotation 205617: source `4m`, crop visibly west;
- train annotation 178676: source `9m`, crop visibly north;
- train annotation 48360: source `5m`, crop visibly `4m`;
- valid annotation 29613: source `4m`, crop visibly `6m`;
- valid annotation 68754: source `7m`, crop visibly `6m`.

This bucket is much cleaner than the legacy dataset, but it is still review-only. The current classifier was trained using a sampled jp_v2-derived corpus, so jp_v2 is not an entirely independent evaluation population.

## Human-review and correction workflow

Audit mode writes:

- `flagged_samples.csv` and `flagged_samples.jsonl`;
- ranked contact sheets with source context, GT bbox, crop, labels, confidence and margin;
- `correction_review_template.csv`;
- `summary.json`.

The correction template contains no approval by default. A reviewer must explicitly set `approve` and enter `approved_label`. The proposed classifier label is kept separate from the human-approved field.

`apply-corrections` additionally requires the source annotation SHA-256 from the audit manifest to match the current source JSON. It refuses to target the original source path, writes a new compact COCO JSON, and emits a correction sidecar recording every changed annotation.

No corrected COCO JSON was produced by this investigation because no human-approved correction manifest was supplied.

## Recommended corrected-corpus path

The evidence is strong enough to justify a corrected detector-training corpus, but only through review:

1. Review the high-confidence candidates first, then ambiguous candidates where useful, using the contact sheets and manifest.
2. Record explicit human approvals and corrected semantic labels.
3. Run `apply-corrections` separately for each source split to new versioned COCO JSON files; keep the originals immutable.
4. Preserve the source SHA and generated correction sidecar with that corpus version.
5. Rebuild the single-class NanoDet dataset from those reviewed source JSONs, so detector geometry stays unchanged while the corrected source provenance remains reproducible.

The legacy source should not be auto-corrected from classifier confidence because the deployment classifier is visibly out of domain for many of its tile designs. The jp_v2 candidate set is much smaller and visually higher precision, but still requires the same approval gate.

## Audit artifacts

Legacy source audit:

```text
.local/recognition/coco_annotation_audit/coco_mahjong-production-c8/
  summary.json
  flagged_samples.csv
  flagged_samples.jsonl
  correction_review_template.csv
  contact_sheets/
  visual_spotcheck.json
```

jp_v2 audit:

```text
.local/recognition/coco_annotation_audit/coco_mahjong_jp_v2-production-c8/
  summary.json
  local_artifact_index.json
  flagged_samples.csv
  flagged_samples.jsonl
  correction_review_template.csv
  contact_sheets/
  visual_spotcheck.json
```

Combined accounting summary:

```text
.local/recognition/coco_annotation_audit/combined-production-c8-summary.json
```
