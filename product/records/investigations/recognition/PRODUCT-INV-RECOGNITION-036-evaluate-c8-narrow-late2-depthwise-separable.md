# PRODUCT-INV-RECOGNITION-036: Evaluate late-stage depthwise-separable C8 narrow

- **status**: superseded
- **date**: 2026-10-06
- **trigger**: PRODUCT-INV-RECOGNITION-032 showed that C8 narrow retained most of the C8 robustness while reducing iPhone p50 from 13.29 ms to about 3.44 ms. The next question was whether the two late expensive equivariant 3x3 stages could be replaced by field-wise DW3x3 plus equivariant PW1x1 to approach Plain-class latency without discarding C8 symmetry.
- **scope**: Replace both stage 3 and stage 4 3x3 equivariant convolutions of C8 narrow with field-wise grouped DW3x3 followed by equivariant PW1x1, while keeping the accepted random360 e150 seed42 training recipe and full eight-stage evaluation suite fixed.
- **non_scope**: Stage-local attribution, further width reduction, C4/C2 comparison, quantization, detector changes, production promotion, or multi-seed architecture confirmation.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-032
  - PRODUCT-INV-RECOGNITION-033
- **follow_up_candidates**:
  - Localize the observed quality loss by testing stage3-only and stage4-only substitution separately.
  - Retain the same C8 narrow field schedule while isolating which late stage is sensitive to field-wise separation.
- **follow_up_results**:
  - PRODUCT-INV-CLASSIFIER-001

## Investigation scope

Test whether both late C8 narrow stages can be made field-wise depthwise-separable while preserving the quality profile that made C8 narrow attractive.

The candidate keeps C8 fields 4/8/16/32 and changes only stages 3 and 4. Each affected stage uses field-wise grouped equivariant DW3x3 followed by equivariant PW1x1, with equivariant BatchNorm/ReLU after both operations. GroupPooling, global average pooling, and the classifier head remain unchanged.

## What was investigated

| evidence | ref |
|---|---|
| Study | `tile-classifier/c8-narrow-late2-dw3-pw1-random360-e150-full-eval-v1` |
| Study Plan | `tile-classifier/c8-narrow-late2-dw3-pw1-random360-e150-full-eval-v1-plan-8ed3b75da8913b03` |
| Study Result | `tile-classifier/run-10ddcc8ecfb9481c80133ffc5ebb8c94` |
| ClearML controller | `1d9f9878c53c4a958c9ec9ab93fe5a47` |
| Architecture | `tile-classifier/tile-c8-gray35-narrow-late2-dw3-pw1-v1` |
| source commit | `21156c61c7f0acff85260bca00eb12cc3ea8d9c0` |
| runtime registry | `8` |
| training worker | `dev-wsl-gpu3060` / RTX 3060 |

Training controls match the accepted C8 narrow comparison recipe:

- corpus: `tile-classifier/gray35-jp500-seed42-v3-jp189-v1`;
- Train Protocol: `tile-classifier/tile-shape-train-gpu-v6`;
- 150 epochs, effective batch 128;
- AdamW learning rate 0.001, weight decay 0.0001;
- `random360-only-v1` augmentation;
- seed 42;
- AMP and TF32 enabled.

The run completed training plus the same eight evaluation stages used by PRODUCT-INV-RECOGNITION-032.

## Findings

### Latency improves substantially

| metric | C8 narrow baseline | late2 DW3/PW1 | direction |
|---|---:|---:|---|
| CPU p50 | 0.9170 ms | **0.5526 ms** | about 40% lower |
| iPhone p50 | 3.435 ms | **1.690 ms** | about 51% lower |
| iPhone p95 | 3.520 ms | **1.830 ms** | about 48% lower |

Replacing both late equivariant 3x3 stages therefore removes roughly half of the browser inference latency remaining after the original C8 width reduction.

### The latency gain is not free

| metric | C8 narrow baseline | late2 DW3/PW1 |
|---|---:|---:|
| dense64 manual angle mean | **0.979201** | 0.967813 |
| full-class mean condition accuracy | **0.957037** | 0.930815 |
| full-class worst condition accuracy | **0.844444** | 0.791111 |
| Manzu mean condition accuracy | **0.973050** | 0.889362 |
| Manzu worst condition accuracy | **0.893617** | 0.702128 |
| validity balanced accuracy | **0.933911** | 0.903858 |
| all-real accuracy | **0.999855** | 0.999702 |
| all-real worst-class recall | **0.998808** | 0.998316 |

The strongest regression is in the Manzu diagnostic. Mean condition accuracy falls by more than eight percentage points and worst-condition accuracy falls by about nineteen points.

The degradation is broad enough that the combined stage3+stage4 substitution should not replace the plain C8 narrow baseline solely for its latency gain.

## Cross-cutting interpretation

The result shows that equivariant late-stage spatial mixing is a major part of C8 narrow inference cost, but at least one of the two late stages is also carrying important discriminative capacity.

The combined modification is therefore useful as a latency bound, not as the preferred architecture. It demonstrates that C8 symmetry itself is not the only source of cost: the form of the late equivariant spatial convolutions matters materially.

Because the candidate changes two stages at once, this run cannot identify whether stage 3, stage 4, or their interaction causes most of the quality loss. That attribution requires a separate controlled study and is recorded in PRODUCT-INV-RECOGNITION-037.

## Recommendation

Do not promote the combined late2 DW3/PW1 Architecture.

Keep the 1.69 ms iPhone p50 as evidence that a substantially faster C8-family classifier is feasible.

Split the modification by stage before making any further architecture decision. A useful candidate must recover much of the original C8 narrow robustness while preserving a meaningful fraction of the observed latency reduction.

## Evidence boundary

The completed StudyResult is visible in ClearML and all nine child Tasks completed. At the time this Investigation was written, the Study/Plan/Architecture definitions are recoverable from source commit `21156c61c7f0acff85260bca00eb12cc3ea8d9c0`, while the completed StudyResult YAML is not present in the current working-tree `mldb_data` snapshot. This Investigation preserves the immutable StudyResult ID and ClearML controller ID rather than fabricating a local result file.

## Open questions

- Which of stage 3 or stage 4 is responsible for most of the robustness loss?
- Can one late-stage substitution reach roughly 2.5 ms iPhone p50 while retaining baseline-quality Manzu and validity behavior?
- Is further field-width reduction still worthwhile after selective late-stage separation?
