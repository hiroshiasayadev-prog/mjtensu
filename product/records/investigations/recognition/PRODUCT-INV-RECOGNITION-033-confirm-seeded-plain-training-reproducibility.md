# PRODUCT-INV-RECOGNITION-033: Confirm seeded Plain training reproducibility

- **status**: concluded
- **date**: 2026-10-06
- **trigger**: PRODUCT-INV-RECOGNITION-027 corrected the classifier training path so seed 42 controls initialization, but the selected Plain 0.5x late-DW model had not been independently rerun from the same sealed Plan on separate workers. Before relying on small architecture deltas, the project needed direct evidence that the corrected training path reproduces the same trained weights rather than merely similar metrics.
- **scope**: Execute the same sealed Plain 0.5x late DW3x3 + PW1x1 batch-128 random360 e150 Plan twice and compare final weight identity plus the eight canonical evaluation stages.
- **non_scope**: Multiple random seeds, heterogeneous GPU-family determinism, architecture selection, batch-size comparison, latency determinism, production promotion, or claims about every classifier Architecture.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-027
  - PRODUCT-INV-RECOGNITION-025
- **follow_up_candidates**:
  - Treat the corrected seeded training path as reproducible for this Plan across the two tested RTX 3060 workers.
  - Keep heterogeneous-GPU reproducibility as a separate boundary before using sub-percentage quality deltas for final model selection.

## Investigation scope

Test whether the corrected seeded classifier training path produces the same trained model when the exact same Study Plan is executed independently twice.

The selected Architecture is the Plain 0.5x late-capacity model with late DW3x3 refinement and PW1x1 expansion to 256 channels.

Training is fixed at batch 128, random360-only, 150 epochs, seed 42, AdamW learning rate 0.001, weight decay 0.0001, AMP enabled, and TF32 enabled.

## What was investigated

| evidence | value |
|---|---|
| Study | `tile-classifier/plain-w500-late-dw-seed-reproducibility-v1` |
| Study Plan | `tile-classifier/plain-w500-late-dw-seed-reproducibility-v1-plan-80085f0fdbf256ad` |
| Study Result A | `tile-classifier/run-c325ff0a0d0147c180b38f55d79a3409` |
| Study Result B | `tile-classifier/run-1f32498f05c547308a8e7b1dbe746c94` |
| source commit | `6ebd6602647a06117bd9549cc7507ae5931b18a2` |
| runtime registry | `8` |
| training worker A | `precision5820-gpu3060` |
| training worker B | `dev-wsl-gpu3060` |

Both runs completed the same eight evaluation stages:

- dense64 angle robustness;
- full-class diagnostic;
- validity rejection;
- Manzu diagnostic;
- full-class occlusion;
- ONNX CPU latency;
- iPhone ORT-Web latency;
- exhaustive all-real crop recall.

## Findings

### Final trained weights are bitwise identical

Both independent training runs produced a 736,286-byte state dict with the same SHA-256:

`59a607b36e20f23617fe2f820353a96b7fc6543e2a01ab74ceb2e81616d48207`

The artifact URIs differ because each execution produced its own canonical artifact location, but the weight bytes are identical.

### Quality evaluations reproduce

| metric | run A | run B |
|---|---:|---:|
| dense64 manual angle mean | 0.972674 | 0.972674 |
| full-class mean condition accuracy | 0.923556 | 0.923556 |
| full-class worst condition accuracy | 0.760000 | 0.760000 |
| validity balanced accuracy | 0.919483 | 0.919483 |
| Manzu mean condition accuracy | 0.904965 | 0.904965 |
| Manzu worst condition accuracy | 0.659574 | 0.659574 |
| all-real accuracy | 0.999830 | 0.999830 |
| all-real worst-class recall | 0.998782 | 0.998782 |

Classification/count metrics are identical at the recorded precision.

A few floating reduction statistics such as mean true margin differ only at the few-micro-unit level. Those differences do not change any observed prediction-count metric or the final weight identity.

### Latency is measurement evidence, not determinism evidence

CPU p50 differed between the two completed evaluations:

- run A: 0.3605 ms;
- run B: 0.3028 ms.

iPhone p50 was 1.0 ms in both runs and p95 was 2.0 ms in both runs; the recorded means were 1.175 ms and 1.155 ms.

Runtime latency therefore should continue to be treated as an environmental measurement rather than a deterministic output of the seeded training procedure.

### Cross-worker result

The two trainings ran on different RTX 3060 workers: `precision5820-gpu3060` and `dev-wsl-gpu3060`.

Within that tested hardware class, the corrected seeded path produced identical final weights across independent executions.

## Cross-cutting interpretation

PRODUCT-INV-RECOGNITION-027 established that uncontrolled initialization had invalidated some earlier causal comparisons.

This investigation strengthens the corrected path: the selected Plain late-DW condition is not merely statistically similar across two reruns; its final serialized weights are identical.

This does not prove bitwise reproducibility across different GPU families or every Architecture. PRODUCT-INV-RECOGNITION-032 already records a heterogeneous-worker boundary for C8-family runs.

## Recommendation

Use the seeded v6 training path as the controlling classifier training path for subsequent same-condition comparisons.

Do not spend additional runs repeating this exact Plan solely to re-establish determinism.

Do not use CPU or browser latency variation as evidence that seeded model training failed to reproduce.

Before a final model-selection decision based on very small quality deltas, pin worker hardware or explicitly repeat finalists on the same hardware class.

## Follow-up artifact candidates

- No new correction artifact is required for this exact Plain condition.
- A worker-pinned finalist comparison remains appropriate if future model selection hinges on sub-percentage quality differences.

## Open questions

- Does identical-weight reproducibility hold across RTX 3060 and RTX 3090 training for the same Architecture and source revision?
- Which finalist comparisons actually need multi-run confirmation once effect sizes become small?
