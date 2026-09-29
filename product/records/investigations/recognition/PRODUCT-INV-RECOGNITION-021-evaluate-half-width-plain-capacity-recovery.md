# PRODUCT-INV-RECOGNITION-021: Evaluate half-width Plain capacity recovery

- **status**: concluded
- **date**: 2026-09-30
- **trigger**: PRODUCT-INV-RECOGNITION-020 showed that replacing an existing full spatial convolution with depthwise-separable processing saves latency but does not recover enough quality.
- **scope**: Compare two additive capacity-recovery strategies on the `0.5x + late256` Plain backbone and measure their CPU and corrected iPhone latency.
- **non_scope**: Production promotion, detector changes, augmentation changes, multi-seed confirmation, quantization, and further speed-first micro-optimization.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-020
- **follow_up_candidates**:
  - Production model-selection judgment if this candidate family is later considered for promotion

## Investigation scope

Test whether the large speed advantage of the half-width Plain backbone can be retained while restoring quality through cheap additional processing.

Keep the existing half-width spatial stages intact.

Include the corrected iPhone latency re-evaluation because it measures the same three trained models. MLDB created a second Study only because it cannot append a new Evaluation to an existing StudyResult.

## Out of scope

- Production model promotion.
- Detector changes.
- Augmentation changes.
- Multi-seed statistical confirmation.
- Quantization.
- Further speed-first micro-optimization.

## Background

PRODUCT-INV-RECOGNITION-020 did not support further factorization of existing full spatial convolutions.

The remaining hypothesis was additive: preserve the favorable `16/32/64/96` topology and add cheap capacity only after expensive spatial resolution has already been reduced.

## What was investigated

Three trained conditions were compared.

```text
baseline:
  ... -> Conv3x3 64->96 -> PW1x1 96->256 -> GAP

late depthwise refinement:
  ... -> Conv3x3 64->96 -> DW3x3 96->96 -> PW1x1 96->256 -> GAP

stage3 temporary expansion:
  16x16: DW3x3 32->32 -> PW1x1 32->128
   8x8 : PW1x1 128->64 -> Conv3x3 64->96 -> PW1x1 96->256
```

| primary evidence | ref |
|---|---|
| MLDB Study | `tile-classifier/plain-capacity-recovery-screen-v1` |
| StudyResult | `tile-classifier/run-01fc07b6027d4f00a1c8dd15119779ce` |
| source commit | `6dd26897dc99d0ceea6b69b095af7104496f9d99` |

The original iPhone latency evaluation used protocol v1. Its single-run `performance.now()` timing collapsed into integer-millisecond buckets.

A sealed v2 protocol was therefore created. It measures `100` sequential runs per block across `40` blocks and reports per-run block-average p50/p95/mean.

The same trained models were re-evaluated without retraining.

| corrected latency evidence | ref |
|---|---|
| MLDB Study | `tile-classifier/plain-capacity-recovery-iphone-latency-v2-review-v1` |
| StudyResult | `tile-classifier/run-bde5ea6da58047eeab6760c1eac402e1` |
| source commit | `0d5bb0c7d6e57f360e30787763591598c49c99a1` |

## Findings

### Accuracy and CPU latency

| model | CPU p50 | angle mean | full front | full mean | full worst | full mean margin | Manzu mean | Manzu worst | Manzu margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.5x + late256` | `0.309 ms` | `0.952` | `0.956` | `0.881` | `0.680` | `4.83` | `0.845` | `0.468` | `2.89` |
| `0.5x + late DW3x3 + PW256` | `0.316 ms` | `0.962` | `0.967` | `0.906` | `0.702` | `6.23` | `0.888` | `0.596` | `5.27` |
| stage3 expand128/compress64 | `0.274 ms` | `0.943` | `0.942` | `0.867` | `0.673` | `4.69` | `0.861` | `0.660` | `3.10` |

The late depthwise refinement adds about `0.007 ms` CPU p50 relative to the same-run baseline.

The late refinement improves every listed global accuracy and margin metric.

The stage3 expansion is faster on CPU and has a strong Manzu worst-condition result. Its global accuracy and margin recovery are weaker.

### Corrected iPhone latency

The v1 run produced coarse samples, including impossible `0 ms` observations for the stage3 expansion candidate. The v1 p50 values are therefore unsuitable for fine ranking inside this fast model family.

The corrected v2 results are:

| model | iPhone v2 p50 | p95 | mean | p50 vs baseline |
|---|---:|---:|---:|---:|
| `0.5x + late256` | `1.07 ms` | `1.11 ms` | `1.076 ms` | baseline |
| `0.5x + late DW3x3 + PW256` | `1.12 ms` | `1.17 ms` | `1.131 ms` | about `+4.7%` |
| stage3 expand128/compress64 | `0.95 ms` | `1.00 ms` | `0.959 ms` | about `-11%` |

The apparent v1 `1 ms -> 2 ms` regression for late depthwise refinement was a timer-quantization artifact.

The corrected device cost is about `0.05 ms` p50 above the same-run baseline.

### Comparison with the original 1.0x Plain random360 reference

The closest 1.0x reference is the PRODUCT-INV-RECOGNITION-016 baseline from `tile-classifier/run-de1563ab21f7445fbcfa1fbea9e31d83`.

The comparison uses a different Study execution. Small differences are not evidence of statistical superiority.

| metric | Plain `1.0x` reference | `0.5x + late DW refinement` |
|---|---:|---:|
| CPU p50 | `0.966 ms` | `0.316 ms` |
| manual angle mean | `0.944` | `0.962` |
| full front accuracy | `0.951` | `0.967` |
| full mean condition accuracy | `0.882` | `0.906` |
| full worst condition accuracy | `0.711` | `0.702` |
| full mean condition margin | `5.12` | `6.23` |
| Manzu mean condition accuracy | `0.868` | `0.888` |
| Manzu worst condition accuracy | `0.511` | `0.596` |

The only listed metric below the 1.0x reference is full worst-condition accuracy. The difference is about `0.9` percentage point.

## Cross-cutting observations

Adding spatial processing at 8x8 is more effective than factorizing the earlier full spatial convolution tested in PRODUCT-INV-RECOGNITION-020.

The `16/32/64/96` topology preserves its large CPU advantage.

The corrected iPhone result shows that the late depthwise refinement has a small device-latency cost.

MAC count alone does not predict ORT Web/WASM ranking. The stage3 expansion candidate is faster than baseline on iPhone despite its added pointwise operations.

## Follow-up judgment candidates

- Whether `tile-plain-gray35-w500-late256-late-dw3-pw1-v1` should enter a later production model-selection process.
- Whether multi-seed confirmation is needed before any promotion judgment.
- Whether further architecture work should remain closed until a concrete quality or deployment failure appears.

## Recommendation

The late-DW half-width Plain candidate appears preferable within this experiment family.

Further speed-first simplification does not appear justified by the measured gains.

Any production adoption should be decided in a later artifact rather than in this investigation.

## Follow-up artifact candidates

- Production model-selection ADR or equivalent product judgment if promotion is opened.
- Additional investigation only if a concrete deployment or quality deficit requires new evidence.

## Open questions

- Does the late-DW result persist across additional seeds if statistical confirmation becomes necessary?
- Does the candidate preserve its advantage in the full production recognition pipeline?
