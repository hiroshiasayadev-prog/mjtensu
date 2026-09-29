# PRODUCT-INV-RECOGNITION-020: Evaluate stage2 depthwise-separable substitution

- **status**: concluded
- **date**: 2026-09-29
- **trigger**: PRODUCT-INV-RECOGNITION-019 found stage2 to be the least damaging place to remove a full 3x3, but pure 1x1 replacement still reduced Manzu accuracy and margin.
- **scope**: Compare the unchanged `0.5x + late256` stage2 full convolution against a stage2 `DW3x3 + PW1x1` factorization.
- **non_scope**: Factorizing stage3 or stage4, adding late spatial refinement, width changes, production promotion, and multi-seed confirmation.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-019
- **follow_up_candidates**:
  - PRODUCT-INV-RECOGNITION-021
- **follow_up_results**:
  - PRODUCT-INV-RECOGNITION-021

## Investigation scope

Test whether stage2 can retain useful spatial filtering with a cheaper depthwise-separable block.

Replace only the full `3x3 16->32` stage2 convolution.

## Out of scope

- Factorizing stage3 or stage4.
- Adding new late spatial processing.
- Changing channel width.
- Production model promotion.
- Multi-seed statistical confirmation.

## Background

PRODUCT-INV-RECOGNITION-019 showed that a pure stage2 1x1 replacement is the least damaging spatial-mixing ablation.

The 1x1 replacement removes all neighboring-pixel processing. A depthwise 3x3 followed by a pointwise 1x1 restores per-channel spatial filtering while retaining a cheaper channel mixer.

## What was investigated

The candidate replaces:

```text
Conv3x3 16->32
```

with:

```text
DW3x3 16->16, groups=16
PW1x1 16->32
```

All later stages, pooling, late256 expansion, head, training conditions, and evaluations remain unchanged.

| evidence | ref |
|---|---|
| MLDB Study | `tile-classifier/plain-stage2-depthwise-separable-screen-v1` |
| StudyResult | `tile-classifier/run-93c70cce7fae4c05b4d211326c147130` |
| source commit | `e2b6b03ed095d43bc9fb784604ad5e7a4fd2f02e` |

## Findings

| model | CPU p50 | angle mean | full mean acc. | full worst | full mean margin | Manzu mean | Manzu worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| `0.5x + late256` | `0.302 ms` | `0.949` | `0.886` | `0.687` | `5.18` | `0.840` | `0.511` |
| stage2 `DW3x3 + PW1x1` | `0.250 ms` | `0.936` | `0.863` | `0.662` | `4.05` | `0.799` | `0.574` |

The factorized stage is faster on CPU.

The candidate does not consistently recover the quality of the unchanged full convolution.

Some worst-case behavior improves relative to the pure stage2 1x1 ablation, but global mean accuracy and margin remain below the baseline.

## Cross-cutting observations

The arithmetic saving is much larger than the measured whole-model latency saving.

Other layers and backend/operator overhead remain significant.

Replacing an existing full spatial convolution is less attractive than preserving the fast half-width topology and spending a small amount of extra compute where it is cheapest.

## Follow-up judgment candidates

- Whether low-resolution spatial refinement can recover quality without replacing an existing full convolution.
- Whether temporary channel expansion provides a better quality/cost tradeoff.

## Recommendation

Further stage replacement does not appear preferable.

An additive capacity-recovery experiment on the unchanged half-width backbone appears more useful.

## Follow-up artifact candidates

- PRODUCT-INV-RECOGNITION-021 for additive half-width capacity recovery.

## Open questions

- Is late 8x8 depthwise refinement cheap enough to improve margin without losing the half-width latency advantage?
- Does temporary stage3 channel expansion recover more quality?
