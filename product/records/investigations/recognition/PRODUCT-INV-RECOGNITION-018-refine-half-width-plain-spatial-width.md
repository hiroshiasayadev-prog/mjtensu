# PRODUCT-INV-RECOGNITION-018: Refine Plain spatial width above 0.5x

- **status**: concluded
- **date**: 2026-09-29
- **trigger**: PRODUCT-INV-RECOGNITION-017 found the `0.5x + late256` region attractive enough to justify a finer width search before accepting the latency of `0.625x` and wider backbones.
- **scope**: Resolve the `0.5x` to `0.625x` spatial-width region at small increments while keeping late capacity approximately fixed.
- **non_scope**: Stage-kernel changes, depthwise substitution, production promotion, multi-seed confirmation, and iPhone runtime acceptance.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-017
- **follow_up_candidates**:
  - PRODUCT-INV-RECOGNITION-019
- **follow_up_results**:
  - PRODUCT-INV-RECOGNITION-019

## Investigation scope

Test whether modest channel-width increases above the `16/32/64/96` half-width schedule produce a useful intermediate latency/quality point.

## Out of scope

- Stage-level kernel replacement.
- Depthwise-separable substitution.
- Production model promotion.
- Multi-seed statistical confirmation.
- iPhone runtime acceptance.

## Background

The coarse grid left a large latency gap between `0.5x` and `0.625x`.

`0.5x + late256` was already competitive enough in quality to justify testing small spatial-width increases before moving to a wider backbone.

## What was investigated

The main late256 series covered approximately:

- `0.50000x`: `16/32/64/96`;
- `0.53125x`: `17/34/68/102`;
- `0.56250x`;
- `0.59375x`: `19/38/76/114`;
- `0.62500x`.

A `0.625x + late192` point was retained as a nearby late-capacity comparison.

| evidence | ref |
|---|---|
| MLDB Study | `tile-classifier/plain-spatial-width-fine-screen-v1` |
| StudyResult | `tile-classifier/run-bc6fe4f59d1946daa7dbd95f8ac85d1c` |
| source commit | `be302e6fa078f030d7f7342e14d31ef4f676e27b` |

## Findings

| model | representative channels | CPU p50 | full mean acc. | full worst | Manzu mean |
|---|---|---:|---:|---:|---:|
| `0.5x + late256` | `16/32/64/96` | `0.302 ms` | `0.889` | `0.722` | `0.882` |
| `0.53125x + late256` | `17/34/68/102` | `0.541 ms` | `0.875` | `0.700` | `0.861` |
| `0.5625x + late256` | slightly wider | `0.580 ms` | `0.887` | `0.693` | `0.871` |
| `0.59375x + late256` | `19/38/76/114` | `0.605 ms` | `0.902` | `0.733` | `0.901` |
| `0.625x + late256` | wider again | `0.711 ms` | `0.896` | `0.747` | `0.855` |
| `0.625x + late192` | same spatial width | `0.672 ms` | `0.885` | `0.713` | `0.850` |

The largest finding is the latency discontinuity immediately above `0.5x`.

Moving from `16/32/64/96` to `17/34/68/102` increases CPU p50 from about `0.302 ms` to `0.541 ms` without a corresponding quality gain.

The exact ORT/MLAS kernel cause was not profiled.

## Cross-cutting observations

The useful region does not behave like a smooth width-versus-latency curve.

Backend-sensitive channel shapes appear important enough to dominate small arithmetic changes.

The empirical result supports preserving `16/32/64/96` as the efficiency reference.

## Follow-up judgment candidates

- Whether structural simplification can reduce cost while retaining the favorable half-width channel schedule.
- Which spatial stages require full 3x3 mixing.

## Recommendation

Further fractional-width interpolation does not appear useful.

A stage-level spatial-mixing ablation on `0.5x + late256` appears preferable.

## Follow-up artifact candidates

- PRODUCT-INV-RECOGNITION-019 for stage spatial-mixing ablation.

## Open questions

- Which stages require neighboring-pixel mixing?
- Can any 3x3 stage be replaced by 1x1 channel mixing without material quality loss?
