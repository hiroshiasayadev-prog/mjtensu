# PRODUCT-INV-RECOGNITION-017: Evaluate Plain spatial width and late-channel allocation

- **status**: concluded
- **date**: 2026-09-29
- **trigger**: PRODUCT-INV-RECOGNITION-016 showed that late 1x1 expansion can recover quality cheaply on narrowed Plain CNNs, but did not establish how spatial width and late channel capacity should be traded.
- **scope**: Broadly compare Plain spatial widths and late expansion widths to identify useful latency/quality regions.
- **non_scope**: Fine-width interpolation, stage-kernel ablation, depthwise substitution, production promotion, multi-seed confirmation, and iPhone acceptance.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-016
- **follow_up_candidates**:
  - PRODUCT-INV-RECOGNITION-018
- **follow_up_results**:
  - PRODUCT-INV-RECOGNITION-018

## Investigation scope

Evaluate how spatial backbone width and low-resolution late channel expansion trade off CPU latency, classification accuracy, and robustness.

Use a broad grid. Do not resolve small width increments inside this investigation.

## Out of scope

- Fine-grained width search between selected coarse points.
- Stage-level kernel replacement.
- Depthwise-separable substitution.
- Production model promotion.
- Multi-seed statistical confirmation.
- iPhone runtime acceptance.

## Background

PRODUCT-INV-RECOGNITION-016 found two useful patterns.

- Narrowing spatial channels is the strongest CPU latency lever.
- Late 1x1 expansion recovers substantial margin at low additional cost.

The remaining question was how much spatial width should be retained before late expansion.

## What was investigated

Representative conditions covered:

- Plain `1.0x`;
- `0.5x` with late widths `192`, `224`, and `256`;
- `0.625x` with late widths `192`, `224`, and `256`;
- `0.75x` with multiple late widths;
- `0.875x` with multiple late widths.

All trials used the same corpus family, `random360-only-v1`, 100 epochs, seed 42, and common evaluation protocols.

| evidence | ref |
|---|---|
| MLDB Study | `tile-classifier/plain-spatial-late-channel-grid-screen-v1` |
| StudyResult | `tile-classifier/run-1d73b84383734a2c9e8e9a282056bcaf` |
| source commit | `d2431e2a717393def4b9cb8835cd112c9a4c69e5` |

## Findings

| representative model | CPU p50 | angle mean | full mean acc. | full worst acc. | full mean margin |
|---|---:|---:|---:|---:|---:|
| Plain `1.0x` | `0.873 ms` | `0.946` | `0.901` | `0.749` | `5.65` |
| `0.5x` | `0.275 ms` | `0.905` | `0.808` | `0.560` | `3.04` |
| `0.5x + late192` | `0.370 ms` | `0.938` | `0.873` | `0.687` | `4.78` |
| `0.5x + late224` | `0.327 ms` | `0.942` | `0.881` | `0.684` | `5.00` |
| `0.5x + late256` | `0.331 ms` | `0.945` | `0.885` | `0.709` | `5.17` |
| `0.625x + late192` | `0.697 ms` | `0.958` | `0.908` | `0.756` | `5.77` |
| `0.75x + late192` | `0.707 ms` | `0.962` | `0.909` | `0.749` | `6.06` |
| `0.875x + late224` | `0.962 ms` | `0.966` | `0.925` | `0.784` | `6.54` |

Late capacity is substantially cheaper than restoring width across the full spatial backbone.

`0.5x + late256` retains a large CPU latency advantage and recovers most of the quality lost by plain `0.5x`.

Wider `0.625x-0.875x` candidates reach stronger single-seed quality, but CPU latency rises sharply.

## Cross-cutting observations

The coarse points do not show a smooth width-versus-quality knee.

The `0.5x` region remains unusually attractive because its latency is much lower than the next tested width.

The coarse screen cannot determine whether a small width increase above `0.5x` preserves that latency advantage.

## Follow-up judgment candidates

- Whether the region immediately above `0.5x` contains a better latency/quality point.
- Whether the observed latency gap is smooth or caused by backend-sensitive channel shapes.

## Recommendation

A dedicated fine-width screen between `0.5x` and `0.625x` appears preferable before selecting a wider backbone.

## Follow-up artifact candidates

- PRODUCT-INV-RECOGNITION-018 for fine-width refinement.

## Open questions

- Is there a useful spatial-width knee immediately above `0.5x`?
- Does CPU latency scale smoothly across small width increments?
