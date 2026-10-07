# PRODUCT-INV-RECOGNITION-037: Localize C8 narrow separable regression by stage

- **status**: superseded
- **date**: 2026-10-06
- **trigger**: PRODUCT-INV-RECOGNITION-036 cut C8 narrow iPhone p50 from about 3.44 ms to 1.69 ms by making both stages 3 and 4 field-wise DW3x3 + equivariant PW1x1, but produced a large Manzu and full-class quality regression. A controlled stage-local study was required before rejecting the separable idea entirely.
- **scope**: Compare stage3-only and stage4-only field-wise DW3x3 + equivariant PW1x1 substitutions under one two-trial Study, with all other architecture, training, seed, and evaluation conditions held fixed.
- **non_scope**: Combined late2 retest, earlier-stage substitution, further width changes, quantization, multi-seed confirmation, detector work, or production promotion.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-032
  - PRODUCT-INV-RECOGNITION-033
  - PRODUCT-INV-RECOGNITION-036
- **follow_up_candidates**:
  - Keep stage3-only as the selective-separable reference candidate.
  - Do not spend another run on stage4-only without a materially different hypothesis.
  - If classifier latency remains too high, compare stage3-only against narrower C8 field schedules rather than combining stage3 and stage4 separation again.
- **follow_up_results**:
  - PRODUCT-INV-CLASSIFIER-002

## Investigation scope

Identify which late C8 narrow stage is responsible for the accuracy loss seen when both stages are made field-wise depthwise-separable.

Trial 1 changes only stage 3 from the C8 narrow baseline. Trial 2 changes only stage 4. Both preserve C8 fields 4/8/16/32, GroupPooling, global average pooling, and the 35-logit classifier head.

## What was investigated

| evidence | ref |
|---|---|
| Study | `tile-classifier/c8-narrow-stage3-stage4-dw3-pw1-random360-e150-full-eval-v1` |
| Study Plan | `tile-classifier/c8-narrow-stage3-stage4-dw3-pw1-random360-e150-full-eval-v1-plan-a1606e5b1e7aae34` |
| Study Result | `tile-classifier/run-626b78a7fc4d4150b3e86260143b1d9b` |
| ClearML controller | `5d05c851172c4d828c6b5f980464402b` |
| stage3 Architecture | `tile-classifier/tile-c8-gray35-narrow-stage3-dw3-pw1-v1` |
| stage4 Architecture | `tile-classifier/tile-c8-gray35-narrow-stage4-dw3-pw1-v1` |
| source commit | `be9e51c8a3e7f2f529507347af08f57f389cf3cc` |
| runtime registry | `8` |

Training controls match PRODUCT-INV-RECOGNITION-036 and the accepted C8 narrow recipe: 150 epochs, effective batch 128, AdamW 0.001 / 0.0001, `random360-only-v1`, seed 42, AMP/TF32 enabled.

Training workers were both RTX 3060 class: stage3 on `precision5820-gpu3060`, stage4 on `dev-wsl-gpu3060`. PRODUCT-INV-RECOGNITION-033 established identical-weight reproducibility for one corrected seeded Plain Plan across these two RTX 3060 workers, but this investigation still treats small architecture deltas cautiously.

Both trials completed training and all eight evaluation stages.

## Findings

### Stage 3 preserves much more quality than stage 4

| metric | C8 narrow baseline | stage3-only | stage4-only |
|---|---:|---:|---:|
| dense64 manual angle mean | 0.979201 | **0.983437** | 0.979549 |
| full-class mean condition accuracy | 0.957037 | **0.962519** | 0.931111 |
| full-class worst condition accuracy | 0.844444 | **0.893333** | 0.777778 |
| Manzu mean condition accuracy | **0.973050** | 0.964539 | 0.933333 |
| Manzu worst condition accuracy | **0.893617** | 0.829787 | 0.787234 |
| validity balanced accuracy | **0.933911** | 0.903858 | 0.917090 |
| all-real accuracy | 0.999855 | **0.999893** | 0.999731 |
| all-real worst-class recall | 0.998808 | **0.998834** | 0.998627 |

Stage3-only is close to or better than the C8 narrow baseline on dense-angle, full-class, and all-real metrics. It still regresses Manzu worst-condition accuracy and validity rejection, so it is not a free replacement.

Stage4-only is materially worse on the two diagnostics that already failed in the combined late2 experiment. Its full-class mean and worst-condition scores, plus Manzu mean and worst-condition scores, move strongly toward the combined late2 failure profile.

This localizes most of the separable quality damage to stage 4 rather than stage 3.

### Both selective variants retain useful latency savings

| metric | C8 narrow baseline | stage3-only | stage4-only | combined late2 |
|---|---:|---:|---:|---:|
| CPU p50 | 0.9170 ms | 0.7318 ms | **0.7010 ms** | 0.5526 ms |
| iPhone p50 | 3.435 ms | 2.550 ms | **2.520 ms** | 1.690 ms |
| iPhone p95 | 3.520 ms | 2.690 ms | **2.590 ms** | 1.830 ms |

Stage3-only retains roughly one quarter of the original C8 narrow iPhone latency reduction opportunity while avoiding most of the stage4-associated quality loss.

Stage4-only is only about 0.03 ms faster than stage3-only at iPhone p50. That negligible speed advantage does not justify its much worse full-class and Manzu behavior.

## Cross-cutting interpretation

The late C8 narrow stages are not interchangeable. Stage 4 is substantially more sensitive to replacing the full equivariant 3x3 with field-wise DW3x3 plus equivariant PW1x1.

The combined late2 result from PRODUCT-INV-RECOGNITION-036 is therefore not evidence that field-wise separation is generally incompatible with C8 narrow. It is evidence that applying it indiscriminately to both late stages is too aggressive.

Stage3-only becomes the useful selective-separable reference. It improves latency materially and preserves the strongest rotation/full-class/all-real behavior, while exposing narrower residual weaknesses in Manzu and validity rejection that can be compared against other speed-reduction strategies.

## Recommendation

Reject stage4-only as the primary continuation path.

Do not return to the combined stage3+stage4 late2 Architecture without a new mechanism specifically addressing stage4 quality loss.

Keep stage3-only as the current selective depthwise-separable C8 narrow candidate for future latency/quality comparison. Compare it against further C8 field-width reduction before deciding which route should become the production candidate.

## Evidence boundary

The completed two-trial Study and all eighteen child Tasks are present in ClearML. At the time this Investigation was written, the Study/Plan/Architecture definitions are recoverable from source commit `be9e51c8a3e7f2f529507347af08f57f389cf3cc`, while the completed StudyResult YAML is not present in the current working-tree `mldb_data` snapshot. The immutable StudyResult and controller IDs above are preserved as evidence.

## Open questions

- Can stage3-only recover the remaining Manzu/validity gap with a small targeted change rather than restoring the full stage3 convolution?
- Does additional C8 width reduction beat stage3-only on the same quality/latency frontier?
- Is the stage4 sensitivity caused mainly by loss of spatial mixing, loss of cross-field mixing before the final group pooling, or both?
