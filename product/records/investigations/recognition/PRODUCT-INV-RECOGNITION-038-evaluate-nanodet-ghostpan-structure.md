# PRODUCT-INV-RECOGNITION-038: Evaluate NanoDet GhostPAN structure

- **status**: concluded
- **date**: 2026-10-06
- **trigger**: The NanoDet detector still exhibited duplicate, merged, spurious, and bbox-explosion pathologies after data/training work. Before later batch-size studies fixed the GhostPAN baseline, a controlled architecture screen tested whether the neck fusion/downsample structure itself was the dominant cause.
- **scope**: Hold ShuffleNetV2 1.0x, NanoDet Plus head, assignment/loss, input size, corpus, seed, and training protocol fixed while comparing four GhostPAN variants: baseline depthwise downsample, full-convolution downsample, residual fusion, and two-block fusion.
- **non_scope**: Batch-size optimization, score-threshold tuning, NanoDet-R/OBB angle branches, data-recipe changes, classifier changes, functional-video behavior, or production promotion.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-010
  - PRODUCT-INV-RECOGNITION-024
- **follow_up_candidates**:
  - Keep the baseline GhostPAN as the controlled reference for training-recipe studies.
  - Treat two-block fusion as a possible later architecture revisit only if a same-worker rerun is justified.
  - Do not continue residual fusion or full-conv downsample without a new hypothesis.

## Investigation scope

Determine whether simple GhostPAN structural changes reduce detector pathology without sacrificing ordinary bbox precision/recall.

All candidates use the same 320x320 AABB NanoDet Plus detector and differ only in GhostPAN downsample/fusion structure.

## What was investigated

| evidence | ref |
|---|---|
| Study | `nanodet/ghostpan-screen-v1` |
| Study Plan | `nanodet/ghostpan-screen-v1-plan-c83bbbebd8106287` |
| Study Result | `nanodet/run-66eaf4d106b84f3eb57a7b5501501fc5` |
| ClearML controller | `3176a05fb6d541bdb754bfae369fb6ac` |
| source commit | `8d93406f6b3de83f84cfc38e578c41bae3478504` |
| runtime registry | `7` |

The four candidates were:

| candidate | controlled GhostPAN change |
|---|---|
| baseline | depthwise downsample, no residual fusion, one block |
| full-conv | replace depthwise downsample with full convolution |
| residual | depthwise downsample plus residual fusion |
| two-block | depthwise downsample, no residual fusion, two fusion blocks |

Training used 80 epochs, batch 16, validation every 5 epochs, and seed 42 on `nanodet/mahjong-composite-pathology-320-v1`. Evaluation used `nanodet/bbox-pathology-v2` at score threshold 0.35, NMS IoU 0.6, and the product duplicate-suppression pass.

## Findings

### Product-facing bbox metrics

| candidate | precision | recall | F1 | duplicate GT | spurious | affected images | positive count delta | bbox excess p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.975853 | **0.993220** | 0.984460 | **0.025424** | 0.007494 | **0.316456** | 0.164557 | 2 |
| full-conv | 0.969421 | **0.994068** | 0.981590 | 0.037288 | 0.003306 | **0.316456** | 0.177215 | 2 |
| residual | 0.940610 | 0.993220 | 0.966200 | 0.097458 | 0.011236 | 0.531646 | 0.430380 | 3 |
| two-block | **0.978279** | 0.992373 | **0.985276** | 0.036441 | **0.001671** | **0.316456** | **0.151899** | **1** |

Residual fusion is clearly harmful under this setup. It sharply increases duplicates, affected-image rate, positive count deltas, and bbox excess while reducing F1.

Full-convolution downsampling does not improve the overall trade. It slightly lowers spurious predictions, but loses precision/F1 and increases duplicate/count-excess behavior versus baseline.

Two-block fusion is the only variant that looks competitive. It gives the highest observed product-facing F1, lowest spurious prediction rate, lowest positive-count-delta rate, and lowest bbox-excess p95. However, its F1 gain over baseline is only about 0.0008 and its duplicate-GT rate is worse.

## Evidence boundary

Training was not hardware-pinned across all four candidates. Baseline and residual trained on RTX 3060 workers, while full-conv and two-block trained on RTX 3090. The large residual-fusion regression is still directionally clear, but the very small baseline-versus-two-block delta is not a hardware-isolated causal result.

This boundary is especially important because later NanoDet work chose the baseline architecture as the controlled reference while changing batch size and update budget. The GhostPAN screen therefore does not justify silently replacing the baseline with the two-block variant.

## Cross-cutting interpretation

Simple GhostPAN rewrites do not remove the detector pathology surface. Three of four candidates retain an affected-image rate of about 31.6% even after product suppression, and the residual variant is substantially worse.

The two-block candidate improves count-excess and spurious behavior enough to remain technically interesting, but the improvement is too small and too hardware-confounded to justify changing the reference architecture before training-recipe variables are controlled.

## Recommendation

Retain `nanodet/nanodet-plus-m320-ghostpan-baseline-v1` as the canonical reference architecture for subsequent batch-size and threshold studies.

Reject residual fusion and full-conv downsample as continuation paths from this screen.

Do not promote two-block fusion from this result alone. Revisit it only with a same-worker controlled comparison if later detector work shows that neck structure is again the highest-value lever.

## Open questions

- Does two-block fusion retain its small advantage under same-worker retraining with the corrected COCO corpus and current batch-24 recipe?
- Are the remaining merged/duplicate/meld-domain failures better addressed by training data and thresholding than by GhostPAN depth?
- Does the later region-semantic functional-video diagnostic change which detector pathology matters most for product behavior?
