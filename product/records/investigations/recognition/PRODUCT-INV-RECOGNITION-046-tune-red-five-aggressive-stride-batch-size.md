# PRODUCT-INV-RECOGNITION-046: Tune RED5 aggressive-stride training batch size

- **status**: concluded
- **date**: 2026-10-16
- **trigger**: PRODUCT-INV-RECOGNITION-045 selected the dense stride-aggressive topology but also showed that the batch-128 heuristic from the base classifier transferred poorly to RED5. A dedicated batch sweep was required before judging the new topology's quality ceiling.
- **scope**: Hold the RED5 dense stride-aggressive fields 6/12/24/48 architecture, corpus, seed 42, 50-epoch schedule, optimizer hyperparameters, augmentation, checkpoint selection, and exhaustive evaluation fixed; vary only training batch across 256, 512, 1024, and 2048, using the prior batch-128 run as the reference.
- **non_scope**: Architecture changes; latency remeasurement for identical inference graphs; learning-rate scaling; multi-seed confirmation; or production promotion.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-023
  - PRODUCT-INV-RECOGNITION-027
  - PRODUCT-INV-RECOGNITION-045
- **follow_up_candidates**:
  - Use batch 256 for subsequent stride-aggressive RED5 architecture experiments.
  - Do not transfer base-classifier batch conclusions to RED5 without direct evidence.

## Investigation scope

Identify the preferred optimizer-step batch for the stride-aggressive RED5 architecture.

The inference graph is identical across all batch conditions. Therefore the Study evaluates quality only; CPU/iPhone latency from PRODUCT-INV-RECOGNITION-045 remains applicable to the dense stride-aggressive graph.

## What was investigated

| evidence | ref |
|---|---|
| Study | `tile-classifier/red-five-c8-rgb-full-stride-aggressive-batch-sweep-v1` |
| Study Plan | `tile-classifier/red-five-c8-rgb-full-stride-aggressive-batch-sweep-v1-plan-bedb46278f1e217b` |
| Study Result | `tile-classifier/run-d9f16289572e4f8ca8b144056c90088e` |
| source commit | `d4a65f4a842b86c8b4f8c90791294d2324a7ed7c` |
| runtime registry | `8` |

The completed batch-128 candidate from PRODUCT-INV-RECOGNITION-045 is retained as a reference rather than retrained again.

## Findings

| training batch | specialist errors | false normal | false red | manual | warm |
|---:|---:|---:|---:|---:|---:|
| 128 reference | 4 | 0 | 4 | 100% | 100% |
| **256** | **0** | **0** | **0** | **100%** | **100%** |
| 512 | 5 | 0 | 5 | 100% | 100% |
| 1024 | 10 | 0 | 10 | 100% | 100% |
| 2048 | 6 | 1 | 5 | 100% | 100% |

Batch 256 is the clear winner for this seed and recipe. It produces zero physical specialist errors across the audited 116,080-example scope while retaining 241/241 manual and 24/24 warm accuracy.

The relationship is non-monotonic. Increasing batch beyond 256 progressively changes the error boundary and does not recover the historical large-batch Full6 behavior.

This also confirms that the base 35-class classifier result, where batch 128 had been preferred to 256 and 512, is not a transferable rule for RED5. Dataset balance, binary-specialist objective, repeated manual examples, and architecture interact with optimizer batch differently.

## Recommendation

Use batch 256 as the accepted training setting for further stride-aggressive RED5 experiments.

Do not change learning rate solely to preserve a linear-scaling rule; this sweep intentionally held the rest of the accepted recipe fixed and directly measured the resulting quality.

## Evidence boundary

This is a single-seed architecture-specific optimization result. Zero observed errors on the audited persisted specialist corpus is not proof of zero deployment error probability, and part of that corpus participates in training. Manual and warm holdouts remain separate checks.

Latency was intentionally not rerun for each batch because training batch does not alter the exported inference topology.

## Follow-up judgment candidates

- Freeze batch 256 while testing late selective DW/PW.
- Revisit multi-seed confirmation only after the architecture itself is selected.

## Follow-up artifact candidates

No production promotion is established by this investigation.

## Open questions

- Does the batch-256 advantage remain after late-stage factorization?
- How stable is the zero-error point across seeds once the architecture is finalized?
