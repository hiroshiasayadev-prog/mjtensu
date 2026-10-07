# PRODUCT-INV-RECOGNITION-032: Evaluate cyclic-equivariant width reduction

- **status**: concluded
- **date**: 2026-10-05
- **trigger**: PRODUCT-INV-RECOGNITION-030 and PRODUCT-INV-RECOGNITION-031 found only limited gains from widening pure Plain CNN stages, while C8 retained a large robustness advantage at much higher latency. The next question was whether that advantage required the full C8 activation width.
- **scope**: Compare current C8 with a half-tensor-width C8 variant and a C4 variant matched to the C8-narrow equivariant tensor width while holding training recipe, seed, and evaluation suite fixed.
- **non_scope**: C2, finer C8 width sweeps, architecture changes outside group order and regular-field counts, pooling changes, quantization, multi-seed confirmation, worker pinning, or production promotion.
- **source_refs**:
  - PRODUCT-INV-RECOGNITION-005
  - PRODUCT-INV-RECOGNITION-027
  - PRODUCT-INV-RECOGNITION-030
  - PRODUCT-INV-RECOGNITION-031
- **follow_up_candidates**:
  - Continue narrowing C8 before reducing group order further.
  - Test narrower C8 field schedules such as 3/6/12/24 and 2/4/8/16.
  - Pin training hardware before using small quality deltas for final model selection.

## Investigation scope

Test whether C8 robustness survives a large reduction in equivariant activation width.

Also test whether reducing the symmetry group from C8 to C4 preserves quality at the same equivariant tensor width.

## Out of scope

- C2.
- Finer C8 width sweeps.
- Pooling changes.
- Non-equivariant architecture changes.
- Quantization.
- Multi-seed confirmation.
- Worker-pinned training.
- Production promotion.

## Background

The fast Plain family remained materially below C8 on difficult robustness metrics.

PRODUCT-INV-RECOGNITION-031 showed that targeted Plain stage2 widening can recover some quality.

The Plain gain still did not close the robustness gap.

Current C8 is also expensive on product hardware.

The next useful question was therefore not whether to remove equivariance.

The useful question was whether C8 carries more equivariant channel capacity than needed.

## What was investigated

| model | group | regular fields | equivariant tensor channels |
|---|---:|---|---|
| current C8 | C8 | 8/16/32/64 | 64/128/256/512 |
| C8-narrow | C8 | 4/8/16/32 | 32/64/128/256 |
| C4 width-match | C4 | 8/16/32/64 | 32/64/128/256 |

C8-narrow and C4 carry the same equivariant tensor channel counts at every stage.

The topology otherwise remains in the C8 family:

- 5x5 first equivariant convolution.
- 3x3 later equivariant convolutions.
- Pointwise max pooling between stages.
- GroupPooling after the final equivariant block.
- Global spatial average pooling.
- Classifier head.

GroupPooling produces different invariant channel counts for C8-narrow and C4.

The comparison therefore matches equivariant tensor width.

The comparison does not match every parameter or post-pooling dimension.

Training controls were:

- Corpus: `tile-classifier/gray35-jp500-seed42-v3-jp189-v1`.
- Train Protocol: `tile-classifier/tile-shape-train-gpu-v6`.
- Epochs: 150.
- Batch: 128.
- Learning rate: 0.001.
- Weight decay: 0.0001.
- Augmentation: `random360-only-v1`.
- Seed: 42.
- AMP: enabled.
- TF32: enabled.

Primary quality evidence:

| evidence | ref |
|---|---|
| Study | `tile-classifier/cyclic-equivariance-width-screen-v1` |
| Study Plan | `tile-classifier/cyclic-equivariance-width-screen-v1-plan-045a42cab2bbb6fe` |
| Study Result | `tile-classifier/run-ac1a2514af214cb69977972c0278ab39` |
| source commit | `d27bb767259a27bcff9776a0fdd3f70dc5964119` |
| runtime registry | `8` |
| training | 3/3 completed |

Some latency evaluations required separate recovery runs.

Latency recovery evidence:

| evidence | ref |
|---|---|
| Study | `tile-classifier/cyclic-equivariance-width-screen-latency-recovery-v1` |
| Study Result | `tile-classifier/run-5f7624b9bea84d71821ab0d588e01fee` |
| source commit | `35e40874894b397e1a506f4d4c94dbec49cd8cd2` |

A later C4-only latency rerun also completed:

| evidence | ref |
|---|---|
| Study | `tile-classifier/cyclic-equivariance-width-screen-c4-latency-rerun-v1` |
| Study Result | `tile-classifier/run-ae017fff12ed487aa35b658cfbda0e52` |
| source commit | `9632681416f3e3d48d14db0635c1ce4d8315688a` |

## Findings

### Quality

| metric | current C8 | C8-narrow | C4 width-match |
|---|---:|---:|---:|
| dense64 angle mean | 0.979757 | 0.979201 | **0.980347** |
| full-class mean | 0.952296 | **0.957037** | 0.948889 |
| full-class worst | **0.857778** | 0.844444 | 0.828889 |
| Manzu mean | **0.982979** | 0.973050 | 0.960284 |
| Manzu worst | **0.936170** | 0.893617 | 0.872340 |
| validity balanced | 0.918286 | **0.933911** | 0.902661 |
| all-real accuracy | **0.999925** | 0.999855 | 0.999832 |
| all-real worst-class recall | **0.999326** | 0.998808 | 0.998989 |

C8-narrow retains most observed robustness despite halving equivariant tensor width.

Dense-angle robustness is effectively unchanged.

Observed full-class mean and validity balanced accuracy are higher.

Full-class worst, Manzu, and all-real robustness are lower.

C4 remains strong.

At the matched 32/64/128/256 equivariant tensor width, C4 is below C8-narrow on full-class and Manzu robustness.

The C8-narrow versus C4 direction supports keeping C8 symmetry before reducing group order further.

The quality difference is not a hardware-isolated proof of group-order causality.

### Latency

The three-model latency recovery Study measured:

| model | CPU p50 | iPhone p50 |
|---|---:|---:|
| current C8 | 3.5638 ms | 13.29 ms |
| C8-narrow | **0.9170 ms** | 3.435 ms |
| C4 width-match | 0.9524 ms | **3.420 ms** |

C8-narrow is about 3.9x faster than current C8 on both CPU and iPhone in the recovery Study.

C4 does not provide a meaningful latency advantage over C8-narrow.

The later C4 rerun measured about 0.875 ms CPU p50 and 3.265 ms iPhone p50.

The rerun keeps C4 and C8-narrow in the same practical latency class.

### Worker-hardware reproducibility boundary

The primary Study did not pin training to one GPU type.

Worker assignment was:

| model | training GPU |
|---|---|
| current C8 | RTX 3090 |
| C8-narrow | RTX 3060 |
| C4 width-match | RTX 3060 |

The two reduced-width models also ran on different RTX 3060 workers.

A prior corrected replay trained the same current-C8 Architecture with the same nominal recipe on an RTX 3060.

The two current-C8 trainings produced different final weight hashes.

| run | GPU | weight SHA-256 |
|---|---|---|
| corrected replay C8 | RTX 3060 | `ce095aed59d95ec64fe73814457396df45ab706514460bfecfe4f7684c94761e` |
| cyclic-screen C8 | RTX 3090 | `ede8791ff4235b5a8a5aac3947d1c0855ac9424d7593bc51b0a7dcb5bebdd92e` |

The Architecture, Train Protocol, training parameters, and relevant source files are unchanged between those revisions.

PRODUCT-INV-RECOGNITION-027 did not claim bitwise-identical training across heterogeneous GPUs.

The observed weight-hash difference is consistent with that boundary.

The large inference-latency reduction remains valid evidence.

The broad finding that C8-narrow retains high quality also remains useful.

Sub-percentage-point quality differences should not be treated as hardware-isolated causal effects.

## Cross-cutting observations

The result changes the most promising classifier search direction.

Pure Plain widening recovers limited robustness at meaningful latency cost.

C8-narrow removes most of the original C8 latency burden while retaining strong robustness.

The observed trade favors reducing equivariant width before removing equivariance.

C4 remains valuable as a reference because its latency is similar to C8-narrow.

## Follow-up judgment candidates

- Whether the next width screen should keep C8 and reduce regular-field counts further.
- Whether C4 should remain a reference rather than the main candidate.
- Whether finalist quality comparisons require worker-pinned training.
- Whether C2 is still worth testing after a narrower C8 screen.

## Recommendation

Further C8 narrowing appears preferable to reducing group order now.

Examples, not exhaustive:

- C8 fields 3/6/12/24.
- C8 fields 2/4/8/16.

C4 appears less attractive than C8-narrow because observed latency is similar and robustness is lower.

## Follow-up artifact candidates

- A C8 field-width screen Investigation.
- A worker-controlled finalist comparison before production selection.
- A later production model-selection artifact if one equivariant candidate is adopted.

## Open questions

- How far can C8 field counts be reduced before full-class worst and Manzu robustness collapse?
- Does C8-narrow retain its quality under worker-pinned retraining?
- Does C4 remain weaker than C8-narrow after hardware-controlled training?
- Is C2 useful only after C8 width reaches its practical knee?
