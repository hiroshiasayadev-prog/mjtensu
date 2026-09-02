# PRODUCT-TASK-SYSTEM-002-15: Optimize Recognition classifier preprocessing

- **status**: in_progress
- **date**: 2026-09-02
- **work_item**: PRODUCT-WORK-SYSTEM-002
- **task_type**: correction
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-002-08
  - PRODUCT-INV-RECOGNITION-011
- **outputs**:
  - lower-latency gray64/red-five crop preprocessing
  - target-device preprocessing timing evidence
  - PRODUCT-TASK-SYSTEM-002-15

## Trigger

PRODUCT-INV-RECOGNITION-011 confirmed that MobileNetV3-Small 1.0x reduces isolated iPhone 13 base-classifier inference latency by about 1.5-1.7x versus the current Plain e150 classifier. The same target-device diagnostics show that classifier preprocessing is now the larger cost: a representative 17-candidate frame measured about `161 ms` of base-classifier preprocessing versus about `66 ms` of base-classifier inference.

The production timing boundary already separates candidate crop extraction from classifier preprocessing. Therefore the measured preprocessing cost does not include `production-pipeline.ts` crop `drawImage()` / `getImageData()` work. Code inspection identifies the software Lanczos implementation in `classifier/preprocessing.ts` as the primary suspect.

## Goal

Reduce production classifier preprocessing latency without changing the accepted classifier input contract, normalization, letterbox geometry, border-fill policy, candidate ordering, or red-five refinement semantics.

## Current bottleneck

The current `resampleChannel()` performs a direct two-dimensional radius-3 Lanczos convolution in TypeScript. For every target pixel it evaluates approximately `6 x 6` source taps, repeatedly invoking `Math.sin()` through `lanczos()` even though the kernel is separable and the x/y weights are constant across rows/columns for one resize.

For a typical tile crop resized to roughly `32 x 64`, this is on the order of tens of thousands of weighted taps per channel per crop and a very large number of transcendental-function calls across a 16-24-candidate frame. The RGB red-five path repeats the same resampling operation for three channels.

## Work

1. Replace the direct 2D Lanczos implementation with mathematically equivalent separable filtering:
   - precompute horizontal source indices/normalized weights once per target x;
   - precompute vertical source indices/normalized weights once per target y;
   - run one horizontal 6-tap pass into an intermediate buffer;
   - run one vertical 6-tap pass into the final byte buffer;
   - round/clamp only at the final output byte, preserving the legacy 2D kernel result modulo floating-point operation ordering.
2. Add a regression test that compares the optimized resampler against a test-local copy of the legacy direct 2D algorithm on a nontrivial resize and requires every output byte to agree within one intensity level.
3. Preserve existing grayscale/RGB conversion, border-median fill, letterbox dimensions, normalization, dynamic batch shape, and classifier model contracts.
4. Run focused classifier and production-pipeline tests plus typecheck.
5. Re-measure iPhone 13 diagnostics at representative candidate counts. Primary evidence is `baseClassifierPreprocessingMs`; also record red-five preprocessing where available.
6. If separable Lanczos does not reduce preprocessing sufficiently, follow with a second-stage browser-native resize experiment that performs source-box resize/letterbox through Canvas `drawImage()` and validates classifier accuracy/input parity before replacing the software resampler.

## Done condition

- Optimized preprocessing preserves the established classifier input semantics within the explicit resampling parity tolerance.
- Focused tests and typecheck pass.
- iPhone 13 shows a material reduction in base preprocessing latency relative to the ~161 ms / 17-candidate observation, or the task records evidence that the remaining cost requires the stage-2 Canvas path.

## Implementation: 2026-09-02

`src/recognition/classifier/preprocessing.ts` now computes the radius-3 Lanczos resize as two separable passes instead of one direct `6 x 6` nested convolution per output pixel. Lanczos weights are computed only while building one-dimensional contribution tables; `Math.sin()` is no longer called inside the per-pixel resampling loops. A `Float64Array` intermediate avoids introducing an extra byte-rounding step between horizontal and vertical passes.

`test/recognition-c8-classifier.test.ts` includes the former direct 2D Lanczos algorithm as a test-only reference and checks a nontrivial `5 x 3 -> 4 x 2` resize through the production grayscale preprocessing path with a maximum one-byte difference tolerance.

## Verification: 2026-09-02

User-executed verification from `product/frontend`:

- `npx vitest run test/recognition-c8-classifier.test.ts test/recognition-services.test.ts` — **PASS**, 2/2 files and 30/30 tests.
- `npm run typecheck` — **PASS**.

The implementation/regression gates are therefore PASS. Task completion still depends on target-device timing evidence because the Done condition explicitly requires confirming a material iPhone preprocessing reduction or escalating to the Canvas-native resize stage.
