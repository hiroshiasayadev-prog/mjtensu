# PRODUCT-TASK-SCORING-001-07: Review production scoring boundary

- **status**: done
- **date**: 2026-08-27
- **work_item**: PRODUCT-WORK-SCORING-001
- **task_type**: review
- **estimate**: 0.5d
- **depends_on**:
  - PRODUCT-TASK-SCORING-001-06
- **outputs**:
  - PRODUCT-TASK-SCORING-001-07

## Goal

Independently judge whether the complete Agari fork, WASM ABI, golden corpus, and TypeScript ScoringService implementation are semantically sound and ready for production integration.

## Work

- Review S01 through S06 Evidence and the exact verified source/artifact state.
- Check conformance to the scoring input/result, Scoring API, Agari fork, and Agari adapter contracts.
- Check that TypeScript does not contain a second scoring engine or display-string-dependent control flow.
- Check that fork changes remain narrow and upstream-compatible outside the accepted rule delta.
- Record PASS or NEEDS REVISION and any named findings without repairing them in this Task.

## Done condition

The review records one integrated PASS or NEEDS REVISION verdict with complete findings/evidence for the production Scoring boundary.

## Verification

- Confirm the reviewed fork/WASM/TypeScript state is exactly the S06 verified state.
- Trace scoring semantic judgments to the accepted product/fork contracts and golden Evidence.
- Confirm findings are independent and are not repaired or self-closed here.

## Evidence

- This Task is the independent integrated review for PRODUCT-WORK-SCORING-001.

### Integrated review: 2026-08-27

**Overall verdict: PASS**

No blocking or non-blocking findings were identified in the reviewed production Scoring boundary. This Task made no implementation repairs and did not self-close any finding.

#### Reviewed S06 state

The directly inspected production boundary matches the source/artifact identity recorded by the final S06 PASS:

| identity | reviewed state |
|---|---|
| upstream Agari revision | `a0a9ce15cdf1bea6e7e158bbac1adb4e7a33a547` |
| mjtensu Agari fork revision | `fb362b6db416e67984cdb36f704d8ebf6657662e` |
| canonical production package | `vendor/agari-wasm/` |
| stable ABI | `v1` |
| production WASM | `vendor/agari-wasm/agari_wasm_bg.wasm`, 200739 bytes |
| recorded SHA-256 | `0e3297ed5f6807eac4d7369eb5846bc17e5ea4851470bf9d40c78ec6030e277c` |
| production module path | `@agari-wasm/agari_wasm.js` |
| build provenance | `rustc 1.98.0 (88d9e12ae 2026-08-18)`, `wasm-pack 0.15.0`, release |

`vendor/agari-wasm/provenance.json`, the generated V1 declarations, the production loader aliases, and the 200739-byte committed WASM are present with the same identities recorded by S06. `test/scoring-production-wasm-artifact.test.ts` binds that package to the recorded SHA-256/byte identity and invokes `loadProductionScoringService()` against it. S06 records that integrity/production-loader test as PASS together with the real-WASM corpus, typecheck, lint, architecture, and production build gates. No divergence from that verified boundary was observed in the files inspected by this review.

#### S01-S06 integrated review

| area | verdict | independent review evidence |
|---|---|---|
| S01 fork/artifact ownership | **PASS** | The production dependency is the repo-root vendored package with explicit upstream/fork revisions and reproducible build provenance. Runtime use does not depend on an arbitrary Rust working copy. |
| S02 fork rule semantics | **PASS** | Product-significant rule switches live in Rust `RuleConfig` and are consumed by Rust yaku/fu/limit/dora logic. `RuleConfig::default()` preserves the accepted upstream-compatible defaults; the product profile is supplied explicitly instead of changing hidden global defaults. |
| S03 stable WASM ABI | **PASS** | `abi_v1.rs` exposes explicit request fields for every product scoring condition/rule, machine yaku codes, structured score-level/fu/dora/payment data, and distinct `scored`, `not-winning-shape`, `no-yaku`, `invalid-request`, and `internal-error` outcomes. The generated TypeScript declarations include every runtime field consumed by the adapter. |
| S04 golden contract | **PASS** | The V1 corpus is semantic rather than display-string based and covers the rule/output distinctions consumed at the product boundary. S06 records full real-WASM compatibility PASS for the corpus. |
| S05 TypeScript adapter/service | **PASS** | The input adapter serializes product DTOs to the V1 request and enforces input-contract invariants. The result adapter validates and renames structured machine fields without recomputing Mahjong scoring. Preview preserves expected non-winning/no-yaku outcomes; `calculate` treats those outcomes as violated strict-calculation preconditions rather than engine failures. Engine/init/ABI failures remain explicit adapter failures. |
| S06 production compatibility | **PASS** | The final S06 evidence records green Rust/ABI prerequisites, full real-WASM golden compatibility, production artifact integrity/load verification, typecheck, lint/architecture, and production build consumption of the committed vendor WASM. |

#### Boundary review

1. **Fork scope and authority — PASS**
   - Yaku detection, Fu calculation, score-limit classification, yakuman policy, dora handling, payment arithmetic, and best-scoring interpretation selection remain in Rust/Agari.
   - The accepted product rule delta is represented by explicit Rust rule fields rather than by TypeScript scoring branches.
   - The legacy unversioned Agari WASM API still contains upstream-compatible display-oriented outputs, but the production loader consumes only `score_hand_v1` and `validate_winning_shape_v1`; the legacy display strings are not part of product control flow.

2. **Stable WASM ABI — PASS**
   - The V1 request makes win method, riichi state, situational conditions, winds, dora indicators, and every configurable product rule explicit.
   - Stable result semantics are encoded as discriminated outcomes plus machine yaku codes and structured limit/fu/dora/payment fields.
   - Invalid requests, no-yaku hands, non-winning shapes, and internal boundary failures are not collapsed into one error channel.
   - The adapter does not read undocumented legacy result fields or display labels.

3. **Adapter boundary — PASS**
   - TypeScript performs DTO serialization, physical/input invariant checks, ABI-shape validation, semantic field mapping, and presentation-model normalization only.
   - Duplicate machine yaku identities are aggregated only by preserving their awarded han under the product semantic yaku identity; no yaku, Fu, limit, payment, or point value is inferred from display text or recomputed from Mahjong formulas.
   - Structured Rust payment/fu/limit results remain the calculation authority.

4. **Application/UI ownership — PASS**
   - Application condition policy only constrains contradictory/selectable input combinations and delegates preview/calculation to `ScoringService`; it does not implement role/Fu/limit/payment calculation.
   - UI code consumes semantic `ScoringPreview`/`ScoringCalculation` values and formats them for presentation. Yaku labels and Japanese limit/payment text are output formatting only and are not used to decide scoring semantics.
   - Architecture tests explicitly reject direct concrete Agari WASM imports from UI/Application and record the production source tree as free of architecture violations in the S06 gate.

#### Findings

**None.** The reviewed production Scoring boundary satisfies the accepted input/result, Scoring API, Agari fork, Agari adapter, and testing-strategy contracts and is ready for production integration.
