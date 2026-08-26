# PRODUCT-TASK-SCORING-001-03: Implement stable Agari WASM ABI

- **status**: done
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-SCORING-001
- **task_type**: implementation
- **estimate**: 1.5d
- **depends_on**:
  - PRODUCT-TASK-SCORING-001-02
- **outputs**:
  - mjtensu Agari WASM ABI implementation
  - PRODUCT-TASK-SCORING-001-03

## Goal

Expose the accepted rule-aware Agari scoring behavior through a stable machine-readable WASM ABI, including a scoring-independent winning-shape validation entry point.

## Work

- Add a versioned score request carrying all scoring conditions and explicit rule configuration.
- Return stable tagged scoring outcomes for scored, not-winning-shape, no-yaku, invalid-request, and internal-error states.
- Return stable machine yaku codes plus awarded han for regular yaku.
- Return structured score-level semantics including kiriage mangan and counted/actual yakuman units.
- Preserve aggregate fu breakdown, dora/aka counts, and payment fields through WASM.
- Add a scoring-independent winning-shape validation entry point for ordinary, chiitoitsu, and kokushi completed shapes.
- Add Rust/WASM-facing tests for stable discriminants and result-shape behavior.

## Implementation contract

| target | required change | acceptance criterion | verification |
|---|---|---|---|
| scoring request ABI | Expose all product conditions plus explicit rule config through one versioned request shape. | No product rule/condition depends on an implicit WASM default or display-string convention. | WASM/Rust contract tests over serialized requests. |
| outcome/result ABI | Return stable tagged semantic outcomes, stable yaku codes, awarded regular-yaku han, structured score level, fu, dora, and payments. | TypeScript can branch and normalize without parsing English/Japanese display strings or dividing han to infer yakuman units. | ABI fixture tests and generated binding inspection. |
| winning-shape API | Add a scoring-independent shape-validation function using the same Agari decomposition implementation. | Valid ordinary/chiitoitsu/kokushi shapes return winning; coherent non-winning shape returns not-winning-shape without requiring conditions/yaku. | Dedicated shape-validation tests. |
| failure semantics | Separate invalid-request from internal-error and from normal not-winning/no-yaku outcomes. | Malformed request and expected mahjong semantic outcomes have distinct stable discriminants. | Negative ABI tests. |

## Done condition

The fork exposes the stable scoring and winning-shape WASM APIs required by the Agari fork contract, generated bindings preserve those stable fields, and the focused ABI tests pass.

## Verification

- Run Rust/WASM ABI tests for every result discriminant.
- Run stable-yaku-code and awarded-han fixtures.
- Run structured-limit/fu/dora/payment fixtures.
- Run ordinary/chiitoitsu/kokushi/non-winning shape-validation fixtures.
- Inspect generated TypeScript/WASM binding output for the expected versioned entry points and fields.

## Evidence

- `spec:product.system.contracts.agari_fork` defines the stable WASM semantic contract.
- S02 supplies rule-aware core semantics; this Task exposes them without TypeScript-side reinterpretation.
- V1 implementation is present under `external/agari/crates/agari-wasm/src/abi_v1.rs` and is wired from `agari-wasm/src/lib.rs` without replacing the legacy upstream-compatible exports.
- `score_hand_v1` requires one explicit V1 request carrying the winning tile, every scoring condition, winds, dora/ura indicators, and all nine `RuleConfig` fields; unsupported `double_wind_pair_fu` values are classified as `invalid-request`.
- Stable tagged score outcomes are implemented for `scored`, `not-winning-shape`, `no-yaku`, `invalid-request`, and `internal-error`. The scored payload exposes stable yaku codes, awarded regular-yaku han, structured limit semantics, aggregate fu, dora/ura/aka counts, payment fields, and dealer role without display-string parsing or yakuman-unit inference from han.
- `validate_winning_shape_v1` reuses the same Agari decomposition path and validates ordinary, chiitoitsu, kokushi, and called-meld completed shapes without requiring scoring context or yaku.
- Rust-facing ABI fixtures cover discriminants, strict request/rule shape, rule propagation, stable yaku-code uniqueness, open-yaku awarded han, red-five winning-tile normalization, structured limits including actual/counting yakuman, fu/dora/payment shapes, and winning/non-winning validation.
- The V1 exports suppress wasm-bindgen's default `any` declarations and append explicit V1 TypeScript request/outcome declarations through `typescript_custom_section`, so regenerated bindings contain typed V1 entry points and all stable fields.
- Malformed numeric honor melds are rejected by the core parser instead of panicking, allowing both V1 scoring and winning-shape validation to classify them deterministically as `invalid-request`.
- Verification on 2026-08-26 passed: `cargo test -p agari-wasm abi_v1` ran 22 focused V1 ABI tests with 22 passed, 0 failed.
- Full fork verification passed: `cargo test --workspace` completed 296 core library tests, 31 CLI tests, and 49 WASM tests with 0 failures (376 tests total; doc-test suites also passed with 0 tests).
- `wasm-pack build crates/agari-wasm --target web --out-dir ../../web/src/lib/wasm` completed successfully and regenerated the web package.
- Regenerated `web/src/lib/wasm/agari_wasm.d.ts` contains the full stable V1 request/result types and the consumer-facing typed signatures `score_hand_v1(request: AgariScoreRequestV1): AgariScoreOutcomeV1` and `validate_winning_shape_v1(hand: string): AgariWinningShapeOutcomeV1`; the lower-level `InitOutput` signatures remain wasm-bindgen's raw WebAssembly export representation and are not the public TypeScript contract.
- Regenerated `agari_wasm.js` exports both V1 wrapper functions and forwards them to the corresponding WASM exports.
