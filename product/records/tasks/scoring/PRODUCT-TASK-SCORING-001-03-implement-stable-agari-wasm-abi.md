# PRODUCT-TASK-SCORING-001-03: Implement stable Agari WASM ABI

- **status**: not_started
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
- Execution results are recorded here when the Task is performed.
