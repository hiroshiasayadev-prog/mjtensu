# PRODUCT-TASK-SCORING-001-02: Implement Agari rule semantics

- **status**: not_started
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-SCORING-001
- **task_type**: implementation
- **estimate**: 2d
- **depends_on**:
  - PRODUCT-TASK-SCORING-001-09
- **outputs**:
  - mjtensu Agari fork core rule implementation
  - PRODUCT-TASK-SCORING-001-02

## Goal

Implement the narrow Agari core changes required by the accepted mjtensu `RuleConfig` and rule-aware scoring semantics while preserving upstream decomposition/scoring responsibilities outside that delta.

## Work

- Introduce one coherent rule-config object that reaches every affected yaku/dora/fu/limit path.
- Implement open-tanyao, aka-dora, indicator-dora, and ippatsu switches.
- Implement kiriage mangan for 4 han 30 fu and 3 han 60 fu.
- Implement counted-yakuman on/off behavior with sanbaiman cap when disabled.
- Separate detected yakuman identity from configured yakuman-unit contribution.
- Implement double-yakuman-variant and multiple-yakuman policies independently.
- Implement 2/4-fu double-wind-pair policy.
- Preserve fixed chiitoitsu/pinfu/open-no-extra-fu behavior required by the fork contract.
- Add Rust regression tests for every modified rule branch and interaction required by the fork specification.

## Implementation contract

| target | required change | acceptance criterion | verification |
|---|---|---|---|
| `RuleConfig` propagation | Add one explicit config object consumed by every affected scoring stage. | No product-significant rule switch is implemented only as a WASM/TypeScript postprocess or unrelated global. | `cargo test` plus source-level review of config propagation. |
| yaku/dora switches | Implement open tanyao, aka dora, indicator dora, and ippatsu behavior exactly as the fork contract specifies. | On/off regression pairs return the expected awarded yaku/bonus values without changing unrelated decomposition. | Focused Rust regression tests. |
| limit rules | Implement kiriage and counted-yakuman policy independently of actual yakuman detection. | 4h30f/3h60f and 13+ non-yakuman cases match both configured branches. | Focused Rust regression tests. |
| yakuman units | Separate variant identity from 1/2-unit policy and apply multiple-yakuman aggregation independently. | All four double variants and multiple-yakuman interactions match the configured unit totals while preserving detected identities. | Table-driven Rust yakuman tests. |
| double-wind pair fu | Make 2/4-fu total configurable and reject unsupported config values. | Round+seat same wind produces exactly the configured pair fu; single-value winds/dragons remain unchanged. | Rust fu regression tests. |
| upstream compatibility | Preserve upstream tests for behavior outside the accepted semantic delta. | Upstream test suite remains PASS after the fork changes. | Execute upstream `cargo test` suite plus mjtensu regression tests. |

## Done condition

The Agari core fork implements every rule semantic in `spec:product.system.contracts.agari_fork`, all focused/upstream Rust tests pass, and no unrelated decomposition or TypeScript-side scoring reimplementation is introduced.

## Verification

- Run the complete upstream Agari Rust test suite.
- Run the added mjtensu rule regression matrix.
- Record results for every rule switch and required interaction.
- Confirm no TypeScript/WASM postprocess is being used to compensate for a missing core rule branch.

## Evidence

- `spec:product.system.contracts.agari_fork` is the normative semantic delta.
- PRODUCT-TASK-SCORING-001-01 decided the source/build-management boundary and PRODUCT-TASK-SCORING-001-09 recorded it in canonical ADR/Specification authority; neither changes scoring behavior.
- Execution results are recorded here when the Task is performed.
