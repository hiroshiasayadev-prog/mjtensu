# PRODUCT-TASK-SCORING-001-06: Verify Agari golden compatibility

- **status**: not_started
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-SCORING-001
- **task_type**: verification
- **estimate**: 1d
- **depends_on**:
  - PRODUCT-TASK-SCORING-001-04
  - PRODUCT-TASK-SCORING-001-05
- **outputs**:
  - PRODUCT-TASK-SCORING-001-06

## Goal

Execute the complete scoring golden corpus through the real mjtensu Agari fork, generated WASM, and production TypeScript ScoringService as one objective compatibility gate.

## Work

- Build/load the pinned production Agari fork/WASM artifact.
- Execute every golden fixture through the real TypeScript adapter/service path where applicable.
- Verify stable yaku codes/awarded han, dora/aka counts, FuCalculation, LimitClassification, and payment outputs against explicit expected values.
- Verify all rule-switch pairs and yakuman-policy interactions.
- Verify winning-shape, no-yaku, and non-winning outcomes remain distinct.
- Execute upstream/fork Rust tests and focused TypeScript scoring tests as prerequisites to the integrated verdict.
- Record exact revision/artifact provenance and an overall PASS, FAIL, or validly BLOCKED result.

## Done condition

Every required golden fixture and predefined scoring compatibility check has an observed result and the overall gate is PASS, FAIL, or validly BLOCKED.

## Verification

| check | expected result |
|---|---|
| pinned upstream/fork provenance resolved | exact revisions recorded |
| upstream + fork Rust tests | PASS |
| stable WASM ABI tests | PASS |
| TypeScript scoring focused tests | PASS |
| golden corpus schema/coverage check | PASS |
| full golden corpus through real fork/WASM/adapter | every case matches expected semantic result |
| strict typecheck/lint/architecture gate | PASS |

The overall verdict is PASS only when every required check is PASS.

## Evidence

- `spec:product.system.contracts.testing_strategy` requires real-engine golden verification rather than fake-only scoring tests.
- Exact Agari revisions, WASM artifact identity, corpus version, command outputs, mismatches if any, and the final verdict are recorded here when executed.
