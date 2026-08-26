# PRODUCT-TASK-SCORING-001-09: Author Agari build-management contract

- **status**: done
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-SCORING-001
- **task_type**: authoring
- **estimate**: 0.25d
- **depends_on**:
  - PRODUCT-TASK-SCORING-001-01
  - PRODUCT-TASK-SCORING-001-08
- **outputs**:
  - PRODUCT-TASK-SCORING-001-09
  - PRODUCT-ADR-SYSTEM-003
  - `spec:product.system.contracts.agari_fork`

## Goal

Record the accepted Agari fork source, revision, WASM consumption, provenance, and upgrade-management decision in canonical ADR and Specification authority.

## Work

- Amend PRODUCT-ADR-SYSTEM-003 without supersession because the accepted Agari-fork architecture remains unchanged.
- Extend `spec:product.system.contracts.agari_fork` with the normative source/artifact-management contract selected by PRODUCT-TASK-SCORING-001-01.
- Preserve existing scoring semantics and fork-delta rules unchanged.

## Done condition

PRODUCT-ADR-SYSTEM-003 records the durable rationale and consequences, and `spec:product.system.contracts.agari_fork` normatively defines the selected production source/artifact-management boundary without introducing a new scoring-semantic decision.

## Verification

- Confirm the ADR remains `accepted` and is amended rather than superseded.
- Confirm the Specification requires exact upstream/fork commit SHAs, committed generated release WASM, artifact provenance/integrity metadata, frontend-build independence from Rust tooling, and explicit-only upstream upgrades.
- Confirm no accepted scoring rule or WASM semantic contract changed.

## Evidence

- PRODUCT-ADR-SYSTEM-003 remains `accepted`; it was amended rather than superseded because the selected Agari-fork architecture and scoring ownership remain unchanged.
- The ADR now records the separate fork repository, exact-SHA production pinning, committed generated WASM package, frontend-build independence from Rust tooling, provenance discipline, explicit upstream upgrades, and rejected submodule/per-build-rebuild alternatives.
- `spec:product.system.contracts.agari_fork` now owns the normative source/artifact-management contract, including `vendor/agari-wasm/`, full upstream/fork commit SHAs, release artifact refresh, provenance fields, artifact identity, and compatibility-gated upgrades.
- Existing RuleConfig, stable WASM ABI, winning-shape validation, scoring semantics, and required fork tests were preserved.
