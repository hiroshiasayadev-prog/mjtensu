# PRODUCT-TASK-SYSTEM-001-03: Enforce architecture import boundaries

- **status**: not_started
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-SYSTEM-001
- **task_type**: implementation
- **estimate**: 0.5d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-001-01
- **outputs**:
  - production frontend lint/architecture configuration
  - PRODUCT-TASK-SYSTEM-001-03

## Goal

Make the accepted module/dependency boundaries fail mechanically when production code imports prohibited private or concrete-library paths.

## Work

- Configure ESLint, an architecture-test layer, or an equivalent deterministic mechanism.
- Forbid cross-feature deep imports where only public entry points are allowed.
- Forbid UI/Application imports of `onnxruntime-web` and concrete Agari WASM bindings.
- Forbid Recognition imports of UI implementation.
- Add bounded fixture/probe cases proving the configured rules detect prohibited examples.
- Keep rule implementation aligned with the accepted architecture without introducing a generic catch-all dependency module.

## Implementation contract

| target | required change | acceptance criterion | verification |
|---|---|---|---|
| cross-feature import gate | Enforce public-entry-point-only cross-feature imports. | A prohibited deep import is reported by the static architecture gate while a public-entry import is accepted. | Execute bounded architecture-rule fixtures/probes. |
| concrete library isolation | Enforce ONNX Runtime and Agari binding isolation from UI/Application. | Direct prohibited imports are rejected by the configured static gate. | Execute explicit negative fixture/probe cases. |
| recognition/UI direction | Prevent Recognition implementation from depending on UI implementation. | A Recognition-to-UI private import is rejected. | Execute an architecture-rule fixture/probe. |
| lint integration | Integrate architecture rules into the ordinary lint/static verification command. | The production verification gate cannot pass while a prohibited import remains. | Run the configured lint/static command over the production source root. |

## Done condition

The production static gate deterministically enforces the accepted architecture import boundaries and is part of the ordinary lint/static verification path.

## Verification

- Run the configured lint/static command.
- Run bounded architecture fixtures/probes for allowed public imports and prohibited deep/concrete-library imports.
- Confirm the production source tree passes with no architecture violations.

## Evidence

- `spec:product.system.architecture` fixes the dependency and concrete-library isolation rules.
- `spec:product.system.contracts.testing_strategy` requires those rules to be mechanically testable.
- Execution results are recorded here when the Task is performed.
