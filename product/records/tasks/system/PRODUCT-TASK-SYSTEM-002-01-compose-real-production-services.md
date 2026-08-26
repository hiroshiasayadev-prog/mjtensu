# PRODUCT-TASK-SYSTEM-002-01: Compose real production services

- **status**: not_started
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-SYSTEM-002
- **task_type**: implementation
- **estimate**: 1.5d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-001-05
  - PRODUCT-TASK-RECOGNITION-001-07
  - PRODUCT-TASK-SCORING-001-07
  - PRODUCT-TASK-APPLICATION-001-06
  - PRODUCT-TASK-UI-001-07
- **outputs**:
  - production composition-root/service wiring
  - PRODUCT-TASK-SYSTEM-002-01

## Goal

Wire the independently reviewed Camera, Recognition, Scoring, Application, and UI implementations into the real production composition root without weakening their public-module boundaries.

## Work

- Construct stable CameraService, Recognition model/runtime services, ScoringService, and ScoringSessionService at the application composition boundary.
- Provide those stable service references to React through the accepted service-provider/context or equivalent composition mechanism.
- Bind the reviewed Application Zustand store and router/pages to the real services rather than test fakes.
- Start application-lifetime recognition model prefetch according to the existing runtime/cache ownership contract without blocking initial Top presentation.
- Initialize the Agari WASM/scoring dependency according to the accepted scoring build/lifecycle implementation while keeping ScoringService public operations synchronous once usable.
- Preserve route-owned camera/realtime recognition lifecycle independently of app-lifetime model/scoring resources.
- Add focused integration tests proving the real composition graph can initialize controlled dependencies without private cross-feature imports.

## Implementation contract

| target | required change | acceptance criterion | verification |
|---|---|---|---|
| composition root | Construct the real feature/application services and inject them through public boundaries. | UI and Application consume only public feature interfaces; private Recognition/Scoring implementation paths are imported only by the composition root where allowed. | Integration bootstrap test plus architecture/static gate. |
| app-lifetime resources | Own recognition asset/runtime and scoring WASM readiness at application lifetime rather than page/store lifetime. | Route navigation does not recreate/dispose healthy shared model/WASM resources; opaque resources do not enter Zustand. | Lifecycle integration tests. |
| real service replacement | Replace fake-service wiring used by feature tests with production implementations for the production build. | Production route flow resolves the real Camera/Recognition/Scoring/Application service graph while test builds can still inject deterministic fakes. | Production/test composition tests. |
| startup behavior | Keep Top presentation independent of background recognition-model acquisition and preserve Recognition page parallel camera/runtime startup. | Initial Top is not blocked on recognition ONNX acquisition; Recognition can await the same in-flight prefetch. | Controlled deferred-resource integration tests. |

## Done condition

The production composition root wires every reviewed feature through its public contract, owns lifecycle resources at the accepted scope, and passes focused composition/lifecycle/architecture tests.

## Verification

- Run composition-root integration tests with controlled real/fake boundary substitutions.
- Run app-lifetime versus route-lifetime lifecycle tests.
- Run strict typecheck/lint/architecture checks.
- Build the production application with the real service graph enabled.

## Evidence

- `spec:product.system.architecture` defines composition-root and runtime-resource ownership.
- The four feature Work Item reviews are prerequisites so this Task integrates accepted feature boundaries rather than unfinished implementations.
- Execution results are recorded here when the Task is performed.
