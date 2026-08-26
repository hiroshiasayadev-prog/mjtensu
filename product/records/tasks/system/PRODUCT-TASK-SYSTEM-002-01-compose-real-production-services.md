# PRODUCT-TASK-SYSTEM-002-01: Compose real production services

- **status**: completed
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

### Implementation: 2026-08-27

- `src/app/production-services.ts` now owns the production service graph: one stable `CameraService`, Recognition production service set, initialized production `ScoringService`, `ScoringSessionService`, `CorrectionEditorService`, and dependency-bound Application Zustand store are constructed at the app composition boundary and projected into the existing UI public service-provider bundles.
- `src/main.tsx` now boots that real graph and passes the production Application/Recognition/Scoring service references into `App`; feature-test injection remains available through `App` props and the controlled factories on `createProductionServiceGraph()`.
- `src/recognition/production-services.ts` creates one browser `RecognitionModelAssets` resolver and passes that same resolver into the production runtime, so application-lifetime `prefetch()` and later `RecognitionRuntime.initialize()` share the existing asset-level in-flight deduplication/cache contract. The realtime facade creates its pipeline only after the initialized runtime is actually used.
- Production bootstrap schedules Recognition model prefetch only after the React root has been rendered and does not await model acquisition before Top presentation. Background prefetch failure is deliberately not promoted into Top failure; Recognition initialization retains the visible retry/error boundary.
- `src/camera/browser-camera-service.ts` supplies the concrete browser `CameraService`: it requests environment-facing video with the accepted `1280 x 720` ideal constraints, owns `MediaStream` preview attachment, copies the current usable frame into a per-capture canvas, normalizes camera-open failures, and makes session stop idempotently release every owned track.
- `ApplicationStateProvider` no longer constructs an unused fallback store when the production composition root supplies the real dependency-bound store, keeping service ownership in composition rather than hidden UI fallback construction.
- `test/production-composition.test.tsx` adds controlled composition/lifecycle coverage for stable service projection, the real Application store dependency seam, Top remaining independent of Recognition prefetch/runtime initialization, explicit background prefetch start, and app-lifetime Recognition disposal ownership.
- `test/camera-service.test.ts` adds focused browser Camera contract coverage for capture constraints, copied latest-frame semantics, preview/session teardown, idempotent track release, and normalized permission failure.
- No cross-feature private Recognition or Scoring implementation path was introduced into `app`; composition consumes the top-level public entries. The shared-assets runtime helper remains internal to the Recognition module and is not re-exported from its public entry point.

### Verification: 2026-08-27

Final user-executed verification:

- `npm test -- camera-service.test.ts production-composition.test.tsx recognition-model-runtime.test.ts recognition-services.test.ts` — **PASS**, 4/4 files and 31/31 tests.
- `npm run lint` — **PASS**, architecture boundaries OK across 58 source files.
- `npm run typecheck` — **PASS** after moving the `#root` lookup/null check into `bootstrapProductionApp()` so the DOM container is locally narrowed before `createRoot()`.
- `npm run build` — **PASS**, Vite production build completed and PWA `generateSW` emitted `dist/sw.js` and `dist/workbox-2fbc6a65.js`.
- Vite reported non-blocking warnings about extensionless imports under the future `configLoader: 'native'` behavior and a post-minification chunk larger than 500 kB; neither warning invalidates this Task's production composition/lifecycle acceptance criteria.

All required composition, lifecycle, architecture, typecheck, and production-build gates pass; the Task is complete.
