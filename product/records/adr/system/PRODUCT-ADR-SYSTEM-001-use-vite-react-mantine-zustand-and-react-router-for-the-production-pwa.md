# PRODUCT-ADR-SYSTEM-001: Use Vite, React, Mantine, Zustand, and React Router for the production PWA

- **status**: accepted
- **date**: 2026-08-26
- **migrated_to_spec**: `spec:product.system.architecture`

## Context

The production application is a smartphone-oriented PWA whose primary workload is client-side camera capture, live recognition with ONNX Runtime Web, semantic correction, and local riichi-mahjong scoring.

The product does not currently require server-side rendering, React Server Components, server-owned application state, or server-side scoring/model execution.
At the same time, the UI includes multiple application pages, condition-entry controls, correction flows, dialogs, responsive mobile layout, and a high-frequency live camera overlay.

The system architecture already separates long-lived runtime services from Application scoring-session state and from page-local recognition/correction state.
The frontend framework and state/navigation choices must preserve those ownership boundaries rather than turning one global store or component tree into a second dependency graph.

Model-asset acquisition also spans navigation: the application may begin background ONNX asset prefetch once the initial UI is available, and that work must continue if the user navigates away from Top or enters Recognition before prefetch completes.

## Decision

Use the following production frontend stack:

- Vite as the frontend build/dev tool;
- React as the UI framework;
- TypeScript with strict checking;
- Mantine for ordinary UI components and theming;
- React Router for route/history management;
- Zustand for mutable cross-page Application scoring-session state;
- `vite-plugin-pwa` for PWA packaging and service-worker integration.

Do not use Next.js for the current production application.

### State ownership

Use Zustand only for mutable cross-page Application state, currently centered on `ScoringSessionState | null`.

Do not store ONNX `InferenceSession` objects, camera resources, or other long-lived services in Zustand.
These resources are constructed by the `app` composition root and exposed to React through stable service references, for example through a service-provider/context boundary.

Keep high-frequency and page-local state local to the owning UI surface. This includes current recognition overlays/updates, modal state, and uncommitted correction drafts.
Only committed semantic state that must survive page transitions is promoted into the Application store.

### Routing

Use route paths conceptually equivalent to:

```text
/
/recognition
/conditions
/result
/help
```

Routes are navigation/history state, not persistent scoring-session state.
Conditions and Result require an active scoring session and return to Top when no such session exists.

Automatic navigation from a confirmed Recognition result to Conditions replaces the transient Recognition history entry.
This makes normal back navigation behave as:

```text
Result -> Conditions -> Top
```

rather than implicitly reopening a completed camera-recognition run.
Starting another recognition attempt remains an explicit action that discards the active scoring session according to the Application contract.

### Model prefetch ownership

Background recognition-model asset prefetch is owned by application bootstrap/composition, not by the Top page component.

The application may begin prefetch after the initial UI is available without blocking Top rendering.
Navigation must not cancel or restart that acquisition merely because a route component unmounts.
If Recognition is entered while acquisition remains in flight, runtime initialization waits on or deduplicates against the same asset work before creating inference sessions.

## Rationale

Vite matches the application's client-only deployment model and avoids introducing server-rendering/runtime concepts that the product does not use.
React provides a conventional component model for the Conditions, Result, correction, and recognition surfaces, while Mantine reduces ordinary mobile form/dialog/layout work without constraining the camera overlay to component-library abstractions.

React Router provides browser/PWA history semantics directly instead of requiring a custom screen-state/history synchronization layer.
Replacing the Recognition route on automatic confirmation reflects the lifecycle of Recognition as a transient capture activity rather than a revisitable form step.

Zustand provides selective subscriptions for cross-page mutable Application state without requiring high-frequency mutable state to travel through React Context.
Keeping runtime services outside Zustand prevents opaque lifecycle-managed resources from becoming UI state merely for global reachability.

Separating app-owned model prefetch from page ownership allows download progress to survive route changes and aligns asset lifetime with the existing model-runtime contract.

## Rejected alternatives

### Next.js

Next.js can produce a client-heavy PWA, but the current product does not benefit from SSR, server components, server actions, or a server-owned routing/data layer.
Using it would introduce additional framework/runtime concerns around an application whose core camera, ONNX, and scoring work must execute in the browser anyway.

### React Context as the mutable scoring-session store

A Context-based store can represent the session, but Zustand provides simpler selective subscriptions and avoids broad rerender coupling as Conditions and Result consume different slices of mutable session state.
Context remains suitable for stable service references.

### Put model sessions in Zustand

ONNX sessions are lifecycle-managed runtime resources rather than Application state.
Their size is not a reason to put them in a global store, and doing so would mix resource ownership/disposal with UI state subscriptions.
They remain owned by `RecognitionRuntime` through the composition root.

### Keep all live recognition state in Zustand

Recognition overlay observations update frequently and are relevant only while the Recognition page is active.
Promoting them into a global store would create unnecessary cross-page state and subscriptions without product value.
Only the committed `RecognizedStructure` crosses into Application state.

### Custom screen enum without a router

The page count is small enough that a custom screen enum is possible, but it would require the application to reproduce browser history/back-navigation behavior and route guarding itself.
React Router provides those semantics without changing the scoring-session ownership model.
