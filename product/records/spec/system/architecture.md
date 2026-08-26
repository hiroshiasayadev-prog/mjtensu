# Contract: System architecture

- **id**: `spec:product.system.architecture`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.system`

## What this is

Implementation-facing architecture boundary for the production PWA.
The purpose is to make responsibility leaks and ad hoc cross-feature coupling structurally difficult rather than relying on implementation discipline alone.

## Module boundaries

The production source tree must expose the following logical top-level modules under its application source root:

```text
app/
domain/
camera/
recognition/
scoring/
application/
ui/
```

The exact repository path of the production source root is bootstrap-owned, but code inside that root must preserve these module boundaries.

| module | owns |
|---|---|
| `domain` | Library-independent semantic value types shared across system boundaries. |
| `camera` | Browser camera lifecycle and latest-frame acquisition. |
| `recognition` | Per-frame recognition, model-runtime integration, post-processing, spatial reconstruction, stabilization, and committed recognition output. |
| `scoring` | Library-independent scoring contract plus the isolated adapter to the concrete riichi scoring library. |
| `application` | One scoring session, correction commands, winning-tile selection, scoring orchestration, and page-independent application state. |
| `ui` | Pages, visible components, overlay rendering, user input, and navigation. |
| `app` | Composition root, bootstrap, route assembly, and concrete dependency wiring. |

## Dependency direction

The allowed conceptual direction is:

```text
                 domain
               ↗   ↑   ↖
          camera recognition scoring
               ↖    ↑    ↗
                application
                     ↑
                     ui

app/composition-root wires concrete implementations across these boundaries.
```

More explicitly:

- `domain` must not depend on camera, recognition, scoring, application, UI, ONNX Runtime, or the concrete scoring library.
- `camera`, `recognition`, and `scoring` may depend on `domain` but must not depend on `ui`.
- `application` may consume public camera/recognition/scoring semantics required to coordinate a scoring session but must not own browser rendering, ONNX execution, or scoring-library internals.
- `ui` may consume public application and feature APIs but must not reach into private recognition/scoring/camera implementation modules.
- `app` may import concrete implementations in order to construct the application and is the only intended cross-module composition point.

## Public entry points

Every top-level feature module must expose one public entry point, for example:

```ts
import { ... } from '@/recognition';
import { ... } from '@/scoring';
import { ... } from '@/application';
```

Cross-module imports of private implementation paths are forbidden.

Examples of forbidden imports:

```ts
import { suppressDuplicates } from '@/recognition/pipeline/duplicate-suppression';
import { createOrtSession } from '@/recognition/infra/onnx-runtime';
import { PaiForgeAdapter } from '@/scoring/infra/pai-forge-adapter';
```

Private submodules may import one another within their owning top-level module.

## Concrete-library isolation

- `onnxruntime-web` imports are confined to recognition runtime/infrastructure code.
- The concrete riichi scoring library is confined to the scoring adapter/infrastructure code.
- Browser media APIs used to own camera capture are confined to the camera implementation.
- UI code must consume library-independent recognition/scoring/application contracts.

No concrete-library type may leak into a public cross-module contract unless a later system decision explicitly changes this boundary.

## Composition root

Concrete implementations are assembled in one composition root owned by `app`.
Feature pages and components must receive public services/contracts rather than constructing model sessions, scoring adapters, or camera implementations themselves.

The production frontend stack is:

- Vite;
- React;
- TypeScript in strict mode;
- Mantine for ordinary application UI components and theming;
- React Router for page navigation/history;
- Zustand for mutable Application scoring-session state;
- `vite-plugin-pwa` for PWA packaging/service-worker integration.

Next.js is not part of the production architecture. The application is client-side, camera-heavy, ONNX-runtime-heavy, and does not currently require server rendering or React Server Components.

## Runtime/service versus state ownership

Long-lived runtime services are constructed by the composition root and are not stored in Zustand.
This includes at least:

- `CameraService`;
- `RecognitionModelAssets`;
- `RecognitionRuntime` and its app-lifetime ONNX sessions;
- `ScoringService`;
- `ScoringSessionService`.

React receives these stable service references through a service-provider/context boundary or equivalent composition mechanism. The context carries service references, not high-frequency mutable application state.

Zustand owns the mutable cross-page Application state, currently centered on:

```ts
ScoringSessionState | null
```

Model sessions, browser media resources, and other opaque lifecycle-managed runtime objects must not be placed in the Zustand store merely to make them globally reachable.

High-frequency or page-local state remains local to the owning UI surface. In particular, current camera overlays, `RealtimeRecognitionUpdate`, modal visibility, and uncommitted correction drafts are not promoted to the global Application store unless a later cross-page requirement establishes a reason to do so.

## Routing and navigation

The production routes are conceptually:

```text
/
/recognition
/conditions
/result
/help
```

Route URLs are navigation/history state, not the source of truth for the scoring session.
Routes that require an active scoring session, such as Conditions and Result, must redirect to Top when no session exists rather than inventing or partially reconstructing one from the URL.

Recognition confirmation replaces the transient Recognition history entry when navigating automatically to Conditions. Conceptually:

```text
Top
  -> Recognition
  -> confirmed / replace
Conditions
  -> Result
```

Therefore normal back navigation from Result returns to Conditions, while back navigation from Conditions returns to Top rather than restarting a completed Recognition capture implicitly.
A new Recognition attempt is entered explicitly and discards the prior scoring session according to the Application contract.

## Background model prefetch ownership

Recognition-model asset prefetch is an application-lifetime background task, not a Top-page-owned effect.
The app bootstrap/composition layer may start `RecognitionModelAssets.prefetch()` after the initial Top UI is available without blocking its presentation.

Navigation away from Top must not cancel or restart that acquisition merely because the Top component unmounted.
If Recognition is entered while prefetch is still in flight, `RecognitionRuntime.initialize()` waits on/deduplicates against the same asset acquisition before constructing inference sessions.

## Shared-code rule

A generic `shared/utils`, `common/utils`, or equivalent catch-all module must not be introduced as an architectural dependency bucket.
Code that becomes genuinely reusable must first receive a semantic owner and a responsibility-specific name.

## Mechanical enforcement

The architecture is not documentation-only.
The production project must enforce at least the following in lint/test configuration:

- forbidden deep imports across top-level modules;
- forbidden UI imports from recognition/scoring infrastructure;
- forbidden direct UI imports of `onnxruntime-web` or the concrete scoring library;
- forbidden recognition imports of UI code;
- public-entry-point-only cross-feature imports.

A build must fail when these constraints are violated.

The exact ESLint plugin or architecture-test implementation is implementation-owned.

## Boundary

This file fixes dependency and ownership constraints. Exact feature signatures belong in `contracts/`, and semantic value/state definitions belong in `concepts/`.
