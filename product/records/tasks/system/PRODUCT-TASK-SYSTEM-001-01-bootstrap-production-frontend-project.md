# PRODUCT-TASK-SYSTEM-001-01: Bootstrap production frontend project

- **status**: done
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-SYSTEM-001
- **task_type**: implementation
- **estimate**: 1d
- **depends_on**: []
- **outputs**:
  - production frontend bootstrap files
  - PRODUCT-TASK-SYSTEM-001-01

## Goal

Create the production Vite/React/TypeScript strict project and the seven top-level module/public-entry-point skeleton required by the accepted architecture.

This Task owns the production source/bootstrap surface only. Shared automated-test configuration and bootstrap smoke tests are owned by PRODUCT-TASK-SYSTEM-001-02.

## Work

- Choose the repository-local production frontend root as the bootstrap-owned implementation detail.
- Configure Vite, React, TypeScript strict mode, Mantine, React Router, Zustand, and the `vite-plugin-pwa` dependency selected by PRODUCT-ADR-SYSTEM-001.
- Create the `app`, `domain`, `camera`, `recognition`, `scoring`, `application`, and `ui` source-module boundaries.
- Create one public entry point for each top-level feature module.
- Add only the minimum render/bootstrap code needed for later feature Tasks to compile against the project.
- Do not configure the shared test runners, add bootstrap smoke tests, or implement architecture-import enforcement in this Task.
- Do not activate production service-worker/cache/update behavior; that lifecycle remains owned by PRODUCT-TASK-SYSTEM-002-02.

## Implementation contract

| target | required change | acceptance criterion | verification |
|---|---|---|---|
| production frontend project | Create a Vite + React + TypeScript strict production application using the framework dependencies fixed by PRODUCT-ADR-SYSTEM-001. | A clean dependency install and production build can resolve the application entry point without feature implementation. | Run the configured production build and strict no-emit typecheck. |
| source module skeleton | Create `app`, `domain`, `camera`, `recognition`, `scoring`, `application`, and `ui` boundaries with public entry points. | Later feature code can import each module through its public entry point without requiring private implementation paths. | Compile the seven public entry-point paths; PRODUCT-TASK-SYSTEM-001-02 owns the automated import smoke. |
| minimal application bootstrap | Provide only enough composition/render code to mount the React application without production feature services. | The application entry can initialize without Recognition, Scoring, Application, or feature-UI implementations. | Production build/typecheck; render smoke is owned by PRODUCT-TASK-SYSTEM-001-02. |

## Done condition

The production frontend project, selected stack dependencies, strict TypeScript configuration, seven module boundaries, public entry points, and minimal application bootstrap exist without introducing unresolved feature design decisions or duplicating the shared test-harness responsibility.

## Verification

- Run the production build.
- Run strict no-emit TypeScript checking.
- Confirm the seven required module/public-entry-point paths resolve; the automated proof is supplied by PRODUCT-TASK-SYSTEM-001-02.

## Evidence

- Production frontend root selected as `product/frontend/`.
- Added Vite/React/TypeScript project configuration and ADR-selected frontend dependencies in `product/frontend/package.json`.
- Added strict TypeScript 7-compatible configuration and `@/* -> ./src/*` import alias without the removed `baseUrl` option.
- Pinned the frontend development runtime to the Node 24 LTS line (`^24.15.0`) and recorded `.node-version` as `24.19.0`; Node 25 is outside the selected supported toolchain.
- Added `src/app`, `src/domain`, `src/camera`, `src/recognition`, `src/scoring`, `src/application`, and `src/ui`, each with one public `index.ts` entry point.
- Added minimal `src/main.tsx` and `src/app/App.tsx` bootstrap code; no feature routes, stores, runtime services, or PWA lifecycle semantics were implemented.
- `npm ci` PASS on Node `v24.19.0` / npm `11.12.1`; dependency install completed with 0 audit vulnerabilities.
- `npm run typecheck` PASS with strict TypeScript checking for production and test configuration.
- `npm run test:e2e` executed the production build successfully with Vite `v8.2.2`: 820 modules transformed and `dist/` emitted successfully.
- The seven public entry points are exercised by PRODUCT-TASK-SYSTEM-001-02's public-entry smoke and passed under the shared test harness.
