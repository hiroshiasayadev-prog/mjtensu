# PRODUCT-TASK-SYSTEM-001-02: Implement shared test harness

- **status**: in_progress
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-SYSTEM-001
- **task_type**: implementation
- **estimate**: 1d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-001-01
- **outputs**:
  - production frontend test configuration
  - PRODUCT-TASK-SYSTEM-001-02

## Goal

Provide the shared Vitest, Testing Library, and Playwright harness that all production feature Work Items use for focused and browser-level tests.

This Task exclusively owns bootstrap/public-entry smoke coverage. PRODUCT-TASK-SYSTEM-001-01 owns the production application skeleton but does not create or configure automated tests.

## Work

- Configure Vitest for TypeScript unit and contract tests.
- Configure Testing Library and the DOM test environment for React component behavior.
- Configure Playwright for deterministic browser E2E execution against the built production application.
- Add shared structurally typed fake-service and deterministic async support without embedding feature-specific semantic rules or importing concrete runtime libraries.
- Add a non-placeholder component smoke proving the application root renders without production feature services.
- Add a bounded smoke that imports all seven top-level modules through their public entry points.
- Add one non-placeholder Playwright smoke proving the built application opens and exposes the bootstrap UI.
- Expose stable package scripts for unit/component, browser, build, and strict typecheck verification.

## Implementation contract

| target | required change | acceptance criterion | verification |
|---|---|---|---|
| Vitest | Configure deterministic TypeScript unit/contract execution for the production project. | Bounded unit tests pass under the configured runner. | Run `npm test`. |
| Testing Library | Configure React component tests against the production UI environment. | The application root can be rendered and queried through user-visible semantics without production feature services. | Run the component bootstrap smoke under `npm test`. |
| Playwright | Configure browser E2E execution against a production Vite build. | A browser smoke can open the built application and assert one visible bootstrap outcome. | Run `npm run test:e2e`. |
| shared test support | Provide generic typed fake-service composition and deterministic async-control foundations without feature-specific semantics. | Later UI/Application tests can implement public Camera/Recognition/Scoring service interfaces with deterministic fakes without importing concrete runtime internals. | Compile and execute the bounded fake-service support tests. |
| public-entry smoke | Import `app`, `domain`, `camera`, `recognition`, `scoring`, `application`, and `ui` through their public entry points. | All seven architecture entry paths resolve through the shared test/tool configuration. | Run the public-entry smoke under `npm test` and strict test typecheck. |

## Done condition

The shared production test harness can execute unit, component, and browser smoke tests; owns all bootstrap/public-entry smoke coverage; and exposes deterministic fake/async foundations compatible with later public product contracts without duplicating feature semantics.

## Verification

- Run `npm install` from `product/frontend/` to materialize the dependency lock/install state.
- Run `npm run typecheck`.
- Run `npm test` and confirm the bootstrap, public-entry, and fake-service support tests pass.
- Run `npm run test:e2e:install` once when Chromium is not already installed for Playwright.
- Run `npm run test:e2e` and confirm the production-build browser smoke passes.

## Evidence

- Added `vitest.config.ts` with jsdom and Testing Library setup in `test/setup.ts`.
- Added `test/app.smoke.test.tsx` as the single application-root component smoke owned by this Task.
- Added `test/public-entry-points.test.ts` to resolve all seven architecture modules through public entry points.
- Added generic fake-service/deferred support under `test/support/` with focused tests in `test/fake-service.test.ts`.
- Added `playwright.config.ts` and `test/e2e/bootstrap.spec.ts`; the E2E command builds first and serves `dist/` through Vite preview.
- Added stable `test`, `test:watch`, `test:e2e`, `test:e2e:install`, `build`, and `typecheck` package scripts.
- Command execution results remain to be recorded before changing this Task to `done`.
