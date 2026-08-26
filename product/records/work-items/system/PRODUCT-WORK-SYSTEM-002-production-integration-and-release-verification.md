# PRODUCT-WORK-SYSTEM-002: Production integration and release verification

- **status**: not_started
- **date**: 2026-08-26
- **source_refs**:
  - PRODUCT-ADR-SYSTEM-002
  - `spec:product.system.contracts.pwa_cache_update`
  - `spec:product.system.contracts.testing_strategy`
  - PRODUCT-WORK-RECOGNITION-001
  - PRODUCT-WORK-SCORING-001
  - PRODUCT-WORK-APPLICATION-001
  - PRODUCT-WORK-UI-001
- **impact_refs**: []
- **tasks**:
  - PRODUCT-TASK-SYSTEM-002-01
  - PRODUCT-TASK-SYSTEM-002-02
  - PRODUCT-TASK-SYSTEM-002-03
  - PRODUCT-TASK-SYSTEM-002-04
  - PRODUCT-TASK-SYSTEM-002-05
  - PRODUCT-TASK-SYSTEM-002-06

## Goal

Integrate the completed Recognition, Scoring, Application, and UI implementations into one production PWA and execute the browser, PWA, real-device, and performance release gates.

## Boundary

This Work Item owns concrete service composition, production asset/package wiring, PWA cache/update implementation, real-service browser integration, target-device verification, and release-level acceptance evidence.

It does not reopen feature semantics already fixed by Specifications. A failed release gate that requires changed product behavior returns to an explicit decision/spec route rather than being silently patched inside integration.

## Impact Scope

| target | impact |
|---|---|
| production composition root | Wire real Camera, Recognition, Scoring, Application, and UI implementations. |
| production model/WASM assets | Pin and package concrete model-set and Agari WASM artifacts for the build. |
| PWA/service worker | Implement build-pinned shell/model cache and update lifecycle. |
| browser integration tests | Execute real-service integration and PWA offline/update checks. |
| iPhone 13 acceptance | Record real-camera/full-pipeline functionality and performance evidence. |

## Task flow

```text
SYSTEM W001 review PASS
RECOGNITION W001 review PASS
SCORING W001 review PASS
APPLICATION W001 review PASS
UI W001 review PASS
        |
        +-> I01 real service composition
        +-> I02 production asset + PWA cache/update wiring

I01 + I02 -> I03 real-service browser/PWA integration verification
I01 + I02 -> I04 target-device functional acceptance
I03 + I04 -> I05 target-device complete-pipeline performance/release gate
I05 -> I06 independent integrated release review
```

Real-device functional and browser/PWA verification may proceed in parallel once the production build is wired.

## Task Candidates

| task | task type | responsibility | dependency |
|---|---|---|---|
| PRODUCT-TASK-SYSTEM-002-01 | implementation | Compose the real production services/runtime/store/router dependency graph. | all feature/bootstrap integrated reviews |
| PRODUCT-TASK-SYSTEM-002-02 | implementation | Implement concrete production asset pinning plus PWA cache/update behavior for shell, models, and Agari WASM. | all feature/bootstrap integrated reviews |
| PRODUCT-TASK-SYSTEM-002-03 | verification | Execute real-service browser E2E plus PWA offline/update/build-coherence checks. | I01, I02 |
| PRODUCT-TASK-SYSTEM-002-04 | verification | Execute iPhone 13 real-camera/full-flow functional acceptance. | I01, I02 |
| PRODUCT-TASK-SYSTEM-002-05 | verification | Measure the complete production recognition pipeline on iPhone 13 and decide the accepted release performance gate without inventing a new latency threshold. | I03, I04 |
| PRODUCT-TASK-SYSTEM-002-06 | review | Independently review the integrated production release state and all required release Evidence. | I05 |

## Completion Condition

- Real production services are composed without violating public module boundaries.
- Concrete model-set and Agari WASM artifacts are reproducibly pinned to the production build.
- PWA shell/model caching and update behavior matches the cache/update contract.
- Real-service browser/PWA integration verification is PASS.
- iPhone 13 functional acceptance is PASS.
- Complete three-model production recognition timing is measured on iPhone 13 and the accepted 100 ms request-cadence gate is PASS, or the Work Item remains incomplete pending performance/spec resolution.
- The independent integrated release review is PASS with no unresolved findings.

## Evidence

- PRODUCT-ADR-SYSTEM-002 fixes build-pinned manifests and deferred model caching.
- The PWA cache/update and production testing contracts define the integration/release gates.
- The four feature Work Items supply independently reviewed implementation boundaries consumed by this Work Item.
