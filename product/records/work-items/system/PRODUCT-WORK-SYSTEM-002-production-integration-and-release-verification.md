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
  - PRODUCT-TASK-SYSTEM-002-07
  - PRODUCT-TASK-SYSTEM-002-08
  - PRODUCT-TASK-SYSTEM-002-09
  - PRODUCT-TASK-SYSTEM-002-10
  - PRODUCT-TASK-SYSTEM-002-11
  - PRODUCT-TASK-SYSTEM-002-12
  - PRODUCT-TASK-SYSTEM-002-13
  - PRODUCT-TASK-SYSTEM-002-14
  - PRODUCT-TASK-SYSTEM-002-15

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
I04 findings -> I07 correct target-device Recognition startup/layout findings
I04 overlay findings -> I09 correct target-device Recognition overlay/tile feedback
I09 -> I10 correct mobile Conditions information architecture
I09 -> I11 correct mobile Result information architecture
I07 + I09 + I10 + I11 -> continue/reverify I04
I04 duplicate-suppression finding -> I12 correct merged-bridge duplicate resolution
I04 detector-artifact finding + I12 corrected postprocess -> I13 promote real-capture detector -> continue/reverify I04
I04 meld-grouping geometry finding -> I14 replace greedy meld-row grouping -> continue/reverify I04
I04 performance finding -> I08 optimize production Recognition throughput
I05 classifier timing + INV-011 -> I15 optimize classifier preprocessing -> continue/reverify I05
I03 + I04 + I08 + I15 -> I05 target-device complete-pipeline performance/release gate
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
| PRODUCT-TASK-SYSTEM-002-07 | correction | Correct iPhone 13 Recognition first-use startup failure and target-device capture-surface/region-layout findings, then return to I04. | I04 findings |
| PRODUCT-TASK-SYSTEM-002-08 | correction | Batch production tile-classifier/red-five inference and remove the target-device sequential throughput bottleneck before I05 measurement. | I04 F-MAJ-04 |
| PRODUCT-TASK-SYSTEM-002-09 | correction | Correct target-device live Recognition overlay differentiation and replace internal tile codes with lightweight user-facing tile identity, then return to I04. | I04 F-MAJ-05, F-MAJ-06; I07 done |
| PRODUCT-TASK-SYSTEM-002-10 | correction | Rework the Conditions mobile information architecture around tile-face winning-tile selection, clear section hierarchy, navigation, and persistent current-yaku/calculation feedback. | I04 F-MAJ-07; I09 |
| PRODUCT-TASK-SYSTEM-002-11 | correction | Rework Result into compact tile/yaku/score cards with han-band feedback, dominant final points, on-demand fu detail, and persistent correction/restart actions. | I04 F-MAJ-08; I09 |
| PRODUCT-TASK-SYSTEM-002-12 | correction | Correct detector duplicate suppression so a large merged bridge cannot transitively collapse multiple spatially distinct tile detections to one confidence winner. | I04 F-MAJ-09 |
| PRODUCT-TASK-SYSTEM-002-13 | correction | Promote the real-capture fine-tuned detector after exact-runtime validation shows substantially better real meld recall with no held-out composite meld regression. | I04 F-MAJ-10; I12 |
| PRODUCT-TASK-SYSTEM-002-14 | correction | Replace jitter-sensitive greedy meld-row cutting with bounded `±45°` common-direction search and complete-linkage-style exact-cover partition scoring. | I04 F-MAJ-11 |
| PRODUCT-TASK-SYSTEM-002-15 | correction | Reduce the classifier preprocessing bottleneck by replacing direct 2D software Lanczos with equivalent separable filtering, then re-measure iPhone 13 preprocessing before considering browser-native resize. | I08; I05 timing; INV-011 |

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
