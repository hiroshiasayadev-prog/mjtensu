# PRODUCT-TASK-SYSTEM-002-05: Verify iPhone 13 recognition performance

- **status**: not_started
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-SYSTEM-002
- **task_type**: verification
- **estimate**: 1d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-002-03
  - PRODUCT-TASK-SYSTEM-002-04
- **outputs**:
  - PRODUCT-TASK-SYSTEM-002-05

## Goal

Measure the complete production recognition path on iPhone 13 and objectively determine whether the accepted 100 ms recognition-request cadence is sustainable without overlapping acceptance-owning evaluations or an accumulating stale-frame queue.

## Work

- Measure the production path with the actual detector, 35-class classifier, and red-five specialist as invoked by representative frames.
- Record device/browser/PWA mode, production build, model-set version, and selected execution provider per model.
- Record a bounded timing distribution or equivalent sample evidence for complete recognition evaluations rather than detector-only or desktop extrapolation.
- Observe scheduler behavior under sustained live capture, including active-evaluation concurrency and stale-frame queue behavior.
- Compare observed behavior only to the already accepted 100 ms request-cadence contract; do not invent a new p95 latency threshold in this Task.
- Record PASS, FAIL, or validly BLOCKED and route a failure to performance implementation or an explicit product/spec decision rather than silently accepting slower behavior.

## Done condition

The target-device performance evidence is complete enough to determine PASS, FAIL, or validly BLOCKED for the accepted recognition cadence and scheduler constraints.

## Verification

| check | expected result |
|---|---|
| environment/build/model-set/provider identities recorded | complete |
| complete production recognition evaluation timing sample | recorded |
| detector + base classifier + red-five specialist represented according to actual invocation path | PASS |
| recognition request cadence | 100 ms target sustainable under accepted scheduler semantics |
| acceptance-owning evaluation concurrency | at most one |
| stale-frame queue | no accumulating required queue |

PASS requires the accepted cadence to be sustainable without violating the one-active-evaluation/stale-frame behavior. If it is not, this Task records FAIL rather than defining a looser threshold.

## Evidence

- `spec:product.recognition.runtime_recognition` owns the 100 ms request-cadence contract.
- `spec:product.system.contracts.testing_strategy` requires complete three-model target-device measurement and explicitly prohibits inventing a new performance threshold here.
- Raw timing observations, summary statistics, provider selections, scheduler observations, and final verdict are recorded here when executed.
