# Contract: Production testing strategy

- **id**: `spec:product.system.contracts.testing_strategy`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product.system`

## What this is

Cross-feature verification contract for the production mjtensu PWA.

This contract defines which behavior is tested at unit, contract/integration, browser-E2E, and real-device levels; which test tools are used; how deterministic fakes are used to keep feature work parallel; and which verification evidence is required before a production release can be accepted.

It does not require every behavior to be proven through a full browser or device end-to-end test. Each semantic rule should be verified at the lowest deterministic layer that can prove it, with higher layers reserved for integration and device/browser behavior.

## Test toolchain

The production frontend uses:

- Vitest for TypeScript unit and contract/integration tests;
- Testing Library for React component behavior;
- Playwright for browser-level E2E tests;
- `cargo test` for the mjtensu Agari fork;
- Rust/WASM plus TypeScript contract tests for the generated Agari WASM ABI;
- TypeScript compiler checks with `tsc --noEmit` or an equivalent strict no-emit typecheck;
- ESLint or an equivalent configured static rule runner for code-quality and architecture-boundary rules.

The exact package scripts and directory names are implementation-owned. A production Work Item may add supporting utilities, fixture builders, and test helpers without creating a second testing policy.

## Verification layers

### L1: Unit and focused component tests

L1 proves deterministic local behavior without real camera, model assets, service workers, or production WASM/network loading unless the tested unit explicitly owns such a boundary.

Examples include:

- canonical tile conversions and validation;
- duplicate suppression;
- crop/composite coordinate transforms;
- meld grouping and reconstruction;
- stabilization state transitions;
- scoring-condition normalization and control availability;
- correction-draft commands and validation targeting;
- scoring-session state transitions and winning-tile preservation;
- React component interactions with deterministic service fakes;
- result-format/presentation mapping from product result types.

L1 is the preferred layer for exhaustive combinatorial and edge-case coverage of pure semantics.

### L2: Contract and integration tests

L2 proves a real implementation boundary while keeping the rest of the system controlled.

Examples include:

- actual ONNX Runtime loading of the production detector/classifier artifacts against fixed fixture images or tensors;
- actual detector/classifier preprocessing and output decoding;
- actual Agari fork WASM loading and stable ABI normalization;
- golden scoring cases through the real Agari fork and TypeScript scoring adapter;
- application services against deterministic RecognitionService/ScoringService fakes;
- public-entry-point and architecture-boundary tests.

L2 tests should use production artifacts when the contract being verified is specifically about artifact/runtime compatibility.

### L3: Browser E2E

L3 uses Playwright to prove route behavior, visible recovery paths, page-to-page state preservation, correction/calculation flows, and PWA/browser integration that cannot be established by component tests alone.

Feature E2E should prefer deterministic fake services where real camera/model execution is not the behavior under test. This permits UI, Application, Recognition, and Scoring implementation to proceed in parallel behind the already-defined public contracts.

At minimum the browser E2E suite covers:

- Top -> Recognition -> Conditions -> Result primary navigation using controlled recognition/scoring results;
- Recognition preparation and recoverable camera/runtime failure surfaces;
- automatic Recognition -> Conditions history replacement semantics;
- Conditions correction and winning-tile selection;
- no-yaku and invalid-input recovery behavior;
- Result -> condition correction -> Result;
- Result -> recognition correction -> immediate recalculation or Conditions fallback;
- new-recognition session replacement;
- route guards when no scoring session exists;
- Help navigation that does not create or corrupt a session.

Real recognition model execution is not required in every browser E2E case.

### L4: Real-device acceptance

L4 uses the target mobile browser/PWA environment with actual camera capture, actual production ONNX models, actual Agari WASM, and production service composition.

The initial release acceptance device is iPhone 13 Safari/PWA because that device has already been used for recognition feasibility work.

Real-device verification covers at least:

- camera permission/startup and landscape Recognition surface;
- production model loading and provider selection/fallback behavior observable on the device;
- live detector/classifier overlays and automatic stable-result transition;
- full Recognition -> Conditions -> Result scoring flow;
- retry after a recoverable runtime/camera initialization failure when such a failure can be induced safely;
- installed/offline behavior after the required shell/model assets have been cached;
- recognition runtime performance evidence for the complete production three-model pipeline.

L4 acceptance is a release verification responsibility rather than a substitute for deterministic lower-level tests.

## Implementation-Task test ownership

Every `implementation` Task owns the focused automated tests and fixtures required to prove its own Implementation contract.

An implementation Task is not complete merely because production code exists. Its completion boundary requires:

```text
bounded production change
+ focused automated tests/fixtures for that change
+ declared verification passing
```

Production code and its focused tests may remain in one implementation Task when they share one acceptance boundary, as permitted by the Design Records Task authoring contract.

A separate `verification` Task is created when the acceptance gate itself needs independent aggregation, release gating, reusable evidence, post-completion evaluation, device execution, or cross-task failure routing.

Independent `review` Tasks remain distinct from objective verification Tasks. Review owns semantic/code soundness and findings; verification owns predefined pass/fail checks.

## Deterministic fakes and parallel feature work

Public contracts are the parallelization boundary.

UI and Application work must not wait for real ONNX or Agari implementation when their required dependency behavior can be represented by deterministic fakes conforming to the public contracts.

At minimum the test support may provide deterministic fakes for:

- `CameraService`;
- `RecognitionRuntime` / realtime recognition updates;
- `ScoringService`;
- application-lifetime model/runtime preparation state where a UI test needs it.

Fakes must emit product/public contract values rather than concrete ONNX, Agari, browser-media, or adapter-internal types.

A fake-driven passing E2E suite does not prove production adapter compatibility. Real adapter/artifact compatibility belongs to L2 and final L4 verification.

## Recognition test requirements

Recognition implementation must cover deterministic semantics below the full-model layer, including at least:

- fixed-region/composite coordinate mapping;
- detector output decode and region assignment;
- duplicate suppression, including confidence winner selection and non-overlapping neighbors remaining distinct;
- 35-class base-classifier invalid/background handling;
- red-five specialist invocation only for base `5m`, `5p`, and `5s`;
- crop resize/letterbox/normalization parity with the exported classifier artifact contract;
- completed-hand and dora left-to-right ordering;
- meld row grouping and supported tilt behavior;
- concealed-kan two-observation reconstruction into four logical members;
- unresolved/illegal meld identities remaining downstream-correctable rather than recognition-rejected;
- visible observation count/capture-completeness eligibility;
- semantic stabilization independent of bounding-box jitter;
- three-consecutive-eligible-structure commit;
- at most one acceptance-owning inference evaluation and stale-frame dropping/no required frame queue;
- initialization idempotence, concurrent initialization deduplication, retry after failure, and route-leave lifecycle semantics;
- configured execution-provider fallback order.

Actual model contract tests use a bounded fixed fixture set to prove the production ONNX artifacts can be loaded and produce expected output structure/known predictions. Model accuracy research is not rerun as part of every production unit suite.

## Scoring golden-corpus requirements

Before Scoring implementation is accepted, a versioned deterministic golden corpus must cover the product rule and result surface through the real Agari fork.

The corpus covers at least:

- ordinary four-meld-one-pair wins;
- Chiitoitsu;
- Kokushi Musou and the 13-sided variant;
- red-five aka dora;
- indicator dora;
- riichi and double riichi;
- ippatsu;
- menzen tsumo;
- rinshan kaihou;
- chankan;
- haitei and houtei;
- tenhou and chiihou;
- open/closed yaku han differences including kuisagari cases;
- every aggregate fu category exposed by the product `FuCalculation`;
- fixed Chiitoitsu 25 fu;
- pinfu tsumo 20 fu;
- open no-extra-fu 30-fu minimum behavior;
- 4 han 30 fu and 3 han 60 fu with kiriage on/off;
- non-yakuman 13+ han with counted-yakuman on/off;
- every supported double-yakuman variant under both multiplier policies;
- multiple simultaneous actual yakuman with stacking on/off;
- interaction of multiple-yakuman and double-variant policies;
- double-wind pair 2-fu and 4-fu policies;
- ron payment for dealer and non-dealer where applicable;
- dealer tsumo payment;
- non-dealer tsumo payment;
- `not-winning-shape` distinct from `no-yaku`;
- dora-only bonus not creating a scoring yaku;
- stable yaku-code and result normalization through the TypeScript adapter.

Golden expected values are reviewable fixtures, not snapshots of opaque display strings. Yaku identities, han, fu breakdown, limit classification, dora/aka counts, and payment values are asserted semantically.

## Application test requirements

Application tests cover at least:

- scoring-session creation from a committed recognition structure;
- initial Tsumo / East round / East seat / Riichi None / all-situational-false conditions;
- rightmost completed-hand tile as the initial winning-tile selection only;
- user selection of another completed-hand tile;
- preservation of winning tile across identity replacement with stable `TileInstanceId`;
- replacement default when the selected tile is removed or moved out of completed hand;
- condition normalization through the one shared policy;
- score-result invalidation after score-relevant mutation;
- correction-draft temporary malformed states remaining page-local;
- correction commit refusing unsupported/non-winning structure;
- no-yaku remaining distinct from structural-correction validity;
- Result correction behavior that either recalculates or routes to Conditions without restoring stale score state.

## Architecture/static test requirements

The production project must make the architecture rules mechanically testable.

At minimum a failing static/build gate is required for:

- cross-feature private/deep imports where only public entry points are permitted;
- direct UI import of `onnxruntime-web`;
- direct UI/Application import of the concrete Agari WASM binding;
- recognition import of UI implementation;
- storage of ONNX sessions, media resources, or other opaque runtime objects in the Zustand session store when prohibited by the architecture contract;
- TypeScript strict type errors;
- configured lint errors.

The exact ESLint plugin, custom rule, or architecture-test implementation is implementation-owned.

## PWA/browser integration requirements

Release-level browser verification covers:

- application-shell startup;
- build-pinned recognition model manifest coherence;
- deferred ONNX acquisition rather than service-worker-install blocking;
- offline Recognition availability after shell and model assets are cached;
- update availability without forced destruction of an active scoring session;
- no old-JavaScript/new-manifest mixing within one running build;
- route-history behavior defined by `spec:product.ui.screen_flow`.

## Performance acceptance

The recognition product contract targets one recognition request every 100 ms.

The release verification must measure the complete production path on the target real device, including detector plus both classifiers where applicable, rather than extrapolating from desktop or detector-only benchmarks.

This contract does not invent a p95 latency threshold beyond the accepted cadence requirement before that end-to-end measurement exists.

Release verification records at least:

- device/browser/PWA mode;
- model-set version;
- execution provider selected per model;
- measured end-to-end recognition evaluation timing distribution or equivalent bounded sample evidence;
- whether the 100 ms request cadence can be sustained without overlapping acceptance-owning evaluations or an accumulating stale-frame queue.

If the accepted cadence cannot be sustained, release verification does not silently pass. The work returns to performance implementation or an explicit product/spec decision to change the cadence contract.

## Fixture and snapshot policy

Test fixtures must be deterministic and reviewable.

Use semantic fixtures for tiles, structures, scoring cases, normalized runtime results, and fixed recognition images/tensors.

Snapshots may support stable serialized/display output, but a snapshot must not be the only assertion for critical scoring amounts, yaku identities, correction validity, navigation outcomes, recognition commit semantics, or runtime error kind.

Large training datasets and the full classifier/detector research corpus are not copied into the production test suite. Use bounded fixtures selected for the contract being tested.

## Release gate

A production release is eligible for acceptance only when:

- strict typecheck passes;
- lint/architecture gates pass;
- L1 focused tests pass;
- required L2 actual-artifact/WASM contract tests pass;
- required L3 Playwright E2E passes;
- PWA offline/update checks pass;
- required L4 iPhone 13 device checks are recorded as PASS;
- the complete production recognition performance gate is PASS;
- no unresolved independent-review finding remains on a required release Work Item.

A lower-level test passing does not waive a required higher-level release gate, and a manual device success does not waive deterministic unit/contract failures.

## Boundary

| concern | owner |
|---|---|
| Cross-feature production test strategy and release verification layers | This contract. |
| Feature semantics being tested | The owning recognition/scoring/application/UI Specifications. |
| Concrete Task completion and Evidence | Work Item / Task records. |
| Exact test code, fixture paths, scripts, CI implementation | Production implementation. |
| Model research accuracy and training validation | Recognition investigations/training tooling, not this production test contract. |
