# PRODUCT-TASK-SYSTEM-002-04: Verify iPhone 13 functional acceptance

- **status**: in_progress
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-SYSTEM-002
- **task_type**: verification
- **estimate**: 1d
- **depends_on**:
  - PRODUCT-TASK-SYSTEM-002-01
  - PRODUCT-TASK-SYSTEM-002-02
- **outputs**:
  - PRODUCT-TASK-SYSTEM-002-04

## Goal

Execute target-device functional acceptance of the installed production PWA on iPhone 13 using the real camera, production recognition models, production Agari WASM, and full scoring flow.

## Work

- Install/open the production PWA on iPhone 13 Safari/PWA mode and record environment/build identities.
- Verify camera permission/startup and the landscape Recognition capture layout.
- Verify actual model loading/provider selection and live recognition overlays.
- Verify a stable recognized structure automatically transitions to Conditions without a shutter or extra confirmation.
- Verify winning-tile/condition correction and successful score calculation through real Agari WASM.
- Verify Result presentation and at least one condition correction/recalculation path.
- Verify installed/offline behavior after required shell/model assets have been cached.
- Exercise a recoverable camera/runtime retry path when it can be induced safely and reproducibly; record BLOCKED for only that subcheck if the environment cannot induce it without changing the production build.
- Record expected/observed outcomes and one overall PASS, FAIL, or validly BLOCKED verdict.

## Done condition

Every predefined target-device functional check has an observed result and the overall verification verdict is PASS, FAIL, or validly BLOCKED under the production testing strategy.

## Verification

| check | expected result |
|---|---|
| iPhone 13 Safari/PWA production startup | PASS |
| camera permission/startup | PASS |
| landscape fixed capture regions | PASS |
| production model initialization/provider selection | PASS |
| live boxes/identity/meld feedback | PASS |
| three-stable-result automatic Conditions transition | PASS |
| Conditions selection/condition edit + real score calculation | PASS |
| Result display + recalculation path | PASS |
| cached offline application/Recognition availability | PASS |
| recoverable retry path | PASS or explicitly BLOCKED only when safe induction is unavailable |

The overall result follows the predefined acceptance-gate rules; an unexecuted required check is not silently treated as PASS.

## Evidence

- `spec:product.system.contracts.testing_strategy` selects iPhone 13 Safari/PWA as the initial real-device release acceptance environment.
- Device OS/browser/PWA mode, build/model-set/WASM identities, selected execution providers, screenshots/log notes where useful, observed results, and final verdict are recorded here when executed.
- Timing/performance acceptance is deliberately separate in I05.

### Acceptance session: 2026-08-27

Production identities fixed before target-device execution:

- production build asset version: `e15bf73e46ef0d48`
- production asset manifest: `production-assets-e15bf73e46ef0d48.json`
- recognition model set: `recognition-v1-2026-08-27`
- detector runtime spec: `nanodet-plus-m-320-v1`
- tile-classifier runtime spec: `c8-tile-35-v1`
- red-five-classifier runtime spec: `c8-red-five-v1`
- Agari upstream commit: `a0a9ce15cdf1bea6e7e158bbac1adb4e7a33a547`
- Agari fork commit: `fb362b6db416e67984cdb36f704d8ebf6657662e`
- Agari WASM SHA-256: `0e3297ed5f6807eac4d7369eb5846bc17e5ea4851470bf9d40c78ec6030e277c`

Target-device observations remain pending. Do not infer device OS, installed-PWA mode, selected ONNX execution providers, camera behavior, recognition success, offline behavior, or retry behavior from desktop/browser verification.

The current production UI proves runtime readiness but does not surface the selected provider identity. Therefore the provider-selection row cannot be marked PASS from UI observation alone; target-device provider evidence must be obtained through a device-observable diagnostic mechanism rather than inferred from manifest preference order.

#### Target-device observation record

| check | observed result |
|---|---|
| iPhone 13 / iOS version / Safari or installed-PWA mode recorded | pending |
| iPhone 13 Safari/PWA production startup | pending |
| camera permission/startup | **FAIL** — after granting camera permission, the first Recognition attempt consistently enters a recoverable failure and requires manual retry before recognition can proceed; exact normalized error owner/category still needs capture |
| landscape fixed capture regions | **FAIL** — on iPhone 13 landscape the current width-driven `16:9` capture surface plus page chrome exceeds the usable viewport, clips the capture surface vertically, and makes physical alignment difficult |
| recognition-region physical placement usability | **FAIL** — the visible dora/completed-hand rows are separated by an impractically large vertical gap on device, making simultaneous physical placement difficult |
| production model initialization/provider selection | pending — provider identity must be observed, not inferred |
| live boxes/identity/meld feedback | **FAIL** — exact debug capture identified both a duplicate-suppression defect (F-MAJ-09 / I12) and insufficient production-detector meld localization on the same target frame (F-MAJ-10 / I13). The corrected duplicate policy prevents the seven post-NMS candidates from collapsing transitively to one, but the composite-augmented detector still retains only four meld candidates including one oversized merged localization; exact-runtime validation selects the real-capture fine-tune for promotion. |
| three-stable-result automatic Conditions transition | **PASS** — a valid 14-tile closed structure (`123456789m111p22s`) with dora indicator `4p` automatically transitioned to Conditions once the physical tile count was corrected |
| Conditions selection/condition edit + real score calculation | pending |
| Result display + recalculation path | pending |
| cached offline application/Recognition availability | pending |
| recoverable retry path | pending |

No overall PASS verdict is recorded until the target-device observations above are executed and the currently recorded functional FAIL findings are corrected and reverified.

#### Recognition-page iPhone 13 acceptance details

The following checks are part of I04 rather than optional polish. They derive from the Recognition-page/runtime contracts and target-device usability of the fixed semantic capture surface.

| acceptance check | required result |
|---|---|
| first-use permission flow | Granting camera permission on a healthy supported device proceeds into usable camera/recognition startup without a spurious failure that requires a second manual retry. |
| camera/runtime startup ownership | Camera and recognition runtime initialize independently; one side becoming ready or failing does not unnecessarily tear down/restart the healthy side. |
| preview availability during model preparation | Once the camera session is usable, the live preview remains available while Recognition runtime preparation continues. |
| landscape viewport fit | In iPhone 13 landscape, the entire recognition capture surface and its three semantic frames are simultaneously visible without vertical scrolling or clipping. |
| capture surface primacy | Recognition uses the available landscape viewport as the primary camera surface; page chrome does not materially reduce or obscure the area needed to align tiles. |
| safe exit/recovery affordance | A user can abandon Recognition or invoke owner-specific recovery while the camera surface uses the landscape viewport; an app-level viewport-filling layout must retain an accessible exit/recovery control. |
| visible/input-boundary correspondence | A tile visibly placed fully inside a semantic frame is not cut by an additional hidden crop and the overlay maps to the same source boundary used by recognition. |
| practical semantic-region placement | The completed-hand, dora, and meld frames can be populated simultaneously in a normal tabletop setup on iPhone 13 without requiring impractical separation or repositioning solely because of UI geometry. |
| outside-region masking | Camera content outside the three active semantic regions is visibly dimmed/masked while the regions remain clear. |
| all regions continuously active | Completed hand, dora, and meld regions remain active; empty dora/meld regions are accepted without toggles. |
| live feedback | Retained detector candidates show box + recognized identity/unresolved state; recognized/unresolved feedback is readily distinguishable on the target camera surface, and meld grouping feedback is visually distinguishable from individual candidate boxes. Exact colors are implementation-owned. |
| stabilization behavior | Bounding-box jitter alone does not prevent stabilization; three consecutive equal eligible structures commit once with no shutter/OK action. |
| downstream validity boundary | Non-winning/no-yaku structure does not keep Recognition open merely because scoring validity is downstream. |
| route/resource cleanup | Leaving Recognition stops/releases the page-owned realtime run and camera session. |

A behavior that technically reaches Conditions only after device-specific workarounds is not accepted as PASS when the workaround violates the startup, visible-surface, or recovery contract above.

#### Findings routed from this acceptance session

- **F-MAJ-01 — spurious first-use Recognition failure**: after granting camera permission on the target device, the initial Recognition attempt consistently requires a manual retry. A healthy permission/startup path must enter usable Recognition directly; the exact normalized failing owner/category remains to be captured during correction.
- **F-MAJ-02 — landscape capture surface does not fit the target viewport**: the current width-driven `16:9` surface plus page chrome exceeds the usable iPhone 13 landscape height, clips the camera/capture surface vertically, and makes alignment materially difficult.
- **F-MAJ-03 — semantic-region placement is impractical on device**: the current visible dora/completed-hand layout leaves excessive vertical separation for the physical tabletop workflow even though their source regions are independently remapped into the fixed detector composite.
- **F-MAJ-04 — target-device Recognition throughput is far below the accepted cadence**: observed live behavior is roughly `1.2 fps`. Source inspection found detector candidates are classified sequentially with one `await`ed base-classifier inference per candidate plus conditional red-five inference, with per-crop JavaScript preprocessing. This finding is performance-owned and must be corrected/measured separately from I04 functional acceptance.
- **F-MAJ-05 — live Recognition overlays are not sufficiently visually distinguishable**: on the target device the retained candidate bounding boxes appear white regardless of recognized/unresolved state, while meld-group connectors are also white. The current solid/dashed treatment does not provide sufficiently clear visual separation during live camera alignment. The specification does not prescribe exact colors, but it does require detector boxes and meld-group connectors to be visually distinguishable.
- **F-MAJ-06 — live Recognition exposes internal tile codes instead of user-facing tile identity**: recognized candidates currently render canonical implementation labels such as `7z` directly beside the bounding box. The production Recognition surface should present an immediately recognizable tile identity rather than requiring the user to translate internal suit/honor codes. A compact tile-face/icon treatment is preferred when it remains lightweight and locally cached; exact artwork is implementation-owned.
- **F-MAJ-07 — Conditions mobile information architecture is not usable as a scoring-verification surface**: the target-device Conditions page exposes internal instance identifiers such as `recognition:43`, labels the winning-tile selection area ambiguously as `認識牌姿`, renders semantic tile identities as raw codes such as `4p`, duplicates recognized structure and correction presentation without clear visual hierarchy, lacks an ordinary in-flow back affordance, and places current-yaku feedback away from the controls whose effects the user is verifying. The mobile surface must make winning-tile selection, recognized structure, correction, conditions, current-yaku feedback, and calculation hierarchy immediately understandable without exposing internal IDs/codes.
- **F-MAJ-08 — Result mobile presentation does not provide a compact, readable scoring-result hierarchy**: the target-device Result surface does not yet present tile evidence, awarded yaku, fu/han/limit, final points, payment detail, and correction/restart actions with the compact hierarchy expected of a read-only result screen. Tile evidence should use tile faces rather than canonical codes; dora, completed hand/winning tile, and melds need distinct compact rows; yaku names/han need a tightly aligned readable treatment; the final point value must dominate the score block; fu detail must remain available on demand without occupying the default result; and correction/restart actions should remain persistently accessible without competing visually with the score itself.
- **F-MAJ-09 — detector duplicate suppression collapses distinct meld tiles through a merged bridge box**: a production iPhone debug capture reproduced the poor meld recognition with exact detector tensors and raw output. Ordinary IoU NMS retained seven meld-region candidates, but the browser duplicate stage connected all seven through the intersection/smaller-area relation and retained only one confidence winner. Desktop CPU inference on the exact captured tensor reproduced the same raw output, so this is not an iPhone WASM divergence. A large detector box that overlaps multiple smaller candidates which are not themselves duplicate-overlapping must be rejected as a merged bridge before confidence-based pairwise duplicate resolution; transitive connected-component collapse is not accepted. Correction is routed to PRODUCT-TASK-SYSTEM-002-12.
- **F-MAJ-10 — pinned composite-augmented detector is weaker than the available real-capture fine-tune at the deployed runtime operating point**: after correcting duplicate semantics, the recorded iPhone tensor still leaves only four meld candidates from the current production detector and retains an oversized `108.98 x 50.13` merged localization. Exact-runtime held-out comparison at confidence `0.35`, IoU NMS `0.60`, semantic-region assignment, merged-bridge rejection, and pairwise duplicate suppression shows the real-capture fine-tune improves real-capture overall F1 from `0.9725` to `0.9884` and real-capture meld F1 from `0.7273` to `0.9600`; on composite validation it slightly improves overall F1 from `0.9831` to `0.9858` while meld performance remains identical at `0.9968`. The previous aggregate-AP rationale is therefore superseded for the deployed runtime operating point. Artifact promotion is routed to PRODUCT-TASK-SYSTEM-002-13.
- **F-MAJ-11 — meld-row grouping is too sensitive to detector bbox-center jitter**: debug capture `mjtensu-recognition-debug-2026-08-27T17-37-35-246Z.json` recognizes five meld tiles as `7m 8m 9m` and `1p 1p` but reports `unresolved-meld-geometry`. Removing only the redundant per-row fitted-angle hard gate passed deterministic tests but produced no practical target-device improvement, exposing the broader issue: the current algorithm sorts projected observations by `v`, greedily appends only to the current row, and rejects the whole candidate direction if jitter leaves any singleton/oversized row. Correction is routed to PRODUCT-TASK-SYSTEM-002-14, which replaces greedy row cutting with bounded `±45°` common-direction search plus complete-linkage-style row-candidate/exact-cover partition scoring.

### Acceptance usability adjustment verification: 2026-08-27

During target-device acceptance, two observability/usability issues were corrected before continuing the device gate:

- Recognition now surfaces whether the current frame is below commit eligibility, has unresolved meld geometry, or is in stable-result confirmation instead of presenting every non-confirmed state only as `認識しています`.
- Conditions-page tile correction now automatically installs each valid corrected structure; the extra `牌姿を反映` action is no longer required there. Result-origin recognition correction retains its explicit commit because that action owns recalculation/navigation behavior.

User-executed regression/build verification after these changes:

- `npx vitest run test/recognition-page.test.tsx test/tile-correction-ui.test.tsx` — **PASS**, 26/26 tests.
- `npm run typecheck` — **PASS**.
- `npm run build` — **PASS**, Vite 8.2.2 production PWA build completed and generated `sw.js` / Workbox assets.
- application bundle for this acceptance build: `assets/index-D8t1Pnba.js`
- production asset-manifest identity remains `e15bf73e46ef0d48`; that identity pins Recognition/Agari production artifacts and is not by itself a unique source/UI bundle identifier.
- Vite native-config-loader extension notices and the >500 kB chunk notice are non-blocking warnings for this functional acceptance task.

Because the production PWA deliberately leaves an update worker waiting while an existing controlled client is active, target-device continuation must ensure the updated application bundle is actually loaded rather than assuming an already-open installed PWA activated the new worker.

### Detector-promotion continuation identity: 2026-08-28

The next iPhone 13 continuation session must use the detector-promotion build rather than the earlier `recognition-v1-2026-08-27` acceptance build.

- recognition model set: `recognition-v2-2026-08-28`
- production detector: `nanodet-plus-m-320-real-capture-ft10-l10.onnx`
- detector SHA-256: `9587a02dd1bbccfc14a925dc69c66b3c4a34ab628552b840ec113f7899dbf883`
- production build asset manifest: `production-assets-973ac08b074fc268.json`
- application bundle observed in the deterministic production build: `assets/index-CLYLTmbD.js`
- desktop/browser exact-capture regression: **PASS**, `7` retained meld candidates and no retained meld box exceeding the bounded `60`-pixel extent check on the recorded failure tensor
- browser real-artifact gate: **PASS**, model set `recognition-v2-2026-08-28`, all three models on `wasm-simd`, no provider fallbacks

Before judging F-MAJ-09/F-MAJ-10 on device, confirm the installed/served PWA has adopted this new build/model-set identity. Then repeat the physical meld arrangement that produced the prior failure and record whether distinct tile-localized overlays are retained in live Recognition.
