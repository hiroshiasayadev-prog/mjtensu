# PRODUCT-TASK-SCORING-001-06: Verify Agari golden compatibility

- **status**: done
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-SCORING-001
- **task_type**: verification
- **estimate**: 1d
- **depends_on**:
  - PRODUCT-TASK-SCORING-001-04
  - PRODUCT-TASK-SCORING-001-05
- **outputs**:
  - PRODUCT-TASK-SCORING-001-06

## Goal

Execute the complete scoring golden corpus through the real mjtensu Agari fork, generated WASM, and production TypeScript ScoringService as one objective compatibility gate.

## Work

- Build/load the pinned production Agari fork/WASM artifact.
- Execute every golden fixture through the real TypeScript adapter/service path where applicable.
- Verify stable yaku codes/awarded han, dora/aka counts, FuCalculation, LimitClassification, and payment outputs against explicit expected values.
- Verify all rule-switch pairs and yakuman-policy interactions.
- Verify winning-shape, no-yaku, and non-winning outcomes remain distinct.
- Execute upstream/fork Rust tests and focused TypeScript scoring tests as prerequisites to the integrated verdict.
- Record exact revision/artifact provenance and an overall PASS, FAIL, or validly BLOCKED result.

## Done condition

Every required golden fixture and predefined scoring compatibility check has an observed result and the overall gate is PASS, FAIL, or validly BLOCKED.

## Verification

| check | expected result |
|---|---|
| pinned upstream/fork provenance resolved | exact revisions recorded |
| upstream + fork Rust tests | PASS |
| stable WASM ABI tests | PASS |
| TypeScript scoring focused tests | PASS |
| golden corpus schema/coverage check | PASS |
| full golden corpus through real fork/WASM/adapter | every case matches expected semantic result |
| strict typecheck/lint/architecture gate | PASS |

The overall verdict is PASS only when every required check is PASS.

## Evidence

- `spec:product.system.contracts.testing_strategy` requires real-engine golden verification rather than fake-only scoring tests.
- Exact Agari revisions, WASM artifact identity, corpus version, command outputs, mismatches if any, and the final verdict are recorded here when executed.

### Verification attempt: 2026-08-26

**Overall verdict: BLOCKED**

S06 cannot execute its objective real-fork/WASM/production-adapter compatibility gate yet because both declared prerequisites are incomplete in the current working tree.

| check | observed result |
|---|---|
| S04 golden corpus prerequisite | **BLOCKED** — `PRODUCT-TASK-SCORING-001-04` is still `in_progress`; its corpus/schema files exist, but its own Evidence explicitly says objective TypeScript/Vitest execution is pending. |
| S05 production adapter prerequisite | **BLOCKED** — `PRODUCT-TASK-SCORING-001-05` is still `not_started`, and `product/frontend/src/scoring/` currently contains only the public `index.ts` contract rather than the production Agari loader/adapter/service required by this gate. |
| production WASM artifact/provenance | **BLOCKED** — the required committed `vendor/agari-wasm/` production package and machine-readable provenance manifest are not present in the mjtensu tree, so there is no production artifact identity to verify or load. |
| local Agari source provenance available for diagnosis | observed upstream checkout base `a0a9ce15cdf1bea6e7e158bbac1adb4e7a33a547`; S02 commit `dcf34dd30dccf8bbe249efbee05fbd5056dede63`; current S03 local fork HEAD `3e1ff9fe24e867b444cea244f55ebfbc0357ae22`. This is local checkout evidence only, not a substitute for the required production provenance manifest. |
| upstream + fork Rust tests | not re-executed by S06 because the production artifact/adapter gate is not runnable. S03 records its earlier `cargo test --workspace` PASS (376 tests) and focused ABI PASS (22 tests). |
| stable WASM ABI tests | not re-executed by S06; S03 records the earlier focused ABI PASS. |
| TypeScript scoring focused tests | **BLOCKED** — S05 implementation/tests are not present yet. |
| golden corpus schema/coverage check | implementation is present, but S04 records execution pending; S06 does not promote that to PASS without observed execution evidence. |
| full golden corpus through real fork/WASM/adapter | **BLOCKED** — no production adapter/service and no committed production WASM package are available to run. |
| strict typecheck/lint/architecture gate | not executed as an S06 acceptance gate because the required S05 production path is absent. |

No scoring mismatch verdict is possible from the current tree: the compatibility runner would have to invent S05 API/artifact details, which would violate the declared dependency and would not test the production path.

Resume S06 after S04 records its corpus verification PASS and S05 lands the real production loader/adapter/service plus committed WASM artifact/provenance. At that point this section must be replaced or extended with the exact commands/results for every required check and a final PASS or FAIL verdict.

### Verification resumed: 2026-08-27

- S05 is now `done`; the production input/result adapter, scoring service, WASM initialization boundary, focused tests, corrected default rule profile, typecheck, and architecture evidence are present.
- S04 is now `done`; its corpus schema/coverage verification passed 5/5 tests on 2026-08-27.
- `product/frontend/test/scoring-golden-real-wasm.test.ts` now executes every V1 golden fixture through the real generated Agari WASM module and the production TypeScript `loadAgariScoringService` / adapter / ScoringService path. It checks the dedicated winning-shape API, preview outcome, and exact normalized `ScoringCalculation` for scored cases.
- The generated local WASM package currently exists at `external/agari/web/src/lib/wasm/` and is sufficient for the objective real-engine compatibility runner without introducing a fake engine.
- The canonical committed production `vendor/agari-wasm/` package and provenance manifest required by S01/S09 are still absent. The final S06 verdict therefore cannot become PASS until artifact placement, SHA-256/size/toolchain provenance, and `loadProductionScoringService()` consumption are verified.
- First real-WASM execution on 2026-08-27 reached the actual engine: 5/49 cases passed and all 44 scored-result cases failed at payment normalization with `ron payment fields are inconsistent`, `dealer-tsumo payment fields are inconsistent`, or `non-dealer-tsumo payment fields are inconsistent`. This was traced to stable ABI serialization rather than score semantics: `AgariPaymentInfoV1` declares absent payment branches as `null`, while the default `serde_wasm_bindgen::to_value` representation serializes Rust `Option::None` as JavaScript `undefined`. The TypeScript adapter correctly rejects that undeclared runtime shape.
- `external/agari/crates/agari-wasm/src/abi_v1.rs` was corrected to serialize stable V1 responses with `serde_wasm_bindgen::Serializer::new().serialize_missing_as_null(true)`, preserving the declared `number | null` ABI. The generated WASM must be rebuilt before repeating the integrated corpus gate.
- After rebuilding that ABI correction, the focused ABI suite passed 22/22 and the real-WASM corpus advanced to 37/49 PASS. The remaining 12 failures separated into three bounded causes rather than a general scoring mismatch: Pinfu-tsumo `raw_total` was emitted as 0 despite `base=20,total=20`; two Ittsu fixtures incorrectly expected +2 wait fu for a 78->9 ryanmen completion; the Baiman fixture omitted an actually awarded Pinfu; and six failures compared equivalent yaku identities in a different engine iteration order.
- `external/agari/crates/agari-core/src/scoring.rs` now keeps Pinfu-tsumo aggregate fu internally consistent by setting `raw_total=20`, with a regression assertion in `test_fu_pinfu_tsumo`.
- The Ittsu golden cases now expect zero wait fu and corrected raw totals, because the selected 9s completes 789s from 78s as ryanmen. The Baiman case now includes Pinfu, 9 total han, and the corresponding 20-fu Pinfu-tsumo breakdown while remaining Baiman with unchanged payments.
- The real-WASM runner now compares yaku identity/awarded-han semantics independent of array iteration order. This follows `spec:product.scoring.result`, which permits ordered **or otherwise presentation-ready** yaku, and `spec:product.system.contracts.testing_strategy`, which requires semantic yaku identity/han compatibility rather than one concrete engine ordering.

### Integrated compatibility result: 2026-08-27

**Overall verdict: BLOCKED on production artifact provenance only. Scoring compatibility checks are PASS.**

After the bounded ABI/core/fixture corrections above, the complete executable scoring gate is green:

| check | observed result |
|---|---|
| upstream + fork Rust tests | **PASS** — `cargo test --workspace`: agari core 296/296, CLI 31/31, agari-wasm 49/49, doc tests 0 failures. |
| stable WASM ABI tests | **PASS** — included in workspace result; all 22 `abi_v1` tests pass within the 49 agari-wasm tests. |
| generated release WASM | **PASS** — `wasm-pack build crates/agari-wasm --target web --out-dir ../../web/src/lib/wasm` completed successfully in release profile with wasm-opt. |
| golden corpus schema/coverage | **PASS** — `scoring-golden-corpus.test.ts`: 5/5. |
| full golden corpus through real generated WASM + production TypeScript adapter/service | **PASS** — `scoring-golden-real-wasm.test.ts`: 49/49. |
| combined focused TypeScript golden run | **PASS** — 2 files / 54 tests / 0 failures. |
| strict TypeScript check | **PASS** — `npm run typecheck`. |
| lint / architecture gate | **PASS** — `Architecture import boundaries: OK (51 source files checked)`. |
| pinned production artifact/provenance | **BLOCKED** — the tested generated package still lives under `external/agari/web/src/lib/wasm/`; the contract-required committed `vendor/agari-wasm/` package and machine-readable provenance manifest are not present. The corrected fork revision is now committed as `fb362b6db416e67984cdb36f704d8ebf6657662e`; generated WASM identity is SHA-256 `0e3297ed5f6807eac4d7369eb5846bc17e5ea4851470bf9d40c78ec6030e277c`, 200739 bytes, built with `rustc 1.98.0 (88d9e12ae 2026-08-18)` and `wasm-pack 0.15.0`. The remaining provenance blocker is repository identity: `git remote -v` still points both fetch/push `origin` to upstream `https://github.com/agari-industries/agari.git`, so there is no truthful canonical mjtensu `forkRepository` to record for this fork-only commit. |
| `loadProductionScoringService()` against committed production artifact | **BLOCKED** — no contract-authoritative committed `vendor/agari-wasm/` artifact exists yet to consume. |

The scoring implementation is semantically compatible with the complete V1 golden corpus. The fork corrections are now committed and the generated artifact hash/size/toolchain identity is known. S06 cannot truthfully report overall PASS until the remaining source/artifact-management contract is satisfied: establish a canonical mjtensu fork repository identity for commit `fb362b6db416e67984cdb36f704d8ebf6657662e`, materialize the generated package under `vendor/agari-wasm/`, record the complete provenance manifest, and verify the production loader consumes that committed package without requiring the Rust checkout.

### Production artifact finalization: 2026-08-27

**Overall verdict: PASS**

- The canonical fork repository is established and configured locally as `origin = https://github.com/hiroshiasayadev-prog/mjtensu-agari.git`; upstream remains `https://github.com/agari-industries/agari.git`. This resolves the previously missing `forkRepository` provenance field for fork commit `fb362b6db416e67984cdb36f704d8ebf6657662e`.
- Repo-root `vendor/agari-wasm/` is the canonical committed package location, and `vendor/agari-wasm/provenance.json` records schema V1, upstream/fork repositories and exact revisions, ABI `v1`, WASM SHA-256 `0e3297ed5f6807eac4d7369eb5846bc17e5ea4851470bf9d40c78ec6030e277c`, byte size `200739`, `rustc 1.98.0 (88d9e12ae 2026-08-18)`, `wasm-pack 0.15.0`, and release profile.
- `loadProductionScoringService()` now consumes the statically analyzable `@agari-wasm/agari_wasm.js` vendor dependency. Vite/TypeScript/Vitest resolve `@agari-wasm` directly to repo-root `vendor/agari-wasm`, making the committed package a normal frontend dependency while ordinary builds remain independent of Rust/Cargo/wasm-pack and the external fork checkout.
- `test/scoring-production-wasm-artifact.test.ts` verifies the committed binary against manifest SHA-256/byte identity and invokes `loadProductionScoringService()` itself against the vendor module/binary before scoring an accepted golden fixture.
- Final production-artifact verification passed on 2026-08-27: `scoring-production-wasm-artifact.test.ts` 2/2, `scoring-golden-corpus.test.ts` 5/5, and `scoring-golden-real-wasm.test.ts` 49/49; 3 files / 56 tests / 0 failures overall.
- `npm run typecheck` passed.
- `npm run lint` passed with `Architecture import boundaries: OK (51 source files checked)`.
- `npm run build` passed with Vite 8.2.2; the production bundle emitted `dist/assets/agari_wasm_bg-CaQzJWDG.wasm` at 200.73 kB, demonstrating that the committed vendor WASM is consumed by the production build path.
- All S06 verification checks are therefore PASS: exact source/artifact provenance is resolved, Rust/ABI prerequisites are green, the full golden corpus matches through the real engine and production TypeScript service path, and the canonical committed production artifact is integrity-checked and bundled successfully.
