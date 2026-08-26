# PRODUCT-ADR-SYSTEM-003: Use an Agari fork as the production riichi scoring engine

- **status**: accepted
- **date**: 2026-08-26
- **depends_on**: PRODUCT-ADR-SYSTEM-001
- **supersedes**:
- **migrated_to_spec**: `spec:product.system.contracts.agari_fork`

## Context

The production PWA needs a local browser scoring engine that can evaluate a completed Japanese riichi hand with explicit winning tile, open/concealed meld semantics, red fives, dora indicators, round/seat wind, ron/tsumo, riichi/double-riichi, ippatsu, rinshan, chankan, haitei/houtei, tenhou/chiihou, fu detail, limits, and payment breakdown.

The product also owns an explicit `ScoringRuleProfile`. The current default requires open tanyao, aka/dora/ippatsu, kiriage mangan, kazoe-yakuman disabled with sanbaiman cap, multiple actual yakuman, double-yakuman variants scored as one yakuman, double-wind pair 2 fu, chiitoitsu 25 fu, pinfu-tsumo 20 fu, and open pinfu-shaped ron minimum 30 fu.

The previously considered `@pai-forge/riichi-mahjong` package does not provide enough of the required situation-yaku/rule/dora behavior through its public scoring API to remain a thin scoring dependency.

Agari provides a Rust scoring core plus a browser-targetable WASM wrapper. Its upstream implementation already includes hand decomposition, ordinary/chiitoitsu/kokushi shapes, winning-tile wait interpretation, situational yaku, red fives, dora/ura/aka breakdown, fu breakdown, actual yakuman stacking, and ron/tsumo payment calculation.

Upstream Agari still fixes several rule choices that differ from the product profile and exposes some WASM result semantics through display strings rather than stable machine discriminants.

## Decision

Use a pinned mjtensu-maintained fork of `agari-industries/agari` as the production riichi scoring engine.

Compile the fork's WASM target for the PWA and isolate it behind the library-independent `ScoringService` boundary.

Maintain the canonical mjtensu Agari fork in a separate Git repository rather than vendoring the Rust source tree or using a Git submodule inside mjtensu. Production provenance records the exact full upstream-base commit SHA and exact full fork commit SHA; mutable branches and tags are not production pins.

Build the browser package explicitly from the selected fork revision in release mode and commit the generated JavaScript/WASM/type package into mjtensu under `vendor/agari-wasm/`. Ordinary frontend development and production builds consume that committed package and do not rebuild Agari or require Rust tooling.

Commit machine-readable artifact provenance alongside the generated package, including source revisions, stable ABI version, WASM SHA-256/size, toolchain versions, and release profile. Upstream upgrades are explicit maintenance events and are accepted only after the upstream-compatible/fork tests and mjtensu scoring golden compatibility gate pass.

Keep the fork narrow. Preserve upstream decomposition, yaku, fu, and payment algorithms except where explicit product rule semantics require parameterization.

The required fork delta is normative in `spec:product.system.contracts.agari_fork` and includes:

- explicit rule configuration for product-variable scoring rules;
- kiriage-mangan support;
- configurable counted-yakuman behavior;
- configurable double-yakuman variants;
- configurable multiple-yakuman aggregation;
- configurable double-wind pair fu;
- explicit open-tanyao, dora, aka-dora, and ippatsu switches;
- rule-aware actual-yakuman unit calculation rather than relying only on 13/26-han encoding;
- stable tagged WASM outcomes/yaku codes/score-level codes;
- a scoring-independent winning-shape WASM entry point.

The product adapter maps its one undifferentiated indicator set to Agari's ordinary indicator input and supplies no Agari ura-indicator input. This preserves the product's existing responsibility: the user supplies the indicators that should count, while the product does not retain their visible/ura/kan source.

## Rationale

Agari matches the product's required scoring domain substantially more closely than PaiForge while already supporting browser deployment through WASM.

Using the existing Rust scoring implementation avoids creating a second mahjong solver and score calculator in the TypeScript application. The remaining upstream/product mismatches are concentrated in rule-policy parameterization and browser ABI normalization rather than in core hand evaluation.

Maintaining rule configuration inside the Rust scoring engine is preferable to applying score corrections in TypeScript after Agari returns. Post-hoc correction would duplicate scoring rules across two languages and could produce inconsistencies between best-decomposition selection, yaku/fu totals, limit classification, and payment.

A stable fork-owned WASM ABI also prevents UI/Application logic from depending on English error text or result display names.

Keeping the fork source in its own repository preserves a clean upstream merge/rebase boundary. Committing only the generated browser package into mjtensu lets Recognition, Application, and UI work proceed without a Rust checkout or `wasm-pack`, while exact source SHAs and artifact hashing keep the generated binary traceable and reproducible.

## Rejected alternatives

### Continue with PaiForge

PaiForge can provide useful hand decomposition and part of ordinary scoring, but the required product semantics would require substantial surrounding scoring logic or a deeper fork. In particular, the previously inspected public path did not cover the full situational-yaku, red-five, and rule-profile behavior required by the product.

That would make the adapter responsible for too much mahjong logic and weaken the intended scoring-library isolation.

### Implement scoring directly in TypeScript

A product-owned TypeScript engine would remove the external runtime dependency but would require mjtensu to own decomposition, yaku, fu, limit, and payment correctness from scratch.

The current product does not gain enough from that ownership to justify replacing an existing near-fit engine.

### Use upstream Agari without a fork

Upstream behavior currently differs from the product profile for kiriage mangan, counted yakuman, double-yakuman variants, and double-wind pair fu. Its WASM wrapper also returns important semantic distinctions through strings.

Correcting those differences only in the TypeScript adapter would create duplicated or post-hoc scoring semantics. A narrow fork keeps one scoring authority.

### Keep the fork as a Git submodule in mjtensu

A submodule would preserve source pinning but would make ordinary mjtensu checkout/bootstrap and parallel frontend work depend on nested-repository state that the frontend does not otherwise need.

The product only consumes the browser artifact at runtime, so source ownership remains cleaner in the dedicated fork repository while mjtensu pins the generated package through explicit provenance.

### Rebuild Agari during every frontend build

Making the normal PWA build invoke Cargo/`wasm-pack` would couple unrelated frontend work to the Rust toolchain and increase build/setup cost without improving scoring correctness.

An explicit artifact-refresh workflow keeps scoring-source changes reproducible while ordinary frontend builds consume a fixed reviewed artifact.

### Use a full game/simulation engine

A larger riichi game engine could also calculate scores but would introduce table/game-state concerns not required by the product's completed-hand calculator flow.

Agari's focused scoring core is a better responsibility match.

## Consequences

The production build gains a Rust/WASM scoring artifact in addition to the TypeScript application and ONNX recognition artifacts.

Scoring-service construction must ensure the WASM module is initialized before exposing the synchronous `ScoringService` operations. WASM loading itself remains a bootstrap/runtime concern and does not make preview/calculation asynchronous.

The project owns maintenance of a small Agari fork and must deliberately merge upstream changes rather than silently following upstream `main`.

The mjtensu repository contains generated Agari browser artifacts in addition to source code. Artifact refreshes must update the generated package and provenance together; reviewers can tie the consumed WASM back to the exact fork and upstream revisions.

Ordinary frontend builds remain independent of Rust/Cargo/`wasm-pack`, while maintainers performing an Agari artifact refresh need the pinned Rust/WASM build toolchain.

Every fork update must run upstream tests plus mjtensu fork/rule/ABI compatibility tests and the production scoring golden corpus before the new artifact becomes authoritative.

The adapter becomes simpler than the prior PaiForge design: Agari scoring operates on tile kinds plus red notation, so no temporary mapping from `TileInstanceId` to a concrete 136-tile library ID is required.

## Evidence

Upstream Agari documents a Rust scoring pipeline containing decomposition, wait detection, yaku/dora, fu, and payment logic and provides a separate `agari-wasm` crate for browser use.

The upstream WASM request accepts winning tile, ron/tsumo, riichi/double-riichi, ippatsu, round/seat wind, dora/ura indicators, last-tile, rinshan, chankan, tenhou, and chiihou conditions. Its output exposes yaku, han, fu, dora/ura/aka breakdown, payment, counted-yakuman status, and aggregate fu breakdown.

Upstream score-level logic currently treats 13+ non-yakuman han as counted yakuman and has ordinary mangan thresholds without kiriage. Upstream pair-fu logic awards 4 fu to a double-value wind pair. Upstream yaku values encode Daisuushii, Kokushi 13-wait, Suuankou Tanki, and Junsei Chuuren as 26-han-equivalent double yakuman. These are the principal rule-policy mismatches addressed by the fork contract.
