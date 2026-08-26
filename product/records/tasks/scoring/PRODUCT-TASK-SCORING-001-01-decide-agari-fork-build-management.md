# PRODUCT-TASK-SCORING-001-01: Decide Agari fork build management

- **status**: done
- **date**: 2026-08-26
- **work_item**: PRODUCT-WORK-SCORING-001
- **task_type**: decision
- **estimate**: 0.5d
- **depends_on**: []
- **outputs**:
  - PRODUCT-TASK-SCORING-001-01

## Goal

Fix the production boundary for storing the mjtensu Agari fork, pinning its source revisions, building WASM, and consuming that WASM reproducibly from mjtensu.

## Work

- Selected a separate Git repository as the canonical source of the mjtensu Agari fork; mjtensu does not vendor the Rust source or use a Git submodule for the fork.
- Selected exact full commit SHAs as the production pin for both the imported upstream base revision and the mjtensu fork revision. Branch names and tags may remain human-facing labels but are not production identity authorities.
- Selected an explicit release-WASM refresh workflow: build the chosen fork revision with the fork test suite, produce the release WASM package, and commit the generated JS/WASM/type package into mjtensu for ordinary frontend builds to consume without requiring a Rust toolchain.
- Selected a provenance manifest carrying upstream commit, fork commit, ABI version, WASM SHA-256 and byte size, Rust/wasm-pack versions, and release build profile. A generation timestamp may be retained only as evidence, not artifact identity.
- Selected explicit-only upstream upgrades: no automatic tracking of upstream `main`; an upgrade must choose a new upstream/fork revision and pass upstream/fork tests plus the mjtensu scoring golden corpus before the production pin/artifact is accepted.
- Routed durable rationale as an amendment of PRODUCT-ADR-SYSTEM-003 rather than a new or superseding ADR because the accepted Agari-fork architecture remains unchanged. The normative build/provenance rules are routed to `spec:product.system.contracts.agari_fork`.

## Done condition

All four owned decisions are decided, deferred, or validly blocked, and the selected implementation route is explicit enough that S02 and S03 need no additional contract choice.

## Verification

- Exact upstream and fork revisions are identified by full Git commit SHA.
- The consumed production WASM is tied to the fork revision by the committed provenance manifest and WASM SHA-256.
- Ordinary mjtensu frontend build/test workflows consume the committed generated package and do not require Rust or `wasm-pack`.
- Upstream changes cannot silently enter production because revision changes and artifact refresh are explicit.
- The decision changes source/artifact management only and does not alter the accepted Agari scoring semantics.

## Evidence

| item | state | outcome |
|---|---|---|
| fork source location | decided | Separate canonical Agari fork Git repository; no Rust-source vendoring or Git submodule in mjtensu. |
| source revision pinning | decided | Record the exact full upstream-base SHA and exact full mjtensu-fork SHA; mutable branches/tags are not production pins. |
| WASM build/consumption | decided | Explicit release build from the pinned fork; generated WASM package is committed into mjtensu and consumed directly by ordinary frontend builds. |
| production provenance metadata | decided | Record source SHAs, ABI version, WASM hash/size, toolchain versions, and release profile; timestamp is evidence-only. |
| upstream upgrade policy | decided | Explicit upgrade only; upstream/fork tests and mjtensu scoring golden corpus gate acceptance. |
| ADR routing | decided | Amend PRODUCT-ADR-SYSTEM-003; do not supersede it. Reflect current normative rules in `spec:product.system.contracts.agari_fork`. |

The selected boundary preserves the accepted scoring semantics while making the production source and generated artifact reproducible.
