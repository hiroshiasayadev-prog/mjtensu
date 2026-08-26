# Overview: mjtensu system specification

- **id**: `spec:product.system`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product`

## What this is

Placement router for internal system concepts and implementation-facing contracts that must remain stable enough to prevent feature code from reconstructing product semantics ad hoc.

Product specifications remain the authority for externally meaningful behavior. This system specification fixes the internal vocabulary, dependency boundaries, state machines, and public TypeScript-facing contracts used to implement that behavior.

## Document kinds

| kind | owns |
|---|---|
| `architecture.md` | Module boundaries, dependency direction, adapter isolation, public-entry-point rules, and enforceable import constraints. |
| `concepts/` | Canonical internal vocabulary, semantic distinctions, invariants, and state models. |
| `contracts/` | Public module boundaries and implementation-facing signatures. Contracts may use exact TypeScript shapes when that precision prevents accidental coupling. |

## Current topics

| title | kind | ref | summary |
|---|---|---|---|
| System architecture | Architecture | `spec:product.system.architecture` | Dependency direction and module-isolation rules. |
| Canonical tile model | Concept | `spec:product.system.concepts.canonical_tile_model` | Tile identity, tile instances, recognition meld drafts, and logical meld representation. |
| Recognition model | Concept | `spec:product.system.concepts.recognition_model` | Per-frame observations, recognition drafts, and committed structures. |
| Recognition state | Concept | `spec:product.system.concepts.recognition_state` | Realtime eligibility, stabilization, and automatic confirmation. |
| Coordinate system | Concept | `spec:product.system.concepts.coordinate_system` | Camera-frame normalized geometry used across camera, recognition, and UI. |
| Recognition API | Contract | `spec:product.system.contracts.recognition_api` | One-frame pipeline and realtime-recognizer signatures. |
| Camera API | Contract | `spec:product.system.contracts.camera_api` | Browser camera ownership, preview/session lifecycle, latest-frame acquisition, and 720p ideal capture preference. |
| Scoring API | Contract | `spec:product.system.contracts.scoring_api` | Draft preview, strict calculation input/output, explicit rule profile, and concrete scoring-library isolation. |
| Agari scoring adapter | Contract | `spec:product.system.contracts.agari_adapter` | Private translation from mjtensu scoring semantics to/from the production Agari fork, including tile/meld serialization, condition/rule mapping, and result normalization. |
| mjtensu Agari fork | Contract | `spec:product.system.contracts.agari_fork` | Required semantic delta from upstream Agari: explicit product rule config, rule-aware yakuman/limit behavior, stable WASM result codes, and scoring-independent winning-shape validation. |
| Application scoring-session API | Contract | `spec:product.system.contracts.application_session_api` | Active session state, semantic commands, result invalidation, preview orchestration, and calculation orchestration. |
| Recognition model runtime | Contract | `spec:product.system.contracts.model_runtime` | Model-set manifest, background asset prefetch, app-lifetime inference sessions, and per-model provider fallback. |
| Runtime errors | Contract | `spec:product.system.contracts.runtime_errors` | Feature-owned camera, recognition-runtime, and scoring failure taxonomies separated from normal semantic states. |
| Scoring condition policy | Contract | `spec:product.system.contracts.scoring_condition_policy` | Shared condition normalization and control-availability rules used by Application and UI. |
| Tile correction editor API | Contract | `spec:product.system.contracts.correction_editor_api` | Permissive correction draft, semantic edit commands, local validation targets, and validated structure commit. |
| PWA cache and update lifecycle | Contract | `spec:product.system.contracts.pwa_cache_update` | Build-pinned model manifest, deferred ONNX caching, offline behavior, and non-disruptive service-worker updates. |
| Production testing strategy | Contract | `spec:product.system.contracts.testing_strategy` | Unit, contract, browser-E2E, real-device, architecture, scoring-golden, and release verification requirements. |

## Placement rules

- Product-visible behavior belongs in `product/records/spec/{recognition,scoring,application,ui}` first.
- A system concept explains what an internal value or state means and what invariants it carries.
- A system contract fixes the boundary through which another module may consume that concept.
- Concrete model architecture, checkpoints, thresholds, datasets, and experiments remain in ADRs, investigations, or implementation unless they alter a system boundary.
- Concrete framework component names, CSS values, and private helper functions do not belong here.

## Current pending contracts

No additional cross-feature system contract from the current design pass is intentionally pending. Product `YakuId`, `FuCalculation`, and `LimitClassification` declarations are fixed by `spec:product.system.contracts.scoring_api`; future changes to those public result semantics revise that existing contract rather than introducing a new cross-feature contract.
