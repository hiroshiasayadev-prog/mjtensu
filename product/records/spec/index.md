# Overview: mjtensu product specifications

- **id**: `spec:product`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `root`

## What this is

Placement router for the current mjtensu product contract.
The product recognizes a scored Japanese riichi mahjong hand from a live PWA camera flow, collects non-image scoring conditions, calculates score, and presents an inspectable result.

## Current contract

| area | owns | must route elsewhere |
|---|---|---|
| `recognition/` | Runtime camera-recognition input, semantic tile-recognition output, fixed capture-region contract, ordering, stabilization, and recognition acceptance. | Training experiments, dataset construction, scoring rules, scoring-session state, page composition. |
| `scoring/` | Library-independent scoring input and scoring result semantics. | Camera behavior, recognition implementation, page navigation, concrete scoring-library APIs. |
| `application/` | One scoring session, preservation and replacement of recognized data and conditions, correction/recalculation behavior, and new-session disposal. | Camera rendering, page layout, scoring formulas, concrete framework state. |
| `ui/` | PWA screen flow, page composition, visible controls, semantic layout, and reusable visible responsibilities. | Learned-model internals, scoring formulas, concrete component files, CSS values, framework state implementation. |
| `system/` | Internal architecture, canonical concepts, state models, and implementation-facing public contracts/signatures. | Product-visible behavior, model-training evidence, concrete private helpers, framework/CSS detail. |

## End-to-end flow

```text
Top
  -> realtime recognition
  -> conditions
  -> score calculation
  -> result
       -> recognition correction -> recalculate -> result
       -> condition correction   -> recalculate -> result
       -> new recognition        -> discard session -> recognition
```

Recognition is shutterless. The camera flow commits a recognition result only after the runtime stability contract succeeds.

## Topics

| title | kind | ref | summary |
|---|---|---|---|
| Runtime recognition | Overview | `spec:product.recognition` | Live capture regions and semantic recognition output. |
| Scoring | Overview | `spec:product.scoring` | Library-independent scoring input and result contracts. |
| Application | Overview | `spec:product.application` | One scoring session and correction/recalculation behavior. |
| UI | Overview | `spec:product.ui` | PWA screens, screen flow, and reusable visible responsibilities. |
| System | Overview | `spec:product.system` | Internal architecture, canonical concepts, and implementation-facing contracts. |

## Placement rules

| question | route |
|---|---|
| What camera content is accepted and what semantic tile result is produced? | `recognition/`. |
| What information is required to calculate score and what semantic result is returned? | `scoring/`. |
| What state survives correction, recalculation, help navigation, or return from result? | `application/`. |
| What is visible on a page, where is it placed, and what can the user operate? | `ui/`. |
| Which detector, classifier, dataset, threshold, model checkpoint, or training procedure is used? | ADR / investigation / implementation, not product spec unless it changes an externally meaningful recognition contract. |
| Which TypeScript component, prop, hook, store, route file, or CSS token is used? | Implementation or internal design. |
| Which internal concept, module boundary, dependency rule, or public TypeScript signature constrains implementation? | `system/`. |

## Dependency direction

- UI consumes application session behavior and scoring/recognition semantics but does not own them.
- Application coordinates recognition results and scoring inputs/results without owning camera rendering or scoring formulas.
- Scoring is independent of camera and UI implementation.
- Recognition is independent of score presentation and scoring-library implementation.
- Concrete ML models and concrete scoring libraries implement these contracts but are not their semantic authority.

## Related records

| ref | relation |
|---|---|
| PRODUCT-ADR-RECOGNITION-001 | Establishes the shutterless staged realtime recognition direction and stability behavior. |
| PRODUCT-ADR-RECOGNITION-002 | Establishes the fixed hand, dora, and meld capture regions and composite detector layout. |
