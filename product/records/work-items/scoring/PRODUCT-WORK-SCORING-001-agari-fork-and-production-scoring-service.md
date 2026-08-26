# PRODUCT-WORK-SCORING-001: Agari fork and production scoring service

- **status**: done
- **date**: 2026-08-26
- **source_refs**:
  - PRODUCT-ADR-SYSTEM-003
  - `spec:product.scoring.input`
  - `spec:product.scoring.result`
  - `spec:product.system.contracts.scoring_api`
  - `spec:product.system.contracts.agari_fork`
  - `spec:product.system.contracts.agari_adapter`
  - `spec:product.system.contracts.testing_strategy`
- **impact_refs**: []
- **tasks**:
  - PRODUCT-TASK-SCORING-001-01
  - PRODUCT-TASK-SCORING-001-02
  - PRODUCT-TASK-SCORING-001-03
  - PRODUCT-TASK-SCORING-001-04
  - PRODUCT-TASK-SCORING-001-05
  - PRODUCT-TASK-SCORING-001-06
  - PRODUCT-TASK-SCORING-001-07
  - PRODUCT-TASK-SCORING-001-08
  - PRODUCT-TASK-SCORING-001-09

## Goal

Deliver the production Agari-based riichi scoring engine, stable WASM boundary, golden scoring corpus, and library-independent TypeScript ScoringService required by the product scoring contracts.

## Boundary

This Work Item owns the production dependency/fork integration decision, the narrow mjtensu Agari fork changes required by the accepted fork contract, the stable WASM ABI, golden scoring fixtures, the TypeScript Agari adapter/ScoringService, and scoring compatibility verification.

It does not own Recognition, Application session orchestration, UI presentation, whole-game settlement outside the scoring contract, or final PWA/device release composition.

## Impact Scope

| target | impact |
|---|---|
| Agari production dependency | Pin and maintain the upstream/fork provenance and build boundary selected by T01. |
| Agari fork core/WASM | Add accepted rule configuration, stable result codes, and scoring-independent winning-shape validation. |
| production `scoring` module | Implement WASM loading, product-to-Agari serialization, normalization, and public ScoringService. |
| scoring golden corpus | Add deterministic semantic fixtures covering product rules/results. |

## Task flow

```text
S01 decide fork/dependency/build-management boundary
   -> S08 route canonical authoring
   -> S09 amend ADR/spec build-management authority
   -> S02 implement Agari RuleConfig/scoring semantic delta
   -> S03 implement stable WASM ABI and shape validation

SYSTEM T05 bootstrap review PASS + accepted scoring specs -> S04 author/implement versioned golden corpus fixtures
SYSTEM T05 bootstrap review PASS + S03 -> S05 TypeScript Agari adapter and ScoringService
S04 + S05 -> S06 objective golden/ABI compatibility verification -> S07 independent integrated review
```

S04 may proceed in parallel with fork implementation because its expected semantic results are defined by the accepted product contracts rather than by current upstream output strings.

## Task Candidates

| task | task type | responsibility | dependency |
|---|---|---|---|
| PRODUCT-TASK-SCORING-001-01 | decision | Fix the production Agari fork repository/dependency pin/build-artifact management boundary. | none |
| PRODUCT-TASK-SCORING-001-08 | coordination | Route the accepted build-management decision through canonical authoring before fork implementation. | S01 |
| PRODUCT-TASK-SCORING-001-09 | authoring | Amend PRODUCT-ADR-SYSTEM-003 and `spec:product.system.contracts.agari_fork` with the accepted source/artifact-management boundary. | S01, S08 |
| PRODUCT-TASK-SCORING-001-02 | implementation | Implement the accepted Agari core rule configuration and rule-aware scoring semantic delta. | S09 |
| PRODUCT-TASK-SCORING-001-03 | implementation | Implement the stable WASM scoring ABI and scoring-independent winning-shape validation API. | S02 |
| PRODUCT-TASK-SCORING-001-04 | implementation | Materialize the versioned semantic golden scoring corpus and fixture runner inputs. | SYSTEM T05 |
| PRODUCT-TASK-SCORING-001-05 | implementation | Implement the TypeScript Agari loader/adapter and public ScoringService. | SYSTEM T05, S03 |
| PRODUCT-TASK-SCORING-001-06 | verification | Execute the complete golden corpus through the real fork/WASM/TypeScript adapter and verify stable ABI/result semantics. | S04, S05 |
| PRODUCT-TASK-SCORING-001-07 | review | Independently review the complete scoring-engine and adapter implementation. | S06 |

## Completion Condition

- Production Agari provenance, pinning, and build-consumption behavior are explicitly decided and recorded.
- The Agari fork implements every rule semantic required by `spec:product.system.contracts.agari_fork` without unrelated scoring-engine rewrites.
- The stable WASM scoring and winning-shape APIs exist and do not require display-string parsing.
- The versioned golden corpus covers the complete minimum matrix in the production testing strategy.
- The public TypeScript ScoringService exposes only library-independent product types.
- Real fork/WASM/adapter golden verification is PASS.
- The independent integrated review is PASS with no unresolved findings.

## Evidence

- PRODUCT-ADR-SYSTEM-003 selects an Agari fork as the production scoring engine.
- The Agari fork and adapter Specifications define the required semantic delta and isolation boundary.
- The scoring input/result and Scoring API Specifications define product-owned inputs/results.
- The production testing strategy requires a real-engine golden corpus before scoring acceptance.
- All registered tasks PRODUCT-TASK-SCORING-001-01 through PRODUCT-TASK-SCORING-001-09 are `done`.
- PRODUCT-TASK-SCORING-001-06 records final overall PASS for the production artifact/provenance gate, complete 49/49 real-WASM golden compatibility, production loader integrity verification, typecheck, lint/architecture, and production build consumption of the committed WASM package.
- PRODUCT-TASK-SCORING-001-07 records an independent integrated review verdict of PASS with no findings and confirms the complete production Scoring boundary is ready for production integration.
- Therefore every Completion Condition of PRODUCT-WORK-SCORING-001 is satisfied as of 2026-08-27.
