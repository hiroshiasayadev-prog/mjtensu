# Overview: Recognition

- **id**: `spec:product.recognition`
- **status**: draft
- **date**: 2026-08-26
- **parent**: `spec:product`

## What this is

Owner of the runtime contract that turns the live camera view into one stable semantic riichi-mahjong recognition result.

## Current contract

| concern | contract |
|---|---|
| Interaction | Recognition is continuous and shutterless. |
| Camera orientation | Recognition uses the landscape capture layout. |
| Semantic regions | Completed hand, dora indicators, and melds are captured separately through fixed visible regions. |
| Acceptance | Stabilization starts only from frames containing at least 10 valid visible non-dora tile observations in total and at least 2 valid completed-hand observations; a recognition structure is committed after the same eligible structure is observed for three consecutive recognition evaluations. Scoring validity is not part of recognition acceptance. |
| Ordering | Completed-hand tiles and dora indicators are left-to-right; meld groups are top-to-bottom and tiles inside each group are ordered along the group row. |
| Tile identity | Ordinary riichi tiles and red fives remain distinguishable. |
| Invalid candidates | The base grayscale classifier has 35 outcomes: 34 tile identities plus invalid/background. Invalid/background outcomes do not become recognized tiles. |
| Live output | Recognition exposes current candidate boxes, recognized tile identities/unresolved states, and meld-group geometry for camera overlay. |
| Committed output | Application receives an ordered recognition structure independently of live detector geometry. |

## Topics

| title | kind | ref | summary |
|---|---|---|---|
| Runtime recognition | Contract | `spec:product.recognition.runtime_recognition` | Capture layout, recognition structure, meld grouping, feedback boundary, and temporal commit semantics. |
| Recognition pipeline | Contract | `spec:product.recognition.pipeline` | Per-frame detector/classifier stages and the live-observation plus recognition-structure output boundary. |

## Non-goals

- Detector or classifier training procedure.
- Dataset schemas and annotation tooling.
- Concrete model architecture, checkpoint, confidence threshold, or runtime provider.
- Scoring-rule semantics.
- Page composition outside the visible recognition contract.

## Boundary

| concern | owner |
|---|---|
| Runtime camera-to-semantic-result contract | `spec:product.recognition` |
| Score-calculation input meaning | `spec:product.scoring` |
| Session preservation and correction | `spec:product.application` |
| Recognition-page composition | `spec:product.ui.pages.recognition` |
| ML model choice and validation evidence | ADR / investigation / implementation. |

## Related records

| ref | relation |
|---|---|
| PRODUCT-ADR-RECOGNITION-001 | Establishes staged realtime recognition and three-result stabilization. |
| PRODUCT-ADR-RECOGNITION-002 | Establishes the fixed semantic capture regions. |
| PRODUCT-ADR-RECOGNITION-004 | Establishes the current 35-class base-classifier pipeline and keeps scoring validity downstream of Recognition. |
