# Contract: Runtime recognition

- **id**: `spec:product.recognition.runtime_recognition`
- **status**: draft
- **date**: 2026-08-28
- **parent**: `spec:product.recognition`

## What this is

Current runtime contract for accepting camera content and producing one stable recognized tile structure plus live recognition observations.

## Capture-region contract

The landscape camera preview exposes three fixed semantic regions.

```text
+--------------------------------------+
|  +-----------------+  +-----------+ |
|  | Dora indicators |  |           | |
|  +-----------------+  |           | |
|                       |   Melds   | |
|  +-----------------+  |           | |
|  | Completed hand  |  |           | |
|  +-----------------+  +-----------+ |
+--------------------------------------+
```

| region | visible aspect ratio | semantic use |
|---|---:|---|
| Completed hand | `17:4` | One horizontal row containing the concealed hand and winning tile. |
| Dora indicators | `17:4` | One row containing every dora indicator the user wants counted for this winning hand, including kan-dora or ura-dora indicators when applicable. |
| Melds | `1:1` | Zero through four meld groups stacked vertically. |

The visible region itself is the recognition boundary. Content displayed inside a frame must not be hidden by an additional inner crop.
All three semantic regions are always active during recognition. An empty dora or meld region is a valid state and does not require a separate enable/disable control.

The current detector-composite mapping is the `320 x 320` layout established by PRODUCT-ADR-RECOGNITION-002:

| region | x | y | width | height |
|---|---:|---:|---:|---:|
| Completed hand | 7 | 0 | 306 | 72 |
| Dora indicators | 7 | 74 | 306 | 72 |
| Melds | 74 | 148 | 172 | 172 |

Changing the concrete detector implementation does not change the visible semantic-region contract unless this specification is revised.

## Recognition outputs

Each evaluation produces live observation data and a recognized structure.

The live observation data may include detector boxes, current tile identities or unresolved classifications, and meld-group geometry needed by the Recognition page. These observations are not scoring input and may change with frame-to-frame geometry.

A committed recognized structure contains:

| field concept | meaning |
|---|---|
| Completed-hand tiles | Ordered recognized tile instances from left to right. The winning tile is not inferred here; Application assigns the initial winning-tile selection. |
| Dora indicators | Ordered recognized indicator tiles from left to right. Recognition does not distinguish visible, kan, ura, or kan-ura source; the user supplies the complete set that should count for this winning hand. |
| Meld groups | Spatially reconstructed groups from top to bottom, preserving member order and inferred meld semantics when those semantics are unambiguous. |
| Tile identity | Canonical riichi tile identity including distinction between ordinary fives and red fives. |

Invalid/background classifier outcomes do not become recognized tiles, but their current detector observations may still be exposed to the Recognition page as unresolved live feedback.
The recognized structure is a recognition draft: it is not required to already be a legal winning hand or to contain a yaku.

## Ordering rules

- Completed hand: left to right.
- Dora indicators: left to right.
- Meld groups: top to bottom.
- Tiles inside one meld group: left to right.
- The order of separate meld groups must remain stable in the committed result.

## Meld grouping

Tiles recognized inside the meld region are reconstructed into zero through four spatial meld groups before a recognized structure is committed.

- Meld grouping is based on the spatial arrangement of recognized tile bounding boxes.
- Recognition must tolerate a common meld-row tilt of up to `±22.5°` from horizontal.
- Separate meld rows must remain distinguishable under that tilt; recognition is not required to support more extreme row rotation.
- The `±22.5°` value defines the required common-row direction/search support boundary. It is not a mandatory rejection threshold for the fitted angle of every already-reconstructed short row: detector bounding-box center jitter may move a two- or three-member row's fitted angle beyond that value while the spatial partition remains stable.
- Meld groups are ordered from top to bottom.
- Tiles inside one group are ordered from left to right along that group's row direction.
- Grouping first reconstructs spatial rows; scoring legality is not a grouping criterion.
- A two-member group containing the same base tile identity, ignoring ordinary-versus-red-five distinction, is interpreted as the visible evidence for a concealed kan.
- A concealed kan reconstructed from two visible tile observations is represented as one logical concealed-kan meld even though only the two face-up tiles produced detector boxes.
- Three- and four-member groups may receive chi/pon/open-kan interpretation when their current identities make that interpretation unambiguous.
- A spatially reconstructed group is not rejected solely because its current recognized identities do not form a legal scoring meld. Conditions owns correction and scoring owns winning-hand validity.
- If the observed points cannot be assigned to stable spatial meld groups, the frame does not produce a committable recognized structure.

The exact geometric grouping algorithm, including how row direction is estimated, how bounding-box centers are clustered, and what numerical clustering tolerances are used, is an implementation detail.

## Invalid and duplicate candidates

Recognition post-processing removes candidates that must not become recognized tile instances. This includes:

- the base classifier's invalid/background outcome;
- detections assigned only to padding or separators;
- detector duplicates removed by `spec:product.recognition.pipeline`, including merged bridge boxes that overlap multiple otherwise distinct candidates and ordinary pairwise duplicate losers;
- candidates outside the three semantic capture regions.

Removing one of these candidates does not imply that the remaining recognized structure is a valid winning mahjong hand. Winning-shape and yaku validity are downstream concerns.

## Realtime evaluation and stabilization

- Recognition evaluates continuously while the recognition page is active.
- The target evaluation cadence remains one recognition request every 100 milliseconds.
- At most one recognition evaluation owns acceptance at a time; queued stale camera frames are not required.
- Stabilization may begin only when the evaluated frame contains at least `10` valid visible non-dora tile observations after duplicate suppression and classification.
- At least `2` of those valid observations must be in the completed-hand region.
- The minimum-count gate counts actual valid observations in the completed-hand and meld regions. Dora indicators and invalid/background outcomes do not count. Logical concealed-kan expansion does not increase the visible-observation count: a concealed kan reconstructed from two face-up observations contributes two observations to this gate.
- These minima are capture-completeness guards, not winning-hand-validity checks; they permit the minimum visible case of four concealed kans: two completed-hand observations plus two visible observations for each concealed kan.
- A recognized structure becomes committed only after the same structure is produced for three consecutive evaluations.
- A differing structure or an evaluation that cannot reconstruct a stable recognition structure breaks the current stabilization run.
- Scoring legality, winning-shape validity, and yaku existence are not recognition-acceptance criteria.
- No shutter action is required to commit the result.
- After commitment, Application receives exactly one stable recognized structure and the UI leaves the live recognition state.

"Same recognized structure" compares tile identities, their region membership, completed-hand order, dora-indicator order, and meld grouping/reconstruction. Detector-box jitter alone must not prevent stabilization when the recognized structure is unchanged.

## Live feedback boundary

While no result is stable, the recognition surface may show current detections and actionable guidance.
When recognition can localize a problem to a tile or region, feedback should identify that location.
The UI must not require the user to understand detector confidence, model names, or classifier internals in order to recover.

## Non-goals

- Exact visual styling of frames and diagnostics.
- Exact detector confidence threshold or NMS implementation.
- Model architecture, classifier channel representation, checkpoint, or training dataset.
- Winning-shape validity, yaku validity, fu calculation, or point calculation.
- Winning-tile selection.

## Boundary

| concern | owner |
|---|---|
| Fixed semantic regions and committed recognition structure | This contract. |
| Per-frame detector/classifier stages and live observation output | `spec:product.recognition.pipeline`. |
| Live page composition and user-visible feedback placement | `spec:product.ui.pages.recognition`. |
| Winning-tile default and scoring-session state | `spec:product.application.scoring_session`. |
| Score input and validation | `spec:product.scoring.input`. |
| Training/model decisions | Recognition ADRs, investigations, and implementation. |

## Related records

| ref | relation |
|---|---|
| PRODUCT-ADR-RECOGNITION-001 | Origin of shutterless realtime evaluation and three-result stabilization. |
| PRODUCT-ADR-RECOGNITION-002 | Origin of the capture-region and `320 x 320` composite layout. |
| PRODUCT-ADR-RECOGNITION-004 | Current decision for the 35-class base-classifier pipeline and the separation of recognition acceptance from scoring validity. |
