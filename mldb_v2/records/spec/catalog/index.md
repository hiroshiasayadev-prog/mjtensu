# Index: MLDB v2 catalog

- **id**: `spec:mldb.v2.catalog`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2`

## What this is

Defines reusable semantic/input/model assets used before concrete execution.

## Definitions

| kind | responsibility |
|---|---|
| Namespace | Coarse research/experiment concept and repository/backend grouping. |
| Task | Prediction semantics. |
| Corpus | Immutable materialized samples and splits. |
| Architecture | Unweighted model structure for one Task. |

Train Protocol, Evaluation Protocol, and Study are reusable definitions but live in their focused
topics because their contracts are larger.

## Topics
| ref | responsibility |
|---|---|
| `spec:mldb.v2.catalog.namespace_format` | Namespace metadata and identity. |
| `spec:mldb.v2.catalog.task_format` | Prediction Task semantics. |
| `spec:mldb.v2.catalog.corpus_format` | Immutable Corpus metadata and manifest linkage. |
| `spec:mldb.v2.catalog.architecture_format` | Architecture definition metadata and executable ownership. |
| `spec:mldb.v2.catalog.architecture_build` | Architecture companion callable contract. |

## Common identity rules

Reusable entity IDs use `<namespace>/<local-id>`. Versioned semantic entities use a human-visible
`-vN` revision suffix. A semantic or executable behavior change after sealing requires a new
revision.

References are forward-only; entities do not maintain reverse-reference lists.
