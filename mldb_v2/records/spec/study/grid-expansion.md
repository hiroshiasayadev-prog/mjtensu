# Contract: Study grid expansion

- **id**: `spec:mldb.v2.study.grid_expansion`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.study`
- **contract_class**: `algorithm`

## Training/model trial expansion

Expansion is deterministic and side-effect free. For a training source, trial ordering is the
Cartesian product from outermost to innermost:

1. Architectures in Study declaration order;
2. explicitly swept Train Protocol parameter keys in ascending Unicode code-point order;
3. each key's `values` in Study declaration order;
4. seeds in Study declaration order.

Parameters omitted from the Study do not create axes; their protocol defaults are inserted into the
complete resolved parameter mapping. Existing-Model trials preserve authored Model order.

Trials are assigned `trial-0001`, `trial-0002`, ... in this exact order.
## Evaluation-coordinate expansion

For each materialized model trial, evaluation coordinates are appended in this exact order:

1. evaluation stages in Study declaration order;
2. that stage's explicitly swept Evaluation Protocol parameter keys in ascending Unicode code-point
   order;
3. each key's `values` in Study declaration order.

A stage with no explicit parameter axes creates exactly one Evaluation coordinate using all protocol
defaults. Each coordinate stores the complete resolved parameter mapping and is assigned
`eval-0001`, `eval-0002`, ... within its parent trial across all stages.

Duplicate Architectures, Models, seeds, or decoded values inside one axis are rejected before
expansion. Empty required lists are invalid. Evaluation expansion never changes trial numbering.

Expansion MUST NOT create backend Tasks, query backend state, execute companion Python, or mutate
canonical definitions.
