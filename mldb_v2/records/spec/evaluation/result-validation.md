# Contract: Evaluation result validation

- **id**: `spec:mldb.v2.evaluation.result_validation`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.evaluation`
- **contract_class**: `validation`

## Acceptance

A backend-completed evaluation is not yet a completed MLDB Evaluation Result.

MLDB accepts it only after:

1. resolving the exact planned Evaluation Protocol;
2. validating every required metric key;
3. validating scalar type and finite-number requirements where applicable;
4. rejecting undeclared values from the formal metric mapping;
5. validating every required formal artifact;
6. verifying ArtifactRef size/digest;
7. verifying Model, Corpus, plan, trial, and stage lineage.

Extra backend telemetry is ignored for canonical formal-result purposes.

If acceptance fails, the canonical Evaluation Result is `failed` with an acceptance diagnostic.
MLDB MUST NOT silently downgrade a missing required metric into success.

Formal comparison tools SHOULD use `result.metrics` rather than training telemetry.
