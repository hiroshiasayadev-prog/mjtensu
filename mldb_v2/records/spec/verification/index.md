# Overview: MLDB v2 verification

- **id**: `spec:mldb.v2.verification`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2`

## What this is

Defines gates that prevent draft/broken executable assets or malformed backend outputs from
entering formal experiments/history.

## Topics

| ref | responsibility |
|---|---|
| `spec:mldb.v2.verification.definition_lifecycle` | validate, verify, and seal definitions. |
| `spec:mldb.v2.verification.executable_integrity` | Sibling and project-source integrity for executable definitions. |
| `spec:mldb.v2.verification.executable_asset_tests` | Required per-asset pytest gate and deterministic test placement. |
| `spec:mldb.v2.verification.result_acceptance` | convert collected backend outcome into canonical result. |
