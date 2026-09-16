# Contract: Result acceptance

- **id**: `spec:mldb.v2.verification.result_acceptance`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.verification`
- **contract_class**: `validation`

## Purpose

Backend completion is evidence that a process ended, not proof of a valid MLDB result. Acceptance consumes only terminal candidates conforming to `spec:mldb.v2.backend.candidate_outcome`.

## Common checks

Before terminal canonical persistence, result acceptance verifies:

- Study Result, Plan, trial, and stage identity;
- pinned Git commit/provenance;
- exact referenced definitions;
- applicable implementation and Corpus manifest digests;
- Model/weights lineage;
- required output presence;
- artifact URI/size/SHA integrity;
- terminal backend-attempt summary.

Training completion additionally validates canonical weights before generating Model identity.

Evaluation completion additionally validates the Evaluation Protocol metric/artifact contract.

A validation failure creates or finalizes a failed formal result with diagnostic classification; it
MUST NOT persist malformed successful payload as completed.

Result acceptance never mutates reusable definitions.
