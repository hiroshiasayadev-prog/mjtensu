# Overview: MLDB v2 training

- **id**: `spec:mldb.v2.training`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2`

## What this is

Defines reusable training behavior, terminal training outcomes, and canonical learned Model
identity. Backend queue/agent behavior is outside this topic.

## Topics

| ref | responsibility |
|---|---|
| `spec:mldb.v2.training.train_protocol_format` | Reusable executable training procedure. |
| `spec:mldb.v2.training.train_interface` | Common Train callable boundary. |
| `spec:mldb.v2.training.canonical_weights` | Canonical learned-state format and loading. |
| `spec:mldb.v2.training.training_result_format` | Terminal formal training outcome. |
| `spec:mldb.v2.training.model_format` | Learned identity and canonical weights reference. |
