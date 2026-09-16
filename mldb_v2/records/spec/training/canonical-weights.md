# Reference: Canonical learned weights

- **id**: `spec:mldb.v2.training.canonical_weights`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.training`
- **contract_class**: `reference`

## Format

MLDB v2 canonical learned weights use format `pytorch-state-dict/v1`: a plain mapping from string
state keys to `torch.Tensor` values. The mapping is serialized directly with `torch.save` after all
tensors are detached and moved to CPU.

Optimizer, scheduler, scaler, epoch counters, protocol configuration, trainer objects, and wrapped
checkpoint structures are forbidden from the canonical learned-weight artifact.
## Acceptance and loading

Before publication, MLDB constructs a fresh module with the selected Architecture `build()` and
requires strict state-dict compatibility with the Train Protocol return value. Every state key must
be a string and every value a tensor.

The serialized bytes are uploaded to canonical object storage and represented by an ArtifactRef
containing URI, byte size, SHA-256, and format. Training cannot become `completed` until upload and
integrity verification succeed.

Model loading downloads/verifies the exact bytes, calls
`torch.load(..., map_location="cpu", weights_only=True)`, revalidates the plain mapping, constructs a
fresh Architecture module, and calls `load_state_dict(state, strict=True)`. Key repair and
non-strict fallback are prohibited.
