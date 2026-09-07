# MLDB-ADR-SCHEMA-018: Define PyTorch state-dict serialization

- **status**: accepted
- **date**: 2026-09-04
- **depends_on**: MLDB-ADR-SCHEMA-003, MLDB-ADR-SCHEMA-005, MLDB-ADR-SCHEMA-006
- **supersedes**:
- **migrated_to_spec**: 2026-09-04

## Context

MLDB Training Run already defines one canonical learned artifact at `artifacts/weights.pt` with format identifier `pytorch-state-dict`.

The existing decision establishes that this artifact contains only the learned state selected by the Train Protocol and excludes optimizer, scheduler, epoch, protocol configuration, and arbitrary trainer state.

The serialization and loading contract is still ambiguous. Without a concrete contract, two implementations could both claim `pytorch-state-dict` while writing incompatible wrapper mappings, whole serialized modules, or other checkpoint payloads.

MLDB needs one small canonical PyTorch representation that preserves the existing separation between Architecture structure, Training Run learned state, and Model identity.

## Decision

Define `pytorch-state-dict` in MLDB v1 as a plain CPU tensor state mapping serialized with PyTorch `torch.save`.

### Canonical state value

The canonical in-memory state is the selected trained module's `state_dict()` after every stored tensor is detached and represented on CPU.

The persisted state must contain only string keys mapped to `torch.Tensor` values.

Modules whose `state_dict()` contains non-tensor extra state are not compatible with the MLDB v1 canonical learned-state format.

The canonical state must not be wrapped in another mapping such as `model_state_dict`, `checkpoint`, or `state`.

Conceptually:

```python
state = {
    key: value.detach().cpu()
    for key, value in trained_module.state_dict().items()
}
torch.save(state, weights_path)
```

Generic runtime validation must reject a non-string state key or a non-tensor state value before canonical persistence.

### Canonical serialization

The Training Run runtime serializes the accepted mapping directly to:

```text
artifacts/weights.pt
```

using PyTorch `torch.save`.

MLDB v1 does not require deterministic byte-for-byte serialization of the same semantic state across PyTorch versions, processes, or environments.

The Training Run SHA-256 and byte size identify and protect the exact artifact bytes that were actually persisted for that Run. They are not a semantic hash of tensor values and must not be expected to match an independently reserialized equivalent state.

### Canonical loading

Model loading resolves the canonical artifact and loads it on CPU using PyTorch `torch.load` with restricted weights loading enabled:

```python
state = torch.load(
    weights_path,
    map_location="cpu",
    weights_only=True,
)
```

The loaded value must satisfy the same plain `Mapping[str, torch.Tensor]` contract before use.

MLDB then constructs a fresh selected Architecture and loads the state using strict matching:

```python
module = architecture.build()
module.load_state_dict(state, strict=True)
```

A load failure, non-conforming payload, or strict state mismatch invalidates learned-state consumption rather than being silently repaired or loaded non-strictly.

### Format boundary

`pytorch-state-dict` is the canonical learned-state representation for MLDB v1, not a deployment or exchange format.

TorchScript, ONNX, safetensors, framework checkpoints, and future exported formats may be introduced through separate artifact or export contracts without changing historical Training Run artifacts.

## Rationale

`torch.save(state_dict)` is the smallest representation that matches the current PyTorch Architecture and Train Protocol interfaces.

Persisting only tensor state keeps Architecture responsible for executable structure and prevents canonical learned artifacts from depending on pickled project classes or trainer objects.

Restricted `torch.load(..., weights_only=True)` aligns loading with the tensor-only artifact contract and avoids treating arbitrary pickled objects as part of the canonical Model representation.

Strict `load_state_dict` preserves the existing invariant that the learned result belongs to the exact selected Architecture.

Not requiring byte-deterministic reserialization avoids turning PyTorch's internal serialization details into an MLDB compatibility contract. Integrity is instead anchored to the exact bytes produced by the immutable Training Run.

## Rejected alternatives

### Serialize the complete `torch.nn.Module`

A whole serialized module would mix learned state with executable structure, depend on Python class importability, and undermine Architecture as the authoritative model-structure asset.

### Store a framework checkpoint wrapper

Mappings such as `model_state_dict`, optimizer state, epoch counters, or trainer metadata are useful working checkpoints but are not the canonical learned result. They remain under the Training Run `work/` directory.

### Use safetensors as the v1 canonical format

Safetensors could provide a future tensor-only representation, but MLDB v1 already uses PyTorch as its Architecture and training boundary. Introducing another dependency and format is unnecessary for the initial canonical learned-state contract.

### Require byte-identical serialization for equivalent states

The purpose of the artifact SHA-256 is integrity of the persisted Training Run result, not cross-version canonical hashing of semantic tensor values. Requiring identical bytes across PyTorch implementations would create an unnecessary compatibility burden.

## Consequences

Training Run tooling must validate that canonical state keys are strings and canonical state values are tensors.

Canonical tensors are detached and stored on CPU.

The runtime writes the state mapping directly with `torch.save` and records the resulting artifact SHA-256 and byte size.

Model loading uses `torch.load(..., map_location="cpu", weights_only=True)`, validates the loaded plain tensor mapping, constructs the selected Architecture, and applies `load_state_dict(..., strict=True)`.

Whole-module serialization, arbitrary checkpoint wrappers, non-tensor extra state, and non-strict canonical loading are outside `pytorch-state-dict` v1.

No guarantee is made that independently serializing semantically equivalent state produces the same SHA-256.

## Evidence

Current MLDB Training Run and Model specs already separate Architecture construction from canonical learned weights and require strict compatibility with a fresh Architecture before Training Run completion.

The project's existing PyTorch training and evaluation scripts commonly persist and restore model `state_dict` values, making direct PyTorch state serialization compatible with established implementation practice.
