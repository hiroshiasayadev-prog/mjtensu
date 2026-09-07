# Reference: Canonical learned weights

- **id**: `spec:mldb.training.canonical_weights`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.training`

## What this is

Defines the one canonical learned-weight artifact produced by a completed Training Run.

The runtime owns validation and serialization of this artifact after `train(context)` returns the protocol-selected trained module.

## Artifact contract

| property | contract |
|---|---|
| owner | Completed Training Run. |
| path | `mldb_data/training_runs/<training-run-id>/artifacts/weights.pt`. |
| Run-relative path | `artifacts/weights.pt`. |
| format identifier | `pytorch-state-dict`. |
| contents | Learned PyTorch module state only. |
| integrity | SHA-256 and byte size recorded in `run.yaml`. |
| downstream identity | The automatically created Model resolves these bytes through its originating Training Run. |

There is exactly one canonical learned-weight artifact for each completed Training Run.

## State acceptance

Before serialization, the runtime must:

1. require the Train Protocol return value to be `torch.nn.Module`;
2. construct a fresh module through the selected Architecture `build()` interface;
3. require strict state-dict compatibility between the returned trained module and that fresh Architecture;
4. reject the training result when strict compatibility fails.

The strict compatibility check prevents a Train Protocol from returning a different model structure than the selected Architecture.

Checkpoint-selection policy remains inside the Train Protocol. The returned module must already contain the protocol-selected learned state.

## Serialization rules

`pytorch-state-dict` v1 is a plain mapping of string state keys to `torch.Tensor` values.

- The runtime serializes the accepted learned state, not the Train Protocol.
- Every state key must be a string.
- Every state value must be a `torch.Tensor`.
- Every tensor must be detached and moved or copied to CPU representation before canonical persistence.
- A module that exposes non-tensor extra state through `state_dict()` is not compatible with this v1 format.
- The mapping must be serialized directly with `torch.save(state, weights_path)`.
- The canonical mapping must not be wrapped under a checkpoint key such as `model_state_dict`, `checkpoint`, or `state`.
- The canonical artifact must not contain optimizer state.
- The canonical artifact must not contain scheduler state.
- The canonical artifact must not contain gradient-scaler state.
- The canonical artifact must not contain epoch counters or protocol configuration.
- The canonical artifact must not contain arbitrary framework trainer objects.
- Framework-specific best, last, resume, or periodic checkpoints remain under the Training Run `work/` directory.

Conceptually:

```python
state = {
    key: value.detach().cpu()
    for key, value in trained_module.state_dict().items()
}
torch.save(state, weights_path)
```

After writing `weights.pt`, the runtime calculates its SHA-256 and byte size and records both under `result.weights` in the Training Run.

The format does not promise byte-identical serialization of semantically equivalent state across PyTorch versions or independent executions. The recorded SHA-256 protects the exact persisted bytes of this Training Run rather than acting as a semantic tensor-state hash.

## Loading rules

Canonical Model loading uses restricted PyTorch weights loading on CPU:

```python
state = torch.load(
    weights_path,
    map_location="cpu",
    weights_only=True,
)
module = architecture.build()
module.load_state_dict(state, strict=True)
```

The loaded value must again satisfy the plain string-to-tensor mapping contract before `load_state_dict` is called.

Canonical loading must not silently repair keys, coerce arbitrary payload objects, or use non-strict state matching.

## Completion rule

A Training Run must not become `completed` until all of the following succeed:

| condition | required result |
|---|---|
| Train Protocol return type | `torch.nn.Module`. |
| fresh Architecture construction | Succeeds. |
| strict state compatibility | Succeeds. |
| plain tensor-state validation | Every state key is a string and every state value is a tensor. |
| canonical serialization | `artifacts/weights.pt` exists after direct `torch.save` persistence. |
| artifact integrity calculation | SHA-256 and byte size available. |
| Run result metadata | Matches the persisted artifact. |

Failure of any required condition fails the affected Training Run rather than producing a completed Run with an untrusted learned artifact.

After successful Training Run finalization, MLDB automatically ensures exactly one deterministic Model identity exists for that Run. Model YAML does not duplicate the `.pt` bytes.

## Immutability

Once the Training Run is terminal `completed`:

- `artifacts/weights.pt` must not be modified in place;
- its recorded SHA-256 and byte size are immutable execution facts;
- a different learned state requires another Training Run;
- downstream evaluation and later model operations should verify the recorded integrity before consuming the artifact when required by their contracts.

## Boundary

| concern | owner |
|---|---|
| Architecture construction | `spec:mldb.catalog.architecture_build`. |
| Train Protocol checkpoint selection | Train Protocol implementation through `spec:mldb.training.train_interface`. |
| Run result metadata fields | `spec:mldb.training.training_run_format`. |
| Transition to `completed` | `spec:mldb.training.training_run_lifecycle`. |
| Model identity and model-loading lineage | `spec:mldb.model`. |
| ONNX or other exported deployment formats | Future export topic. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.training` | Parent training overview. |
| `spec:mldb.training.train_interface` | Supplies the protocol-selected trained module. |
| `spec:mldb.training.training_run_format` | Records canonical artifact metadata. |
| `spec:mldb.catalog.architecture_build` | Constructs the fresh structure used for strict compatibility validation. |
