# Contract: Train interface

- **id**: `spec:mldb.training.train_interface`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.training`
- **contract_class**: `interface`

## What this is

Defines the common callable boundary used by MLDB to execute one resolved Train Protocol.

The interface standardizes invocation and returned learned state without standardizing the internal training loop.

## Request

The executable loader exposes the Train Protocol entrypoint:

```python
def train(context: TrainContext) -> torch.nn.Module:
    ...
```

Conceptually, `TrainContext` provides:

```python
@dataclass(frozen=True)
class TrainContext:
    task: TaskHandle
    corpus: CorpusHandle
    architecture: ArchitectureHandle
    seed: int
    parameters: Mapping[str, Any]
    work_dir: Path
```

| field | contract |
|---|---|
| `task` | Resolved Task shared by the selected Corpus, Architecture, and Train Protocol. |
| `corpus` | Resolved immutable Corpus used by this Training Run. |
| `architecture` | Resolved sealed Architecture used to construct the model structure. |
| `seed` | Validated integer Training Run seed. Boolean is invalid. |
| `parameters` | Complete resolved Train Protocol public-parameter mapping. |
| `work_dir` | Training Run `work/` directory for protocol-owned checkpoints, logs, and temporary files. |

Before invocation, MLDB must resolve the selected assets, validate required integrity, verify their Task agreement, and resolve public parameters.

The protocol must use `context.seed` as the Run seed when its behavior uses randomness. MLDB preserves the validated integer seed without coercion; framework-specific range requirements remain protocol or implementation concerns.
The protocol must consume published values from `context.parameters` rather than substituting hidden defaults.

## Response

A successful call returns exactly one trained `torch.nn.Module`.

The returned module is the protocol-selected learned result of that execution.

| response rule | contract |
|---|---|
| type | `torch.nn.Module`. |
| selected state | Any checkpoint selection or early-stopping choice must already be restored into the returned module. |
| Architecture compatibility | Returned state must be strictly loadable into a fresh module built from the selected Architecture. |
| serialization | The protocol does not serialize MLDB canonical weights. The training runtime owns that step. |

A protocol may use a direct PyTorch loop, specialized detector logic, third-party framework APIs, or subprocess execution.

An external trainer may produce framework-specific checkpoints under `work_dir`. Before returning, the protocol must translate the selected learned state into the selected MLDB Architecture module.

The protocol must not return a path, optimizer checkpoint, trainer object, serialized artifact, or configuration object instead of the trained module.

## Errors

| condition | result |
|---|---|
| Required assets fail resolution or static integrity validation during preflight | Reject the launch request; do not allocate a Training Run. |
| Corpus, Architecture, and Train Protocol do not share one Task during preflight | Reject the launch request; do not allocate a Training Run. |
| Training seed is not an integer or is boolean during preflight | Reject the launch request; do not allocate a Training Run. |
| Public parameter resolution fails | Reject the launch request; do not allocate a Training Run. |
| Sibling protocol implementation cannot load after Run allocation | The affected Training Run fails. |
| `train` entrypoint is missing or not callable after Run allocation | The affected Training Run fails. |
| `train(context)` raises | The affected Training Run fails unless cancellation handling applies. |
| Return value is not `torch.nn.Module` | The affected Training Run fails. |
| Returned state is not strictly compatible with a fresh selected Architecture | The affected Training Run fails. |

A Train Protocol failure must not mutate or cancel unrelated Runs.
Concrete exception types are implementation-owned.

## Rules

- MLDB must not require one universal trainer implementation behind this interface.
- Protocol-specific loss, target assignment, augmentation, optimizer, scheduler, and checkpoint-selection logic remain inside the protocol implementation.
- `work_dir` contents do not become formal MLDB artifacts by existence alone.
- The Train Protocol must not write `artifacts/weights.pt` directly.
- Canonical learned-state validation and serialization occur after successful return.
- Training-time validation used only to select the returned learned state remains part of the Train Protocol.
- Post-training reusable evaluation belongs to Evaluation Protocol rather than this interface.

## Boundary

| concern | owner |
|---|---|
| Train Protocol YAML and implementation hash | `spec:mldb.training.train_protocol_format`. |
| Public parameter resolution | `spec:mldb.runtime.public_parameters`. |
| Architecture `build()` | `spec:mldb.catalog.architecture_build`. |
| Training Run state transitions | `spec:mldb.training.training_run_lifecycle`. |
| Canonical state validation and `.pt` serialization | `spec:mldb.training.canonical_weights`. |
| Concrete context classes and loader implementation | Implementation. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.training` | Parent training overview. |
| `spec:mldb.training.train_protocol_format` | Declares the `train` entrypoint. |
| `spec:mldb.catalog.architecture_build` | Provides the fresh model structure used by protocols and state validation. |
