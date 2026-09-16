# Contract: Train callable interface

- **id**: `spec:mldb.v2.training.train_interface`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.training`
- **contract_class**: `interface`

## Entrypoint

Train Protocol v1 companion modules expose exactly:

```python
def train(context: TrainContext) -> torch.nn.Module:
    ...
```

The runtime constructs one fresh Architecture module before invocation and supplies it through the
context. The protocol does not resolve MLDB files, backend Tasks, or object-store credentials itself.

## Context shape

`TrainContext` has exactly these semantic fields:

```python
@dataclass(frozen=True)
class MaterializedCorpus:
    definition: Corpus
    root: Path

@dataclass(frozen=True)
class TrainContext:
    task: Task
    corpus: MaterializedCorpus
    architecture: Architecture
    model: torch.nn.Module
    seed: int
    parameters: Mapping[str, PublicParameterValue]
    telemetry: TelemetryReporter
    work_dir: Path
```
`Task`, `Corpus`, and `Architecture` above mean immutable parsed canonical definition values; their
concrete Python classes are frozen by Skeleton, not by protocol code.

`corpus.root` is an execution-local directory whose relative file tree matches the sealed Corpus
manifest. `model` is a fresh module returned by the selected Architecture `build()` and contains no
learned state beyond normal initialization. `parameters` is the complete resolved Train Protocol
mapping. `seed` is the exact validated integer Study input; boolean is invalid. `telemetry` is the
backend-neutral reporter from `spec:mldb.v2.common.telemetry`. `work_dir` is an execution-local
writable directory for checkpoints and temporary files.

Backend IDs, queue names, credentials, S3 URIs, Study Result IDs, and ClearML objects are not context
fields.

## Response and ownership

A successful call returns exactly one `torch.nn.Module` containing the protocol-selected learned
state. It may be `context.model` mutated in place or another module, but the returned state MUST be
strictly loadable into a fresh module from the selected Architecture.

Checkpoint/early-stopping selection must already be applied before return. The protocol may use any
internal trainer/framework strategy but MUST NOT return a path, trainer, optimizer checkpoint,
serialized bytes, or configuration object. A successful call MUST have emitted at least one valid
scalar telemetry point satisfying `spec:mldb.v2.common.telemetry`; the generic contract does not
prescribe which metric or cadence is appropriate.

Protocol code owns loss, target assignment, augmentation, optimizer, scheduler, checkpoint selection,
and training-time validation. Generic MLDB owns fresh Architecture construction, post-return strict
compatibility validation, canonical state-dict serialization/publication, and result acceptance.

Files under `work_dir` are non-canonical unless a later explicit formal contract promotes them. A
protocol exception or invalid return fails only that logical training stage.
