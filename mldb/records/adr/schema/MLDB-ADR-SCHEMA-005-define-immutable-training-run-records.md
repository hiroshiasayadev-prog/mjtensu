# MLDB-ADR-SCHEMA-005: Define immutable Training Run records

- **status**: accepted
- **date**: 2026-09-03
- **depends_on**: MLDB-ADR-SCHEMA-001, MLDB-ADR-SCHEMA-002, MLDB-ADR-SCHEMA-003, MLDB-ADR-SCHEMA-004
- **supersedes**:
- **migrated_to_spec**:

## Context

Task, Corpus, Architecture, and Train Protocol define reusable MLDB assets, but they do not identify one concrete training execution.

A concrete execution chooses one immutable Corpus, one sealed Architecture, one sealed Train Protocol, and execution-specific values such as the random seed. It then runs on a particular host and software environment, may succeed or fail, and may produce protocol-specific checkpoints, logs, histories, generated framework configuration, and other working files. On successful return, MLDB itself serializes the protocol-selected trained module into one canonical learned-weight artifact.

Without a first-class Training Run record, a learned checkpoint can only be understood by reconstructing command lines, output-directory names, training logs, or ad hoc experiment metadata. MLDB therefore needs an immutable execution record that connects one training attempt to the exact reusable assets that were selected.

Training Run is an execution-history entity rather than a reusable definition. It is not revised in place like Architecture or Train Protocol. Re-running the same inputs creates another Training Run with another Run ID.

## Decision

Introduce `TrainingRun` as the MLDB record for one concrete invocation of one Train Protocol against one Corpus and one Architecture.

A Training Run records the selected immutable inputs, run-specific execution values, lifecycle state, execution timing/environment, and the training artifacts produced by that attempt.

Training Run does not define training behavior. Train Protocol remains authoritative for the reusable training procedure.

### Physical placement

Training Runs live outside the Brewprint Design Records tree under:

```text
mldb_data/
  training_runs/
```

Each Training Run occupies one directory named by the Training Run ID:

```text
mldb_data/training_runs/<training-run-id>/
  run.yaml
  work/
  artifacts/
```

The protocol receives `work/` through `TrainContext.work_dir`. It may write framework checkpoints, logs, histories, generated framework configuration, and other protocol-specific execution files there.

`artifacts/` is owned by generic MLDB run tooling. Train Protocol code must not write the canonical learned-weight artifact directly. On successful return from `train(context)`, MLDB validates the returned trained module and serializes its canonical learned `state_dict` under `artifacts/`.

A protocol may create arbitrary files beneath `work/`, but `run.yaml` and MLDB-owned canonical artifacts are the authoritative generic Training Run outputs.

Example:

```text
mldb_data/training_runs/tr-20260903-001/
  run.yaml
  work/
    best-framework-checkpoint.pt
    last-framework-checkpoint.pt
    history.jsonl
  artifacts/
    weights.pt
```

### Training Run ID

Training Run IDs are event identifiers, not versioned asset identifiers. They do not use the terminal `-vN` grammar used by Task revisions, Architectures, or Train Protocols.

The initial human-readable ID form is:

```text
tr-YYYYMMDD-NNN
```

where `YYYYMMDD` is the local calendar date on which the Training Run record is created and `NNN` is a zero-padded per-date sequence number starting at `001`.

Examples:

```text
tr-20260903-001
tr-20260903-002
tr-20260904-001
```

An ID is immutable once allocated and must be unique within `mldb_data/training_runs/`.

If concurrent allocation later makes the simple daily sequence inconvenient, the ID grammar may be revised by a later schema decision without changing existing Run IDs.

### Lifecycle

Training Run status is one of:

```text
running
completed
failed
cancelled
```

The Training Run record is created with `status: running` before the Train Protocol is invoked.

The record may be updated while the run is active to capture execution facts that become known during execution. It must eventually transition from `running` to exactly one terminal state:

```text
completed
failed
cancelled
```

Terminal Training Runs are immutable. A terminal run must not return to `running`, and its selected inputs, seed, timestamps, result, failure information, and artifact identity must not be rewritten to describe a different execution.

Retrying a failed or cancelled run creates a new Training Run ID even when all selected inputs and the seed are unchanged.

### Metadata format

Training Run metadata is YAML and uses schema identifier:

```text
mjtensu.mldb/training-run/v1
```

A completed example is:

```yaml
schema: mjtensu.mldb/training-run/v1

id: tr-20260903-001
status: completed

corpus: gray35-jp500-v3-jp189-v1
architecture: mobilenet-v3-small-f8-r1-v1
train_protocol: tile-classifier-adamw-cosine-v1

study:
  run: sr-20260904-001
  trial: trial-0001

parameters:
  epochs: 150
  batch_size: 1024
  learning_rate: 0.0003
  weight_decay: 0.0001

execution:
  seed: 42
  started_at: 2026-09-03T21:15:02+09:00
  finished_at: 2026-09-03T22:03:41+09:00

environment:
  host: bugratserver
  python: 3.12.0
  torch: 2.8.0
  cuda: "12.8"
  device: NVIDIA GeForce RTX 4090

result:
  weights:
    format: pytorch-state-dict
    path: artifacts/weights.pt
    sha256: 0123456789abcdef...
    bytes: 3873724

work:
  history: work/history.jsonl
  selected_framework_checkpoint: work/best-framework-checkpoint.pt
```

A failed example is:

```yaml
schema: mjtensu.mldb/training-run/v1

id: tr-20260903-002
status: failed

corpus: gray35-jp500-v3-jp189-v1
architecture: mobilenet-v3-small-f8-r1-v1
train_protocol: tile-classifier-adamw-cosine-v1

parameters:
  epochs: 150
  batch_size: 1024
  learning_rate: 0.0003
  weight_decay: 0.0001

execution:
  seed: 43
  started_at: 2026-09-03T22:10:00+09:00
  finished_at: 2026-09-03T22:11:14+09:00

failure:
  type: CUDAOutOfMemoryError
  message: CUDA out of memory during training.
```

### Required fields

Every Training Run record must contain:

- `schema`;
- `id`;
- `status`;
- `corpus`;
- `architecture`;
- `train_protocol`;
- `parameters`;
- `execution.seed`;
- `execution.started_at`.

A terminal Training Run additionally requires `execution.finished_at`.

A `completed` Training Run additionally requires:

- `result.weights.format`;
- `result.weights.path`;
- `result.weights.sha256`;
- `result.weights.bytes`.

For Training Run v1, `result.weights.format` is `pytorch-state-dict`.

A `failed` Training Run should record `failure.type` and `failure.message` when the failure can be represented safely as text.

`environment` and `work` are optional generic mappings. Their internal fields may grow as tooling matures without requiring every training framework to expose the same environment or working-file vocabulary.

### Input relationships

A Training Run references exactly one:

- Corpus;
- Architecture;
- Train Protocol.

The Training Run does not repeat `task` because Task is already referenced by Corpus, Architecture, and Train Protocol. Before execution, MLDB must verify that all three selected assets reference the same Task.

The referenced Corpus is immutable by its Corpus identity and artifact hash.

The referenced Architecture and Train Protocol must be `sealed`, and their implementation hashes must validate before the run begins.

Training Run records must not copy Architecture or Train Protocol implementation source into `run.yaml`. The immutable asset IDs and their own integrity records are the authoritative dependency links.

### Study lineage

A Training Run created from a Study Run may additionally contain:

```yaml
study:
  run: sr-20260904-001
  trial: trial-0001
```

`study.run` references the Study Run that materialized the training job, and `study.trial` identifies the Study-local training coordinate in that Study Run's immutable plan.

The mapping is optional because Training Runs may also be launched directly outside a Study.

When present, both fields are required and must correspond to one training trial in the referenced Study Run plan. The Training Run's Corpus, Architecture, Train Protocol, seed, and fully resolved parameters must exactly match that planned trial.

Study lineage is immutable execution provenance. It does not change Training Run lifecycle or result semantics.

### Run-specific seed

Random seed is a Training Run value rather than a Train Protocol parameter.

The same reusable Train Protocol may therefore be executed multiple times with different seeds without creating new Train Protocol revisions:

```text
same Corpus
same Architecture
same Train Protocol
seed 42 -> tr-20260903-001
seed 43 -> tr-20260903-002
seed 44 -> tr-20260903-003
```

The launcher supplies `execution.seed` to the selected Train Protocol through `TrainContext.seed`.

A Train Protocol must use the supplied seed as its run seed rather than silently replacing it with a protocol-local default when the protocol uses randomness.

### Resolved Train Protocol parameters

Training Run records the complete resolved values of the public parameter interface declared by its referenced Train Protocol.

For each Run, the caller may supply values only for keys published under `TrainProtocol.parameters`. Generic MLDB tooling starts from every published `default`, replaces the explicitly supplied values, rejects unknown keys, and records the resulting complete mapping under Training Run `parameters` before invoking the protocol.

For example, a sweep may execute the same Train Protocol with different published values such as learning rate or batch size without creating a new Train Protocol revision. Parameters not published by the protocol cannot be changed through Training Run.

`parameters` is therefore not an unrestricted override mapping. Fixed loss behavior, augmentation, optimizer family, checkpoint selection, or any other behavior that the Train Protocol does not publish remains fixed by the sealed protocol implementation.

The launcher passes the exact recorded mapping through `TrainContext.parameters`. A terminal Training Run's resolved parameter values are immutable execution facts.

### Invocation sequence

The generic MLDB launch flow is conceptually:

```text
allocate Training Run ID
  -> create run.yaml with status=running
  -> create work/ and artifacts/
  -> resolve Corpus / Architecture / Train Protocol
  -> verify Task compatibility and asset integrity
  -> resolve and record the complete Train Protocol parameter mapping
  -> construct TrainContext using the Run seed, resolved parameters, and work/ directory
  -> invoke TrainProtocol.train(context)
  -> receive trained torch.nn.Module or failure
  -> validate returned state against a fresh selected Architecture instance
  -> serialize canonical CPU state_dict to artifacts/weights.pt
  -> hash and record the canonical weight artifact
  -> finalize run.yaml as completed / failed / cancelled
  -> for completed Runs, ensure the deterministic Model record exists
  -> freeze the terminal Training Run
```

The exact command-line interface and implementation of the launcher are outside this ADR.

### Canonical learned weights

A completed Training Run contains exactly one canonical learned-weight artifact produced by generic MLDB tooling from the `torch.nn.Module` returned by the Train Protocol.

Before serialization, MLDB must verify that the returned value is a `torch.nn.Module` and that its `state_dict` is strictly loadable into a fresh module returned by the selected Architecture's `build()` entrypoint. This rejects a Train Protocol that accidentally returns a different model structure.

MLDB then serializes a CPU copy of the returned module's learned `state_dict` to:

```text
artifacts/weights.pt
```

Training Run v1 records this as `format: pytorch-state-dict`, together with its relative path, SHA-256, and byte size.

The canonical artifact contains learned model state only. It must not contain optimizer state, scheduler state, gradient-scaler state, epoch counters, protocol configuration, or arbitrary framework objects.

The Train Protocol remains responsible for deciding which learned state is the result. For example, it may restore its best internally selected checkpoint into the module before returning. Generic MLDB tooling does not implement checkpoint-selection policy.

On successful completion, MLDB automatically creates exactly one first-class Model record for this immutable Training Run result. Model identity and placement are defined by MLDB-ADR-SCHEMA-006; no separate manual promotion or registration step is required to create the Model.

Framework-specific best/last/periodic checkpoints remain protocol-owned working files under `work/` and need not use a common format.

### Work files and artifact handling

Train Protocol implementations may write arbitrary run-specific outputs beneath `work/`. The optional `work` mapping in `run.yaml` is a convenience inventory of notable protocol-owned files and is intentionally free-form. MLDB v1 does not require every generated working file to be enumerated.

`artifacts/` is reserved for generic MLDB-owned outputs. Training Run v1 requires only the canonical `artifacts/weights.pt` learned-state artifact on successful completion.

The canonical result weight path, SHA-256, and byte size are required so that the learned result cannot silently change after the run becomes terminal.

Terminal canonical artifacts must not be modified in place. Protocol-owned `work/` files are historical run material and should likewise not be rewritten after terminal completion when doing so would alter the historical evidence. Derived or post-training assets that need independent lifecycle and identity should be registered through later Model, Evaluation, Export, or Artifact entities instead of rewriting the historical run.

### Environment capture

Training Run records the concrete execution environment on a best-effort basis.

Useful fields may include host, operating system, Python version, PyTorch version, CUDA version, accelerator name, framework version, or other protocol-specific runtime information.

MLDB v1 does not claim that this free-form environment mapping is a complete reproducible environment lockfile. Exact environment packaging, container identity, or dependency snapshots may be introduced later if the project demonstrates a need for stronger environment reproduction.

The environment record exists primarily to make execution differences visible when diagnosing run behavior.

### Failure and cancellation

A failed Training Run remains a valid historical record even when no usable checkpoint was produced.

Failure metadata may include exception type and a concise message. Large tracebacks or logs should remain run artifacts rather than being copied wholesale into `run.yaml`.

A cancelled Training Run similarly remains in MLDB with its terminal state and available execution/artifact information.

MLDB must not discard unsuccessful runs merely because they cannot produce a Model. Failed runs can provide useful evidence about OOM limits, numerical instability, incompatible settings, or environment failures.

### Immutability boundary

Training Run differs from Architecture and Train Protocol lifecycle:

- Architecture and Train Protocol may be freely edited while `draft` and become immutable when `sealed`;
- Training Run is created as an execution record, progresses only through its lifecycle, and becomes immutable immediately on reaching a terminal state;
- Training Run never receives a `-vN` revision.

If a completed Run record was factually recorded incorrectly because of an MLDB tooling defect, correction must preserve the historical meaning rather than repurpose the Run ID for a different execution. A later migration mechanism may annotate or supersede invalid records if such a need arises.

## Rationale

Training Run is the natural join point between the reusable MLDB definitions established by the preceding ADRs.

By recording Corpus, Architecture, Train Protocol, and seed explicitly, one checkpoint can be traced to the exact training inputs without relying on directory naming conventions or reconstructed shell commands.

Keeping seed at the Run level permits repeated stochastic trials under one unchanged Train Protocol. Recording only values from the protocol's explicit public parameter interface likewise permits useful parameter sweeps while preventing Training Run from becoming a second unrestricted training-configuration system.

A directory per Run provides a natural containment boundary for protocol-owned working material and MLDB-owned canonical learned weights. Separating `work/` from `artifacts/` makes it clear which files have framework-specific meanings and which result is standardized by MLDB.

Having the Train Protocol return a trained `nn.Module` lets MLDB own canonical serialization uniformly across in-process PyTorch trainers and externally delegated trainers. Requiring a hash and byte size for the resulting state-dict artifact establishes the integrity boundary used by the automatically generated Model identity.

Keeping the environment mapping lightweight avoids blocking automation on a universal environment schema while still preserving the information most likely to explain differences between executions.

## Rejected alternatives

### Put seed in Train Protocol

Seed varies between repeated executions without changing the reusable training procedure. Treating seed as protocol identity would require otherwise identical protocols or revisions solely to execute stochastic repeats.

Seed is therefore owned by Training Run and supplied through `TrainContext.seed`.

### Permit arbitrary per-run protocol overrides

An unrestricted override map would allow one Run to alter behavior the Train Protocol never declared variable, splitting the effective training definition between protocol code and ad hoc Run configuration.

Training Run therefore accepts only keys explicitly published by the sealed Train Protocol. This still permits queue-driven sweeps over intentional parameters such as learning rate or batch size while preserving the protocol as the authority over what may vary.

### Add a `-vN` revision to Training Run

A Training Run is a historical event rather than a reusable definition. Re-execution is another event and should receive another Run ID, not another revision of the previous event.

### Delete failed runs

Failure is useful experimental information. Discarding failed runs would hide known OOM, compatibility, or numerical failures and make repeated troubleshooting harder.

Failed and cancelled Training Runs remain immutable historical records.

### Treat every framework checkpoint as a Model

Training may produce many transient or periodic framework checkpoints. Promoting every snapshot to a first-class Model would create unnecessary inventory noise and would preserve framework-specific serialization as the Model boundary.

Train Protocol may keep such snapshots under `work/`. MLDB instead creates one canonical learned state from the module returned by the protocol, and successful Training Run finalization automatically creates exactly one Model that references that canonical result.

### Require a complete environment lockfile in every Run

Different frameworks and hosts expose environment information differently, and the project has not yet demonstrated a need to reproduce entire execution environments byte-for-byte.

Training Run v1 therefore captures useful environment metadata without making environment lockfiles a validity requirement.

## Consequences

Future Training Run tooling should be able to:

- allocate a unique `tr-YYYYMMDD-NNN` ID;
- create the Training Run directory and initial `run.yaml`;
- validate the selected Corpus, Architecture, and Train Protocol;
- verify that all three resolve to the same Task;
- verify Corpus and sealed implementation hashes before execution;
- resolve protocol defaults plus caller-supplied values into a complete Training Run `parameters` mapping;
- reject parameter keys not published by the selected Train Protocol;
- pass the Run seed through `TrainContext.seed`;
- pass resolved parameters through `TrainContext.parameters`;
- pass the Run `work/` directory through `TrainContext.work_dir`;
- invoke the selected Train Protocol through `train(context)`;
- require a trained `torch.nn.Module` return value on success;
- validate the returned state against a fresh selected Architecture instance;
- serialize a canonical CPU `state_dict` to `artifacts/weights.pt`;
- calculate and persist its format, SHA-256, and byte size;
- capture timing and best-effort environment metadata;
- finalize status as `completed`, `failed`, or `cancelled`;
- automatically ensure exactly one deterministic Model record exists for each completed Run;
- retain failed/cancelled Run directories;
- reject mutation of terminal Run facts and recorded canonical result artifacts.

The initial launcher does not need to understand optimizer, loss, scheduler, augmentation, detector configuration, or external-framework internals. Those remain inside the selected Train Protocol.

Every completed Training Run exposes one canonical, hash-identified PyTorch state-dict result that can be traced to immutable Corpus, Architecture, and Train Protocol inputs plus its run-specific seed and execution record. MLDB-ADR-SCHEMA-006 assigns exactly one automatically generated Model identity to each such completed Run.

## Evidence

The current classifier training script writes run-level `config.json`, `history.jsonl`, `best.pt`, `last.pt`, periodic checkpoints, summaries, timing, GPU information, and the training seed into one output directory. These are concrete execution facts rather than Architecture or Corpus properties.

The current rotated FCOS trainer likewise records a run configuration, seed, runtime information, history, `last.pt`, `model_best.pt`, best-epoch metadata, elapsed time, and failure-relevant execution behavior. Its output shape demonstrates the usefulness of one Run directory with a small generic record plus framework-specific artifacts.

The current NanoDet orchestration creates condition-specific run directories, invokes NanoDet training externally, locates the resulting best checkpoint, and already records checkpoint/config hashes in completion metadata. This demonstrates that Training Run can wrap external-framework execution without requiring the generic MLDB launcher to own the internal training loop.
