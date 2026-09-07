# MLDB-ADR-SCHEMA-004: Define executable Train Protocol assets

- **status**: accepted
- **date**: 2026-09-03
- **depends_on**: MLDB-ADR-SCHEMA-001, MLDB-ADR-SCHEMA-002, MLDB-ADR-SCHEMA-003
- **supersedes**:
- **migrated_to_spec**:

## Context

MLDB needs to record how models are trained while supporting substantially different training implementations.

The project already has conventional image-classifier training, a custom rotated FCOS training loop, and NanoDet training driven by NanoDet-specific configuration and command-line tooling. These workflows share some concepts such as epochs, optimization, checkpoints, and device handling, but they differ materially in dataset loading, batch structure, loss construction, target assignment, validation, scheduler stepping, model freezing, augmentation, checkpoint selection, and external-framework integration.

Attempting to encode every training behavior into one universal MLDB trainer would require a growing set of model-family-specific branches and configuration fields. It would also force external frameworks such as NanoDet through a project-owned training loop even when their own trainer is the more natural and reliable execution path.

The actual common requirement is smaller: MLDB must be able to select a Train Protocol and invoke it automatically with a resolved Task, Corpus, Architecture, and execution work location. The Train Protocol may then implement the training procedure directly, use helper functions contained in the same protocol file, or delegate to an external framework.

Train Protocol should therefore standardize the execution boundary rather than the internal training algorithm.

## Decision

Introduce `TrainProtocol` as a versioned executable MLDB asset that defines how a compatible Architecture is trained for one Task.

A Train Protocol consists of a YAML metadata record and a sibling Python implementation file. The YAML provides identity, lifecycle, Task binding, a human-readable summary, and important parameter notes. The Python file is the authoritative executable training procedure.

MLDB v1 standardizes one Train Protocol entrypoint:

```python
def train(context: TrainContext) -> torch.nn.Module:
    ...
```

The returned module is the trained model selected by the Train Protocol as the result of that execution. Generic MLDB tooling invokes the entrypoint, validates the returned module against the selected Architecture, and serializes the canonical learned weights. The Train Protocol does not own canonical MLDB weight serialization and does not need to use a common internal training loop.

### Physical placement and basename contract

Train Protocol assets live outside the Brewprint Design Records tree under:

```text
mldb_data/
  train_protocols/
```

Each Train Protocol occupies two sibling files:

```text
mldb_data/train_protocols/<train-protocol-id>.yaml
mldb_data/train_protocols/<train-protocol-id>.py
```

Example:

```text
mldb_data/train_protocols/tile-classifier-adamw-cosine-v1.yaml
mldb_data/train_protocols/tile-classifier-adamw-cosine-v1.py

mldb_data/train_protocols/rotated-fcos-standard-v1.yaml
mldb_data/train_protocols/rotated-fcos-standard-v1.py
```

The basename is the Train Protocol ID. The YAML does not repeat the Python filename because it is derived from this placement rule.

### ID and revision grammar

Every Train Protocol ID must end with an explicit positive-integer MLDB revision suffix:

```text
<train-protocol-base-id>-v<positive-integer>
```

Examples:

```text
tile-classifier-adamw-cosine-v1
rotated-fcos-standard-v1
nanodet-real-capture-finetune-v1
```

Only positive integer revisions are valid. The final `-vN` is the MLDB Train Protocol revision rather than an epoch number, experiment number, or external-framework version.

### Lifecycle

Train Protocol status is one of:

```text
draft
sealed
```

A `draft` Train Protocol may be edited freely while its implementation is being developed and tested.

A `sealed` Train Protocol is immutable as an executable training definition and must never return to `draft`.

Any executable training-behavior change after sealing requires a new Train Protocol revision. This includes changes to optimizer construction, loss behavior, augmentation, scheduler behavior, checkpoint selection, external commands, or any other code path that can materially change training output.

Editorial metadata corrections that do not alter executable behavior or protocol meaning do not require a new revision.

### Metadata format

Train Protocol metadata is YAML and uses schema identifier:

```text
mjtensu.mldb/train-protocol/v1
```

Example:

```yaml
schema: mjtensu.mldb/train-protocol/v1

id: tile-classifier-adamw-cosine-v1
status: sealed

task: tile-shape-classification-35-v1

name: Standard tile classifier AdamW cosine training

description: >
  Standard training procedure for compatible tile classifiers using
  AdamW, cosine learning-rate decay, mixed precision, and geometric
  augmentation.

implementation:
  entrypoint: train
  sha256: 0123456789abcdef...

parameters:
  epochs:
    default: 150
  batch_size:
    default: 1024
  learning_rate:
    default: 0.001
  weight_decay:
    default: 0.0001

notes:
  optimizer: AdamW
  scheduler: cosine
  amp: true
  tf32: true
  augmentation:
    rotation_deg: 22.5
    perspective: 0.08
    shear: 0.08
    stretch: 0.12
  checkpoint_selection: >
    Select the best checkpoint using the protocol's full configured
    manual-validation angle sweep.
```

The required fields are:

- `schema`;
- `id`;
- `status`;
- `task`;
- `name`;
- `description`;
- `implementation.entrypoint`;
- `parameters`.

`implementation.sha256` is required when `status: sealed` and may be omitted while `status: draft`.

`parameters` must be a mapping. Each key names one Train Protocol parameter that callers are explicitly allowed to vary between Training Runs. Each parameter entry must be a mapping containing `default`.

An empty `parameters: {}` mapping is valid when the protocol intentionally exposes no run-varying training parameters.

Additional fields inside a parameter entry, such as a human-readable description, type hint, minimum, maximum, or suggested values, may be recorded when useful. MLDB v1 does not require or universally interpret such advisory fields beyond the required `default` value.

`notes` and other human-readable supplemental metadata are optional.

### YAML responsibility and public parameter interface

The YAML is the authoritative Train Protocol identity, lifecycle, Task binding, implementation integrity record, and public parameter interface.

The YAML is not required to encode every executable detail of the training procedure. Only values that are intentionally exposed for per-Run variation belong under `parameters`.

For example, a protocol may expose `learning_rate` and `batch_size` while keeping loss implementation, augmentation policy, optimizer family, checkpoint-selection logic, and other behavior fixed directly in the protocol Python file. This preserves the rule that MLDB exposes only the useful common automation surface rather than attempting to normalize the complete training algorithm.

For every Training Run, generic MLDB tooling resolves a complete parameter mapping by taking the Train Protocol defaults and replacing only keys explicitly supplied for that Run. Unknown keys are invalid. The fully resolved mapping is passed to the protocol through `TrainContext.parameters` and is recorded by Training Run.

The sibling Python implementation remains authoritative for all executable training behavior not represented by the public parameter interface. Protocol code must consume exposed values from `TrainContext.parameters`; it must not silently substitute a different protocol-local default for a published parameter.

`notes` may summarize important fixed behavior such as optimizer family, scheduler structure, loss design, augmentation policy, mixed-precision policy, checkpoint selection, or external-framework behavior without making those facts automatically variable between Runs.

MLDB v1 deliberately permits fixed training behavior to be written directly in Python when exposing it as a generic Run parameter would add more complexity than value.

### Python implementation responsibility

The sibling `<train-protocol-id>.py` file is the authoritative executable training definition.

It must expose:

```python
def train(context: TrainContext) -> torch.nn.Module:
    ...
```

The implementation must return the trained `torch.nn.Module` that the protocol has selected as the result of the execution. If the protocol performs checkpoint selection internally, it must restore the selected learned parameters into the returned module before returning.

The implementation may use any internal organization appropriate for the task, including:

- a direct PyTorch training loop;
- helper functions defined inside the same Train Protocol file;
- model-family-specific loss and target-assignment code;
- custom data loading and batching;
- external framework APIs;
- subprocess invocation of an external trainer such as NanoDet;
- framework-specific configuration files generated temporarily during execution.

Generic MLDB tooling does not inspect or standardize those internal choices.

### TrainContext contract

`TrainContext` is the common input supplied by MLDB to a Train Protocol.

The v1 context must provide access to at least:

```text
task
corpus
architecture
seed
parameters
work_dir
```

Conceptually:

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

The concrete helper classes may be defined later by MLDB implementation tooling, but the semantic contract is fixed here.

The resolved Architecture handle must provide the standardized Architecture construction operation established by MLDB-ADR-SCHEMA-003, conceptually:

```python
model = context.architecture.build()
```

The Corpus handle must provide the registered immutable Corpus artifact and its metadata. The Task handle must provide the referenced Task metadata. `seed` is supplied by the concrete Training Run and is not part of reusable Train Protocol identity. `parameters` is the complete resolved mapping of this protocol's published parameters for the concrete Run. `work_dir` is an execution-owned scratch/output destination in which the protocol may write framework checkpoints, logs, histories, generated configuration, and other protocol-specific execution files. Canonical MLDB learned weights are not written by the protocol into `work_dir`; they are serialized by the MLDB launcher after `train(context)` returns.

Train Protocol code must not need hard-coded knowledge of the repository filenames of the selected Architecture or Corpus in order to resolve those MLDB assets.

### Returned trained module contract

A successful Train Protocol call returns one trained `torch.nn.Module`.

The returned module represents the protocol-selected learned result of the execution. The protocol is responsible for all logic required to decide what that result is, including any best-checkpoint selection, early stopping, or restoration of externally produced framework weights.

For a conventional in-process PyTorch trainer, the protocol may simply return the trained model after restoring its selected best state. For an external framework such as NanoDet, the protocol may run the external trainer, identify its selected checkpoint, construct the selected MLDB Architecture through `context.architecture.build()`, load the selected learned parameters into that module, and return it.

The protocol must not return an optimizer checkpoint, path, framework trainer object, or serialized model artifact in place of the module.

After return, generic MLDB tooling validates that the module's state is strictly loadable into a fresh instance of the selected Architecture and then performs canonical learned-weight serialization. Training Run owns the resulting artifact identity and hash.

### Task, Corpus, and Architecture relationships

A Train Protocol references exactly one Task through `task` because its loss semantics, target handling, validation logic, or other behavior may depend on the prediction problem.

A Train Protocol does not permanently reference one Corpus or one Architecture.

Corpus and Architecture are supplied through `TrainContext` when the protocol is executed. This allows one reusable protocol to train multiple compatible Architectures against the same Task and allows the same protocol to be applied to later compatible Corpus revisions.

For example, one standard classifier Train Protocol may be reused for Plain CNN, MobileNet, and C8 Architectures when their declared interfaces and the selected Corpus are compatible.

Training Run records the exact combination of:

```text
Task
Corpus
Architecture
Train Protocol
execution-specific values
```

used for one concrete training execution.

### Compatibility validation

The generic launcher should reject obvious incompatibilities before invoking `train(context)` when they can be determined from existing MLDB metadata.

At minimum, future validation may check:

- Train Protocol Task equals the resolved Architecture Task;
- Train Protocol Task equals the resolved Corpus Task;
- referenced Task, Corpus, Architecture, and Train Protocol exist;
- required assets are sealed when the later run lifecycle requires sealed inputs;
- sealed implementation and Corpus hashes are valid;
- Corpus representation is compatible with the Architecture input interface where the declared metadata is sufficient;
- Architecture output kind is compatible with any coarse requirement explicitly declared by the Train Protocol.

MLDB does not attempt to statically prove arbitrary Python training code correct. Protocol-specific validation may remain inside `train()` when the condition cannot be represented usefully in generic metadata.

### Project-owned execution behavior is self-contained

A sealed Train Protocol must be self-contained with respect to project-owned executable training behavior. Code that determines loss computation, augmentation, optimizer or scheduler behavior, sample exposure, training-loop semantics, checkpoint selection, or other result-affecting project logic must live in the sibling `<train-protocol-id>.py` file itself rather than being imported from another mutable project-local implementation module.

The protocol may import MLDB infrastructure that only provides non-training behavior such as the `TrainContext` type, asset resolution, metadata access, filesystem utilities, or generic logging transport. It may also import and invoke third-party frameworks or trainers such as PyTorch or NanoDet. Exact third-party versions are recorded by the concrete Training Run environment.

This rule intentionally permits duplication between sealed Train Protocol files when necessary. MLDB v1 prefers self-contained, hash-verifiable training definitions over project-local DRY abstractions whose behavior could change beneath an unchanged protocol hash.

### Implementation integrity

For a sealed Train Protocol, `implementation.sha256` is the SHA-256 digest of the sibling `<train-protocol-id>.py` file.

MLDB tooling may verify this hash before execution. A digest mismatch invalidates a sealed Train Protocol until the implementation is restored or a new revision is created.

As with sealed Architecture assets, a Train Protocol must not hide material executable behavior in mutable project-local dependencies. Its project-owned training behavior is contained directly in the hashed Train Protocol Python file.

Third-party framework code and exact runtime package versions are execution-environment concerns and are not completely captured by the Train Protocol source hash.

### Responsibility boundary

Train Protocol owns the reusable training procedure.

It may define or implement:

- loss computation;
- target assignment;
- optimizer construction;
- learning-rate and scheduler behavior;
- epoch and step policy;
- batch size and data-loader behavior;
- training-time preprocessing and augmentation;
- initialization and pretrained-weight loading;
- freezing/unfreezing policy;
- AMP and numerical-mode settings;
- gradient clipping or accumulation;
- framework-specific checkpoint generation and selection logic used internally during training;
- training-time validation needed to select a checkpoint;
- delegation to external training frameworks.

Train Protocol does not own:

- the semantic target contract, which belongs to Task;
- materialized sample identity and representation, which belong to Corpus;
- unweighted model topology, which belongs to Architecture;
- one concrete execution's seed, timestamps, host, random outcome, logs, and final status, which belong to Training Run;
- the learned model identity, which belongs to Model;
- reusable standalone post-training evaluation definitions, which belong to later Evaluation Protocol entities;
- exported ONNX or other deployment artifacts, which belong to later export/artifact entities.

Training-time validation used only for checkpoint selection may remain inside Train Protocol. A reusable benchmark intended to compare trained Models independently should be represented later as an Evaluation Protocol rather than embedded as the only definition inside Train Protocol.

### Revision triggers

After sealing, a new Train Protocol revision is required for any executable change that may alter training behavior, including but not limited to:

- loss implementation or weighting changes;
- target-assignment changes;
- optimizer type or optimizer-parameter changes;
- learning-rate or scheduler changes;
- changes to the public parameter key set or any published parameter default;
- fixed epoch, batch-size, or step-policy changes not represented by published parameters;
- augmentation changes;
- initialization or pretrained-weight-loading changes;
- freeze/unfreeze schedule changes;
- AMP, TF32, gradient clipping, or accumulation changes;
- data-loader behavior that changes sample exposure or ordering semantics;
- checkpoint-selection changes;
- external-framework configuration changes;
- substantive changes to the `train()` implementation or its tracked execution dependencies.

Draft implementation and test fixes do not require revision churn while the protocol remains `draft`.

## Rationale

Standardizing the Train Protocol entrypoint rather than the training loop gives MLDB one reliable automation surface without pretending that all model families train the same way.

`train(context) -> torch.nn.Module` is small enough to wrap conventional PyTorch classifiers, custom detection training, and external frameworks while also giving MLDB a framework-neutral point at which to take ownership of canonical learned-weight serialization. Plain CNN, MobileNet, and C8 models can share one classifier Train Protocol because Architecture construction is already standardized as `build() -> torch.nn.Module`. Custom FCOS can use the same MLDB launch mechanism while retaining its specialized loss and validation implementation. NanoDet can participate by translating its selected external checkpoint back into the selected MLDB Architecture module before return.

Keeping the public parameter interface deliberately small avoids recreating the configuration language of every training framework inside MLDB while still allowing useful automated sweeps. Only explicitly published keys may vary between Runs; fixed behavior remains in Python. This supports queue-driven parameter exploration without turning Training Run into an unrestricted override mechanism.

Not binding Train Protocol permanently to one Architecture or Corpus preserves useful reuse. The exact Architecture/Corpus/Protocol combination is an execution fact and is better recorded by Training Run.

This boundary deliberately accepts some duplication. Keeping result-affecting project-owned training behavior inside each sealed protocol makes the protocol hash meaningful and avoids introducing a separate executable-code dependency graph.

## Rejected alternatives

### Build one universal MLDB training loop

The current classifier and custom FCOS trainers already differ in batch representation, loss construction, scheduler stepping, validation, freezing, augmentation, and checkpoint selection. NanoDet additionally uses its own trainer and configuration system.

A universal loop would accumulate model-family-specific switches and would force external frameworks into abstractions that do not match them.

MLDB therefore standardizes invocation, not the internal loop.

### Fully normalize all training settings into YAML

A universal schema for losses, optimizers, schedulers, augmentations, target assigners, detector post-processing, freeze schedules, and external-framework settings would duplicate existing framework configuration systems and continually expand as new methods are introduced.

Train Protocol YAML therefore exposes free-form summary parameters while executable behavior remains in Python.

### Treat YAML as the complete executable specification

Requiring every executable behavior to be reconstructable from YAML would either prevent specialized training logic or require MLDB to implement a generic training-programming language.

The sibling Python file is instead authoritative.

### Bind one Train Protocol to one Architecture

That would require separate protocol identities for Plain CNN, MobileNet, and C8 even when the exact same classifier training procedure applies to all of them.

Architecture is therefore supplied at execution time through `TrainContext`.

### Bind one Train Protocol to one Corpus

A frozen Corpus revision is an execution input, not necessarily part of the reusable training procedure. Binding it permanently would duplicate otherwise identical protocols whenever the dataset is refreshed.

Corpus is therefore supplied at execution time through `TrainContext`.

### Share result-affecting project-local trainer helpers between sealed protocols

This would allow a shared helper to change training behavior while the Train Protocol file and its hash remained unchanged, weakening protocol identity.

MLDB v1 therefore keeps result-affecting project-owned training behavior inside each sealed Train Protocol file. Only non-training MLDB infrastructure and third-party framework code may remain external.

## Consequences

Future MLDB Train Protocol tooling can remain deliberately small. It should be able to:

- discover sibling `.yaml` and `.py` files under `mldb_data/train_protocols/`;
- verify ID and basename equality;
- enforce the terminal `-v<positive-integer>` grammar;
- validate `draft` / `sealed` lifecycle values;
- resolve the referenced Task;
- require and verify `implementation.sha256` for sealed protocols;
- import the Python implementation and verify a `train` entrypoint;
- resolve the complete public Train Protocol parameter mapping from protocol defaults plus Run-supplied values;
- reject unknown Run parameter keys;
- construct a `TrainContext` containing resolved Task, Corpus, Architecture, the Training Run seed, resolved parameters, and work-directory handle;
- perform generic compatibility and integrity checks that can be derived from MLDB metadata;
- invoke `train(context)`;
- verify that the return value is a `torch.nn.Module` whose state is strictly compatible with a fresh instance from the selected Architecture;
- serialize the returned module's canonical learned `state_dict` through MLDB-owned Training Run tooling;
- tolerate arbitrary `parameters`, notes, and protocol-specific metadata it does not understand.

MLDB does not need to implement a universal trainer before Train Protocol assets are useful. Existing project training scripts may be adapted into self-contained Train Protocol implementations incrementally; once sealed, result-affecting project-owned behavior must reside inside the protocol Python file.

Training Run uses this stable invocation contract to automate the exact selection and execution of Task, Corpus, Architecture, and Train Protocol assets while recording the concrete execution outcome.

## Evidence

The current `tools/recognition/train_tile_shape_classifier.py` constructs Plain, C8, and resolution-preserving MobileNet classifiers inside one conventional classifier training workflow. It uses cross-entropy, AdamW, cosine annealing, AMP, classifier-specific augmentation, angle-sweep validation, and checkpoint-selection logic. This demonstrates that multiple Architectures can use one reusable Train Protocol when their Task and forward interface are compatible.

The current `tools/recognition/train_rotated_fcos_nano.py` uses a substantially different data loader and collate structure, a detector-specific `compute_loss`, per-step warmup/cosine scheduling, backbone freezing, gradient clipping, rotated-box decoding, detector validation, early stopping, and its own checkpoint-selection key. This demonstrates why specialized protocol code must remain possible even when Architecture construction is standardized.

The current NanoDet workflow uses NanoDet's own `tools/train.py` with generated YAML configuration and invokes it through a subprocess from project orchestration code. This demonstrates that a useful Train Protocol abstraction must permit delegation to an external trainer rather than require all training to execute inside a common project-owned PyTorch epoch loop.
