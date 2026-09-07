# MLDB-ADR-SCHEMA-003: Define versioned PyTorch Architecture assets

- **status**: accepted
- **date**: 2026-09-03
- **depends_on**: MLDB-ADR-SCHEMA-001
- **supersedes**:
- **migrated_to_spec**:

## Context

MLDB needs a stable identity for model structure independently from Corpus choice, training conditions, learned weights, evaluation results, and exported runtime artifacts.

The project already contains substantially different model families: conventional convolutional tile classifiers, MobileNetV3-style classifiers, C8-equivariant classifiers, NanoDet detectors, and a custom rotated FCOS detector. Their internal graphs, heads, feature pyramids, output tensors, helper classes, and configuration concepts differ enough that attempting to encode the complete network graph in one universal YAML structure would create a project-specific neural-network description language.

PyTorch already provides the useful common runtime abstraction for these models: an instantiated architecture is a `torch.nn.Module`. The common MLDB contract can therefore standardize how an Architecture is identified, described, instantiated, and frozen while leaving the internal module graph to Python.

Architecture must also remain separate from Train Protocol. Loss functions, target assignment, optimizers, learning rates, schedulers, augmentation, epoch counts, initialization policy, pretrained-weight selection, and other training behavior do not define the unweighted model structure and should not be duplicated as Architecture defaults.

## Decision

Introduce `Architecture` as a versioned MLDB asset that defines one unweighted model structure for one Task.

For Architecture v1, PyTorch is the supported framework. Each Architecture consists of a YAML metadata record and a sibling Python implementation file with the same Architecture ID basename.

### Physical placement and basename contract

Architecture assets live outside the Brewprint Design Records tree under:

```text
mldb_data/
  architectures/
```

Each Architecture occupies two sibling files:

```text
mldb_data/architectures/<architecture-id>.yaml
mldb_data/architectures/<architecture-id>.py
```

Example:

```text
mldb_data/architectures/plain-cnn-v1.yaml
mldb_data/architectures/plain-cnn-v1.py

mldb_data/architectures/mobilenet-v3-small-f8-r1-v1.yaml
mldb_data/architectures/mobilenet-v3-small-f8-r1-v1.py
```

The basename is the immutable Architecture ID. The YAML does not repeat the Python path because it is derived from this basename rule.

### Architecture ID and revision grammar

Every Architecture ID must end with an explicit positive-integer MLDB revision suffix:

```text
<architecture-base-id>-v<positive-integer>
```

Examples:

```text
plain-cnn-v1
plain-cnn-v2
mobilenet-v3-small-f8-r1-v1
mobilenet-v3-small-f8-r2-v1
rotated-fcos-nano-v1
```

The final `-vN` suffix is the MLDB Architecture revision. It is independent of model-family names or variant labels that may themselves contain version-like text such as `mobilenet-v3`, `f8-r1`, or `f8-r2`.

Only positive integers are allowed. Forms such as `-v1.1`, `-v1-beta`, or an omitted revision suffix are invalid.

Variant names and MLDB revisions have different meanings. `f8-r1-v1` and `f8-r2-v1` are different design variants. `f8-r1-v1` and `f8-r1-v2` are successive revisions of the same named design line.

### Lifecycle

Architecture status is one of:

```text
draft
sealed
```

A `draft` Architecture is under implementation or verification and may be edited in place without changing its revision suffix. Unit-test failures, shape-contract defects, implementation mistakes, and ordinary design iteration may therefore be corrected while the Architecture remains `draft`.

A `sealed` Architecture is immutable as an executable model definition. A sealed Architecture must never return to `draft`.

Any executable implementation change after sealing requires a new Architecture ID with a new revision. This includes bug fixes as well as intentional graph changes when they change the Python implementation used to construct the model.

Editorial changes that do not alter Architecture identity or executable meaning, such as spelling corrections in `description`, do not require a new revision.

### Architecture metadata format

Architecture metadata is YAML and uses schema identifier:

```text
mjtensu.mldb/architecture/v1
```

Example:

```yaml
schema: mjtensu.mldb/architecture/v1

id: mobilenet-v3-small-f8-r1-v1
status: sealed

task: tile-shape-classification-35-v1

name: Resolution-preserving MobileNetV3 f8-r1
family: mobilenet-v3

description: >
  MobileNetV3-style grayscale tile classifier that reaches an 8x8
  feature resolution and preserves that resolution through the late backbone.

implementation:
  framework: pytorch
  entrypoint: build
  sha256: 0123456789abcdef...

interface:
  input:
    kind: image-tensor
    layout: NCHW
    channels: 1
    spatial: [64, 64]
  output:
    kind: classification-logits

structure:
  summary: >
    Grayscale stride-2 stem followed by MobileNetV3 inverted-residual
    blocks, an 8x8 late feature map, 1x1 projection, global average
    pooling, and a fully connected classification head.
  traits:
    - inverted-residual
    - depthwise-separable-convolution
    - squeeze-excitation
    - global-average-pooling

parameters:
  final_feature_resolution: 8
  late_repeats: 1
  final_channels: 576
  classifier_hidden: 1024
```

The required fields are:

- `schema`;
- `id`;
- `status`;
- `task`;
- `name`;
- `family`;
- `description`;
- `implementation.framework`;
- `implementation.entrypoint`;
- `interface.input`;
- `interface.output`;
- `structure.summary`.

`implementation.sha256` is required when `status: sealed` and may be omitted while `status: draft`.

`structure.traits` and `parameters` are optional.

### YAML responsibility

The YAML is the authoritative Architecture identity, lifecycle, Task binding, coarse interface contract, and human-readable summary.

The YAML is not a complete network-graph description and must not duplicate every layer, block, tensor edge, or constructor statement from the Python implementation.

`family` is a non-empty searchable identifier such as `plain-cnn`, `mobilenet-v3`, `nanodet`, `rotated-fcos`, `resnet`, or another locally useful family name. MLDB v1 does not prescribe a global family enum.

`structure.summary` is intentionally human-readable. It should make the broad model construction understandable without opening the Python file.

`structure.traits`, when present, is a searchable list of descriptive terms. MLDB does not interpret these terms as a network type system.

`parameters`, when present, is free-form Architecture-specific summary metadata. A Plain CNN may record channel widths, while a MobileNet variant may record final spatial resolution and repeat count. Generic MLDB validation does not interpret the contents of `parameters`.

### Python implementation responsibility

The sibling `<architecture-id>.py` file is the authoritative executable definition of the model structure.

For `implementation.framework: pytorch`, the file must expose the no-argument entrypoint:

```python
def build() -> torch.nn.Module:
    ...
```

`implementation.entrypoint` is `build` in Architecture v1.

Calling `build()` must construct and return a `torch.nn.Module`. Architecture-specific constructor options are fixed inside the Architecture implementation rather than supplied by the caller.

The following pattern is valid:

```python
def build() -> torch.nn.Module:
    return PlainTileClassifier(
        class_count=35,
        channels=(32, 64, 128, 192),
    )
```

The following pattern is not the Architecture v1 contract:

```python
def build(*, channels, class_count, activation) -> torch.nn.Module:
    ...
```

A parameterized model factory can be useful during Architecture development, but an MLDB Architecture represents one resolved structure rather than a family generator. Distinct resolved structures receive distinct Architecture identities.

### Implementation integrity

For a sealed Architecture, `implementation.sha256` is the SHA-256 digest of the sibling `<architecture-id>.py` file.

MLDB tooling may verify this digest before consuming a sealed Architecture. A digest mismatch makes the Architecture invalid until the implementation is restored or a new Architecture revision is created.

A sealed Architecture implementation must be self-contained with respect to project-owned executable model logic. All project-owned code that determines the constructed model topology or forward behavior must live in the sibling `<architecture-id>.py` file itself. A sealed Architecture must not import such behavior from another mutable project-local Python module.

A sealed implementation may import framework or third-party library code such as PyTorch, torchvision, NanoDet, or escnn. Exact third-party package versions are execution-environment facts and belong to later execution/run environment records rather than Architecture identity. MLDB infrastructure imports that only provide non-model behavior such as asset handles, type definitions, or metadata access are also permitted.

### Framework and construction boundary

Architecture v1 standardizes PyTorch construction through `build() -> torch.nn.Module`; it does not standardize the internal module classes.

This contract is sufficient for conventional classifiers, MobileNet variants, C8 models, NanoDet wrappers, custom FCOS models, and future PyTorch model families because each can hide framework- or library-specific construction behind the same entrypoint.

For example, a NanoDet Architecture wrapper may load its resolved NanoDet model configuration internally and return the resulting `nn.Module`. A custom FCOS Architecture may directly instantiate its backbone, FPN, and head. A classifier Architecture may directly instantiate a conventional `nn.Module`.

### Forward interface

Architecture v1 does not require one universal `forward()` return structure beyond the semantics declared by `interface`.

Different model families legitimately expose different raw outputs. Examples include:

- classification logits as `Tensor[N, K]`;
- a tuple of multi-level dense prediction tensors for FCOS-style detection;
- NanoDet-specific dense head outputs.

`interface.input` and `interface.output` therefore provide a coarse compatibility contract rather than a complete static tensor type system.

For image models, `interface.input` should declare the applicable materialized input facts required for compatibility checking, such as `kind`, `layout`, `channels`, and fixed `spatial` dimensions when the Architecture requires them.

`interface.output.kind` is a non-empty semantic identifier such as `classification-logits` or `multilevel-dense-prediction`. Additional output metadata may be added when useful, but MLDB v1 does not attempt to encode every output tensor shape for every detector family.

Later validation may use this coarse interface together with Task, Corpus, and Train Protocol metadata to reject obvious incompatibilities before training.

### Task binding

Each Architecture references exactly one Task through `task`.

Architecture does not duplicate the Task's label vocabulary or target semantics. It only defines a model structure intended to implement that Task.

For a categorical Task, the resolved Architecture must provide an output structure compatible with the Task's target contract. For a classifier this normally means a classification head whose output class dimension equals the Task's ordered label count.

### Training boundary

Architecture owns model structure, not the training procedure.

Architecture metadata must not define or recommend authoritative values for:

- loss functions or loss weights;
- target assignment rules;
- optimizer type or parameters;
- learning rate;
- scheduler or warmup;
- epoch count;
- batch size;
- augmentation;
- gradient clipping or accumulation;
- AMP policy;
- EMA policy;
- checkpoint selection;
- pretrained-weight selection;
- training seed;
- initialization policy used by a tracked training procedure.

Those settings belong to Train Protocol so that one training configuration remains visible in one place and can be applied consistently across compatible Architectures.

`build()` must not silently load learned or pretrained weights as part of Architecture identity. Construction may leave parameters in the framework or model implementation's ordinary initial state, but the initialization or weight-loading policy used for a tracked training procedure is owned by Train Protocol.

Loss computation is also outside the Architecture contract. Common training logic may use generic losses for compatible outputs, while specialized detector training may use a Train-Protocol-specific training step or loss adapter. The need for specialized loss logic does not change the Architecture construction interface.

### Revision triggers

After sealing, a new Architecture revision is required for executable changes including, but not limited to:

- convolution kernel, stride, dilation, padding, or group changes;
- channel-width changes;
- block addition, removal, or reordering;
- backbone, neck, FPN, or head topology changes;
- activation or normalization changes;
- squeeze-excitation or attention changes;
- pooling strategy changes;
- classifier or detector head changes;
- output-branch changes;
- changes to `build()` or any project-owned code in the Architecture file that determines topology or forward behavior;
- implementation bug fixes that change the constructed or executed model.

A new Architecture revision is not required for changes solely to a later Train Protocol, Corpus, learned weights, Evaluation Protocol, benchmark environment, export format, or deployment selection.

## Rationale

Using `torch.nn.Module` as the common construction boundary reuses PyTorch's existing model abstraction instead of creating an MLDB-specific neural-network graph language.

The no-argument `build()` contract makes an Architecture ID resolve to one concrete structure. Callers do not need model-family-specific constructor knowledge, and a future common trainer can instantiate Plain CNN, MobileNet, NanoDet, or custom FCOS Architectures through the same mechanism.

Keeping raw `forward()` output families distinct avoids forcing detectors and classifiers into an artificial common tensor shape. The coarse YAML interface is sufficient for inventory, human understanding, and basic compatibility validation while specialized training behavior remains available where needed.

The explicit `-vN` suffix makes Architecture revision visible in filenames, metadata, run records, and command output without requiring a metadata lookup. Allowing unrestricted edits only in `draft` avoids revision-number proliferation during normal implementation and testing while sealing protects downstream references from silent code drift.

Hashing the sealed implementation detects accidental edits to the complete project-owned Architecture implementation. Requiring that implementation to be self-contained prevents a small wrapper file from giving a false impression of immutability while imported project-local model code changes underneath it.

Separating Architecture from Train Protocol keeps training defaults from being distributed across model code, runner defaults, and experiment configuration. The later Train Protocol record can therefore remain the single authoritative place for how a compatible Architecture is trained.

## Rejected alternatives

### Encode the complete network graph in YAML

Plain CNN, MobileNetV3, NanoDet, rotated FCOS, and future architectures have substantially different structural concepts. A universal layer/block YAML would either be too weak to represent real models or evolve into a second neural-network programming language that duplicates Python.

MLDB instead keeps Python as the executable graph definition and YAML as identity, interface, lifecycle, and summary metadata.

### Allow parameterized Architecture factories

A factory such as `build(channels=..., class_count=..., activation=...)` would allow one Architecture ID to produce multiple parameter topologies. That weakens identity and makes it harder to know which structure a downstream record actually used.

MLDB Architecture v1 therefore requires a resolved no-argument `build()` entrypoint.

### Require one universal forward return type

A classifier and a dense detector do not naturally return the same data structure. Standardizing them by wrapping every output into a large generic container would add adapter complexity without improving model identity.

The common contract stops at `torch.nn.Module`; YAML records a coarse output kind instead.

### Put loss and training defaults in Architecture

Losses, optimizers, augmentation, initialization, and related settings can vary while the network graph remains unchanged. Storing them in Architecture would split the authoritative training configuration across assets and encourage hidden override precedence.

They are assigned to Train Protocol instead.

### Increment revision for every draft edit

Normal model development includes test failures, incorrect shapes, implementation mistakes, and refactoring before an Architecture is ready for downstream use. Incrementing `-vN` for every such edit would create meaningless revision churn.

Draft Architectures may therefore be edited freely. Immutability begins at `sealed`.

### Permit sealed wrappers to import project-local topology implementations

Hashing only a small wrapper while allowing its imported project-owned implementation to change would not actually freeze the Architecture. The wrapper hash could remain stable while the effective model graph changes.

Sealed Architecture implementations must therefore contain their project-owned topology and forward implementation directly in the Architecture Python file. MLDB v1 does not introduce a separate Code Asset dependency mechanism.

## Consequences

Future Architecture tooling can remain small and generic. It should be able to:

- discover sibling `.yaml` and `.py` files under `mldb_data/architectures/`;
- verify Architecture ID and basename equality;
- enforce the terminal `-v<positive-integer>` grammar;
- validate `draft` / `sealed` lifecycle values;
- resolve and validate the referenced Task;
- require and verify `implementation.sha256` for sealed Architectures;
- import the Python implementation and verify a no-argument `build` entrypoint;
- verify that `build()` returns `torch.nn.Module`;
- optionally run a smoke forward using the declared input interface where practical;
- tolerate Architecture-specific `parameters`, `structure.traits`, and additional interface metadata it does not understand.

A future common trainer can instantiate all supported PyTorch Architectures through the same `build()` entrypoint. The trainer can share device movement, optimizer handling, scheduler handling, AMP, gradient handling, checkpointing, and logging while delegating model-output-specific loss or training-step behavior to Train Protocol where required.

Architecture itself remains independent of those trainer details.

## Evidence

The current `tools/recognition/tile_shape_classifier.py` implements conventional and C8 classifiers as `torch.nn.Module` classes whose forward path returns classification logits.

The current `tools/recognition/resolution_preserving_mobile_models.py` implements resolution-preserving MobileNetV3 classifiers as `torch.nn.Module` while exposing Architecture-specific constructor parameters such as final feature resolution and late repeat count. Those parameters can be resolved inside one MLDB Architecture `build()` wrapper.

The current `tools/recognition/rotated_fcos_nano.py` implements `RotatedFCOSNano` as `torch.nn.Module`, but its forward path returns a tuple of three multi-level dense prediction tensors and its specialized loss computation lives outside the module forward method. This demonstrates why `build() -> nn.Module` is a useful common boundary while one universal forward-output or loss contract is not.

The current NanoDet workflow uses NanoDet-specific configuration and construction code. That model can still participate through an MLDB Architecture wrapper that resolves its configuration internally and returns the resulting PyTorch module, without requiring generic MLDB code to understand NanoDet's internal graph description.
