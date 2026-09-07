# Concept: Study Model expansion

- **id**: `spec:mldb.study.grid_expansion`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.study`

## What this is

Defines how one sealed Study materializes an ordered set of Study-local trials from either a training grid or an authored list of existing Models.

Study v1 does not define adaptive search or queue scheduling policy.

## Concept model

A Study chooses exactly one Model source.

Training source:

```text
Architecture
x every declared Train Protocol parameter values list
x training seed
        ↓
planned training trial
        ↓
Training Run -> Model
```

Existing-Model source:

```text
existing Model declaration order
        ↓
trial references existing Model
```

Every materialized trial receives the same declared evaluation-stage set after evaluation parameters are fully resolved.

## Rules

### Pre-materialization validation

Before allocating a Study Run or producing a valid immutable plan, Study execution validation must verify the common Study-format contract and the selected Model source.

For `model.train`, validation must verify:

- the training Corpus exists and satisfies required integrity checks;
- the Train Protocol exists, is sealed, and satisfies required implementation integrity checks;
- every selected Architecture exists, is sealed, and satisfies required implementation integrity checks;
- training Corpus, Train Protocol, and every Architecture reference the same Task;
- every declared training parameter key is public in the selected Train Protocol;
- `model.train.architectures` contains no duplicate Architecture IDs;
- every training parameter value list is non-empty, contains only JSON-compatible public-parameter values, and contains no duplicate decoded values;
- training seeds are non-empty, unique, integer-valued, and not boolean.

For `model.existing`, validation must verify:

- the list is non-empty and contains no duplicate Model IDs;
- every Model exists;
- every Model resolves through a completed Training Run and valid canonical learned weights;
- all selected Models resolve to one common Task.

For both source modes, validation must verify:

- `evaluations` is non-empty;
- every evaluation Corpus exists and satisfies required integrity checks;
- every Evaluation Protocol exists, is sealed, and satisfies required implementation integrity checks;
- every evaluation Corpus and Evaluation Protocol references the same Task as the Study Models;
- every fixed evaluation parameter key is public in its selected Evaluation Protocol;
- every fixed evaluation parameter value satisfies the JSON-compatible public-parameter value domain.

A Study that fails these checks must be rejected before Study Run allocation and must not be queued.

### Training-source expansion

For every Cartesian-product coordinate in `model.train`, materialization creates exactly one trial.

Each training-derived trial records:

- one Architecture;
- the training Corpus;
- the Train Protocol;
- one validated integer seed;
- one concrete value for every Study-declared training parameter axis;
- defaults for every public Train Protocol parameter omitted from the Study.

The resulting `training.parameters` mapping is the complete mapping defined by `spec:mldb.runtime.public_parameters`.

Training-source ordering from outermost to innermost axis is:

```text
Architecture declaration order
  -> training parameter keys in locale-independent ascending lexicographic order
    -> each parameter's values in declaration order
      -> training seeds in declaration order
```

Training seed is the innermost axis.

The exact parameter key strings are sorted independently of YAML mapping insertion order or locale. Reordering only the keys beneath `model.train.parameters` therefore does not change trial numbering.

Architecture order, each `values` list order, and seed order remain semantically significant authored list order.

### Existing-Model expansion

For every Model in `model.existing`, materialization creates exactly one trial.

The trial records the exact existing Model ID and creates no training coordinate.

Existing-Model trials preserve authored `model.existing` list order exactly.

No Training Run or new Model is created for these trials.

### Evaluation materialization

Every trial receives one materialized entry for each Study evaluation stage.

For each stage, materialization:

1. retains the Study-local `stage` identifier;
2. retains the selected evaluation Corpus and Evaluation Protocol;
3. resolves fixed Study caller values against that Evaluation Protocol's defaults;
4. stores the complete resolved parameter mapping in the plan.

Evaluation stages preserve Study declaration order within every trial.

For training-derived trials, evaluation execution waits for a successful Training Run to produce the trial Model.

For existing-Model trials, the Model dependency is already satisfied by the selected immutable Model.

### Trial identity and ordering

Every materialized plan assigns sequential Study Run-local IDs:

```text
trial-0001
trial-0002
...
```

Trial IDs begin at `trial-0001`, increase without gaps, and are stable only within one Study Run.

| Model source | trial ordering |
|---|---|
| `model.train` | Canonical Cartesian-product order defined above. |
| `model.existing` | Authored existing-Model list order. |

The same sealed Study with the same referenced immutable inputs must materialize deterministically rather than depending on hash-map iteration, filesystem traversal order, Worker scheduling, or Queue timing.

### Dependency intent

Training-derived trial:

```text
planned training
    ↓
Training Run
    ↓ completed
Model
    ↓
Evaluation stages
```

Existing-Model trial:

```text
existing Model
    ↓
Evaluation stages
```

Evaluation stages for one Model are independent siblings.

Training failure blocks only evaluation intents that require that trial's missing Model. Existing-Model trials have no training dependency to block.

Evaluation failure does not block sibling evaluations or unrelated trials.

## Boundary

| concern | owner |
|---|---|
| Authored Model-source and evaluation syntax | `spec:mldb.study.study_format`. |
| Persisted row representation | `spec:mldb.study.plan_format`. |
| Parameter default/override resolution | `spec:mldb.runtime.public_parameters`. |
| Training Run creation and execution | `spec:mldb.training`. |
| Model identity and loading | `spec:mldb.model`. |
| Evaluation Run creation and execution | `spec:mldb.evaluation`. |
| Queue scheduling, concurrency, retries, and Worker assignment | `spec:mldb.orchestration`. |
| Random/Bayesian/adaptive search | Outside Study v1. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.study` | Parent Study overview. |
| `spec:mldb.study.study_format` | Supplies the authored Model source and evaluation stages. |
| `spec:mldb.study.plan_format` | Persists materialized trials. |
| `spec:mldb.runtime.public_parameters` | Produces complete Run parameter mappings. |
| `spec:mldb.model` | Resolves existing Models selected by a Study. |
