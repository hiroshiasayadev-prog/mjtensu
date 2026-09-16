# Reference: Repository layout

- **id**: `spec:mldb.v2.repository.layout`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.repository`
- **contract_class**: `reference`

## Namespace-first layout

A v2 namespace has this form:

```text
mldb_data/
  <namespace>/
    namespace.yaml
    tasks/
    corpora/
    architectures/
    train_protocols/
    evaluation_protocols/
    studies/
    study_plans/
    training_results/
    models/
    evaluation_results/
    study_results/
```

The recognized domain names are fixed. Empty domain directories MAY be absent.

A direct child of `mldb_data/` is a v2 namespace only when it contains `namespace.yaml`.
Existing v1 flat directories such as `mldb_data/tasks/` therefore remain outside v2 lookup.

## Entity paths

For typed reference `<namespace>/<local-id>`:

| kind | canonical YAML |
|---|---|
| Task | `<namespace>/tasks/<local-id>.yaml` |
| Corpus | `<namespace>/corpora/<local-id>.yaml` |
| Architecture | `<namespace>/architectures/<local-id>.yaml` |
| Train Protocol | `<namespace>/train_protocols/<local-id>.yaml` |
| Evaluation Protocol | `<namespace>/evaluation_protocols/<local-id>.yaml` |
| Study | `<namespace>/studies/<local-id>.yaml` |
| Study Plan | `<namespace>/study_plans/<local-id>.yaml` |
| Training Result | `<namespace>/training_results/<local-id>.yaml` |
| Model | `<namespace>/models/<local-id>.yaml` |
| Evaluation Result | `<namespace>/evaluation_results/<local-id>.yaml` |
| Study Result | `<namespace>/study_results/<local-id>.yaml` |

All paths above are relative to `mldb_data/`.

## Executable-asset tests

Required asset-test directories are namespace-first and deterministic:

```text
mldb_tests/<namespace>/architectures/<local-id>/
mldb_tests/<namespace>/train_protocols/<local-id>/
mldb_tests/<namespace>/evaluation_protocols/<local-id>/
```

No YAML stores a custom asset-test path.

## Executable companions

Python under a namespace domain is invalid unless the relevant domain contract explicitly permits
an executable companion.

When permitted:

```text
<local-id>.yaml
<local-id>.py
```

MUST be same-basename siblings. A `.py` without its owning YAML is invalid. `helpers.py`,
`common.py`, package subdirectories, and other shared implementation modules under `mldb_data/`
are invalid.

Corpus may additionally own a same-basename manifest:

```text
<local-id>.manifest.jsonl
```

as defined by the Corpus storage contract.

## ID/path consistency

The YAML `id` is the full `<namespace>/<local-id>` reference. The namespace segment MUST match the
parent namespace directory and the local segment MUST match the file basename.

The kind/domain is not embedded in the ID because lookup is typed.

Cross-namespace references are valid.

## Implementation-code boundary

Definition companion `.py` may import stable reusable code from normal project source packages.
An MLDB definition MUST NOT designate a script under `tools/` as its reusable implementation
entrypoint.
