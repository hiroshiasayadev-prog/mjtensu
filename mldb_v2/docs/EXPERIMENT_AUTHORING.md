# MLDB v2 Experiment Authoring Guide

This guide explains how to turn an ML question into the smallest correct set of MLDB v2 definitions and then execute it.

Formal schemas live under `../records/spec/`; this file is a decision and workflow guide, not a duplicate schema reference.

## 1. First question: what actually changed?

Use the narrowest entity whose semantics changed:

| Change | Entity to change/create |
|---|---|
| prediction target, labels, problem meaning | Task |
| dataset bytes, samples, split, representation | Corpus |
| model topology / construction / forward semantics | Architecture |
| optimizer, schedule, augmentation procedure, training algorithm, training telemetry semantics | Train Protocol |
| evaluation procedure, metric contract, evaluation artifact contract | Evaluation Protocol |
| combinations of existing assets, parameter values, seeds | Study |

Do not create a new Protocol just because a public parameter value changes. Put value sweeps in the Study matrix.

For formal Evaluation artifacts, decide their Study-level presentation in the Evaluation Protocol rather than in an individual Study. `study_view: hidden` (the default) keeps a diagnostic on the child Evaluation Task, `study_view: select` exposes one Study-level view with Model/trial selection, and `study_view: all` mirrors every Model/trial copy. Changing this field on a sealed Protocol requires a new Protocol revision even though it is presentation-only.

## 2. Reuse before creating

Before adding files:

1. Inspect `mldb_data/<namespace>/` for existing Task, Corpus, Architecture, Protocol, and Study definitions.
2. Inspect the referenced companion Python before assuming a new implementation is needed.
3. Check whether the desired change is already a public parameter.
4. Reuse an existing Evaluation Protocol when its metric semantics already answer the question.

A new definition is justified by a semantic change, not by a desire to create a fresh experiment label.

## 3. Namespace-first repository layout

Reusable definitions and canonical history live under one namespace root:

    mldb_data/<namespace>/namespace.yaml
    mldb_data/<namespace>/tasks/
    mldb_data/<namespace>/corpora/
    mldb_data/<namespace>/architectures/
    mldb_data/<namespace>/train_protocols/
    mldb_data/<namespace>/evaluation_protocols/
    mldb_data/<namespace>/studies/
    mldb_data/<namespace>/study_plans/
    mldb_data/<namespace>/study_results/
    mldb_data/<namespace>/training_results/
    mldb_data/<namespace>/models/
    mldb_data/<namespace>/evaluation_results/

Do not use the old flat v1 `mldb_data_old/` layout as a template for new v2 work.

Executable definitions pair a YAML definition with a same-basename Python entrypoint. Reusable experiment helpers may live under `mldb_data/<same-namespace>/lib/`; every helper imported directly or transitively must be listed in `implementation.sources` with its exact SHA-256. Repository-owned imports outside that namespace-private `lib/` boundary are invalid. Corpus definitions may additionally reference manifests/builders according to their formal contract.

## 4. Authoring flow

For a new experiment or semantic definition change:

1. State the ML question and comparison factors.
2. Identify the existing reusable definitions.
3. Decide the minimal entity changes using the table above.
4. If changing/creating a Train or Evaluation Protocol, perform the telemetry review in `TELEMETRY.md` before sealing.
5. Author the new definition as draft using the closest valid existing definition as a template.
6. Validate the definition/repository structure.
7. Run verification, including executable integrity and relevant asset tests.
8. Seal the reusable definition only after verification passes.
9. Create a Study that expresses the comparison matrix and seeds.
10. Commit/push the exact required canonical inputs, executable companions, and declared namespace-private helpers for source pinning.
11. Plan and run the Study through the public CLI.
12. Compare canonical Evaluation Result metrics and artifacts.

Do not stop at step 9 when the user asked to run the experiment.

## 5. CLI authoring checks

Examples:

    .\mldb.cmd validate train-protocol <namespace>/<id> --json
    .\mldb.cmd verify train-protocol <namespace>/<id> --json
    .\mldb.cmd seal train-protocol <namespace>/<id>

For a Study:

    .\mldb.cmd validate study <namespace>/<study-id> --json
    .\mldb.cmd verify study <namespace>/<study-id> --json
    .\mldb.cmd seal study <namespace>/<study-id>
    .\mldb.cmd plan <namespace>/<study-id>

Bulk verification is allowed, but mutation is intentionally explicit. Multi-target sealing requires both `--namespace` and `--all`.

## 6. Sealed definitions are immutable history

Once a reusable definition is sealed and may be referenced by completed execution history, do not overwrite its semantics in place.

If companion Python changes semantics, create a new version ID and update the implementation SHA through the formal verification/sealing flow. The same rule applies when telemetry series/cadence/step semantics change for a Protocol: those observations are part of Protocol semantic review.

Historical examples:

- classifier `tile-shape-train-gpu-v3` remained sealed; telemetry-enabled behavior became `tile-shape-train-gpu-v4`;
- detector `rotated-fcos-train-gpu-v2` remained sealed; validation-telemetry behavior became `rotated-fcos-train-gpu-v3`.

Do not change a historical companion and merely update its hash under the same sealed ID.

## 7. Common decision examples

### Sweep an existing hyperparameter

Question: compare rotation augmentation at 0, 15, and 30 degrees.

Action: keep the same Architecture and Train Protocol; put `rotation_augment_deg` values in the Study matrix.

### Compare model topology

Question: compare ShuffleNet stem/stride structures.

Action: create/reuse one Architecture ID per topology and compare them in one Study while keeping the same Corpus, Train Protocol, and Evaluation Protocol where possible.

### Change training procedure

Question: switch from AdamW to a materially different optimizer/schedule when the current Protocol does not expose that choice as an intentional public parameter.

Action: new Train Protocol version.

### Change only evaluation coordinates

Question: evaluate more angles using an Evaluation Protocol that already exposes `eval_angles`.

Action: change the Study parameter values, not the Evaluation Protocol.

### Change what training reports to Charts

Question: add/change the scientific meaning, cadence, or step semantics of Protocol telemetry.

Action: review `TELEMETRY.md`; for a sealed Protocol this is a semantic version change, so create a new Protocol ID.

## 8. Study design discipline

A Study should express experiment intent, not reproduce implementation code. Keep comparison factors explicit, use deterministic seeds intentionally, and avoid multiplying definitions when a matrix is sufficient.

Before execution, record what conclusion each Evaluation metric can support. Do not run a large sweep with no planned decision rule.

## 9. Source pinning is part of authoring

A Study is not execution-ready merely because its YAML validates. Formal planning requires every consumed source input to match the selected Git commit.

When authoring changes executable or canonical inputs:

- inspect the exact canonical inputs and same-basename executable companions;
- stage only those required files;
- review the cached diff;
- commit and push to a backend-reachable ref;
- plan while still on that source commit.

Unrelated dirty files may remain. Never broaden the commit just to make `git status` look clean.

## 10. Result interpretation

The experiment conclusion comes from canonical Training/Evaluation/Study records and formal artifacts. ClearML logs/Charts help explain what happened during execution but are not canonical result fields.

Do not manually edit a failed/partial result into success. Fix the actual blocker and use the supported resume/rerun/new-run flow.

## 11. Definition/spec references

For exact fields and validation semantics, follow the matching formal specs under:

- `records/spec/catalog/`
- `records/spec/training/`
- `records/spec/evaluation/`
- `records/spec/study/`
- `records/spec/verification/`
- `records/spec/repository/`

If this guide and a formal contract diverge, the contract and executable implementation must be reconciled before authoring continues.