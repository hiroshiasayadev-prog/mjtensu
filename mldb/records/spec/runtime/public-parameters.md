# Reference: MLDB public parameter resolution

- **id**: `spec:mldb.runtime.public_parameters`
- **status**: draft
- **date**: 2026-09-04
- **parent**: `spec:mldb.runtime`

## What this is

Defines the common public-parameter model used by Train Protocol and Evaluation Protocol execution.

The rule converts one protocol declaration and one caller-supplied override mapping into the complete parameter mapping recorded by a concrete Run and passed to protocol code.

## Parameter model

| concept | contract |
|---|---|
| public parameter | A key declared by the selected protocol under its `parameters` mapping. Only public parameters may vary between Runs. |
| default | Every public parameter declaration contains `default`. The default is used when the caller omits that key. |
| caller value | A value supplied for one published key by a direct Run request or Study materialization. |
| resolved mapping | The complete mapping containing exactly one value for every public parameter and no unknown keys. |
| fixed protocol behavior | Result-affecting behavior not exposed as a public parameter remains fixed by the protocol implementation. |

An empty protocol `parameters` mapping is valid and exposes no Run-varying values.

## Resolution rules

Given protocol declarations `D` and caller overrides `C`, MLDB resolves parameters as follows:

```text
for each published key in D
    if key exists in C
        resolved[key] = C[key]
    else
        resolved[key] = D[key].default

if C contains a key not published by D
    resolution fails
```

| condition | result |
|---|---|
| Published key omitted by caller | Use the declared default. |
| Published key supplied by caller | Use the caller value. |
| Unknown caller key | Reject parameter resolution. |
| Protocol publishes no parameters and caller supplies none | Resolve to an empty mapping. |
| Protocol publishes no parameters and caller supplies any key | Reject parameter resolution. |

The resolved mapping must contain every published key exactly once.
The resolved mapping must not contain protocol metadata such as descriptions, limits, or suggested values.

## Value semantics

MLDB v1 public parameter values use the JSON-compatible data model so the same resolved mapping can be persisted losslessly in YAML Run records and Study Run `plan.jsonl`.

Accepted values recursively consist only of:

- `null`;
- boolean;
- string;
- finite integer or finite floating-point number;
- array of accepted values;
- mapping with string keys and accepted values.

NaN, positive or negative infinity, non-string mapping keys, YAML-specific tagged values, and implementation-specific objects are invalid public parameter values.

- The resolver must preserve the decoded value without coercing strings to numbers, numbers to strings, or other value types.
- The resolver must not infer a missing value from protocol implementation code.
- The resolver must not substitute an implementation-local default for a published parameter.
- Additional declaration fields such as `description`, `type`, `minimum`, `maximum`, or `suggested` are advisory unless another spec gives them normative semantics.
- A protocol may perform domain-specific validation after common resolution when that validation is not standardized by MLDB.

The protocol implementation must consume published parameter values from the resolved mapping supplied by MLDB.

## Execution ownership

| execution path | caller values supplied to this rule | owner of resulting resolved mapping |
|---|---|---|
| Direct Training Run | Explicit Train Protocol overrides from the Run request. | Training Run. |
| Study training trial | One concrete grid coordinate for published Train Protocol keys. | Generated Training Run. |
| Direct Evaluation Run | Explicit Evaluation Protocol overrides from the Run request. | Evaluation Run. |
| Study evaluation stage | Fixed values declared by that Study stage. | Generated Evaluation Run. |

The same resolution rule applies regardless of whether caller values originate from a direct request or Study materialization.

A Study does not create a second parameter namespace. Study keys must already be public keys of the referenced protocol.

## Seed boundary

Training seed is not a Train Protocol public parameter in MLDB v1.
Training Run owns the concrete training seed separately from its resolved parameter mapping.
The training seed is an integer execution input; boolean is invalid, and MLDB v1 defines no universal numeric range for it.

Evaluation Run has no universal seed field.
An Evaluation Protocol that requires a caller-controlled stochastic seed must publish `seed` as an ordinary Evaluation Protocol parameter.

## Failure boundary

Parameter resolution occurs during launch preflight before Training Run or Evaluation Run allocation.

An unknown caller key or malformed public-parameter declaration rejects that launch request, creates no Training Run or Evaluation Run, and prevents the request from reaching protocol code.

Study materialization also resolves protocol parameters before child Run allocation. Failure of one request or planned coordinate's parameter resolution must not mutate or cancel unrelated persisted Runs.

Study materialization must reject a Study whose declared parameter keys are not public keys of the referenced protocol before creating the immutable Study Run plan.

## Boundary

| concern | owner |
|---|---|
| Exact Train Protocol `parameters` YAML shape | Training protocol format spec. |
| Exact Evaluation Protocol `parameters` YAML shape | Evaluation protocol format spec. |
| Study Cartesian grid semantics | Study grid-expansion spec. |
| Training seed storage and lifecycle | Training Run specs. |
| Parameter persistence domain | This reference defines the JSON-compatible value domain. |
| Parameter coercion, generic type declarations, and numeric bounds | Not defined by MLDB v1. |
| Protocol-specific semantic validation | Selected Train Protocol or Evaluation Protocol. |
| Concrete resolver function, mapping class, and exception types | Implementation. |

## Related specs

| ref | relation |
|---|---|
| `spec:mldb.runtime` | Parent runtime overview. |
| `spec:mldb.runtime.asset_resolution` | Resolves the protocol whose public parameter declaration is consumed by this rule. |
| `spec:mldb.runtime.component_model` | Defines common runtime responsibility boundaries. |
