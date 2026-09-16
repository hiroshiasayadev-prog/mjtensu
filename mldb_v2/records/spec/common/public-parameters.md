# Reference: MLDB v2 public parameters

- **id**: `spec:mldb.v2.common.public_parameters`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.common`
- **contract_class**: `reference`

## Value domain

Public parameter values are recursively limited to JSON-compatible values: `null`, boolean,
string, integer, finite floating-point number, array of accepted values, or mapping with string keys
and accepted values. NaN, infinities, non-string mapping keys, tagged YAML values, and Python or
framework objects are invalid.

Values are preserved without coercion. Strings are not parsed as numbers and boolean is not accepted
where another contract requires an integer.

Type-sensitive recursive equality is used for duplicate/enum checks: boolean, integer, floating
number, string, and null are distinct scalar types; arrays are ordered; map key order is insignificant.
Therefore `true`, `1`, and `1.0` are distinct values.

## Parameter declaration

Each protocol `parameters.<key>` is a mapping with required `default` and optional fields below.
| field | contract |
|---|---|
| `default` | Required JSON-compatible value. |
| `type` | Optional: `boolean`, `integer`, `number`, `string`, `array`, `object`, or `null`. |
| `enum` | Optional non-empty list of unique JSON-compatible allowed values. |
| `minimum` | Optional finite numeric inclusive lower bound. |
| `maximum` | Optional finite numeric inclusive upper bound. |
| `description` | Optional human-readable string. |

`minimum`/`maximum` are valid only for `type: integer|number`; boolean is never numeric here.
When both are present, `minimum <= maximum`. The declared default must satisfy every declared
constraint. Unknown declaration fields are invalid in schema v1.

## Resolution

Given caller overrides, MLDB produces a complete mapping containing exactly every published key once.
Omitted keys use the declared default; unknown caller keys are rejected. Every resolved value must
satisfy the declaration constraints.

Protocol implementation code MUST consume values from this resolved mapping and MUST NOT substitute
hidden caller-visible defaults.

Training seed is separate from Train Protocol public parameters. It is an integer and boolean is
invalid. Evaluation has no universal seed; a stochastic Evaluation Protocol publishes ordinary
`seed` parameter when required.
