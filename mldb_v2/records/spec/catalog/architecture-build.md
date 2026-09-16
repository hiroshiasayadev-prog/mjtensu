# Contract: Architecture build interface

- **id**: `spec:mldb.v2.catalog.architecture_build`
- **status**: draft
- **date**: 2026-09-09
- **parent**: `spec:mldb.v2.catalog`
- **contract_class**: `interface`

## Request

Architecture v1 companion modules expose exactly:

```python
def build() -> torch.nn.Module:
    ...
```

`build` takes no caller arguments. One Architecture ID identifies one concrete unweighted
structure; structural switches are not supplied dynamically by Study or Train Protocol.
## Response and rules

A successful call returns one `torch.nn.Module` compatible with the Architecture YAML's coarse
input/output interface. It MUST NOT silently load learned/pretrained state as part of Architecture
identity.

Generic MLDB code MUST NOT branch on model family, rewrite the returned graph, or supply topology
parameters. Architecture construction does not train, compute loss, construct optimizer policy, or
load a Model's learned weights.

Verification rejects a missing/non-callable entrypoint, caller-required arguments, exceptions from
`build()`, or a return value that is not `torch.nn.Module`. Train and Model loading use this same
boundary whenever they require a fresh module.
