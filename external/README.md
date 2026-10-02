# External source repositories

`external/` is the workspace for third-party source repositories that are developed or consumed alongside `mjtensu` but are **not** owned by the main `mjtensu` Git repository.

Each child directory is an independent Git repository with its own history and remotes. The parent `mjtensu` repository ignores those child repositories.

Examples:

```text
external/
├── agari/
├── nanodet-aabb-optimized/
└── <future-fcos-fork>/
```

## Fork naming

Do not use the upstream repository/package/import name unchanged for a modified fork that must coexist with upstream or another fork on MLDB workers.

Use a short name that combines the upstream family with the meaningful modification:

```text
<upstream-family>-<meaningful-modification>
```

Examples:

```text
nanodet-aabb-optimized
nanodet-rotated
```

The modification part should describe the role of the fork, not every individual patch. Do not rename the fork every time another optimization is added.

The same semantic stem should be used consistently:

```text
Git repository:       nanodet-aabb-optimized
Python distribution:  nanodet-aabb-optimized
Python import:        nanodet_aabb_optimized
```

The Python import namespace **must also be renamed**. Renaming only the wheel/distribution while leaving `import nanodet` unchanged does not solve worker-side namespace collisions.

## Git remotes

For a public modified OSS fork, use the normal fork workflow:

```text
origin    -> the maintained public fork
upstream  -> the original OSS repository
```

Source changes are committed and pushed in the child repository, not copied into the parent `mjtensu` repository.

Preserve the upstream license, copyright notices, NOTICE files, and any other redistribution obligations. Code, model weights, and datasets can have different licenses; do not assume the source-code license also covers weights or data.

## MLDB integration

Do not put fork-selection or package-version fields into Architecture, Train Protocol, Evaluation Protocol, or Study YAML.

A fork used by formal MLDB execution is packaged as a uniquely named Python distribution, published as an immutable wheel, and added to the global MLDB runtime registry. See:

```text
mldb_v2/docs/EXTERNAL_FORK_PACKAGES.md
mldb_v2/docs/RUNTIME_REGISTRY.md
```
