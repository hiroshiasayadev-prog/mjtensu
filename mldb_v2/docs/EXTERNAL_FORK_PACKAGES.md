# External OSS Forks and Python Packages

This document defines how modified third-party OSS is kept separate from the `mjtensu` source tree and made available to formal MLDB execution.

The goals are:

- modified OSS remains a normal independently versioned Git repository;
- multiple upstream/fork variants can coexist on one worker without Python import collisions;
- MLDB experiment definitions do not choose repositories or environments;
- the global runtime registry remains the package-version authority for execution.

## 1. Source placement and ownership

Modified third-party repositories live below repository-root `external/`:

```text
mjtensu/
├── external/
│   ├── agari/
│   ├── nanodet-aabb-optimized/
│   └── <future-fcos-fork>/
├── mldb_data/
├── mldb_tests/
└── mldb_v2/
```

Each `external/<name>/` directory is an independent Git repository. It is ignored by the parent `mjtensu` repository and owns its own source history, tags, releases, CI, and license files.

For a public OSS fork, the expected remote layout is:

```text
origin    -> maintained public fork
upstream  -> original OSS repository
```

Do not vendor a modified fork into `mldb_data/`, `mldb_v2/`, or `tools/` merely to make it available to workers.

## 2. Fork/package/import naming

A modified fork must not reuse the upstream package/import name unchanged when it needs to coexist with upstream or another fork in the global worker environment.

Use:

```text
<upstream-family>-<meaningful-modification>
```

The suffix describes the fork's durable role, not every individual patch.

Examples:

```text
Git repository:       nanodet-aabb-optimized
Python distribution:  nanodet-aabb-optimized
Python import:        nanodet_aabb_optimized

Git repository:       nanodet-rotated
Python distribution:  nanodet-rotated
Python import:        nanodet_rotated
```

Do **not** do this:

```text
wheel/distribution: nanodet-aabb-optimized
import:             nanodet
```

Changing only the distribution name is insufficient. The importable top-level package must also have a unique namespace; otherwise installing multiple variants into one registry-managed worker still creates module collisions and order-dependent behavior.

If one fork intentionally contains several compatible features that are versioned and released together, keep one package. Split packages only when the implementations need independent release/version lifecycles or cannot safely coexist behind one import namespace.

## 3. Version and tag convention

Preserve the upstream version when it is meaningful, and append a fork revision.

Recommended distribution version:

```text
<upstream-version>+fork.<revision>
```

Recommended Git tag:

```text
v<upstream-version>-fork.<revision>
```

Example:

```text
version: 1.0.0+fork.1
tag:     v1.0.0-fork.1
wheel:   nanodet_aabb_optimized-1.0.0+fork.1-py3-none-any.whl
```

A tag identifies the source commit used to build the release. Do not stuff a Git SHA into every package version merely for provenance; the release tag already resolves to the source commit, while the MLDB runtime-registry snapshot pins the exact package artifact/version used by a Study Run.

If the fork no longer has a meaningful upstream release version, use an ordinary independent semantic version while retaining the unique fork/package/import name.

## 4. Wheel publication

Formal MLDB execution consumes built Python packages, not a mutable source checkout from `external/`.

For public forks, the default publication path is:

```text
fork source repository
    -> Git commit
    -> Git tag
    -> CI builds wheel
    -> GitHub Release asset
    -> MLDB runtime-registry candidate
```

Do not commit built wheels into the parent `mjtensu` repository.

Published fork tags and release assets are append-only operational artifacts: do not move a published tag or replace a wheel asset in place. If source or wheel bytes change, increment the fork revision and publish a new tag/release. The registry lock must resolve the versioned release artifact and its recorded integrity metadata; do not depend on a mutable `latest` URL.

Do not make workers install the fork directly from an editable checkout or an unpinned branch. A wheel gives the runtime registry one immutable installable artifact and avoids per-worker source builds.

For pure-Python forks, a universal wheel may be sufficient. If a fork later contains compiled/CUDA/native extensions, CI must build an artifact compatible with the actual worker platform; do not pretend an incompatible wheel is portable.

Private forks may require authenticated artifact distribution. Public OSS forks should prefer public release assets unless there is a concrete reason not to publish them.

## 5. Runtime-registry registration

The global runtime registry, not experiment YAML, decides which package artifact/version is active.

When a new fork release is ready:

1. build and publish the uniquely named wheel;
2. start from the current registry candidate `pyproject.toml`;
3. add an exact direct dependency for the released wheel/version;
4. resolve a fresh `uv.lock`;
5. verify the lock contains the intended unique distribution and artifact source;
6. publish the candidate through the runtime-registry `PUT /` path;
7. let the registry service perform its clean `uv sync --locked --no-install-project` validation;
8. smoke-test the new registry version on a migrated worker before using it for expensive ML experiments.

A direct public release wheel may be referenced from the registry candidate by immutable release URL. The resulting `uv.lock` is part of the immutable registry snapshot and must resolve the exact artifact/version used by the worker.

Never add fields such as these to Architecture/Protocol/Study YAML:

```text
repository
fork
package
wheel
venv
environment
runtime_registry_version
```

Executable Architecture/Protocol Python imports the unique module it needs, for example:

```python
import nanodet_aabb_optimized
```

The worker's pinned registry version is responsible for making that import available.

## 6. Relationship to MLDB source pinning

There are two separate provenance axes:

```text
MLDB source commit
    -> Study Plan source pin

Python runtime package set
    -> StudyResult.runtime_registry_version
```

The fork source itself is represented by its independently published package release. The runtime-registry snapshot identifies the exact released package used by a run; the fork's release tag maps that package back to its source commit.

Do not copy the fork Git commit, wheel URL, or package inventory into every TrainingResult/EvaluationResult. The immutable runtime-registry version is the environment-provenance handle.

## 7. Development versus formal execution

Local development may use the child repository directly while implementing or debugging the fork. Before formal MLDB execution, however, the implementation must cross the package boundary:

```text
external/<fork> source
    -> committed/pushed fork source
    -> tagged wheel release
    -> published runtime-registry version
    -> MLDB Study Run
```

This keeps exploratory source editing separate from reproducible worker execution.

## 8. Licensing

Public modification and redistribution are governed by the upstream license. Preserve required license/copyright/NOTICE material and comply with copyleft or source-distribution obligations where applicable.

Treat pretrained weights, datasets, generated assets, and third-party model files as separate licensed artifacts. Do not publish them merely because the source-code fork is public.
