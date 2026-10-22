#!/usr/bin/env bash
set -euo pipefail

MLDB_REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MLDB_REPO_ROOT
cd "$MLDB_REPO_ROOT"

# Load simple repository-local KEY=value defaults without overriding the
# environment inherited by this process. Blank lines and # comments are ignored.
if [[ -f "$MLDB_REPO_ROOT/.env" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ -z "$line" || "$line" == \#* || "$line" != *=* ]] && continue
    key="${line%%=*}"
    value="${line#*=}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ ! -v "$key" ]]; then
      printf -v "$key" '%s' "$value"
      export "$key"
    fi
  done < "$MLDB_REPO_ROOT/.env"
fi

python_is_supported() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1
}

for candidate in "$MLDB_REPO_ROOT/.venv/bin/python" python3.13 python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && python_is_supported "$candidate"; then
    exec "$candidate" -m mldb_v2.src.cli "$@"
  fi
done

# The Linux controller host may intentionally keep an older system Python.
# Fall back to the already-provisioned MLDB runner image rather than silently
# running v2 on an unsupported interpreter.
if command -v docker >/dev/null 2>&1 && docker image inspect mldb-clearml-runner:torch2.5.1-cu124-v1 >/dev/null 2>&1; then
  docker_args=(--rm -i --network host -e "MLDB_REPO_ROOT=$MLDB_REPO_ROOT" -e PYTHONDONTWRITEBYTECODE=1 -v "$MLDB_REPO_ROOT:$MLDB_REPO_ROOT" -w "$MLDB_REPO_ROOT")

  if [[ -f "$MLDB_REPO_ROOT/.env" ]]; then
    docker_args+=(--env-file "$MLDB_REPO_ROOT/.env")
  fi

  # Explicit process environment wins over .env for MLDB/backend credentials
  # and configuration.
  while IFS='=' read -r name _; do
    case "$name" in
      MLDB_*|CLEARML_*|AWS_*|MINIO_*|REQUESTS_CA_BUNDLE|SSL_CERT_FILE)
        docker_args+=(-e "$name")
        ;;
    esac
  done < <(env)

  # Worktrees keep Git metadata in the parent repository. Mount that metadata
  # at the same absolute path so plan/source-pinning works in the fallback.
  if git_common_dir="$(git -C "$MLDB_REPO_ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"; then
    case "$git_common_dir" in
      "$MLDB_REPO_ROOT"/*|"$MLDB_REPO_ROOT") ;;
      *) docker_args+=(-v "$git_common_dir:$git_common_dir") ;;
    esac
  fi

  local_ca="${HOME:-}/.local/share/mkcert/rootCA.pem"
  if [[ -n "${HOME:-}" && -f "$local_ca" ]]; then
    docker_args+=(-v "$local_ca:$local_ca:ro")
  fi

  exec docker run "${docker_args[@]}" --entrypoint python mldb-clearml-runner:torch2.5.1-cu124-v1 -m mldb_v2.src.cli "$@"
fi

printf '%s\n' "mldb: Python >=3.11 is required; no supported local interpreter or MLDB runner image is available." >&2
exit 127
