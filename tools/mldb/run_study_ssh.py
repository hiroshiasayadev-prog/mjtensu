from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mldb.src.api.controller import execute_study
from mldb.src.common.ids import StudyId
from mldb.src.orchestration._coordination import orchestration_exclusion
from mldb.src.orchestration._sqlite_queue import SQLiteQueue
from mldb.src.orchestration.reconciliation import reconcile_study_run
from mldb.src.orchestration.retry_policy import RetryDecision
from mldb.src.repository._local_filesystem import LocalFilesystem
from mldb.src.repository.layout import RepositoryLayout
from mldb.src.study.run import StudyRunStatus



def _queue_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _runtime_now() -> datetime:
    return datetime.now(timezone.utc)

class _NoRetryPolicy:
    def after_preflight_failure(self, job, prior_attempts, *, at: str) -> RetryDecision:
        return RetryDecision(None)

    def after_unsatisfied_attempt(self, job, attempt_history, outcome, *, at: str) -> RetryDecision:
        return RetryDecision(None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch any authored MLDB Study and execute it over SSH."
    )
    parser.add_argument("--host", required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--remote-root", default=".cache/mjtensu-mldb-worker")
    parser.add_argument("--remote-python", default="/srv/bugrat/data-lv/mjtensu/nanodet/nanodet/.venv/bin/python")
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0)
    parser.add_argument("--max-worker-invocations", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    # Refuse to allocate a Study Run until the GPU host is reachable/authenticated.
    subprocess.run(["ssh", args.host, "true"], check=True)

    layout = RepositoryLayout(repo_root)
    filesystem = LocalFilesystem()
    queue = SQLiteQueue(repo_root)
    retry_policy = _NoRetryPolicy()

    launched = execute_study(
        StudyId(args.study_id),
        date.today(),
        _runtime_now(),
        layout,
        filesystem,
        queue,
        admitted_at=_queue_now(),
        failed_at=_runtime_now(),
    )
    print(f"study_run={launched.study_run_id} status={launched.status.value}")
    for invocation in range(1, int(args.max_worker_invocations) + 1):
        command = [
            sys.executable,
            str(repo_root / "tools" / "mldb" / "run_ssh_worker.py"),
            "--host", args.host,
            "--repo-root", str(repo_root),
            "--remote-root", args.remote_root,
            "--remote-python", args.remote_python,
            "--heartbeat-seconds", str(args.heartbeat_seconds),
        ]
        print(f"worker invocation {invocation}")
        worker_error = None
        try:
            subprocess.run(command, cwd=repo_root, check=True)
        except subprocess.CalledProcessError as error:
            worker_error = error

        with orchestration_exclusion():
            current = reconcile_study_run(
                launched.study_run_id,
                _runtime_now(),
                _queue_now(),
                layout,
                filesystem,
                queue,
                retry_policy,
            )
        print(f"study_run={current.id} status={current.status.value}")
        if worker_error is not None:
            raise worker_error
        if current.status is not StudyRunStatus.RUNNING:
            return

    raise RuntimeError("Study remained RUNNING after max worker invocations")


if __name__ == "__main__":
    main()
