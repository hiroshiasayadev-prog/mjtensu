from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mldb.src.api.controller import execute_study, get_study_run
from mldb.src.common.ids import StudyId
from mldb.src.orchestration._sqlite_queue import SQLiteQueue
from mldb.src.repository._local_filesystem import LocalFilesystem
from mldb.src.repository.layout import RepositoryLayout
from mldb.src.repository._process_lock import repository_process_lock
from mldb.src.study.run import StudyRunStatus


def _queue_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _runtime_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit one sealed MLDB Study to the shared Queue and optionally wait "
            "for its terminal Study Run state. SSH execution is owned by the "
            "persistent run_ssh_worker.py process."
        )
    )
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument(
        "--host",
        help="Deprecated compatibility option; the Study submitter does not use SSH.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    repo_root = args.repo_root.resolve()
    layout = RepositoryLayout(repo_root)
    filesystem = LocalFilesystem()
    queue = SQLiteQueue(repo_root)

    if args.host:
        print(
            "note: --host is ignored by Study submission; start "
            "tools/mldb/run_ssh_worker.py --host <host> --loop separately",
            flush=True,
        )

    with repository_process_lock(repo_root, "controller"):
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
    print(f"study_run={launched.study_run_id} status={launched.status.value}", flush=True)
    if args.no_wait:
        return

    last_status: StudyRunStatus | None = None
    while True:
        progress = get_study_run(
            launched.study_run_id,
            _queue_now(),
            layout,
            filesystem,
            queue,
        )
        current = progress.study_run
        if current.status is not last_status:
            print(f"study_run={current.id} status={current.status.value}", flush=True)
            last_status = current.status
        if current.status is not StudyRunStatus.RUNNING:
            return
        time.sleep(float(args.poll_seconds))


if __name__ == "__main__":
    main()
