from __future__ import annotations

import multiprocessing
from pathlib import Path

from mldb.src.common.ids import StudyRunId
from mldb.src.orchestration._sqlite_queue import SQLiteQueue
from mldb.src.orchestration.jobs import TrainingJob, TrainingJobCoordinate


REPO_ROOT = Path(__file__).resolve().parents[2]


def _admit_study_job_process(root: str, study_id: str, start_event, output_queue) -> None:
    queue = SQLiteQueue(Path(root))
    study_run_id = StudyRunId(study_id)
    job = TrainingJob(TrainingJobCoordinate(study_run_id, "trial-0001"))
    start_event.wait(timeout=10.0)
    rows = queue.admit_study_jobs(
        study_run_id,
        frozenset({job}),
        admitted_at="2026-09-09T00:00:00.000000Z",
    )
    output_queue.put((study_id, rows[0].job_id))


def test_study_submitter_never_spawns_or_executes_worker() -> None:
    source = (REPO_ROOT / "tools" / "mldb" / "run_study_ssh.py").read_text(
        encoding="utf-8"
    )
    assert "import subprocess" not in source
    assert "subprocess.run" not in source
    assert "execute_study(" in source
    assert "get_study_run(" in source
    assert "repository_process_lock(repo_root, \"controller\")" in source


def test_distinct_studies_can_admit_to_one_queue_from_separate_processes(tmp_path) -> None:
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    output_queue = context.Queue()
    study_ids = ["sr-20260909-101", "sr-20260909-102"]
    processes = [
        context.Process(
            target=_admit_study_job_process,
            args=(str(tmp_path), study_id, start_event, output_queue),
        )
        for study_id in study_ids
    ]
    for process in processes:
        process.start()
    start_event.set()
    for process in processes:
        process.join(timeout=20.0)
        assert process.exitcode == 0
    admitted = dict(output_queue.get(timeout=5.0) for _ in processes)
    assert set(admitted) == set(study_ids)
    queue = SQLiteQueue(tmp_path)
    for study_id in study_ids:
        rows = queue.jobs_for_study_run(StudyRunId(study_id))
        assert len(rows) == 1
        assert rows[0].logical.coordinate.study_run == StudyRunId(study_id)


def test_ssh_worker_is_persistent_shared_queue_consumer() -> None:
    source = (REPO_ROOT / "tools" / "mldb" / "run_ssh_worker.py").read_text(
        encoding="utf-8"
    )
    assert 'parser.add_argument("--loop", action="store_true")' in source
    assert 'worker_id = args.worker_id or f"ssh:{args.host}"' in source
    assert 'repository_process_lock(repo_root, "controller")' in source
    assert "reconcile_study_run(" in source


def test_worker_reconcile_resolves_parent_job_via_attempt(monkeypatch) -> None:
    import importlib.util
    from types import SimpleNamespace

    worker_path = REPO_ROOT / "tools" / "mldb" / "run_ssh_worker.py"
    spec = importlib.util.spec_from_file_location("mldb_test_run_ssh_worker", worker_path)
    assert spec is not None and spec.loader is not None
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)

    study_run_id = StudyRunId("sr-20260909-777")
    run_id = "ev-20260909-777"
    attempt = SimpleNamespace(attempt_id=17, job_id=31, run_id=run_id)
    coordinate = SimpleNamespace(study_run=study_run_id)
    job = SimpleNamespace(logical=SimpleNamespace(coordinate=coordinate))

    class Queue:
        def attempt_by_id(self, attempt_id):
            assert attempt_id == 17
            return attempt

        def job_by_id(self, job_id):
            assert job_id == 31
            return job

    reconciled = SimpleNamespace(id=study_run_id, status=SimpleNamespace(value="completed"))
    calls = []
    def fake_reconcile(study_id, *args, **kwargs):
        calls.append(study_id)
        return reconciled

    monkeypatch.setattr(worker, "reconcile_study_run", fake_reconcile)
    monkeypatch.setattr(worker, "_now_runtime", lambda: object())
    monkeypatch.setattr(worker, "_now_queue", lambda: "2026-09-09T00:00:00.000000Z")

    assignment = SimpleNamespace(attempt_id=17, run_id=run_id)
    worker._reconcile_assignment(
        assignment,
        layout=SimpleNamespace(),
        filesystem=SimpleNamespace(),
        queue=Queue(),
        retry_policy=SimpleNamespace(),
    )
    assert calls == [study_run_id]
