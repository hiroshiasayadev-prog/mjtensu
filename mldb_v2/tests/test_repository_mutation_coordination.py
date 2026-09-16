import multiprocessing as mp
import os
import queue
import threading

import pytest

from mldb_v2.src.common.ids import StudyResultId
from mldb_v2.src.repository.mutation_coordination import StudyResultMutationCoordinator

_ID_A = StudyResultId("demo/run-" + "a" * 32)
_ID_B = StudyResultId("demo/run-" + "b" * 32)


def _hold_lock(repository_root, study_result_id, ready, release, acquired_queue=None):
    coordinator = StudyResultMutationCoordinator(repository_root)
    with coordinator.acquire(study_result_id=StudyResultId(study_result_id)):
        if acquired_queue is not None:
            acquired_queue.put("acquired")
        ready.set()
        release.wait(10)


def _crash_with_lock(repository_root, study_result_id, ready):
    coordinator = StudyResultMutationCoordinator(repository_root)
    with coordinator.acquire(study_result_id=StudyResultId(study_result_id)):
        ready.set()
        os._exit(17)


def test_same_study_result_threads_serialize(tmp_path) -> None:
    coordinator = StudyResultMutationCoordinator(tmp_path)
    first_ready = threading.Event()
    release_first = threading.Event()
    second_acquired = threading.Event()

    def first():
        with coordinator.acquire(study_result_id=_ID_A):
            first_ready.set()
            assert release_first.wait(timeout=5)

    def second():
        assert first_ready.wait(timeout=5)
        with coordinator.acquire(study_result_id=_ID_A):
            second_acquired.set()

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    second_thread.start()
    assert first_ready.wait(timeout=5)
    assert not second_acquired.wait(timeout=0.2)
    release_first.set()
    assert second_acquired.wait(timeout=5)
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)


def test_different_study_result_ids_are_independent(tmp_path) -> None:
    coordinator = StudyResultMutationCoordinator(tmp_path)
    held = threading.Event()
    release = threading.Event()

    def holder():
        with coordinator.acquire(study_result_id=_ID_A):
            held.set()
            assert release.wait(timeout=5)

    thread = threading.Thread(target=holder)
    thread.start()
    assert held.wait(timeout=5)
    with coordinator.acquire(study_result_id=_ID_B):
        pass
    release.set()
    thread.join(timeout=5)


def test_lock_released_when_context_body_raises(tmp_path) -> None:
    coordinator = StudyResultMutationCoordinator(tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        with coordinator.acquire(study_result_id=_ID_A):
            raise RuntimeError("boom")
    with coordinator.acquire(study_result_id=_ID_A):
        pass


def test_same_study_result_processes_serialize(tmp_path) -> None:
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    release = ctx.Event()
    acquired = ctx.Queue()
    holder = ctx.Process(
        target=_hold_lock,
        args=(str(tmp_path), str(_ID_A), ready, release),
    )
    waiter_ready = ctx.Event()
    waiter_release = ctx.Event()
    waiter = ctx.Process(
        target=_hold_lock,
        args=(str(tmp_path), str(_ID_A), waiter_ready, waiter_release, acquired),
    )
    holder.start()
    assert ready.wait(5)
    waiter.start()
    with pytest.raises(queue.Empty):
        acquired.get(timeout=0.3)
    release.set()
    assert acquired.get(timeout=5) == "acquired"
    waiter_release.set()
    holder.join(5)
    waiter.join(5)
    assert holder.exitcode == 0
    assert waiter.exitcode == 0


def test_process_crash_releases_os_lock_even_when_lock_path_remains(tmp_path) -> None:
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    process = ctx.Process(
        target=_crash_with_lock,
        args=(str(tmp_path), str(_ID_A), ready),
    )
    process.start()
    assert ready.wait(5)
    process.join(5)
    assert process.exitcode == 17

    coordinator = StudyResultMutationCoordinator(tmp_path)
    with coordinator.acquire(study_result_id=_ID_A):
        pass


def test_lock_files_are_operational_state_outside_mldb_data(tmp_path) -> None:
    coordinator = StudyResultMutationCoordinator(tmp_path)
    with coordinator.acquire(study_result_id=_ID_A):
        lock_files = list((tmp_path / ".local" / "mldb_v2" / "study_result_locks").glob("*.lock"))
        assert len(lock_files) == 1
        assert tmp_path / "mldb_data" not in lock_files[0].parents
