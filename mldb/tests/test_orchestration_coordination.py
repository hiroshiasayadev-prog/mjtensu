from __future__ import annotations

import importlib
import os
from pathlib import Path
import tempfile
from threading import Barrier, Event, Thread
from typing import Callable
import unittest

import mldb.src.orchestration as orchestration_package
from mldb.src.orchestration import _coordination
from mldb.src.orchestration._coordination import orchestration_exclusion


_WAIT_SECONDS = 2.0


def _start_thread(target: Callable[[], None], errors: list[BaseException]) -> Thread:
    def run() -> None:
        try:
            target()
        except BaseException as error:
            errors.append(error)

    thread = Thread(target=run, daemon=True)
    thread.start()
    return thread


def _tree_snapshot(root: Path) -> dict[str, bytes | None]:
    snapshot: dict[str, bytes | None] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        snapshot[relative] = None if path.is_dir() else path.read_bytes()
    return snapshot


class OrchestrationCoordinationTests(unittest.TestCase):
    def tearDown(self) -> None:
        # Keep a failed blocking-behavior assertion from poisoning later tests.
        importlib.reload(_coordination)

    def test_nested_reentrancy(self) -> None:
        nested_entered = Event()
        errors: list[BaseException] = []

        def nested_use() -> None:
            with orchestration_exclusion():
                with orchestration_exclusion():
                    nested_entered.set()

        thread = _start_thread(nested_use, errors)

        self.assertTrue(nested_entered.wait(_WAIT_SECONDS))
        thread.join(_WAIT_SECONDS)
        self.assertFalse(thread.is_alive())
        self.assertEqual([], errors)

    def test_concurrent_use_is_serialized(self) -> None:
        first_entered = Event()
        release_first = Event()
        first_exited = Event()
        second_ready = Event()
        second_entered = Event()
        start_gate = Barrier(2)
        order: list[str] = []
        errors: list[BaseException] = []

        def first_use() -> None:
            with orchestration_exclusion():
                order.append("first_enter")
                first_entered.set()
                if not release_first.wait(_WAIT_SECONDS):
                    raise AssertionError("first holder was not released")
            order.append("first_exit")
            first_exited.set()

        def second_use() -> None:
            if not first_entered.wait(_WAIT_SECONDS):
                raise AssertionError("first holder did not enter")
            second_ready.set()
            start_gate.wait(_WAIT_SECONDS)
            with orchestration_exclusion():
                order.append("second_enter")
                second_entered.set()

        first = _start_thread(first_use, errors)
        self.assertTrue(first_entered.wait(_WAIT_SECONDS))
        second = _start_thread(second_use, errors)
        self.assertTrue(second_ready.wait(_WAIT_SECONDS))
        start_gate.wait(_WAIT_SECONDS)

        self.assertFalse(second_entered.wait(0.2))
        release_first.set()
        self.assertTrue(first_exited.wait(_WAIT_SECONDS))
        self.assertTrue(second_entered.wait(_WAIT_SECONDS))

        first.join(_WAIT_SECONDS)
        second.join(_WAIT_SECONDS)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual([], errors)
        self.assertEqual(["first_enter", "first_exit", "second_enter"], order)

    def test_exception_releases_for_other_thread(self) -> None:
        class MarkerError(Exception):
            pass

        with self.assertRaises(MarkerError):
            with orchestration_exclusion():
                raise MarkerError("boom")

        entered = Event()
        errors: list[BaseException] = []

        def acquire_after_exception() -> None:
            with orchestration_exclusion():
                entered.set()

        thread = _start_thread(acquire_after_exception, errors)
        self.assertTrue(entered.wait(_WAIT_SECONDS))
        thread.join(_WAIT_SECONDS)
        self.assertFalse(thread.is_alive())
        self.assertEqual([], errors)

    def test_context_exit_releases_for_other_thread(self) -> None:
        with orchestration_exclusion():
            pass

        entered = Event()
        errors: list[BaseException] = []

        def acquire_after_exit() -> None:
            with orchestration_exclusion():
                entered.set()

        thread = _start_thread(acquire_after_exit, errors)
        self.assertTrue(entered.wait(_WAIT_SECONDS))
        thread.join(_WAIT_SECONDS)
        self.assertFalse(thread.is_alive())
        self.assertEqual([], errors)

    def test_private_surface_is_not_package_exported(self) -> None:
        public_names = {
            name for name in vars(_coordination) if not name.startswith("_")
        }

        self.assertEqual({"orchestration_exclusion"}, public_names)
        self.assertNotIn("orchestration_exclusion", vars(orchestration_package))

    def test_exclusion_does_not_mutate_canonical_or_queue_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            canonical = root / "mldb_data" / "study" / "sentinel.json"
            queue = root / ".local" / "mldb" / "queue.sqlite"
            canonical.parent.mkdir(parents=True)
            queue.parent.mkdir(parents=True)
            canonical.write_bytes(b"canonical-sentinel")
            queue.write_bytes(b"queue-sentinel")
            before = _tree_snapshot(root)
            previous_cwd = Path.cwd()

            try:
                os.chdir(root)
                with orchestration_exclusion():
                    with orchestration_exclusion():
                        pass
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(before, _tree_snapshot(root))


if __name__ == "__main__":
    unittest.main()
