from __future__ import annotations

import unittest

from mldb.src.catalog.task import (
    CategoricalTarget,
    Task,
    TaskInput,
    TaskScope,
    validate_task,
)
from mldb.src.common.ids import TaskId


class CatalogTaskTests(unittest.TestCase):
    def test_valid_task_preserves_authored_label_order_as_class_index_abi(self) -> None:
        labels = ("east", "south", "west", "north")
        task = self._task(labels)

        report = validate_task(task)

        self.assertTrue(report.valid)
        self.assertEqual(labels, task.target.labels)
        self.assertEqual("east", task.target.labels[0])
        self.assertEqual("north", task.target.labels[3])

    def test_duplicate_and_empty_labels_are_rejected_without_reordering(self) -> None:
        labels = ("1m", "", "1m", "2m")
        task = self._task(labels)

        report = validate_task(task)

        self.assertFalse(report.valid)
        self.assertEqual(labels, task.target.labels)
        self.assertEqual(
            ["target.labels[1]", "target.labels[2]"],
            [issue.path for issue in report.issues],
        )

    def test_wrong_schema_is_rejected(self) -> None:
        task = self._task(("a", "b"), schema="mjtensu.mldb/task/v2")

        report = validate_task(task)

        self.assertEqual("schema", report.issues[0].path)

    def _task(
        self,
        labels: tuple[str, ...],
        *,
        schema: str = "mjtensu.mldb/task/v1",
    ) -> Task:
        return Task(
            schema=schema,
            id=TaskId("test-task-v1"),
            name="Test task",
            problem_type="multiclass-classification",
            description="Test",
            input=TaskInput(semantic_unit="sample"),
            target=CategoricalTarget(type="categorical", labels=labels),
            semantics={"meaning": "task-local"},
            scope=TaskScope(includes=("included",), excludes=("excluded",)),
        )


if __name__ == "__main__":
    unittest.main()
