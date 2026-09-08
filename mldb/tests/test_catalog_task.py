from __future__ import annotations

import unittest

from mldb.src.catalog.task import (
    CategoricalTarget,
    RotatedObjectDetectionGeometry,
    RotatedObjectDetectionTarget,
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

    def test_non_categorical_target_mapping_is_preserved_without_global_enum(self) -> None:
        task = Task(
            schema="mjtensu.mldb/task/v1",
            id=TaskId("detector-v1"),
            name="Detector",
            problem_type="future-segmentation",
            description="Future target shape.",
            input=TaskInput(semantic_unit="image"),
            target={"type": "future-segmentation", "mask_semantics": "task-local"},
            semantics={},
            scope=TaskScope(includes=("objects",), excludes=()),
        )

        report = validate_task(task)

        self.assertTrue(report.valid)
        self.assertEqual("future-segmentation", task.target["type"])

    def test_rotated_detection_target_contract_is_valid(self) -> None:
        task = Task(
            schema="mjtensu.mldb/task/v1", id=TaskId("obb-v1"), name="OBB",
            problem_type="rotated-object-detection", description="Detect tiles.",
            input=TaskInput(semantic_unit="screen-image"),
            target=RotatedObjectDetectionTarget(
                type="rotated-object-detection", labels=("mahjong_tile",),
                geometry=RotatedObjectDetectionGeometry(
                    format="cx-cy-w-h-angle-deg", angle_period_deg=180
                ),
            ),
            semantics={}, scope=TaskScope(includes=("tiles",), excludes=()),
        )
        self.assertTrue(validate_task(task).valid)

    def test_rotated_detection_target_rejects_bad_geometry(self) -> None:
        task = Task(
            schema="mjtensu.mldb/task/v1", id=TaskId("obb-v1"), name="OBB",
            problem_type="rotated-object-detection", description="Detect tiles.",
            input=TaskInput(semantic_unit="screen-image"),
            target=RotatedObjectDetectionTarget(
                type="rotated-object-detection", labels=("mahjong_tile",),
                geometry=RotatedObjectDetectionGeometry(format="bad", angle_period_deg=0),
            ), semantics={}, scope=TaskScope(includes=("tiles",), excludes=()),
        )
        self.assertEqual(2, len(validate_task(task).issues))

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
