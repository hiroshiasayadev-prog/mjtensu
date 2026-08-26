from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.recognition.build_nanodet_composite_augmented_dataset import (
    _repository_relative_image_name,
    split_composite_images,
)


class CompositeAugmentedDatasetBuilderTest(unittest.TestCase):
    def test_split_is_deterministic_and_keeps_source_groups_together(self) -> None:
        images = []
        image_id = 1
        groups: dict[int, set[int]] = {}
        for source_image_id, group_size in enumerate((3, 2, 4, 1, 3), start=100):
            groups[source_image_id] = set()
            for _ in range(group_size):
                images.append(
                    {
                        "id": image_id,
                        "file_name": f"images/composite_{image_id:06d}.png",
                        "source_annotation_json": "C:/dataset/instances_train.json",
                        "source_image_id": source_image_id,
                    }
                )
                groups[source_image_id].add(image_id)
                image_id += 1

        payload = {
            "images": images,
            "annotations": [],
            "categories": [{"id": 1, "name": "mahjong_tile"}],
        }
        first = split_composite_images(payload, train_fraction=0.8, seed=42)
        second = split_composite_images(payload, train_fraction=0.8, seed=42)

        self.assertEqual(first, second)
        self.assertFalse(first.train_image_ids & first.val_image_ids)
        self.assertEqual(
            first.train_image_ids | first.val_image_ids,
            {image["id"] for image in images},
        )
        for group_image_ids in groups.values():
            self.assertTrue(
                group_image_ids <= first.train_image_ids
                or group_image_ids <= first.val_image_ids
            )
        self.assertLessEqual(
            abs(len(first.train_image_ids) - round(len(images) * 0.8)),
            max(len(group) for group in groups.values()),
        )

    def test_image_name_is_rewritten_relative_to_repository_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory).resolve()
            source_root = repository_root / ".local" / "recognition" / "dataset"
            source_root.mkdir(parents=True)

            result = _repository_relative_image_name(
                repository_root,
                source_root,
                "images/composite_000001.png",
            )

            self.assertEqual(
                result,
                ".local/recognition/dataset/images/composite_000001.png",
            )


if __name__ == "__main__":
    unittest.main()
