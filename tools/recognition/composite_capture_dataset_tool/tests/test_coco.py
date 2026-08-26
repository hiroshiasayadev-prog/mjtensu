from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.recognition.composite_capture_dataset_tool.coco import CocoDataset


class CocoDatasetTest(unittest.TestCase):
    def test_loads_and_resolves_nested_posix_file_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_root = root / "data"
            image_path = image_root / "source" / "train" / "image.png"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (20, 10)).save(image_path)
            annotation_path = root / "instances.json"
            annotation_path.write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "id": 1,
                                "file_name": "source/train/image.png",
                                "width": 20,
                                "height": 10,
                            }
                        ],
                        "annotations": [
                            {
                                "id": 3,
                                "image_id": 1,
                                "category_id": 7,
                                "bbox": [1, 2, 3, 4],
                            }
                        ],
                        "categories": [{"id": 7, "name": "tile"}],
                    }
                ),
                encoding="utf-8",
            )
            dataset = CocoDataset.load(annotation_path, image_root)
            self.assertEqual(dataset.resolve_image_path(dataset.image_at(0)), image_path)
            self.assertEqual(len(dataset.annotations_for_image(1)), 1)

    def test_filters_to_japanese_v2_20250_images_and_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_root = root / "data"
            annotation_path = root / "instances.json"
            records = [
                (1, "coco_mahjong_jp_v2/train/20250722_203605.jpg"),
                (2, "coco_mahjong_jp_v2/train/not_target.jpg"),
                (3, "coco_mahjong/train2017/20250722_203605.jpg"),
            ]
            for _image_id, file_name in records:
                image_path = image_root.joinpath(*file_name.split("/"))
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (20, 10)).save(image_path)
            annotation_path.write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "id": image_id,
                                "file_name": file_name,
                                "width": 20,
                                "height": 10,
                            }
                            for image_id, file_name in records
                        ],
                        "annotations": [
                            {
                                "id": image_id,
                                "image_id": image_id,
                                "category_id": 1,
                                "bbox": [1, 2, 3, 4],
                            }
                            for image_id, _file_name in records
                        ],
                        "categories": [{"id": 1, "name": "tile"}],
                    }
                ),
                encoding="utf-8",
            )

            dataset = CocoDataset.load(
                annotation_path,
                image_root,
                image_path_prefix="coco_mahjong_jp_v2/train/",
                image_name_pattern="20250*.jpg",
            )

            self.assertEqual([image["id"] for image in dataset.images], [1])
            self.assertEqual(len(dataset.annotations_for_image(1)), 1)
            self.assertEqual(dataset.annotation_count, 1)
            self.assertEqual(dataset.source_image_count, 3)
            self.assertEqual(dataset.source_annotation_count, 3)

    def test_rejects_unsafe_file_name_during_resolution(self) -> None:
        dataset = CocoDataset(
            annotation_path=Path("annotations.json"),
            image_root=Path("data"),
            images=[{"id": 1, "file_name": "../secret.png", "width": 1, "height": 1}],
            annotations_by_image_id={},
            categories=[],
            annotation_count=0,
        )
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            dataset.resolve_image_path(dataset.image_at(0))


if __name__ == "__main__":
    unittest.main()
