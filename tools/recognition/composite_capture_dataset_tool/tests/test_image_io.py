from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.recognition.composite_capture_dataset_tool.image_io import (
    EXIF_ORIENTATION_TAG,
    load_coco_image,
)


class LoadCocoImageTest(unittest.TestCase):
    def test_applies_exif_orientation_before_display_and_composition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "oriented.jpg"
            source = Image.new("RGB", (40, 20), "red")
            for x in range(20, 40):
                for y in range(20):
                    source.putpixel((x, y), (0, 0, 255))
            exif = Image.Exif()
            exif[EXIF_ORIENTATION_TAG] = 6  # 90 degrees clockwise
            source.save(image_path, quality=100, subsampling=0, exif=exif)

            loaded = load_coco_image(image_path)

            self.assertEqual(loaded.raw_size, (40, 20))
            self.assertEqual(loaded.oriented_size, (20, 40))
            self.assertEqual(loaded.exif_orientation, 6)
            self.assertTrue(loaded.exif_transpose_applied)
            top_pixel = loaded.image.getpixel((10, 5))
            bottom_pixel = loaded.image.getpixel((10, 35))
            self.assertGreater(top_pixel[0], top_pixel[2])
            self.assertGreater(bottom_pixel[2], bottom_pixel[0])

    def test_unoriented_image_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "plain.png"
            Image.new("RGB", (20, 10), "white").save(image_path)

            loaded = load_coco_image(image_path)

            self.assertEqual(loaded.raw_size, (20, 10))
            self.assertEqual(loaded.oriented_size, (20, 10))
            self.assertIsNone(loaded.exif_orientation)
            self.assertFalse(loaded.exif_transpose_applied)


if __name__ == "__main__":
    unittest.main()
