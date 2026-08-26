from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


EXIF_ORIENTATION_TAG = 274


@dataclass(frozen=True)
class LoadedCocoImage:
    image: Image.Image
    raw_size: tuple[int, int]
    oriented_size: tuple[int, int]
    exif_orientation: int | None
    exif_transpose_applied: bool


def load_coco_image(image_path: Path) -> LoadedCocoImage:
    """Load a source image in the visual orientation used by annotation tools.

    JPEG files from cameras may retain an EXIF Orientation tag while their pixel
    matrix remains unrotated. COCO annotations are generally authored against the
    visually oriented image, so Pillow's EXIF transpose must be applied before the
    image is displayed, cropped, or composed.
    """

    with Image.open(image_path) as source:
        source.load()
        raw_size = source.size
        orientation_value = source.getexif().get(EXIF_ORIENTATION_TAG)
        exif_orientation = (
            int(orientation_value) if orientation_value is not None else None
        )
        oriented = ImageOps.exif_transpose(source)
        oriented.load()
        image = oriented.convert("RGB")

    return LoadedCocoImage(
        image=image,
        raw_size=raw_size,
        oriented_size=image.size,
        exif_orientation=exif_orientation,
        exif_transpose_applied=(
            exif_orientation not in (None, 1)
        ),
    )
