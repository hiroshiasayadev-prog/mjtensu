# Composite capture COCO dataset tool

This Tkinter tool creates manual 320×320 test inputs for the fixed capture layout in `PRODUCT-ADR-RECOGNITION-002`.

## Layout

| Region | Destination | Aspect |
|---|---:|---:|
| Completed hand | `(x=7, y=0, w=306, h=72)` | `17:4` |
| Dora indicators | `(x=7, y=74, w=306, h=72)` | `17:4` |
| Melds | `(x=74, y=148, w=172, h=172)` | `1:1` |

All other pixels are black. Disabled regions remain black. Each selected crop is optionally rotated clockwise by 0°, 90°, 180°, or 270°, then uniformly resized and placed in its fixed destination.

JPEG EXIF Orientation is applied before source annotations are displayed or transformed. This keeps camera images in the same visual coordinate system normally used when their COCO boxes were authored.

## Run

From the repository root:

```powershell
py -m tools.recognition.composite_capture_dataset_tool
```

The explicit default paths and source-image filters are:

```text
--annotations .local\recognition\nanodet_single_class_dataset\annotations\instances_train.json
--image-root data
--output-directory .local\recognition\composite_capture_test_dataset
--image-path-prefix coco_mahjong_jp_v2/train/
--image-name-pattern img_00*.jpg
--min-retained-area-ratio 0.6
```

The GUI therefore opens only Japanese v2 training images whose basename matches `img_00*.jpg`. Annotations belonging to excluded images are not exposed to the GUI or output composer.

Override them as needed:

```powershell
py -m tools.recognition.composite_capture_dataset_tool `
  --annotations .local\recognition\nanodet_single_class_dataset\annotations\instances_val.json `
  --image-root data `
  --output-directory .local\recognition\composite_capture_test_dataset `
  --image-path-prefix coco_mahjong_jp_v2/valid/ `
  --image-name-pattern img_00*.jpg
```

Pillow is the only non-standard dependency:

```powershell
py -m pip install Pillow
```

## Controls

1. Select the active region with its radio button.
2. Set clockwise rotation. Changing rotation clears that region's existing crop because the required source aspect may change.
3. Drag on the source image. The rectangle is constrained to an exact integer multiple of the required aspect ratio.
4. Enable or disable any combination with the checkboxes.
5. `Random preset crops` enables all three regions, selects two distinct hand/dora presets and one meld preset, and resets their rotations to 0°.
6. Preview or save. Saving does not clear the selections, so multiple composites can be written from the same source image.

The default annotation policy is `center`: a tile bbox is considered when its center is inside the crop. A considered bbox is retained only when the crop contains **more than 60%** of the original source bbox area. A bbox with exactly 60% or less remaining is excluded. Retained bboxes are clipped to the source image and crop before rotation and scaling. Alternative selection policies are available through `--annotation-selection-policy contained` and `--annotation-selection-policy intersect`, and the area threshold can be changed with `--min-retained-area-ratio`.

Already-saved composites retain enough provenance to rebuild their annotations without regenerating the PNG files. Apply the current 60% rule and exit without opening Tkinter:

```powershell
py -m tools.recognition.composite_capture_dataset_tool --repair-existing-only
```

Before `instances.json` is changed, the tool copies it into `annotations/backups/`. Normal GUI startup also performs the same rebuild automatically before accepting new saves.

## Output

```text
<output-directory>/
  images/
    composite_000001.png
    ...
  annotations/
    instances.json
```

The output COCO dataset has one category:

```json
{
  "id": 1,
  "name": "mahjong_tile",
  "supercategory": "mahjong_tile"
}
```

Image records include source-image and capture-region provenance plus `min_retained_area_ratio`. Annotation records include `source_annotation_id` and `capture_region`. Source segmentation fields are intentionally not copied because this tool transforms bounding boxes only.

## Validate inputs and tests

Validate path resolution and the first image without opening Tkinter:

```powershell
py -m tools.recognition.composite_capture_dataset_tool --check-inputs
```

Run the unit tests:

```powershell
py -m unittest discover -s tools/recognition/composite_capture_dataset_tool/tests -v
```
