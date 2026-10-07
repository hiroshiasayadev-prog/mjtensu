# Blender synthetic NanoDet corpus

Headless Blender generator for the fixed-layout 320x320 mjtensu Mahjong tile detector input.

## Geometry

- Tile body: 27 x 38 x 19 mm.
- Automatic-table felt field: 815 x 815 mm; outer table target: 960 x 960 mm.
- World `-Y` is the seated player's side; `+X` is the player's right.
- Melds are anchored at the player's lower-right and accumulate upward.
- Completed hand and dora are placed relative to that player sector.
- The render camera frames only the player's near-side working area; it does not frame the whole table.

The source-video crop rectangles are not inferred from generated tile positions. `generate.py` ports the guide geometry from `pwa_capture_dataset/src/layout.ts` (`computeDisplayRegionRects`, object-fit cover behavior) into render pixels. The resulting fixed 640x360 source rectangles for the production reference viewport are approximately:

- dora: `(55.4667, 83.5733, 348.16, 81.92)`
- completed hand: `(55.4667, 174.0267, 348.16, 81.92)`
- melds: `(412.16, 83.5733, 172.3733, 172.3733)`

These are then mapped to the canonical destinations from `capture_layout.v1.json`. Pixels outside those destinations are exact black.

## Lighting

Lighting is room-like and intentionally avoids unreadable glare. The `partial` profile uses an off-frame raised bar on the player's/camera side to cast a soft shadow into the capture area; the blocker itself is kept outside the photographed area.

## Generate

```bash
.local/recognition/blender_synthetic_tooling/blender-4.2.3-linux-x64/blender -b \
  --python tools/recognition/blender_synthetic_nanodet/generate.py -- \
  --config tools/recognition/blender_synthetic_nanodet/config.production.json \
  --output .local/recognition/blender_synthetic_nanodet/final \
  --count 10000 --seed 760701
```

For parallel generation use disjoint `--start-index` ranges plus `--skip-assemble`, then one final `--assemble-only` run.

## Validate / QA

Run with Blender's bundled Python (Pillow and NumPy are installed there):

```bash
PY=.local/recognition/blender_synthetic_tooling/blender-4.2.3-linux-x64/4.2/python/bin/python3.11
$PY tools/recognition/blender_synthetic_nanodet/validate_corpus.py \
  .local/recognition/blender_synthetic_nanodet/final --minimum-images 10000
$PY tools/recognition/blender_synthetic_nanodet/qa.py \
  --root .local/recognition/blender_synthetic_nanodet/final --min-images 10000
```

The corpus is COCO single-class (`mahjong_tile`) and each image also has a sidecar scene record with tile identity, face, meld group/type, exact projected bbox, camera, lighting, material, and fixed source crop metadata.
