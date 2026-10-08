# Blender synthetic NanoDet corpus

The canonical, Git-managed source for generating the fixed-layout 320×320 mjtensu Mahjong detector corpus lives in `tools/blender/synthetic/`. Generated images, COCO annotations, scene records, previews and temporary rendering files belong under the ignored `tools/blender/synthetic/.outputs/example-ceiling1p8m/`, **not** in Git. Historical delivery/snapshot scripts in this directory are retained as source history; **`generate.py` + `config.production.json` + `layout_xy_optimizer.py`** are the current production generator.

## Accepted generation requirements

- **Reproducibility:** `--seed` defaults to `760601`; each image index has deterministic coverage and retry streams. Runs with disjoint `--start-index` ranges should generate the same scene for that index when the code/configuration/Blender assets are unchanged.
- **Meld count 0–4:** every five consecutive indices contain each count **exactly once**; the order is shuffled deterministically from seed and five-image block number (`balanced_shuffled_blocks`). Do not rely on a random weighted sampler to achieve population balance.
- **Normal winning hand:** whenever `completed_hand` is nonempty, the visible completed tiles plus melds form a legal standard **4 sets + 1 pair** (each kan is one set and has four physical tiles). With `m` melds, the concealed completed hand contains `14 - 3m` tiles. A **completed_hand-empty case is not required to be winning**. Tile inventory limits and red-five limits apply across hand, melds and dora indicators.
- **Independently empty regions:** `completed_hand` and `dora_indicators` are independently empty with probability 20% each. A zero-meld case is also allowed. Nonempty dora indicators normally number 1–5 with weights `0.46/0.26/0.16/0.08/0.04`.
- **Meld variety:** chi / pon / open-kan / closed-kan. In a nonzero-meld image, a white-dragon meld is intentionally forced approximately 22% of the time.
- **Placement variation:** `neat:ordinary:messy = 2:6:2`. Do not collapse all examples to neat layouts.
- **Camera:** height 0.40m, preferred lens 17.7mm; if the fixed crop cannot contain all tile bboxes, retry lens values 16/14/12/10mm while retaining the same tile placement, camera height and roll. Clockwise image roll relative to table bottom is **uniformly 0–15°**, determined by seed. Aim near the completed hand when nonempty, while keeping capture geometry valid. Keep plausible oblique camera view and room lighting.
- **Physical frame contact:** if melds exist, **at least one bottommost meld tile physically touches the actual black bottom frame** and **at least one rightmost meld tile physically touches the actual black right frame**. Use the world-space black-frame mesh inner faces and rounded tile-body mesh outline, not the `INNER_X/Y` fake boundary. No penetration of the frame. The contacting tiles may be the same tile. Contact gaps must be within the generator's 1µm tolerance.
- **Completed-to-meld gap:** if both are present, the horizontal clearance between physical tile footprint extents is exactly **three tile widths = 81mm**, as enforced by the generator.
- **Crop:** port PWA `computeDisplayRegionRects` / `object-fit:cover` from `pwa_capture_dataset/src/layout.ts`. Keep the region source rectangles **fixed independently of tile positions**, using the reference viewport `750×334` on `640×360` source images; compose to `320×320`. Every projected tile bbox must fit inside its **own** source crop. Overlap into any **other** source region is permitted only when **strictly less than 10% of that tile's full projected bbox area**.
- **XY optimization:** `layout_xy_optimizer.py` translates camera XY and, if necessary, dora XY to meet the fixed crops; never adjust fixed crop boundaries, tile count, physical meld contact, camera height or roll to force a pass. Favor black-frame visibility in the meld crop. Failed attempts are rejected, not silently accepted.
- **Auditability:** record per-image seed, condition, winning-shape flag, actual black-frame contact distances and contacting tile ids, camera pose/rotation, optimizer offsets, tile identities, group/face, crop geometry, projected bboxes, lighting, and provenance. Validate before publishing any large corpus.

## Geometry

- Tile body 27×38×19mm; table felt 815×815mm; outer border approximately 960×960mm.
- World `-Y` is the seated player's near side; `+X` is their right.
- Right/bottom black-frame inner faces currently at world `X=+406mm` and `Y=-406mm`. They are queried from the *actual meshes* at generation time rather than assumed in the contact logic.
- Fixed PWA crop rectangles for the reference are approximately: dora `(55.4667,83.5733,348.16,81.92)`, completed `(55.4667,174.0267,348.16,81.92)`, melds `(412.16,83.5733,172.3733,172.3733)`.
- Composite destinations: completed `(7,0,306,72)`, dora `(7,74,306,72)`, melds `(74,148,172,172)`; elsewhere pure black.

## Generation

From repository root (Blender and CC0 tile assets currently staged in `.local/recognition/blender_synthetic_tooling/`):

```bash
.local/recognition/blender_synthetic_tooling/blender-4.2.3-linux-x64/blender -b \
  --python tools/blender/synthetic/generate.py -- \
  --config tools/blender/synthetic/config.production.json \
  --output tools/blender/synthetic/.outputs/example-ceiling1p8m \
  --count 10000 --seed 760601
```

For production, use the bounded parallel coordinator instead of manually starting multiple generator processes. The coordinator exclusively locks `run.lock`, allocates disjoint 125-image ranges, launches four short-lived Blender workers with two render threads each, resumes existing records/images, retries failed ranges, and only assembles and runs final QA after all indices exist:

```bash
python3 -u tools/blender/synthetic/parallel_generate.py \
  --output tools/blender/synthetic/.outputs/example-ceiling1p8m \
  --config tools/blender/synthetic/config.production.json \
  --count 10000 --seed 760601 --workers 4 --chunk-size 125 --render-threads 2
```

Logs and `parallel_status.json` live inside the ignored corpus output. The previous sequential `run-resume.sh` must not be started concurrently. On restart run the same coordinator command: complete image indices are skipped. The single-process `--count 10000` example above is retained as a fallback, not the preferred production workflow.

## QA

Blender Python includes NumPy and Pillow:

```bash
PY=.local/recognition/blender_synthetic_tooling/blender-4.2.3-linux-x64/4.2/python/bin/python3.11
$PY tools/blender/synthetic/validate_corpus.py tools/blender/synthetic/.outputs/example-ceiling1p8m --minimum-images 10000
$PY tools/blender/synthetic/qa.py --root tools/blender/synthetic/.outputs/example-ceiling1p8m --min-images 10000
```

The output is single-class COCO (`mahjong_tile`) with a per-image scene record and a provenance file. Do not interpret a successful annotation/bbox count as sufficient proof of a physically plausible frame contact: inspect sample full views and compare the recorded actual-frame contact values.

## Lighting

Lighting uses two broad AREA light sources with a soft shadow gradient and restrained exposure. The production height is **1.8m above the tabletop**, approximating a household ceiling at ~2.5m above the floor with a ~0.7m table; the light energy multiplier is **1.0**. A height-only comparison of 1.8m and 2.5m showed that higher placement reduces the washed-out appearance at constant power, whereas multiplying power to compensate restores it. All test conditions are retained under `.outputs/height-*` with their exact `condition.json`; comparison sheets are `.outputs/height-lighting-comparison-index000014.png` and `index000015.png`. Warm/neutral/cool and dim/normal/bright variations remain; the former hard-shadow and artificial occluder profiles are disabled because they created unrealistically dark seams between pale tile bodies. The existing `.outputs/final-delivery-v3` retains the historical strong-shadow images for review and must not be mixed into a new lighting trial. Its previous validation was **FAIL** because the minimum source-crop bbox retention was 0.999653, short of the required complete containment. The current generator rejects these tiny source-crop overflows rather than accepting a 0.01px tolerance. The 8-image preview for the revised lighting is `.outputs/lighting-diffuse-v1/`, with a side-by-side reference image at `compare_legacy_diffuse.png`. This preview does **not** constitute a fully validated production corpus.

## Verified delivery (2026-10-08)

`tools/blender/synthetic/.outputs/final-delivery-v4-ceiling1p8m/` contains the completed 10,000-image corpus. `parallel_validate.log` and `parallel_qa.log` both report **PASS**: 146,785 COCO annotations; 9,500 train / 500 validation images; 0 invalid/out-of-bounds boxes, region intrusions, physical overlaps, or black-padding violations. Generated images and logs remain ignored by Git and are not part of the source commit.
