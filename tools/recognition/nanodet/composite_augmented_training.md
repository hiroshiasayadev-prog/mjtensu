# Composite-augmented NanoDet training

## Server layout

The training configuration assumes the canonical `/srv/bugrat/data-lv/mjtensu` layout:

```text
/srv/bugrat/data-lv/mjtensu/
  data/
  .local/recognition/
    nanodet_single_class_dataset/
    composite_capture_test_dataset/
    nanodet_composite_augmented_dataset/
    nanodet_runs/
  nanodet/
    experiment-configs/
    pretrained/nanodet-plus-m_320.pth
    nanodet/
      .venv/
      tools/train.py
```

The generated COCO image file names are relative to `/srv/bugrat/data-lv/mjtensu`, because base images live under `data/` and composite images live under `.local/recognition/composite_capture_test_dataset/`.

## Build the merged annotations

From the mjtensu repository root on Windows or Linux:

```powershell
py tools\recognition\build_nanodet_composite_augmented_dataset.py
```

Linux equivalent:

```bash
python tools/recognition/build_nanodet_composite_augmented_dataset.py
```

Defaults:

- composite split: 80% train / 20% validation
- split seed: 42
- split unit: `(source_annotation_json, source_image_id)`
- base train and base validation records remain in their existing partitions
- images are referenced in place and are not copied

Generated files:

```text
.local/recognition/nanodet_composite_augmented_dataset/
  annotations/
    instances_train.json
    instances_val.json
    instances_composite_train.json
    instances_composite_val.json
  provenance.json
```

`instances_composite_val.json` is intended for focused post-training evaluation of held-out composite layouts.

## Copy to the server from Windows PowerShell

Create the destination directories:

```powershell
ssh bugrat@192.168.11.22 "mkdir -p /srv/bugrat/data-lv/mjtensu/nanodet/experiment-configs /srv/bugrat/data-lv/mjtensu/.local/recognition"
```

Copy the training config:

```powershell
scp "C:\Users\imved\projects\mjtensu\tools\recognition\nanodet\configs\e1_nanodet_plus_m_320_composite_augmented_amp40.yml" `
  bugrat@192.168.11.22:/srv/bugrat/data-lv/mjtensu/nanodet/experiment-configs/
```

Copy the composite images and their source COCO metadata:

```powershell
scp -r "C:\Users\imved\projects\mjtensu\.local\recognition\composite_capture_test_dataset" `
  bugrat@192.168.11.22:/srv/bugrat/data-lv/mjtensu/.local/recognition/
```

Copy the generated merged train/validation annotations:

```powershell
scp -r "C:\Users\imved\projects\mjtensu\.local\recognition\nanodet_composite_augmented_dataset" `
  bugrat@192.168.11.22:/srv/bugrat/data-lv/mjtensu/.local/recognition/
```

The existing `data/`, `.local/recognition/nanodet_single_class_dataset/`, `nanodet/pretrained/`, and `nanodet/nanodet/` directories are also required.

## Train

```bash
cd /srv/bugrat/data-lv/mjtensu/nanodet/nanodet
.venv/bin/python tools/train.py \
  /srv/bugrat/data-lv/mjtensu/nanodet/experiment-configs/e1_nanodet_plus_m_320_composite_augmented_amp40.yml \
  --seed 42
```

Run output:

```text
/srv/bugrat/data-lv/mjtensu/.local/recognition/nanodet_runs/
  E1_plus_m_320_composite_augmented_amp40_seed42/
```

The run starts from the official NanoDet `nanodet-plus-m_320.pth`, uses AMP and batch size 96, trains for 40 epochs, validates every 5 epochs, and completes a 40-epoch cosine schedule.

## Local checks

```powershell
py -m unittest discover -s tools\recognition\tests -v
```
