# Rotated detector bootstrap

少数のhuman-reviewed OBBから、Mahjong tile用の小型rotated detectorをbootstrapする。

## 0. Smoke tests

```powershell
python -m pytest `
  tools/recognition/tests/test_build_rotated_detector_corpus.py `
  tools/recognition/tests/test_build_rotated_detector_augmented_dataset.py `
  tools/recognition/tests/test_rotated_fcos_nano.py
```

## 1. Human OBB corpus

`dataset.pre-obb-review.sqlite` と現在の `dataset.sqlite` を比較し、OBB review後にgeometryが変わったcaptureだけを抽出する。

```powershell
python tools/recognition/build_rotated_detector_corpus.py `
  --include-drafts `
  --overwrite
```

出力:

```text
.local/recognition/rotated_detector_corpus/annotations/human_all.json
.local/recognition/rotated_detector_corpus/annotations/train.json
.local/recognition/rotated_detector_corpus/annotations/val.json
.local/recognition/rotated_detector_corpus/provenance.json
```

train/valは原則 `campaign_id + layout_id` 単位で分離する。review済みlayoutが1個しかない場合だけ、実験用に `--allow-capture-split` を明示する。

## 2. OBB-preserving rotation augmentation

320x320 compositeのsemantic regionは固定し、各region内の画像とOBBを同じscale/rotation/translationで変換する。GTの四隅がregion外へ切れる変換は採用しない。

```powershell
python tools/recognition/build_rotated_detector_augmented_dataset.py `
  --copies-per-image 12 `
  --max-rotation-deg 45 `
  --max-shrink-fraction 0.20 `
  --overwrite
```

出力:

```text
.local/recognition/rotated_detector_augmented_dataset/annotations/train.json
.local/recognition/rotated_detector_augmented_dataset/preflight/contact_sheet.jpg
.local/recognition/rotated_detector_augmented_dataset/provenance.json
```

`contact_sheet.jpg` は学習前に必ず目視する。

## 3. Train

デフォルト構成:

```text
backbone: ShuffleNetV2 0.5x (ImageNet pretrained)
FPN: 64 ch
head: 2 x depthwise-separable conv
levels: stride 8 / 16 / 32
class: objectness 1 class
regression: dx, dy, log(w), log(h), sin(2theta), cos(2theta)
input: 320x320 RGB
```

```powershell
python tools/recognition/train_rotated_fcos_nano.py `
  --epochs 80 `
  --batch-size 32
```

出力:

```text
.local/recognition/rotated_fcos_runs/rfcos_nano_s05_f64_seed42/model_best.pt
.local/recognition/rotated_fcos_runs/rfcos_nano_s05_f64_seed42/model_best.onnx
.local/recognition/rotated_fcos_runs/rfcos_nano_s05_f64_seed42/model_best_metrics.json
.local/recognition/rotated_fcos_runs/rfcos_nano_s05_f64_seed42/history.jsonl
```

ImageNet weightsを取得できないhostでは:

```powershell
python tools/recognition/train_rotated_fcos_nano.py --no-pretrained-backbone
```

## First decision point

最初はarchitecture tuningをしない。まずx0.5/64chでhuman valに対して以下を見る。

- recall
- precision
- rotated IoU
- angle error
- 実際のcropで背景が減っているか

ここで勝てることを確認してから、pseudo-label生成と64ch -> 32ch等の軽量化へ進む。
