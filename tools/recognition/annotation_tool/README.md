# Capture tile annotation tool

保存済みcaptureのsemantic-region cropへ、任意角度の牌矩形annotationを付けるPC用Web UI。

## Features

- 保存時のNanoDet bboxから自動初期化。
- 期待牌数に基づく等間隔の仮矩形生成。
- 矩形の移動、resize、追加、削除。
- dragによる連続角度の回転。Shiftを押しながら回転すると5度単位でsnapする。
- 選択矩形の左右分割・上下分割。
- 座標順でtaskの期待牌を自動対応し、bbox上へ表示。
- 手牌は左から右、ドラは上段から下段かつ行内左から右、副露は上から下かつgroup内左から右で対応する。
- 暗槓は`face == front`の中央2牌だけをannotation対象とする。
- group別の期待数、実数、regionからのはみ出しを検証する。
- 条件を満たすまで「保存して次へ」を無効化する。
- 編集中の内容をSQLiteへ`draft`として自動保存する。

annotation正本は次のrotated rectangleで保存する。

```json
{
  "id": "uuid",
  "centerX": 120.5,
  "centerY": 84.0,
  "width": 31.2,
  "height": 47.8,
  "angleDeg": -8.4
}
```

NanoDet再学習用には外接AABBへ変換でき、牌分類用には角度補正cropへ利用できる。

## Run

repository rootでcapture APIを起動する。

```powershell
python -m tools.recognition.capture_dataset_api.server
```

別PowerShellでannotation UIを起動する。

```powershell
cd tools\recognition\annotation_tool
npm install
npm run dev
```

PCのbrowserで次を開く。

```text
http://127.0.0.1:5174/
```

既存annotationまたはrectifierのAI候補を回転矩形として見直す場合はOBB review modeを使う。
rectifier候補は`AI未レビュー`、明示的に「OBB保存して次へ」を押したcaptureは`レビュー済`として表示する。
一覧をクリックして移動しただけではレビュー済みへ昇格せず、編集内容はAI未レビューのままdraft保存される。
既存矩形を初期値として、四隅dragまたは幅/高さの数値入力でサイズを詰め、回転handleまたは角度入力で傾きを合わせる。

```text
http://127.0.0.1:5174/?mode=obb-review
```

production build:

```powershell
npm run build
npm run preview
```

## Persistence

既存の次のSQLiteへ保存する。

```text
.local/recognition/capture_dataset/dataset.sqlite
```

追加table:

```text
capture_annotation
detection_refresh
```

保存状態は`draft`または`complete`。OBB review provenanceはannotation JSONの`review.state`に
`model_suggested`または`human_reviewed`として保持する。画像ファイル自体は変更しない。

rectifierで未レビューcaptureへAI候補を投入する前に、まずdry-runする。

```powershell
python tools\recognition\apply_rotated_box_rectifier_suggestions.py --dry-run
```

対象件数を確認後、実際にSQLiteへ反映する。

```powershell
python tools\recognition\apply_rotated_box_rectifier_suggestions.py
```

反映前DBは`.local/recognition/capture_dataset/dataset.pre-rectifier-suggestions.sqlite`へ一度だけbackupする。
pre-OBB backupとのgeometry差分があるcaptureは人手レビュー済みとして保護し、AI候補では上書きしない。

## Refresh detector candidates

fine-tune済みONNXで、annotationがまだ存在しないcaptureだけを再推論できる。
既存の`draft`と`complete`は対象外で、書換え直前にもannotation有無を再確認する。
SQLiteは更新前に`capture_dataset/backups/`へbackupされる。

配置21以降を更新する例:

```powershell
python tools\recognition\nanodet\refresh_unannotated_capture_detections.py `
  --from-layout 21 `
  --include-drafts `
  --confidence-threshold 0.35
```

更新後はannotation UIを再読み込みする。API serverの再起動は不要。
`draft`の保存済み矩形自体は維持されるため、そのcaptureで新候補へ戻したい場合だけ
「検出結果で再設定」を押す。
