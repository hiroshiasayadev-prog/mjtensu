# Capture dataset local API

Python standard libraryだけで動くWindows local API。次のcampaignをSQLiteへseedし、PWAから受け取った画像とmetadataを保存する。

- `initial-120`: 本番3領域の30配置×4環境
- `tile-catalog-warm-4-v2`: 全37牌を原画1枚で撮る暖色照明4条件。スマホNanoDetは目視確認用のみ

## Run

repository rootで実行する。

```powershell
python -m tools.recognition.capture_dataset_api.server
```

既定値:

```text
listen: 127.0.0.1:8787
storage: .local/recognition/capture_dataset
```

起動時に次も生成する。

```text
campaign-initial-120.json
campaign-tile-catalog-warm-4-v2.json
```

SQLiteの`capture_expected_tile_slot` viewから、capture IDに対するregion・row/group ordinal・tile ordinal・tile codeを直接取得できる。

変更例:

```powershell
python -m tools.recognition.capture_dataset_api.server `
  --host 127.0.0.1 `
  --port 8787 `
  --storage-root .local\recognition\capture_dataset
```

## Endpoints

```text
GET    /api/health
GET    /api/campaigns/<campaign-id>/overview
GET    /api/campaigns/<campaign-id>/next-task
POST   /api/captures
DELETE /api/campaigns/<campaign-id>/last-capture

GET /api/annotation-campaigns
GET /api/annotations/captures?campaignId=initial-120
GET /api/annotations/captures/<capture-id>
GET /api/annotation-asset?path=<storage-relative-image-path>
PUT /api/annotations/captures/<capture-id>
```

`POST /api/captures`は次のmultipart fieldsを受け取る。

```text
manifest       required application/json
original       required image/jpeg
composite      required image/png
hand_crop      optional image/png
dora_crop      optional image/png
meld_crop      optional image/png
```

`uploadClientId`でidempotentに再送できる。異なるupload IDで同じtaskを二重保存する要求は`409 Conflict`になる。原画JPEGの実寸はmanifestと一致し、composite PNGは`320 x 320`でなければ保存しない。

annotation APIは保存済みcaptureとregion cropを列挙し、次の任意角度矩形を`capture_annotation` tableへ保存する。

```json
{
  "status": "draft",
  "document": {
    "schemaVersion": 1,
    "captureId": "cap_xxx",
    "boxes": {
      "completed_hand": [
        {
          "id": "uuid",
          "centerX": 120.5,
          "centerY": 70.0,
          "width": 31.2,
          "height": 47.8,
          "angleDeg": -8.4
        }
      ],
      "dora_indicators": [],
      "melds": []
    }
  }
}
```

`complete`保存時は、表向き牌だけの期待数、ドラ行、副露groupごとの枚数、rotated rectangleのregion内包を検証する。暗槓の裏向き2牌は期待数へ含めない。

## Test

```powershell
python -m unittest discover -s tools\recognition\capture_dataset_api\tests -v
```
