# Tile catalog capture PWA

既存の`pwa_capture_dataset`とは独立した、牌カタログ撮影専用PWA。

## Responsibility split

スマホ側:

- 37牌を1枚の写真として撮影する
- NanoDetをライブ実行し、黄色いbboxを目視確認用に表示する
- bbox数は参考表示だけで、37件一致を保存条件にしない
- 行分割、牌ラベル確定、crop分割はしない
- 保存するdetector候補は空。スマホ推論結果はannotation初期値に使わない

PC側:

- アップロード後の320画像へNanoDetを再実行する
- その結果をannotation toolの初期bboxとして使う
- 漏れ、重複、位置を手動修正する
- 上から4行を`10 / 10 / 10 / 7`として、各行を左から右へ牌ラベルへ割り当てる

牌順:

```text
萬子: 1m 2m 3m 4m 5m 6m 7m 8m 9m 赤5m
筒子: 1p 2p 3p 4p 5p 6p 7p 8p 9p 赤5p
索子: 1s 2s 3s 4s 5s 6s 7s 8s 9s 赤5s
字牌: 東 南 西 北 白 發 中
```

撮影条件:

1. 暖色・通常・影なし・正面
2. 暖色・暗め・影なし・正面
3. 暖色・通常・部分影・正面
4. 暖色・通常・影なし・少し斜め

牌は4枚の撮影中ずっと並べたままでよい。

## Run

APIを再起動する。

```powershell
cd C:\Users\imved\projects\mjtensu
python -m tools.recognition.capture_dataset_api.server
```

新PWAを起動する。

```powershell
cd C:\Users\imved\projects\mjtensu\tools\recognition\pwa_tile_catalog_capture
npm install
npm run dev
```

terminalに表示されたHTTPS URLをiPhoneから開く。

## PC detector candidates

4枚の撮影後、repository rootで実行する。

```powershell
.\.venv\Scripts\python.exe `
  .\tools\recognition\nanodet\refresh_unannotated_capture_detections.py `
  --campaign-id tile-catalog-warm-4-v2 `
  --confidence-threshold 0.35
```

`tile-catalog` campaignでは`tile_catalog_layout.v2.json`が自動選択される。

## Annotation

```powershell
cd C:\Users\imved\projects\mjtensu\tools\recognition\annotation_tool
npm run dev
```

campaign selectorで`tile-catalog-warm-4-v2`を選ぶ。原画全体が`全牌`regionとして表示され、PC detector候補から開始する。

## Build

```powershell
npm run build
```
