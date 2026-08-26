# Mjtensu capture dataset PWA

ADR-002の3領域を使い、実機配置画像・deployment composite・NanoDet検出結果を収集するPWA。

## Startup

repository rootからlocal APIを起動する。

```powershell
python -m tools.recognition.capture_dataset_api.server
```

別terminalでPWAを起動する。

```powershell
cd tools\recognition\pwa_capture_dataset
npm install
npm run dev
```

install/offline cacheまで確認する場合はproduction buildを配信する。

```powershell
npm run build
npm run preview -- --host 0.0.0.0 --port 4173
```

`predev`で次を自動的に`public/`へ配置する。

- `.local/recognition/nanodet_runs/E1_plus_m_320_composite_augmented_amp40_seed42/model_best/nanodet-plus-m-320-composite-augmented.onnx`
- `tools/recognition/capture_layout.v1.json`
- model SHA-256 metadata
- PWA icon

model pathを差し替える場合:

```powershell
$env:MJTENSU_CAPTURE_MODEL = "C:\path\to\model.onnx"
npm run dev
```

同じWi-FiのiPhoneから次を開く。

```text
https://<Windows IPv4>:5173/?provider=webgl
```

production previewの場合:

```text
https://<Windows IPv4>:4173/?provider=webgl
```

provider候補:

```text
?provider=webgl
?provider=wasm-simd
?provider=wasm-threaded
```

## Storage

既定保存先:

```text
.local/recognition/capture_dataset/
├─ dataset.sqlite
├─ originals/YYYY/MM/DD/
├─ composites/YYYY/MM/DD/
└─ regions/
   ├─ hand/YYYY/MM/DD/
   ├─ dora/YYYY/MM/DD/
   └─ meld/YYYY/MM/DD/
```

原画は高品質JPEG、deployment compositeと各semantic-region cropはlossless PNGで保存する。source-pixel座標だけでなく、撮影時viewport・`object-fit: cover` geometry・display座標もmanifestへ記録する。

送信前にcapture一式をIndexedDBへ保存する。API送信に失敗したcaptureは、instruction画面の「未送信を再送」から再送できる。

## Commands

```powershell
npm run build
python -m unittest discover -s tools\recognition\capture_dataset_api\tests -v
```
