# MLDB Agent Operating Manual

このファイルは `mjtensu` で ML 実験を回す AI agent 向けの運用手順書である。
MLDB 自体の完成度を上げるための設計資料ではない。

## 0. 最上位目的

**MLDB は ML を回すための道具であり、MLDB を作ること自体は目的ではない。**

ユーザーから ML の比較・探索・検証を依頼されたら、成功条件は次である。

1. 実験条件を定義する。
2. 実 GPU で Training / Evaluation を実行する。
3. canonical な結果を MLDB に残す。
4. 結果を比較してユーザーへ返す。
5. 必要なら次の実験を提案・実行する。

MLDB の API、抽象化、transport、daemon、ADR、spec、UI を増やすことは成功条件ではない。
既存機能で実験が回るなら、それを使う。

新しい MLDB 基盤変更を提案する前に必ず問うこと:

> この変更がないと、今ユーザーが求めている ML 実験は本当に回せないか？

答えが No なら、その変更はしない。

## 1. 現在の実行構成

canonical state と Controller は開発 PC 上の repository に置く。
GPU server は計算専用 Worker として扱う。

```text
Windows development PC
  C:\Users\imved\projects\mjtensu
  ├─ mldb_data/                 canonical definitions / Runs / Models
  ├─ .local/mldb/queue.sqlite  operational Queue
  └─ tools/mldb/               SSH execution bridge
          |
          | SSH / SCP, remote domain execution
          v
GPU server 192.168.11.22
  ~/.cache/mjtensu-mldb-worker/
  ├─ cache/assets/             content-addressed immutable asset cache
  └─ attempt-*/                per-attempt remote work area
```

現在の GPU Python は:

```text
/srv/bugrat/data-lv/mjtensu/nanodet/nanodet/.venv/bin/python
```

この Python には PyTorch/CUDA が入っている。system `python3` を勝手に使わない。

## 2. canonical truth と Queue の扱い

`mldb_data/` にある canonical entity / Run / Model が正である。
`.local/mldb/queue.sqlite` は operational projection であり、canonical DB ではない。

絶対にやらないこと:

- Queue の状態を正として canonical Run を書き換える。
- `queue.sqlite` を直接 `UPDATE` して lifecycle を直す。
- completed Training を Queue loss のために再学習する。
- Run ID / Model ID / StudyRun ID を文字列や時刻から推測して生成する。
- `mldb_data/training_runs/`, `evaluation_runs/`, `study_runs/`, `models/` を手編集する。
- canonical weights を手で差し替える。

Git ignore されている runtime entity は disposable という意味ではない。
Git に commit しないだけで、local MLDB における canonical experiment history である。
勝手に削除しない。

Queue が壊れた・消えた場合は reconciliation / recovery を使う。
completed canonical Training Run と Model が存在するなら、それを再利用する。

## 3. Git に入れるもの / 入れないもの

原則として reusable definition と実装コードは Git 管理する。
materialized data と実行履歴は Git 管理しない。

Git 管理対象:

```text
mldb_data/tasks/*.yaml
mldb_data/corpora/*.yaml
mldb_data/corpora/*.py
mldb_data/architectures/*.yaml
mldb_data/architectures/*.py
mldb_data/train_protocols/*.yaml
mldb_data/train_protocols/*.py
mldb_data/evaluation_protocols/*.yaml
mldb_data/evaluation_protocols/*.py
mldb_data/studies/*.yaml
tools/mldb/*.py
```

Git 管理しない:

```text
mldb_data/corpora/*.sqlite*
mldb_data/training_runs/
mldb_data/evaluation_runs/
mldb_data/study_runs/
mldb_data/models/
.local/mldb/
```

大きな Corpus や weights を `git add -f` しない。

## 4. 現在動作確認済みの Golden Path

牌 shape classifier の compliant vertical slice は 2026-09-09 に実 GPU で完走済み。
通常の実行入口は model family に依存しないこれだけ。

```powershell
.\.venv\Scripts\python.exe tools\mldb\run_study_ssh.py `
  --host 192.168.11.22 `
  --study-id tile-plain-full-feature-smoke-v3
```

Study runner は bootstrap や definition 作成を副作用として行わない。
Corpus SQLite が未 materialize の fresh clone だけ、事前に source DB から materialize する。
その場合は次の source DB が必要。

```text
.local/recognition/tile_classifier_datasets/gray35_jp500_seed42_v3_jp189.sqlite
```

明示的に bootstrap したい場合:

```powershell
.\.venv\Scripts\python.exe tools\mldb\bootstrap_tile_shape.py
```

正常終了では Training と Evaluation の attempt が順に完了し、最後に:

```text
study_run=<id> status=completed
```

となることを確認する。
動作確認済みの historical example:

```text
Classifier:
Study Run:      sr-20260909-001  completed
Training Run:   tr-20260909-001  completed
Model:          mdl-20260909-001
Evaluation Run: ev-20260909-001  completed

Rotated detector:
Study Run:      sr-20260909-002  completed
Training Run:   tr-20260909-002  completed
Model:          mdl-20260909-002
Evaluation Run: ev-20260909-002  completed
```

これは smoke の証跡であり、ID をコードへ hard-code するためのものではない。
1 epoch の smoke なので精度値も benchmark として扱わない。

### Remote asset cache

SSH worker は immutable asset の SHA-256 を cache key にする。
同じ Corpus / implementation / weights を毎 attempt SCP してはいけない。

```text
~/.cache/mjtensu-mldb-worker/cache/assets/<sha-prefix>/<sha256>
```

cache hit なら attempt directory へ hardlink し、転送しない。
同じ 121 MB Corpus を parameter sweep の各 trial ごとに再転送しない。
asset 内容が変われば SHA-256 が変わるため、自動的に別 cache entry になる。

通常のログでは初回だけ `cache miss`、以後は `cache hit` になることを期待する。

## 5. ユーザーから実験依頼を受けた時の手順

例:

> ShuffleNet の stem を 3 条件、rotation augmentation あり/なしで比較して。

AI agent は次の順で進める。

1. 比較要因、levels、seed、評価指標を整理する。
2. repo を実際に検索し、既存 Task / Corpus / Protocol / Architecture を確認する。
3. 再利用できるものは再利用する。
4. 必要な Architecture または Protocol 実装だけ追加する。
5. Study に matrix を記述する。
6. resolver / validation で実行可能性を確認する。
7. SSH Study launcher で実 GPU 実行する。
8. canonical Evaluation Run の metrics を読む。
9. 条件別比較をユーザーへ返す。
10. 結果から次の experiment が明確なら提案する。

**計画だけ作って終了しない。**
ユーザーが review / design only と明示していない限り、実行可能な依頼は実験を回すところまで進める。

また、repo state に関する判断は推測せず Desktop Commander / filesystem で現物を確認する。

## 6. 何を新しく作るべきか

変更内容に対して最小の entity だけ追加する。

- **Task**: 予測対象・label semantics・problem definition が変わる時だけ。
- **Corpus**: dataset bytes、sample、split、representation が変わる時。
- **Architecture**: model topology / forward / model construction が変わる時。
- **Train Protocol**: optimizer、schedule、augmentation、training procedure の意味が変わる時。
- **Evaluation Protocol**: 評価 procedure、metric、artifact contract が変わる時。
- **Study**: 既存要素の組合せ、parameter values、seeds を比較する時。

単なる hyperparameter 値違いで Architecture や Train Protocol を複製しない。
Study の parameter matrix を使う。

例:

```text
rotation_augment_deg = [0, 15, 30]
architecture = [plain, mobile-f8]
seed = [42]
```

なら、基本は 2 Architecture + 1 Train Protocol + 1 Study で表現する。
6 個の Train Protocol を作らない。

## 7. 既存 ML コードを再利用する

MLDB 用に trainer / evaluator を丸ごと書き直さない。
既存 `tools/recognition/` の実験コードを薄く wrap する。

現在の tile-shape Train Protocol は例えば以下を再利用している。

```text
tools/recognition/train_tile_shape_classifier.py
  load_training_cache
  train_one_epoch
  evaluate_all
```

新しい Architecture experiment でも、既存 training loop が使えるならそのまま使う。
MLDB integration のためだけに second trainer を実装しない。

既存 script が CLI import 前提で package import できない場合は、今回のように最小の import compatibility を足す程度に留める。

実験コード変更と MLDB core 変更を混同しない。
モデル比較に必要なのが Architecture 実装だけなら `mldb/src/` は触らない。

## 8. Definition の更新ルール

sealed reusable definition の意味を変える時は、同じ ID を上書きしない。
新しい version ID を作る。

特に Architecture / Train Protocol / Evaluation Protocol の `.py` を変更したら、metadata の implementation SHA-256 も一致させる。
古い completed Run が参照している definition の意味を後から変えない。

新しい definition を作る時は既存の対応する YAML / Python を template として使い、必要な差分だけ変える。
独自 schema や独自 field を発明しない。

programmatic read / validation では public Controller boundary を優先できる。

```python
from mldb.src.api.controller import (
    validate_definition,
    seal_definition,
    execute_study,
    get_study_run,
    cancel_study_run,
    get_entity,
    list_entities,
)
```

ただし tile-shape の通常運用では既存 bootstrap / Study launcher を再利用する方が速い。

## 9. Failure handling

現在の SSH Study launcher は retry を積極的に行わない。
remote attempt が失敗すると、その attempt / child Run は canonical に failed となり、Study は reconciliation により通常 `completed_with_failures` へ閉じる。

その場合:

1. traceback の最初の実原因を直す。
2. 古い Run を手編集しない。
3. 新しい Study Run を起動する。

transport / environment failure でも、既に attempt が開始済みなら履歴を消さない。
SSH 接続確認は Study allocation 前に行うため、接続不能なら通常は新しい Study Run を作らず止まる。

cache corruption が疑われても、まず SHA-256 / byte verification のログを確認する。
bridge は cache hit 判定時に integrity を確認する。

失敗を隠すために Queue DB や Run YAML を書き換えない。

## 10. 結果の読み方

Study 完了後は canonical Run を読む。
PowerShell での最低限の確認例:

```powershell
Get-Content mldb_data\study_runs\<study-run-id>\run.yaml
Get-Content mldb_data\training_runs\<training-run-id>\run.yaml
Get-Content mldb_data\evaluation_runs\<evaluation-run-id>\run.yaml
Get-Content mldb_data\models\<model-id>.yaml
```

比較に使う数値は Evaluation Run の `result.metrics` を優先する。
Training 中に stdout へ出ただけの数値を canonical result の代わりにしない。

2026-09-08 の 1 epoch smoke では次が残った:

```text
manual_accuracy_0deg = 0.2066666667
jp_accuracy_0deg     = 0.7967647059
manual_angle_mean    = 0.1405555556
jp_angle_mean        = 0.4150367647
```

これは E2E 動作確認値であり、model quality の結論には使わない。

## 11. MLDB core を触ってよい条件

`mldb/src/` の core/application logic は現時点で実 GPU E2E を完走している。
通常の ML experiment では freeze 扱いにする。

次のような理由では core を変更しない。

- API をきれいにしたい。
- transport を HTTP にしたい。
- daemon 化したい。
- generic scheduler が欲しい。
- abstraction を増やしたい。
- 将来必要そう。
- MLflow っぽい機能を追加したい。

core を変更してよいのは、**実際に必要な experiment が既存 surface では実行不可能で、その blocker を再現できる場合**だけ。

その場合も最小 patch にする。
新しい Wave / ADR / spec 作業へ自動的に広げない。

`mldb/skeleton/` と accepted records/spec は、ユーザーが明示的に MLDB design work を依頼した場合以外は変更しない。

## 12. 現在の tile-shape reusable assets

既に存在するので、まずこれらを再利用できないか確認する。

```text
Task
  tile-shape-classification-35-v1

Corpus
  gray35-jp500-seed42-v3-jp189-v1

Architecture
  tile-plain-gray35-v2

Train Protocol
  tile-shape-train-gpu-v3

Evaluation Protocol
  tile-shape-eval-angle-v3

Smoke Study
  tile-plain-full-feature-smoke-v3
```

Train Protocol の主要 public parameters:

```text
epochs, batch_size, learning_rate, weight_decay,
rotation_augment_deg, perspective_augment, shear_augment,
stretch_augment, projective_augment_probability,
amp, tf32, cache_device, cache_vram_fraction
```

Evaluation は既定で 0/15/30/45 degree を評価する。

## 13. 新しい tile-shape Study を実行する

`run_study_ssh.py` は任意の sealed Study ID を受け取る。
新しい Study YAML を作ったら専用 launcher を増やさず、その ID を渡す。

```powershell
.\.venv\Scripts\python.exe tools\mldb\run_study_ssh.py `
  --host 192.168.11.22 `
  --study-id <new-study-id>
```

`--study-id` は必須。Study を暗黙選択しない。

例: plain と mobile-f8 を rotation augmentation 0/30 degree で比較するなら、
既存 Task / Corpus / Train Protocol / Evaluation Protocol を再利用し、必要な Architecture を用意して Study の matrix を 4 trial にする。

新しい trial ごとに shell script や Study launcher を増やさない。

## 14. 依頼から entity への変換例

### 「rotation augmentation だけ振りたい」

Architecture は変えない。Train Protocol も変えない。
Study の `rotation_augment_deg` values を増やす。

### 「stem の stride / conv 構造を比較したい」

Architecture が変わる。
各構造を別 Architecture ID にし、同じ Train Protocol / Corpus / Evaluation を使う。

### 「AdamW と SGD を比較したい」

training procedure の意味が変わるため Train Protocol を分けるか、既存 Protocol が optimizer を public parameter として正式に扱えるなら parameter sweep にする。

### 「dataset を追加・再分割したい」

新しい Corpus を作る。既存 completed Run の Corpus を差し替えない。

### 「評価角度を増やしたい」

既存 Evaluation Protocol の `eval_angles` parameter で表せるなら Study parameter だけ変更する。
metric contract 自体を変える場合だけ Evaluation Protocol を新 version にする。

## 15. 実行前後のチェック

### 実行前

```powershell
git status --short
.\.venv\Scripts\python.exe tools\mldb\bootstrap_tile_shape.py
```

bootstrap は classifier Corpus SQLite が無い場合の materialization 専用。definition は生成・seal しない。既存 artifact がある場合は resolver による整合性確認だけを行う。
実験用 source dataset が無い場合は、その場で適当な fake Corpus を作らず、既存 dataset build flow を確認する。

### code を変更した時

```powershell
.\.venv\Scripts\python.exe -m py_compile <changed-python-files>
.\.venv\Scripts\python.exe -m pytest mldb\tests -q
git diff --check
```

MLDB core を触っていない小さな experiment-definition 変更でも、少なくとも resolver / launch 前検証は通す。

### 実行後

Study が `completed` / `completed_with_failures` / `failed` / `cancelled` のどれかに閉じたか確認する。
`running` のまま launcher が終了していたら、そのまま次の Study を投げず原因を調べる。

## 16. Study matrix の最小パターン

parameter sweep は Study に書く。
既存 `tile-plain-full-feature-smoke-v3.yaml` または `rotated-fcos-smoke-v1.yaml` を model family に応じた template にする。

```json
{
  "model": {
    "train": {
      "corpus": "<corpus-id>",
      "protocol": "<train-protocol-id>",
      "architectures": ["<arch-a>", "<arch-b>"],
      "parameters": {
        "rotation_augment_deg": {"values": [0.0, 30.0]},
        "epochs": {"values": [50]}
      },
      "seeds": [42]
    }
  },
  "evaluations": [
    {
      "stage": "angle-holdout",
      "corpus": "<eval-corpus-id>",
      "protocol": "tile-shape-eval-angle-v3",
      "parameters": {"batch_size": 1024}
    }
  ]
}
```

schema / id / status / name / description も既存 Study と同じ contract に従って記述する。

## 17. Commit discipline

ML experiment の作業では unrelated file を stage しない。
必ず `git status --short` を見て、今回の experiment / MLDB integration に必要な path だけ add する。

特に repository root には recognition experiment の scratch file が残ることがある。
名前だけ見てまとめて `git add .` しない。

runtime Run / Model / Corpus SQLite は `.gitignore` により除外されるが、ignore ルールに頼り切らず staged diff を確認する。

```powershell
git diff --cached --stat
git diff --cached --check
```

commit message は「MLDBを改善した」ではなく、何の ML execution capability / experiment を追加したか分かる名前にする。

例:

```text
feat(mldb): run tile shape studies on remote GPU worker
exp(recognition): compare tile classifier stems
```

## 18. 新しい AI session の開始チェックリスト

MLDB を使って ML experiment を依頼されたら、最初に:

1. この `mldb/AGENTS.md` を読む。
2. `git status --short` を確認する。
3. `mldb_data/` の既存 reusable definitions を確認する。
4. 関連する `tools/recognition/` の既存実装を確認する。
5. 最新の canonical Runs / Models が必要なら `mldb_data/*_runs/` / `models/` を読む。
6. 既存 assets で実験を表現できるか判断する。
7. 最小の差分で Study を作って実 GPU へ投げる。

「まず MLDB の設計を整理しましょう」から始めない。
「将来のために API を追加しましょう」から始めない。

## 19. 一行ルール

**MLDB に時間を使うのではなく、MLDB を使って ML に時間を使う。**
