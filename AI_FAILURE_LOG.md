# AI Working Corrections

mjtensuでAIが繰り返した判断ミスから、再発防止に必要な少数の行動規則だけを保持する。
session開始時にこのファイルを読む。

## Priority order

1. **Actual product/runtime truth** — 実際に動いているdeploy、model、artifact、repo state。
2. **Current user intent** — 今回何を比較・変更・確認するか。
3. **Canonical project rules** — AGENTS、authoring guide、spec、ADR。
4. **Current task execution** — 実験、実装、文書更新。
5. **Optional cleanup** — commit、push、整理、周辺改善。

下位の作業が上位の確認を置き換えてはならない。

## General tendencies

### GM-01: Environment-role inference
branch名、過去会話、model名だけから dev / production を推測しやすい。

Guard:
- production/current deploymentを述べる前に、deploy directory、compose file、running container、runtime model artifactを実物確認する。
- mjtensu production deploymentは `/persist/srv-bugrat/mjtensu-product/compose.prod.yaml`。running containerは `mjtensu-product-mjtensu-1`。
- mjtensu dev deploymentは `/persist/srv-bugrat/mjtensu-dev/compose.dev.yaml`。running containerは `mjtensu-dev-mjtensu-dev-1`。
- 現在のproduction base classifierはC8。`c8-tile-35-v1` / `tile-c8-gray35-v3-jp189.onnx` をproduction側の現物で確認してから別modelをproductionと呼ぶ。
- `mjtensu-dev` 側のPlain / MobileNet / ShuffleNet実験・promotion履歴をproduction stateと混同しない。

### GM-02: Authority-before-authoring
既存artifactの見た目や別コピーから書式を推測し、canonical guide確認前に文書を作りやすい。

Guard:
- design record作成前にbrewprintのcanonical authoring guideを読む。
- guide sourceは `brewprint/product/records/spec/design-records/authoring-standards/` を優先する。
- `bin/`、生成物、古いrecordの形だけをauthoring authorityにしない。
- guideで確認したmetadata、section、lifecycleだけを書く。

### GM-03: Unrequested repository mutation
artifact作成後に、依頼されていないcommit / pushまで進めやすい。

Guard:
- commit / pushはuserが明示した場合、またはtask contractが明示要求する場合だけ行う。
- 文書追加・修正依頼はworking tree変更までを既定とする。
- corrective workでもremote historyを勝手に書き換えない。


### GM-04: Unapproved branch/worktree proliferation
実験隔離や並行作業を理由に、user合意なしでbranch/worktreeを増やしやすい。

Guard:
- 新しいbranchまたはworktreeを作る前に、目的・親branch・統合方法をuserへ確認する。
- 実験用worktreeをcanonical `product/records/` のauthoring locationとして扱わない。
- 実験結果は既存branchへ統合する前提を明示し、未merge branchを放置しない。

## Active safeguards

- **Deployment identity check:** prod/dev/current modelの主張はdeploy実物確認後に行う。
- **Guide-before-write:** design recordはcanonical authoring guideを先に読む。
- **Mutation boundary:** edit、commit、pushを別操作として扱う。依頼されていないcommit/pushはしない。
- **Source-of-truth check:** repo state、runtime state、model stateを過去会話だけで確定しない。
- **Correction wording:** 誤りを見つけたら、何が誤りで何を現物確認したかを分けて返す。
- **Branch/worktree gate:** 新規branch/worktreeは目的・親・統合方法をuserと合意してから作る。
- **Condition-visibility gate:** user-facing experiment comparisonでは、varying parameter値をtable列・plot軸・series/condition labelのいずれかに必ず露出する。内部trial/eval IDを実験条件の代替表示名にしてはならない。内部IDはmetadata/fallbackに限定する。
- **Experiment-namespace gate:** 実モデル性能・閾値・アーキテクチャ・E2E挙動を評価するML experimentを `mldb-smoke` に置いてはならない。`mldb-smoke` はintegration/infrastructure smoke専用。実験の配置先は既存の実Project/namespace構造を確認して決め、名前にも `smoke` を付けない。
- **Ambiguity-before-mutation gate:** path/project表現は会話の流れと実システム構造を先に使って解釈する。文脈と実物確認で一意に決まるなら質問で止めない。それでも複数解釈が残る場合だけmutation前に短く確認し、推測でdirectory/project/taskを作成・移動しない。

## Incident index

新しいincidentは原則1行だけ追加する。一般化できる場合だけ上の規則を更新する。

| ID | lesson |
|---|---|
| INC-20260929-01 | `mjtensu-dev` のclassifier状態をproductionと繰り返し誤認した。productionは`mjtensu-product/compose.prod.yaml`と実model artifactで確認する。 |
| INC-20260929-02 | Investigation guide確認前にINV-015を書き、さらに依頼されていないcommit/pushまで行った。authoring authority確認とrepository mutationを分離する。 |
| INC-20260929-03 | 実験隔離のためuser合意なしで4本のbranch/worktreeを作り、未merge成果を分散させた。branch/worktree作成は事前合意し、canonical recordsは本体repoへ書く。 |
| INC-20260930-01 | canonical repoを変更せずCI/CDのdeploy checkoutを直接書き換えてdevへ反映した。deploy checkoutは生成・反映先として扱い、source変更はcanonical repoで行ってcommit/pushしCI/CD経由で反映する。 |
| INC-20261005-01 | Study parameter sweepをmodel-comparison UIへ流し、varying conditionを内部trial IDに隠したため比較不能な表示を作った。user-facing比較ではparameter値を明示し、内部IDを条件名として使わない。 |
| INC-20261005-02 | detector thresholdを実動画E2Eで評価する本実験を `mldb-smoke` に作成した。`mldb-smoke` は非ML実験用であり、本実験の配置先は既存の実Project/namespace構造を確認して決める。 |
| INC-20261005-03 | 会話の流れ上 `mldb/nanodet` がClearML Pipeline Projectを指すことが明白だったのに、文脈を無視してrepo pathと誤解し、`mldb_data/tile-detector`等の無意味なtreeを作成した。まず会話文脈と実Project構造で解釈し、それでも曖昧な場合だけ質問する。 |
| INC-20261005-04 | ClearML Project再編で、MLDB canonical sourceが `project = f"mldb/{namespace}"` とProject配置を決定していることを確認せず、既存ClearML Task/Modelだけ `move_to_project` / project editで移動した。これはpush/次回runで旧Projectを再生成する非canonical修正であり、INC-20260930-01と同型。Project/UI再編は先にsource-side projection/routingを修正・検証し、その後に既存履歴を移行する。 |

## Incident template

`INC-YYYYMMDD-NN | 何を誤ったか / 一般化できるlesson`
