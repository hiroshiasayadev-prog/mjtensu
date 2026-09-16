# mjtensu Agent Notes

ML experiment / MLDB に関する作業を行う場合は、計画・実装・実行の前に `mldb/AGENTS.md` を全文読むこと。

`mldb/AGENTS.md` の最上位目的は、MLDB 自体を発展させることではなく、MLDB を使って実際の ML 実験を最短で回すことである。

MLDB v2 (`mldb_v2/`, namespace-first `mldb_data/`, ClearML execution) を扱う場合は、さらに `mldb_v2/AGENTS.md` と `mldb_v2/README.md` から該当する `mldb_v2/docs/` manual を読むこと。v1 の SSH worker 手順を v2 ClearML execution に混ぜないこと。

repo state に依存する判断は推測せず、実 filesystem / Git state を確認すること。

ML/MLDB 以外の作業について、このファイルは追加ルールを定義しない。
