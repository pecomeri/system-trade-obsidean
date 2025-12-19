---
id: no_weekend_entry
type: filter
scope: time_structure
status: active
tags:
  - fx
  - filter
  - week_transition
---

# no_weekend_entry（参照用）

このノートは `FX/27_filters/no_weekend_entry.md` の参照用ミラーです。
正典は `FX/27_filters/no_weekend_entry.md` とします（内容の更新があればそちらを優先）。

## 定義（Contract）
- 対象は **新規エントリーのみ**（exit/強制クローズは扱わない）
- 禁止ルール（UTC）：
  - 金曜：20:00 UTC 以降は新規エントリー禁止
  - 月曜：02:00 UTC までは新規エントリー禁止
- 判定基準（推奨）：
  - **評価時刻 ts（10秒足）**で entry 可否を判定し、禁止なら発注しない

