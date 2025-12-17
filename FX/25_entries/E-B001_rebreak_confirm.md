---
id: E-B001
type: entry
status: draft
---

# E-B001 rebreak_confirm

## 概要

- 目的：
- どのSetup前提で使うか：

## エントリー条件

- 必須条件（Yes/No）：
- 再ブレイク（rebreak）の定義：

## 定義（Contract）

- Entry の役割：
  - Setup が Yes のとき、機械的に入る点を定める（方向当てではない）

- 前提：
  - `S-001=Yes`
  - 方向は Setup で一意に定まっている（ここで方向を作らない）

- Yes 条件（すべて満たす）：
  1) フェイクアウト方向を否定した構造が 1 つ **確定**している
  2) その否定構造の高値/安値を **再度** ブレイク（rebreak）する
  3) 判定は確定足ベース（未確定足は禁止）

- No 条件（代表例）：
  - 抜けた瞬間に入る（否定構造が確定する前に入る）
  - rebreak の同定が曖昧（何を“否定構造の高値/安値”とするかが固定されていない）
  - 同一足内で「否定の確定」と「再ブレイク」が全部起きた扱いになっている

- 記録するログ項目（判定ログ）：
  - 否定構造の時刻（確定時刻）
  - 初回ブレイク時刻
  - 再ブレイク時刻
  - エントリー時刻（発注/約定の基準時刻）

## 用語定義（Glossary）

- 確定（confirmed）
  - 確定＝足の終値（close）が確定した状態。未確定足（形成中）は判定に使わない。
  - 「確定構造」も同様に、確定足から計算された pivot/構造のみを指す。
- ブレイク（break）
  - ブレイク＝**確定足の終値（close）**が基準線を超えたこと（wick/ヒゲは判定に使わない）。
  - 方向（上抜け/下抜け）は、基準線に対して close が上か下かで決める。
- 構造（structure）
  - 構造＝pivot（スイング高値/安値）の連なりで定義する。
  - pivot の確定ルールは TODO（例：左右 `PIVOT_N=TODO` 本、または別方式）だが、**pivot は確定足のみ**から作る。
- 押しレンジ（pullback range）
  - 押しレンジ上下は Setup 側（`S-001`）で確定したレンジを参照する（ここで再定義しない）。
- レンジ内部に復帰（return inside）
  - 復帰判定は Setup 側（`S-001`）の定義（close で内側）に従う。
- 抜け未遂（false break）
  - 抜け未遂イベントの数え方は Setup 側（`S-001`）の定義に従う（close で外側→次足 close で内側）。
- 否定構造（invalidation structure）
  - 否定構造＝抜け未遂（false break）を受けて「抜け方向を否定した」と言える pivot/構造。
  - 具体的な pivot の確定ルール（どの pivot を採用するか）は TODO だが、**否定構造は必ず“確定足”で確定したもの**に限定する。

## 判定の擬似コード（Pseudo）

入力（概念）：
- `setup_ok, setup_log`（`S-001` の判定結果とログ。`range_high/range_low/false_break` を含む）
- `pivots`（pivot 配列：`(ts, kind=HIGH/LOW, price)`。確定ルールは TODO）
- `bars`（判定に使う確定足の時系列。足種別は実装側で選ぶ）

出力：
- `ok: bool`
- `log: dict`（判定ログ：否定構造時刻、first_break、rebreak、entry_ts）

擬似コード（重要：否定構造の確定足と rebreak 発生足は別足）：

```text
def judge_EB001(setup_ok, setup_log, pivots, bars) -> (ok, log):
  log = {}
  if not setup_ok:
    return (No, log + {"reason":"SETUP_NOT_OK"})

  # 1) 否定構造を1つ確定する（pivot確定ルールは TODO）
  inv = find_invalidation_structure(
          pivots=pivots,
          false_break=setup_log["false_break"],
          rule=TODO)
  if inv is None:
    return (No, log + {"reason":"NO_INVALIDATION_STRUCTURE"})
  log["invalidation_ts"] = inv.confirmed_ts

  # 2) 基準線は「否定構造のpivot高値/安値」
  level = inv.pivot_price  # kindに応じて high/low。扱いは TODO だが参照はここに固定
  log["level"] = level

  # 3) 初回ブレイク（first_break）：確定足closeで level を超える最初の足
  first_break = first_confirmed_close_crossing_level(bars, after_ts=inv.confirmed_ts, level=level)
  if first_break is None:
    return (No, log + {"reason":"NO_FIRST_BREAK"})
  log["first_break_ts"] = first_break.close_ts

  # 4) 再ブレイク（rebreak）：first_break より後の“別足”で再度 close が level を超える
  rebreak = next_confirmed_close_crossing_level(bars, after_ts=first_break.close_ts, level=level)
  if rebreak is None:
    return (No, log + {"reason":"NO_REBREAK"})
  if rebreak.close_ts == inv.confirmed_ts:
    return (No, log + {"reason":"SAME_BAR_AS_INVALIDATION"})  # 保険（実装では起きない想定）
  log["rebreak_ts"] = rebreak.close_ts

  # 5) エントリー時刻/価格の基準を固定する
  #   - 判定は rebreak の確定足で行い、
  #   - 執行は「次の足の始値（open）」を基準にする（スリッページ等は別途 TODO）
  entry = next_bar_open_after(bars, ts=rebreak.close_ts)
  if entry is None:
    return (No, log + {"reason":"NO_NEXT_BAR_FOR_ENTRY"})
  log["entry_ts"] = entry.open_ts

  return (Yes, log + {"reason":"OK"})
```

## タイミング

- トリガー足（確定/未確定）：
- エントリー執行（次バー/同バー）：
- 取り消し条件：

## 想定メリット / リスク

- メリット：
- リスク：

## Exit（暫定）

- SL/TP/時間撤退の方針（数値はここでは固定しない）：

## 実装メモ

- 実装上の注意（再ブレイクの検出、欠損バーの扱い等）：
- デバッグ観点（誤検出のパターン）：

## ステータス

- status: draft / active / deprecated
- 最終更新日：

## 変更履歴

- 2025-12-17 定義（Contract）追記
- 2025-12-17 用語定義（Glossary）・擬似コード（Pseudo）追記
