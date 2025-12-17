---
id: S-001
type: setup
status: draft
timeframes:
  structure: 5min
  execution: 10s
---

# S-001 trend_midrange_fakeout_v

## 構造定義（5分足 / 10秒足）

- 5分足（構造）
  - トレンドの定義：
  - ミッドレンジの定義：
  - フェイクアウトの定義：
- 10秒足（執行補助）
  - どの価格アクションを“執行の合図”とするか：
  - 連続バー/否定の扱い：

## 定義（Contract）

- Setup の役割：
  - Entry 判定に進んで良い「形の候補出し」に限定する（勝ちそう等の主観は含めない）

- Yes 条件（すべて満たす）：
  1) 上位足レジームが明示的に `trend_up` / `trend_down`（`range` / `unknown` は No）
  2) 直近構造が「推進 → 押し → 再推進前」になっている
     - 押しは直近推進の“内部”に収まっている（反転＝高値/安値の更新ではない）
  3) 押し内部で「抜け未遂（ブレイク → 否定）」が **一度だけ** 発生している
  4) 現在価格が押しレンジ内部に復帰している（抜けたまま推進に入ったら No）

- No 条件（代表例）：
  - レジームが `range` / `unknown`
  - 抜け未遂が複数回（ノイズ化）
  - 押しが深すぎて構造崩壊（押しが“反転”に見える）
  - 「事後的にフェイクと呼べるだけ」で、当時点で Yes/No が切れない

- 記録するログ項目（判定ログ）：
  - レジーム値（`trend_up` / `trend_down` / `range` / `unknown`）
  - 押しレンジ上下（価格）
  - 抜け未遂の発生位置（価格・方向）と発生時刻
  - 復帰確認の時刻（押しレンジ内に戻った時刻）

## 用語定義（Glossary）

- 確定（confirmed）
  - 確定＝足の終値（close）が確定した状態。未確定足（形成中）は判定に使わない。
- ブレイク（break）
  - ブレイク＝**確定足の終値（close）**が基準線を跨いだこと（wick/ヒゲは判定に使わない）。
- 構造（structure）
  - 構造＝pivot（スイング高値/安値）の連なりで定義する。
  - pivot の確定ルールは TODO（例：左右 `PIVOT_N=TODO` 本、または別方式）だが、**pivot は確定足のみ**から作る。
- 押しレンジ（pullback range）
  - 押しレンジ上下＝「押し開始〜押し終了」区間の**確定足**の高値/安値（または pivot で定義する場合は該当 pivot）。
  - 押しの開始/終了（区間の切り方）は pivot に基づき TODO（後から変えない前提で決める）。
- レンジ内部に復帰（return inside）
  - 復帰＝確定足の終値（close）が押しレンジ内（`range_low <= close <= range_high`）に戻ったこと。
- 抜け未遂（false break）
  - 抜け未遂イベント＝押しレンジの外側へのブレイク（close で外側）→ **次の確定足**でレンジ内に復帰、の 1 セット。
  - 同一足内のヒゲ（wick）だけの抜けは数えない（イベントにしない）。

## 判定の擬似コード（Pseudo）

入力：
- `regime`（`trend_up/trend_down/range/unknown`）
- `pivots`（pivot 配列：`(ts, kind=HIGH/LOW, price)`。確定ルールは TODO）
- `bars`（判定に使う確定足の時系列。足種別は 5min/10s は実装側で選ぶ）

出力：
- `ok: bool`
- `log: dict`（判定ログ：レンジ上下、イベント時刻など）

擬似コード（数値は TODO で変数化して固定する）：

```text
def judge_S001(regime, pivots, bars) -> (ok, log):
  log = {}

  # 1) レジーム
  if regime not in {"trend_up", "trend_down"}:
    return (No, log + {"reason":"REGIME_NOT_TREND"})

  # 2) 推進→押し→再推進前（pivotで判定。具体ルールは TODO）
  impulse, pullback = segment_impulse_and_pullback(pivots, rule=TODO)
  if impulse is None or pullback is None:
    return (No, log + {"reason":"NO_IMPULSE_OR_PULLBACK"})

  pullback_range = range_of_confirmed_bars(bars, start=pullback.start_ts, end=pullback.end_ts)
  range_high = max(pullback_range.high)
  range_low  = min(pullback_range.low)
  log["range_high"] = range_high
  log["range_low"] = range_low

  # 2b) 押しは推進の内部（反転ではない）
  if is_reversal_vs_impulse(pivots, impulse, pullback, rule=TODO):
    return (No, log + {"reason":"PULLBACK_LOOKS_REVERSAL"})

  # 3) 抜け未遂イベント（closeで外側→次足closeで内側）の回数を数える
  false_break_events = []
  for i in indices_of_confirmed_bars_within(pullback_range):
    c0 = close[i]
    c1 = close[i+1] if i+1 exists else None
    broke_out = (c0 > range_high) or (c0 < range_low)
    returned_inside_next = (c1 is not None) and (range_low <= c1 <= range_high)
    if broke_out and returned_inside_next:
      false_break_events.append({"break_ts": ts[i], "return_ts": ts[i+1], "side": side(c0, range_high, range_low)})

  if len(false_break_events) != 1:
    return (No, log + {"reason":"FALSE_BREAK_COUNT_NOT_ONE", "false_break_count": len(false_break_events)})

  log["false_break"] = false_break_events[0]

  # 4) 現在はレンジ内に復帰済み（closeで判定）
  if not (range_low <= last_confirmed_close(bars) <= range_high):
    return (No, log + {"reason":"NOT_RETURNED_INSIDE"})

  return (Yes, log + {"reason":"OK"})
```

## 関連リンク

- Entry:
  - [[../25_entries/E-A001_immediate.md|E-A001_immediate]]
  - [[../25_entries/E-B001_rebreak_confirm.md|E-B001_rebreak_confirm]]
- Filter:
  - [[../27_filters/F-C001_no_rebreak.md|F-C001_no_rebreak]]

## 実装メモ

- データ前提（例：Dukascopy 10秒足）：
- 既存実装との対応（どの関数/シグナルに落とすか）：
- 期待するデバッグ出力（例：signals/entriesの確認観点）：

## ステータス

- status: draft / active / deprecated
- 最終更新日：

## 変更履歴

- 2025-12-17 定義（Contract）追記
- 2025-12-17 用語定義（Glossary）・擬似コード（Pseudo）追記
