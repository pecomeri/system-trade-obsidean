---
id: F-C001
type: filter
status: draft
---

# F-C001 no_rebreak

## 目的

- 事故抑制のために除外したい状況：

## Yes / No 定義

- Yes（通す）：
- No（弾く）：

## 定義（Contract）

- Filter の役割：
  - 勝率改善装置ではなく「明確な地雷回避」に限定する
  - Reject 率が高すぎる設計は NG（過剰フィルター検知を必須にする）
  - 本 F-C001 は Entry 後段のフィルターとして扱う（Setup単体の弾きではない）
  - `entry_ok=False` のケースは Reject ではなく NA（適用外）とする
  - Reject 率の母数は `entry_ok=True` の試行のみ（NA を含めない）

- Reject 条件（いずれかで Reject）：
  1) 再ブレイクが一定時間内に起きない（時間は TODO）
  2) 再ブレイク前に逆方向の構造が確定して、方向の一貫性が崩れた
  3) 同一価格帯で複数回ブレイクが発生し、ノイズ化している

- Pass 条件：
  - Reject に該当しない場合のみ Pass

- 必須モニタリング（過剰フィルター検知用）：
  - Filter 前後の `trade_count`
  - Reject 率（Reject が増えすぎていないか）

- 記録するログ項目（判定ログ）：
  - Reject 理由コード（例：`NO_REBREAK_TIMEOUT` / `OPPOSITE_STRUCTURE` / `MULTI_BREAK_NOISE`）
  - 判定時刻
  - 直前構造の向き（方向一貫性の確認用）

## 用語定義（Glossary）

- 確定（confirmed）
  - 確定＝足の終値（close）が確定した状態。未確定足（形成中）は判定に使わない。
- ブレイク（break）
  - ブレイク＝**確定足の終値（close）**が基準線を超えたこと（wick/ヒゲは判定に使わない）。
- 構造（structure）
  - 構造＝pivot（スイング高値/安値）の連なりで定義する。
  - pivot の確定ルールは TODO（例：左右 `PIVOT_N=TODO` 本、または別方式）だが、**pivot は確定足のみ**から作る。
- 押しレンジ（pullback range）
  - 押しレンジ上下は Setup 側（`S-001`）で確定したレンジを参照する（Filter で再定義しない）。
- レンジ内部に復帰（return inside）
  - 復帰判定は Setup 側（`S-001`）の定義（close で内側）に従う。
- 抜け未遂（false break）
  - 抜け未遂イベントの定義と数え方は Setup 側（`S-001`）に従う（close で外側→次足 close で内側）。
- 再ブレイク（rebreak）
  - 再ブレイクの定義は Entry 側（`E-B001`）に従う（first_break の後、別足の確定 close で再度 level を跨ぐ）。

## 判定の擬似コード（Pseudo）

入力（概念）：
- `setup_ok, setup_log`（`S-001` の判定結果とログ）
- `entry_ok, entry_log`（`E-B001` の判定結果とログ：`level/first_break_ts/rebreak_ts` を含む）
- `pivots`（pivot 配列。確定ルールは TODO）
- `bars10s`（10秒足の確定足時系列。timeout 単位は 10秒足のバー本数で固定）

出力：
- `status: Pass / Reject / NA`
- `reject_code: str | None`（NA の場合は None）
- `log: dict`（reject 判定時刻、理由、補助情報）

擬似コード（TODO は値だけ未決。単位と参照は固定）：

```text
TIMEOUT_BARS_10S = TODO        # 単位：10秒足バー本数（秒ではなく本数で固定）
BAND_WIDTH_PIPS = TODO         # 「同一価格帯」の幅（pips）。値はTODO
MULTI_BREAK_M = TODO           # 「複数回」の閾値（回数）。値はTODO

def judge_FC001(setup_ok, entry_ok, entry_log, pivots, bars10s) -> (status, reject_code, log):
  log = {}
  if not setup_ok:
    return (Reject, "SETUP_NOT_OK", log)
  if not entry_ok:
    # 本 F-C001 では entry 非成立はフィルタ判定の母数に入れない
    return (NA, None, log + {"reason":"ENTRY_NOT_OK_NA"})

  level = entry_log["level"]
  first_break_ts = entry_log["first_break_ts"]
  rebreak_ts = entry_log.get("rebreak_ts")  # 無い場合がある

  # 1) timeout：rebreak が一定本数内に起きない
  if rebreak_ts is None:
    if bars_since(bars10s, from_ts=first_break_ts, to_ts=last_confirmed_ts(bars10s)) > TIMEOUT_BARS_10S:
      return (Reject, "NO_REBREAK_TIMEOUT", log + {"judge_ts": last_confirmed_ts(bars10s)})
  else:
    if bars_since(bars10s, from_ts=first_break_ts, to_ts=rebreak_ts) > TIMEOUT_BARS_10S:
      return (Reject, "NO_REBREAK_TIMEOUT", log + {"judge_ts": rebreak_ts})

  # 2) 方向一貫性：rebreak 前に逆方向構造が確定
  if opposite_structure_confirmed(pivots, between=(first_break_ts, rebreak_ts or last_confirmed_ts(bars10s)), rule=TODO):
    return (Reject, "OPPOSITE_STRUCTURE", log + {"judge_ts": min_ts_that_triggers_opposite(pivots, rule=TODO)})

  # 3) ノイズ：同一価格帯で複数回ブレイク
  #    - 「同一価格帯」は level±BAND_WIDTH_PIPS（値はTODO）で固定
  #    - カウントは close による break 回数で数える（wickは数えない）
  if break_count_in_band(bars10s, center_level=level, band_width_pips=BAND_WIDTH_PIPS, rule="close_cross") >= MULTI_BREAK_M:
    return (Reject, "MULTI_BREAK_NOISE", log + {"judge_ts": rebreak_ts or last_confirmed_ts(bars10s)})

  return (Pass, None, log + {"judge_ts": rebreak_ts or last_confirmed_ts(bars10s)})
```

## 使われ方（リンク）

- Setup:
  - [[../15_setups/S-001_trend_midrange_fakeout_v.md|S-001_trend_midrange_fakeout_v]]
- Entry:
  - [[../25_entries/E-B001_rebreak_confirm.md|E-B001_rebreak_confirm]]

## 実装メモ

- 判定に必要な情報（5min/1min/10s）：
- 欠損や境界の扱い（セッション跨ぎ等）：

## ステータス

- status: draft / active / deprecated
- 最終更新日：

## 変更履歴

- 2025-12-17 定義（Contract）追記
- 2025-12-17 用語定義（Glossary）・擬似コード（Pseudo）追記
- 2025-12-17 最小追記パッチ（pullback区間固定 / next_bar時間足固定 / NA導入）
