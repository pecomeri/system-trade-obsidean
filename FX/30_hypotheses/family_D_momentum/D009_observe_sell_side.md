---
id: D009
family: family_D_momentum
type: observation_variant
strategy: "[[D004_m1_momentum_burst_break_prev]]"
base_strategy: "[[D004_m1_momentum_burst_break_prev]]"
filters:
  - "[[h1_trend_down]]"
  - "[[h1_trend_up]]"
regimes:
  - none
timeframe_signal: M1
timeframe_exec: 10s
status: draft
result: observing
tags:
  - fx
  - family_D_momentum
  - observation
  - sell
  - direction_split
---

# D009: D004 × Observe Sell Side (H1 Trend Down)

## 目的
- これまで buy 側（H1 trend up 前提）に偏っていた観測を補完し、
  **sell 側（H1 trend down）でも同様の構造が出るか**を確認する。
- D004（M1 Momentum Burst / Break Prev + Body Quality）は固定し、
  「方向（trend up/down）」のみを切り替えた観測を行う。
- 目的は成績改善ではなく、**現象の対称性（long/shortで同じか）**の確認。

---

## 背景
- 現状の config では `allow_sell=false` 等により buy 側しか出ない可能性がある。
- sell 側を観測するには、
  1) allow_sell を有効化
  2) H1 trend down のフィルターを用意
  が必要。

---

## 使用する戦略（固定）
- strategy:
  - `D004_m1_momentum_burst_break_prev`
- エントリー条件・SL/TP・執行は一切変更しない。

---

## フィルター
- 有効：
  - `[[h1_trend_down]]`
- 無効（D009では必須ではない）：
  - W1_only 等の時間帯制約（観測目的によりON/OFFは実装側に委ねるが、まずは24h観測が推奨）

---

## 観測内容（最低限）
1. sell トレードが実際に発生しているか（trades.csvで side=SELL が存在するか）
2. 曜日×時間帯の分布（必要なら D006 と同じCSV）
3. holding_time（D007）や early_vs_survivor（D008）の現象が sell 側でも出るか
   - ただし、まずは「sellが出る」ことの確認を最優先とする

---

## 生成物
- results:
  - `FX/results/family_D_momentum/D009/`
- diagnostics（必要なら）:
  - `entry_by_dow_hour.csv` 等

---

## 実行メモ
- config 注意：
  - `allow_sell=true`（必須）
  - `allow_buy` は ON のまま
  - h1_trend_up, `h1_trend_down` が参照されるようにする

---

## 結果メモ（後で記入）
- buy, sell トレード件数（verify / forward）:
- buy と比較して偏りはあるか:
- 次の観測へ進むか（D007/D008をsell側でも回すか）:
