---
id: D009
family: family_D_momentum
type: observation_variant
strategy: "[[FX/30_hypotheses/program_01_breakout/02_1M_breakout_10s_entry/family_D_momentum/D004_m1_momentum_burst_break_prev]]"
base_strategy: "[[FX/30_hypotheses/program_01_breakout/02_1M_breakout_10s_entry/family_D_momentum/D004_m1_momentum_burst_break_prev]]"
filters:
  - "[[h1_trend_down]]"
  - "[[h1_trend_up]]"
regimes:
  - none
timeframe_signal: M1
timeframe_exec: 10s
status: observed
result: strong_break_exhaustion
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

### 実行結果（事実）
- 実行条件:
  - verify=2024 / forward=2025
  - only_session=None / use_time_filter=False（24h観測）
  - use_h1_trend_filter=True（buyはh1_trend_up / sellはh1_trend_down を適用）
  - allow_sell=True / allow_buy=True
- trades 件数（verify / forward）:
  - 2024 verify: BUY=805 / SELL=507（total=1312）
  - 2025 forward: BUY=590 / SELL=639（total=1229）
- side別PnL（verify / forward）:
  - 2024 verify:
    - BUY: sum_pnl_pips=-702.0 / avg_pnl_pips=-0.87
    - SELL: sum_pnl_pips=-516.0 / avg_pnl_pips=-1.02
  - 2025 forward:
    - BUY: sum_pnl_pips=-840.0 / avg_pnl_pips=-1.42
    - SELL: sum_pnl_pips=-604.0 / avg_pnl_pips=-0.95
- 自己チェック:
  - D004の `monthly.csv` / `monthly_by_session.csv` の sha256 を比較し、変更なし

### D008形式の観測（by side / early_loss vs survivor）
- 生成物:
  - `FX/results/family_D_momentum/D009/diagnostics/early_vs_survivor_features_by_side.csv`
  - `FX/results/family_D_momentum/D009/diagnostics/early_vs_survivor_summary_by_side.md`
- group定義（固定）:
  - holding_time_min = (exit_time - entry_time).total_seconds()/60
  - early_loss: 0 <= holding_time_min <= 3
  - survivor: holding_time_min >= 20
- group件数（period×side）:
  - 2024 verify:
    - BUY: early_loss=50 survivor=456
    - SELL: early_loss=84 survivor=158
  - 2025 forward:
    - BUY: early_loss=43 survivor=293
    - SELL: early_loss=85 survivor=218
- 差が大きい特徴量（side別 top3 / delta_median = median_survivor - median_early）:
  - BUY:
    - break_margin_pips: forward=-1.9000 / verify=-3.0500（sign=match）
    - burst_strength: forward=-0.6365 / verify=0.0097（sign=mismatch）
    - next_m1_body_ratio: forward=-0.0859 / verify=-0.0218（sign=match）
  - SELL:
    - break_margin_pips: forward=-1.6000 / verify=-3.1000（sign=match）
    - body: forward=-0.0325 / verify=-0.0400（sign=match）
    - body_ratio: forward=0.0297 / verify=0.0664（sign=match）


### 結論

- 即死トレードは BUY/SELLともに「break_marginが大きい」方向に偏る
    
- SELLではさらに「bodyが大きい」「body_ratioはsurvivorの方が大きい（ヒゲが短い）」が一貫
    
- burst_strength（BUY）は sign mismatch で保留