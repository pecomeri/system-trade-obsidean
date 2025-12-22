---
id: D011a
family: family_D_momentum
type: filter_variant
strategy: "[[FX/30_hypotheses/program_01_breakout/family_D_momentum/D004_m1_momentum_burst_break_prev]]"
base_strategy: "[[FX/30_hypotheses/program_01_breakout/family_D_momentum/D004_m1_momentum_burst_break_prev]]"
filters:
  - "[[h1_trend_up]]"
  - "[[exhaustion_ratio_top20_buy]]"
regimes:
  - none
timeframe_signal: M1
timeframe_exec: 10s
status: observed
result: overfiltering_exhaustion
tags:
  - fx
  - family_D_momentum
  - filter_variant
  - exhaustion
  - buy
---

# D011a: D004 × Exhaustion Ratio Top20 Exclude (BUY)

## 目的
- D010で確認した「exhaustion ratio が高いほど early_loss に偏る」傾向を踏まえ、
  **exhaustionが強すぎるシグナルを除外**して即死（0–3分SL）を減らせるか検証する。
- strategy（D004）は固定し、外付けフィルターのみ追加する。

---

## 仮説
- break_margin_ratio / break_margin_over_mean_prev_body が極端に高いケースは、
  “強すぎる抜け（行き過ぎ）”で反転しやすく、早期SLに偏る。
- 上位20%を除外すると、trades を極端に減らさずに、sum_pnl のマイナスが縮む可能性がある。

---

## フィルター定義（固定）
- 対象：BUY シグナルのみ（sellは対象外）
- 使用特徴量（優先順）：
  1) break_margin_over_mean_prev_body
  2) break_margin_ratio
- 閾値の決め方（リーク防止）：
  - **verify（2024）の分布で上位20%点（p80）を計算し固定**
  - forward（2025）ではその固定閾値をそのまま適用
- 除外ルール：
  - (feature > threshold_p80) のとき entry を出さない

※ 20% は固定（最適化しない）

---

## 評価（Phase 2）
- D004比で
  - sum_pnl_pips が改善するか
  - trades が減りすぎないか
  - verify/forward が同方向か
- まずは “即死（0–3分）件数” が減っているかを確認

---

## 生成物
- `FX/results/family_D_momentum/D011a/`
- diagnostics（可能なら）：
  - `thresholds.json`（p80閾値の記録）
  - `early_loss_rate.csv`（早期SL率）

---

## 結果メモ（後で記入）
- p80閾値（verify基準）:
- 2024 verify: sum_pnl / trades:
- 2025 forward: sum_pnl / trades:
- early_loss件数/率の変化:


- p80閾値（verify基準）: feature=break_margin_over_mean_prev_body p80=3.242938775509479
- 2024 verify: sum_pnl_pips=-1163.999999999504, trades=948
- 2025 forward: sum_pnl_pips=-1291.999999999581, trades=798
- thresholds.json: FX/results/family_D_momentum/D011a/thresholds.json
- compare CSV（D009比）: FX/results/family_D_momentum/D011a/diagnostics/early_loss_compare_vs_D009_buy.csv
- 自己チェック（D009不変の参照用sha256）:
  - in_sample_2024/monthly.csv: 04d557a8731160a17663bfd5278b6c3dd0ec797041de16dd19aa4a175a2737a3
  - forward_2025/monthly.csv: d898acdaa17b0d9291c9f928a06ffd022a576912e25b2d67a9054d2a109c519e
