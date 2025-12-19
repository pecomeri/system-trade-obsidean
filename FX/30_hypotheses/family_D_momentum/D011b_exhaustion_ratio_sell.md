---
id: D011b
family: family_D_momentum
type: filter_variant
strategy: "[[D004_m1_momentum_burst_break_prev]]"
base_strategy: "[[D004_m1_momentum_burst_break_prev]]"
filters:
  - "[[h1_trend_down]]"
  - "[[exhaustion_ratio_top20_sell]]"
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
  - sell
---

# D011b: D004 × Exhaustion Ratio Top20 Exclude (SELL)

## 目的
- D010で確認した exhaustion ratio の再現性（SELL側でも一致）を踏まえ、
  **exhaustionが強すぎるSELLシグナル**を除外して即死（0–3分SL）を減らせるか検証する。

---

## 仮説
- SELL側でも、exhaustion ratio が極端に高いケースは early_loss に偏る。
- 上位20%除外で期待値が改善する可能性がある。

---

## フィルター定義（固定）
- 対象：SELL シグナルのみ
- 使用特徴量（優先順）：
  1) break_margin_over_mean_prev_body
  2) break_margin_ratio
- 閾値の決め方：
  - verify（2024）で p80 を計算して固定
  - forward（2025）では固定値を適用
- 除外：
  - (feature > threshold_p80) のとき entry を出さない

---

## 評価（Phase 2）
- D009（sell側）比で
  - sum_pnl_pips 改善
  - trades 減少が過大でない
  - verify/forward の同方向
- early_loss率の低下を確認

---

## 生成物
- `FX/results/family_D_momentum/D011b/`

---

## 結果メモ（後で記入）
- p80閾値（verify基準）:
- 2024 verify: sum_pnl / trades:
- 2025 forward: sum_pnl / trades:
- early_loss_rate の D009比:
- thresholds.json:
- compare CSV:

- p80閾値（verify基準）: feature=break_margin_over_mean_prev_body p80=2.972413793102057
- 2024 verify: sum_pnl_pips=-511.99999999960824, trades=753
- 2025 forward: sum_pnl_pips=-771.9999999995537, trades=856
- thresholds.json: FX/results/family_D_momentum/D011b/thresholds.json
- compare CSV（D009比）: FX/results/family_D_momentum/D011b/diagnostics/early_loss_compare_vs_D009_sell.csv
- 自己チェック（D009不変の参照用sha256）:
  - in_sample_2024/monthly.csv: 04d557a8731160a17663bfd5278b6c3dd0ec797041de16dd19aa4a175a2737a3
  - forward_2025/monthly.csv: d898acdaa17b0d9291c9f928a06ffd022a576912e25b2d67a9054d2a109c519e
