---
id: D010
family: family_D_momentum
type: observation_variant
strategy: "[[D004_m1_momentum_burst_break_prev]]"
base_strategy: "[[D004_m1_momentum_burst_break_prev]]"
filters:
  - "[[h1_trend_up]]"
  - "[[h1_trend_down]]"
regimes:
  - none
timeframe_signal: M1
timeframe_exec: 10s
status: observed
result: exhaustion_ratio_consistent
tags:
  - fx
  - family_D_momentum
  - observation
  - exhaustion
  - ratio_normalization
  - buy_sell
---

# D010: Observe Exhaustion Ratios (Normalize Break Strength)

## 目的
- D009で観測された「strong_break_exhaustion（強い抜けほど即死しやすい）」を、
  **銘柄スケール依存を除いた比率**で再検証する。
- break_margin_pips や body の“絶対値”ではなく、
  **相対的な行き過ぎ（exhaustion）**として説明できる形に落とす。

---

## 位置づけ
- D010 は観測用 variant（戦略変更なし）
- D009の次段：比率（ratio）にしても同じ差が出るかを確認
- 結果が再現すれば D011（回避フィルター）へ進む根拠になる

---

## 対象
- D009 の結果（BUY/SELL両側、verify/forward）を優先して観測する
- 可能なら D006（24h観測）でも同様に確認できると強い

---

## グループ定義（固定）
- holding_time_min = (exit_time - entry_time).total_seconds()/60
- early_loss: 0 <= holding_time_min <= 3
- survivor: holding_time_min >= 20

---

## 主要な比率特徴量（必須）
※pips換算が可能ならpips、難しければ価格差でも良い（ただし同一単位で割る）

### 1) break_margin_ratio
- break_margin_ratio = break_margin / body
  - buy: break_margin = close - prev_high
  - sell: break_margin = prev_low - close
  - body = abs(close - open)
- 意味：実体に対して「どれだけ抜けたか」

### 2) burst_strength（再掲）
- burst_strength = body / mean_prev_body
- 意味：過去平均に対する実体の加速度（すでにある指標）

---

## 追加の比率（任意：データがあれば）
### 3) break_margin_over_mean_prev_body
- break_margin_over_mean_prev_body = break_margin / mean_prev_body
- 意味：通常レンジに対する抜け幅の相対度

### 4) break_margin_over_atr（もしATRが容易なら）
- break_margin_over_atr = break_margin / ATR
- ※最適化禁止。ATR期間は既存固定値があればそれを流用

---

## 観測内容（必須）
- period（verify/forward）× side（BUY/SELL）× group（early/survivor）で
  - n
  - mean / median / p25 / p75
を比較する。
- delta_median = median_survivor - median_early を出す。
- verify/forwardで符号が一致する特徴量を抽出する（事実のみ）。

---

## 生成物
- `FX/results/family_D_momentum/D009/diagnostics/exhaustion_ratios_by_side.csv`
- `FX/results/family_D_momentum/D009/diagnostics/exhaustion_ratios_summary.md`

（入力をD009に寄せるため、出力先もD009配下に置く）

---

## 次のアクション（D010完了後）
- verify/forward・BUY/SELLで符号一致する ratio が見つかった場合のみ、
  - D011: exhaustion_avoid_filter（回避フィルター仮説）へ進む
- 一致しなければ、D009の差はスケール依存の可能性として stop / hold

---

## メモ（後で記入）
- 一貫して差が出た ratio:
- 次に切る仮説（最大2つ）:

### D010観測メモ（事実：D009から集計）
- 生成物:
  - `FX/results/family_D_momentum/D009/diagnostics/exhaustion_ratios_by_side.csv`
  - `FX/results/family_D_momentum/D009/diagnostics/exhaustion_ratios_summary.md`
- 一貫して差が出た ratio（verify/forwardで sign_match=一致、BUY/SELL別・最大2つ）:
  - BUY:
    - break_margin_over_mean_prev_body（sign_match=一致）
    - break_margin_ratio（sign_match=一致）
  - SELL:
    - break_margin_over_mean_prev_body（sign_match=一致）
    - break_margin_ratio（sign_match=一致）
- 次に切る仮説候補名（最大2つ、名前だけ）:
  - D011a_exhaustion_ratio_buy
  - D011b_exhaustion_ratio_sell


## 結論（事実）
- break_margin_ratio / break_margin_over_mean_prev_body は、BUY/SELLともに、verify/forwardで delta_median の符号が一致した。
- したがって「強い抜け（exhaustion）が即死と関連する」傾向は、pips絶対値ではなく ratio 正規化でも再現した。

