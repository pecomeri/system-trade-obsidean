---
id: exhaustion_ratio_top20_buy
type: filter
scope: entry_quality
status: draft
tags:
  - fx
  - filter
  - exhaustion
  - ratio
  - buy
---

# exhaustion_ratio_top20_buy

## 目的
- 「強すぎる抜け（exhaustion）」になっている BUY シグナルを除外し、
  即死（0–3分でSL到達）に偏るケースを減らす。

## 背景（観測）
- D009/D010の観測により、BUY側で
  - break_margin_ratio
  - break_margin_over_mean_prev_body
  が early_loss（0–3分）に偏る方向で、verify/forward で符号が一致した。

## 対象
- BUY シグナルのみ（SELLには適用しない）

## 定義（Contract）
- 使用する特徴量（優先順）：
  1) `break_margin_over_mean_prev_body`
  2) `break_margin_ratio`（1が欠損・計算不能の場合のみフォールバック）
- 除外ルール：
  - verify（2024）データの分布から p80（上位20%点）を計算し、その閾値を固定する
  - forward（2025）では、その固定閾値をそのまま適用する（リーク防止）
  - `feature > threshold_p80` のとき、新規エントリーを禁止する
- 除外率：
  - **20%固定**（最適化しない）

## メモ（実装上の注意）
- 「p80の算出」は verify のみから行う（forwardを混ぜない）
- threshold は実行結果に `thresholds.json` として保存し、再現可能にする
- body==0、mean_prev_body==0 等で ratio が NaN になる場合は、NaNは除外しない（=フィルター不適用）を基本とする
  - 例外を作る場合は別フィルターとして切る

## 期待される効果（評価指標）
- early_loss（0–3分）件数/率の低下
- trades の過剰減少が起きないこと
- sum_pnl_pips の改善（verify/forward 同方向）

## 関連
- 観測根拠：
  - D009: strong_break_exhaustion（BUY/SELL）
  - D010: exhaustion_ratio_consistent
- 適用先例：
  - `[[D011a_exhaustion_ratio_buy]]`

## 最新のp80（verify基準）
- break_margin_over_mean_prev_body p80=3.242938775509479（`FX/results/family_D_momentum/D011a/thresholds.json`）
