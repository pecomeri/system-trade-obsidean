---
id: D008
family: family_D_momentum
type: observation_variant

strategy: "[[D004_m1_momentum_burst_break_prev]]"
base_strategy: "[[D004_m1_momentum_burst_break_prev]]"

filters:
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
  - early_loss
  - survivor_bias
  - feature_analysis
---

# D008: D004 × Observe Early-Loss vs Survivor Features

## 目的
- D004（M1 Momentum Burst / Break Prev + Body Quality）を **完全に固定**したまま、
  「即死ゾーン（0–3分でSL到達）」と「生存ゾーン（20分以上）」のトレードを分け、
  トリガー周辺の特徴量に差があるかを観測する。
- 時間で exit を変える前に、**entry品質（構造）側の問題**として説明できる差分を探す。

---

## 位置づけ
- D008 は **観測用 variant**
- strategy / entry / SL/TP / exit ロジックは一切変更しない
- 結果は「即死回避フィルター」や「構造仮説（D008a）」を切る材料とする

---

## グループ定義（固定）
- early_loss（即死）：
  - holding_time_min ∈ [0, 3]
- survivor（生存）：
  - holding_time_min ≥ 20

※境界値は最適化しない（D007の観測に基づく固定）

---

## 比較したい特徴量（候補）
※存在する列はそのまま使い、無い場合は可能な範囲で再計算する

### トリガー強度
- burst_strength = body / mean_prev_body
- body_ratio（D004で使用済み）

### break_prev の余裕
- break_margin_pips（または価格差）：
  - buy: close - prev_high
  - sell: prev_low - close

### 直後の否定（可能なら）
- next_m1_is_opposite_body（次のM1が反対実体か）
- next_m1_body_ratio（次のM1の実体品質）

---

## 観測内容（必須）
- early_loss vs survivor の各特徴量について、
  - n
  - mean / median
  - p25 / p75
を比較表として出す。
- 可能なら差分（early - survivor）も併記する。

---

## 生成物
- `FX/results/family_D_momentum/D004/diagnostics/early_vs_survivor_features.csv`
- `FX/results/family_D_momentum/D004/diagnostics/early_vs_survivor_summary.md`
（D004の結果から観測する。D008としてbacktestを回す必要はない）

---

## 検証観点（このIDでやらないこと）
- 閾値を動かして最適化しない
- その場でフィルターを実装しない
- 「勝てるはず」と結論を急がない

---

## 次のアクション（D008完了後）
- verify / forward で一貫して差が出る特徴量があれば、
  - D008a: 即死回避フィルター仮説（固定条件）として切る
- 差が出なければ、
  - 「即死は特徴量で説明できない」可能性として stop

---

## メモ（後で記入）
- early_loss と survivor の差が大きい特徴量：
- verify/forwardで一貫する差：
- 次に切る仮説（最大2つ）：
