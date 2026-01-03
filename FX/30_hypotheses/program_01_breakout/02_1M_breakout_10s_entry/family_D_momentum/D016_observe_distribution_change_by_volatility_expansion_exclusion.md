---
id: D016
family: family_D_momentum
type: observe
title: D016_observe_distribution_change_by_volatility_expansion_exclusion

base_strategy: "[[D001_m1_momentum_burst_range_only]]"
dataset_source: "[[D009_observe_buysell_side_24h_baseline]]"
depends_on:
  - "[[D013_observe_tail_contribution_and_failure_modes]]"
  - "[[D014_observe_early_loss_classifiability]]"

timeframe_signal: M1
timeframe_exec: 10s

status: planned
result: TBD

tags:
  - fx
  - family_D_momentum
  - observe
  - distribution_shift
  - volatility
  - early_loss
  - meta
---

## 目的

D014 において、early_loss の一部が  
**immediate_volatility_expansion = expanded** に偏ることが確認された。

本観測（D016）では、  
`expanded` を除外した場合に  
**D009 母集団の損益分布がどのように変形するか**を観測する。

---

## 前提（母集団）

- 基本母集団：
  - `results/family_D_momentum/D009/`
  - BUY / SELL 両方
  - 24h

- 派生母集団：
  - D009 から `immediate_volatility_expansion = expanded` を除外

※ 分布が変わることを前提とする

---

## 観測仮説（問い）

- expanded を除外すると、
  1) early_loss 比率は減少するか
  2) tail dependency はどの程度変化するか
- ボラ拡張局面は、
  **損失側だけでなく利益側 tail にも寄与しているか**

---

## 定義（Contract）

### 除外条件

- immediate_volatility_expansion = expanded
- 定義は [[D014_observe_early_loss_classifiability]] に従う

---

## 観測項目（Metrics）

### 観測1：early_loss 比率の変化

- verify / forward
- BUY / SELL
- baseline vs modified

---

### 観測2：Tail share の変化

- D013 と同一定義
- 上位 / 下位 tail の寄与率

---

### 観測3：分布形の歪み

- 分散・集中度の変化（定性的）
- 「高分散戦略」としての性質が維持されているか

---

## 期待される結論パターン（事前固定）

### ケースA：tail も大きく削れる

- expanded は D系の本質的ゾーン
- 除外は戦略の性質を変える

### ケースB：early_loss だけが主に減る

- expanded は主に failure 側
- 切り分け余地あり

### ケースC：変化が小さい

- expanded は支配的ではない

---

## 出力物（Artifacts）

- `results/family_D_momentum/D016/`
  - summary_verify.csv
  - summary_forward.csv
  - tail_compare.csv
  - README.md

---

## 注意

- 本観測は「分布変形の確認」に限定
- 即座にフィルター設計へ接続しない

---

## status / result 記録（完了時に更新）

- status: observed
- result:
  - distribution_shift: TBD
  - early_loss_delta: TBD
  - tail_dependency_change: TBD
  - notes: TBD
