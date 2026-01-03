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

status: observed
result:
  distribution_shift: true
  early_loss_effect:
    buy: decrease (verify/forward 方向一致)
    sell: decrease (verify/forward 方向一致)
  tail_dependency: mixed (buy 側は強まり、sell 側は弱まる方向; 符号反転なし)
  filter_suitability: unknown

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

## 結果まとめ

- early_loss は BUY/SELL とも verify/forward で低下方向（方向一致）。
- tail share は BUY 側で振れ幅が拡大、SELL 側で縮小する傾向。符号反転は確認されない。
- 分布の集中度・歪みは BUY 側で強まり、SELL 側で弱まる方向の混在。
- 「危険ゾーン＝利益ゾーン」の同居構造は維持されるが、形は変形している。

---

## 考察（限定）

- early_loss は一様に低下方向だが、尾部構造は一様に弱まらない。
- tail dependency は BUY 側で強まり、SELL 側で弱まる方向のため、全体としては混在と判断。
- 分布は一部で不安定化（BUY の尾部寄与が拡大）し、安定化とは言い切れない。
- 示唆はあるが結論は保留とし、設計・採用には接続しない。

---

## status / result 記録（完了時に更新）

- status: observed
- result:
  - distribution_shift: true
  - early_loss_delta: BUY/SELL とも verify/forward で低下方向（方向一致）
  - tail_dependency_change: mixed（BUY は強まり、SELL は弱まる方向。符号反転なし）
  - notes:
      - immediate_volatility_expansion = expanded を除外した派生母集団で観測
      - 分布の集中度・歪みは BUY 側で強まり、SELL 側で弱まる
      - filter_suitability は unknown と記録
      - outputs = FX/results/family_D_momentum/D016 (summary_verify.csv / summary_forward.csv / tail_compare.csv / README.md)
