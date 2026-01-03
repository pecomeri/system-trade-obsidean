---
id: D015
family: family_D_momentum
type: observe
title: D015_observe_distribution_change_by_no_retrace_exclusion
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
    SELL: decreased_consistent
    BUY: mixed_direction
  tail_dependency:
    weakened: false
    shape_change: more_concentrated_and_unstable
  filter_suitability: low
tags:
  - fx
  - family_D_momentum
  - observe
  - distribution_shift
  - early_loss
  - retrace
  - meta
---

## 目的

D014 において、early_loss の一部が  
**early_retrace_presence = no_retrace** に偏ることが確認された。

本観測（D015）では、  
`no_retrace` を **明示的に除外した場合に、  
D009 母集団の損益分布がどのように変形するか**を観測する。

重要：
- 本観測は「勝たせる」ことを目的としない
- 除外操作により **分布が変わることを前提**とする
- フィルター化の是非を結論づけない

---

## 前提（母集団）

- 基本母集団：
  - `results/family_D_momentum/D009/`
  - BUY / SELL 両方
  - 24h

- 派生母集団：
  - D009 から `early_retrace_presence = no_retrace` を除外したもの

※ D015 は **派生母集団を明示的に扱う**
※ D009 本体は一切変更しない

---

## 観測仮説（問い）

- `no_retrace` を除外すると、
  1) early_loss 比率はどの程度変化するか
  2) D013 で確認された **tail dependency（尾部依存）** は弱まるか、維持されるか
- 除外操作は、
  **「損失だけでなく利益側の尾部」も同時に削っているか**

---

## 定義（Contract）

### 1) 除外条件

- early_retrace_presence = no_retrace
- 定義は [[D014_observe_early_loss_classifiability]] と同一

### 2) 比較対象

- baseline：D009（除外なし）
- modified：D009 - no_retrace

---

## 観測項目（Metrics）

### 観測1：early_loss 比率の変化

- verify / forward 別
- BUY / SELL 別
- baseline vs modified の比較

---

### 観測2：Tail share の変化

- D013 と同一定義
- 上位 / 下位 tail の損益寄与率
- baseline vs modified の比較

---

### 観測3：分布形の変形

- 損益分布の歪度・集中度の変化（定性的）
- 「危険ゾーン＝利益ゾーン」の同居構造が維持されているか

---

## 期待される結論パターン（事前固定）

### ケースA：early_loss は減るが、tail も大きく削れる

- no_retrace は損失と利益の両方を生む危険ゾーン
- 除外は overfiltering に近い

### ケースB：early_loss は減り、tail は概ね維持される

- no_retrace は主に failure mode
- 構造的に切り分け可能な可能性

### ケースC：ほとんど変化しない

- no_retrace は支配的要因ではない

---

## 出力物（Artifacts）

- `results/family_D_momentum/D015/`
  - summary_verify.csv
  - summary_forward.csv
  - tail_compare.csv
  - README.md（定義・注意点）

---

## 注意

- 本結果を即座にフィルター設計に接続しない
- D009 本体の評価・扱いは変更しない

---

## 結果と考察（日本語追記）

結果：
- early_loss は SELL で verify/forward とも低下、BUY は verify で微増・forward で低下という混在。
- tail share は modified 側で振れ幅が大きくなり、verify/forward・BUY/SELL で符号反転も発生。
- 「危険ゾーン＝利益ゾーン」の同居構造は維持され、尾部依存は弱まらない。

考察：
- no_retrace 除外は母集団の形状を大きく歪めるが、尾部構造そのものは崩れない。
- 早期損失だけが一様に削られる形ではなく、利益側の尾部も同時に変形している。
- overfiltering の兆候は明確ではない（尾部が一方向に縮小する挙動ではない）。

---

## status / result 記録（完了時に更新）

- status: observed
- result:
  - distribution_shift: tail share ratios swing and become more extreme in modified (concentration/skew stronger; sign flips in verify/forward, BUY/SELL)
  - early_loss_delta: SELL decreases in verify/forward; BUY mixed (verify slight up, forward down)
  - tail_dependency_change: not weakened; tail shares remain large and sign-flipped in modified
  - notes:
      - no_retrace exclusion treated as derived population only
      - 危険ゾーン＝利益ゾーンの同居は崩れていない（両側の尾部が残る）
      - overfiltering の兆候は明確ではない（尾部が一様に縮小していない）
      - outputs = FX/results/family_D_momentum/D015 (summary_verify.csv / summary_forward.csv / tail_compare.csv / README.md)
