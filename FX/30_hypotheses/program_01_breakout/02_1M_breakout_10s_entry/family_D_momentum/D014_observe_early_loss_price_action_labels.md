---
id: D014
family: family_D_momentum
type: observe
title: D014_observe_early_loss_classifiability

base_strategy: "[[D001_m1_momentum_burst_range_only]]"
dataset_source: "[[D009_observe_buysell_side_24h_baseline]]"
depends_on:
  - "[[D007_observe_holding_time]]"
  - "[[D013_observe_tail_contribution_and_failure_modes]]"

timeframe_signal: M1
timeframe_exec: 10s

status: observed

result:
  classifiability: partial
  stable_axes:
    initial_mae_stage:
      note: large 側で early_loss 比率が高い
    early_retrace_presence:
      note: no_retrace 側で early_loss 比率が高い
    immediate_volatility_expansion:
      note: expanded 側で early_loss 比率が高い
    time_to_failure_stage:
      note: early_loss 定義内のため比率 1.0 で固定（整合チェック用途）
  limitations:
    immediate_directional_alignment: comparison_not_possible
  notes:
    dataset_source: D009 (24h, allow_sell=true)
    entry_population: unchanged
    filtering_or_optimization: none
    outputs: results/family_D_momentum/D014


tags:
  - fx
  - family_D_momentum
  - observe
  - early_loss
  - classification
  - failure_mode
  - meta
---

## 目的

D013 において、D系（family_D_momentum）が  
**高分散・尾部依存（tail dependency）を持つ構造**であることが観測された。

本観測（D014）では、その構造の一部である  
**early_loss（早期損失）が、entry 後の価格挙動によって  
「分類可能な対象かどうか」**を確認する。

重要：
- 本観測は early_loss を減らすことを目的としない
- フィルター化・最適化・戦略改善には接続しない
- 「分類可能性が存在するかどうか」の有無のみを扱う

---

## 前提（母集団）

- 母集団：
  - `results/family_D_momentum/D009/`
  - BUY / SELL 両方
  - 24h（W1_only ではない）

※ D014 は D009 の trades に対し、  
   **entry 後にラベルを付与して集計するのみ**  
※ entry / exit / SL / TP / 時間帯 / direction は一切変更しない

---

## 観測仮説（問い）

- early_loss は、D系母集団の中で
  **一様な失敗（ノイズ）なのか**、
  それとも **価格挙動に基づいて分類できる構造を持つのか**。
- もし分類可能な軸が存在するとすれば、
  それは verify / forward の両方で **符号一致**するか。

---

## 定義（Contract）

### 1) early_loss

- 定義は [[D007_observe_holding_time]] と同一
- early_loss: 0 <= holding_time_min <= 3

---

### 2) 観測ラベルの原則

- すべて **entry 後にのみ確定**するラベルであること
- 価格挙動のみを使用する（出来高・板・外生情報なし）
- entry 可否を変える条件（フィルター）にならない
- 新しい概念・新語を導入しない
- 閾値最適化・分位探索を行わない

---

## 観測項目（Labels）

### 観測1：Immediate Directional Alignment

entry 直後の価格が、  
期待方向に動いたか、逆方向に動いたかを分類する。

- align：有利方向の価格更新が先に発生
- counter：不利方向の価格更新が先に発生
- flat：どちらも発生しない

※ 価格更新の定義は既存 backtest_core に従う

---

### 観測2：Initial MAE Stage

entry から early_loss 確定までの  
**最大逆行幅（MAE）**を段階化する。

- small
- medium
- large

※ 境界は既存スケールを流用し固定  
※ 分位（pXX）探索は禁止

---

### 観測3：Early Retracement Presence

early_loss に至るまでに、  
**一度でも有利方向への戻りが発生したか**を観測する。

- with_retrace
- no_retrace

※ 戻り幅・割合は使用しない

---

### 観測4：Immediate Volatility Expansion

entry 直後のボラティリティが、
直前状態と比較して拡張したかどうかを分類する。

- expanded
- not_expanded

※ レンジ比較は既存定義を使用  
※ 閾値最適化は禁止

---

### 観測5：Time-to-Failure Stage

entry から early_loss 確定までの時間段階。

- very_fast
- fast
- slow

※ 段階境界は固定（D007 系と整合）

---

## 観測方法

- 各ラベルごとに
  - early_loss の **件数比率**
- verify / forward を分離して算出
- 数値の大小ではなく、
  **「どちら側に偏るか（符号）」のみを評価**

---

## 期待される結論パターン（事前固定）

### ケースA：分類可能性あり

- 特定ラベル側に early_loss が偏る
- verify / forward で符号一致
- early_loss は heterogeneous な構造を持つと判断

### ケースB：期間依存

- verify のみ偏りあり
- forward で消失
- 偶然・非安定と判断

### ケースC：分類不能

- すべてのラベルで偏りなし
- early_loss は D009 内では一様ノイズと判断

---

## 出力物（Artifacts）

- `results/family_D_momentum/D014/`
  - summary_verify.csv
  - summary_forward.csv
  - README.md（定義・リーク回避・使用列）
  - thresholds.json（使用した固定境界の記録）

---

## 次の接続（このノードでは実施しない）

- 本結果を用いたフィルター設計
- early_loss 削減ロジック
- 合成設計への直接反映

※ それらは D014 の外側でのみ検討する

---

## 考察（限定）
- initial_mae_stage は large 側で early_loss 比率が高く、verify / forward で方向一致。
- early_retrace_presence は no_retrace 側で early_loss 比率が高く、verify / forward で方向一致。
- immediate_volatility_expansion は expanded 側で early_loss 比率が高く、verify / forward で方向一致。
- time_to_failure_stage は very_fast / fast が early_loss 定義内のため比率が1.0固定で、分類軸というより整合チェックに留まる。
- immediate_directional_alignment は counter に偏っており、比較に必要な他カテゴリが不足。

## 結論（固定）
- early_loss は一様ノイズではなく、entry後ラベルの一部で **偏り（符号一致）が観測される**ため、分類可能性は「部分的にあり」と記録する。
- ただし本結果は **分類可能性の有無の確認**に留め、フィルター化や最適化には接続しない。

## status / result 記録（完了時に更新）

- status: observed
- result:
  - classifiability: partial (sign-consistent axes exist)
  - stable_axes:
      - axis_name: initial_mae_stage (large側でearly_loss比率が高い)
      - axis_name: early_retrace_presence (no_retrace側でearly_loss比率が高い)
      - axis_name: immediate_volatility_expansion (expanded側でearly_loss比率が高い)
      - axis_name: time_to_failure_stage (very_fast/fastはearly_loss定義内のため比率1.0で固定)
  - notes:
      - dataset_source = D009 (24h, allow_sell=true)
      - Entry population unchanged
      - No filtering or optimization applied
      - immediate_directional_alignment は counter のみで比較不能
      - outputs = results/family_D_momentum/D014 (summary_verify.csv / summary_forward.csv / README.md)
