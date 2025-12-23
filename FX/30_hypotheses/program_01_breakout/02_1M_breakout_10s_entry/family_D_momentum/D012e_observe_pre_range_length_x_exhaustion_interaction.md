---
id: D012e
family: family_D_momentum
type: observe
title: D012e_observe_pre_range_length_x_exhaustion_interaction

base_strategy: "[[D001_m1_momentum_burst_range_only]]"
dataset_source: "[[D009_observe_sell_side]]"
depends_on:
  - "[[D010_observe_exhaustion_ratios]]"
  - "[[D012a_observe_pre_range_length]]"

timeframe_signal: M1
timeframe_exec: 10s

status: observed
result: interaction_effect_mixed

tags:
  - fx
  - family_D_momentum
  - observe
  - precondition
  - range
  - exhaustion
  - interaction
---

## 目的

D012a により確認された  
**「ブレイク前のレンジ蓄積（pre_range_length）が entry 品質を分ける」**  
という事実を前提として、

D009 母集団における

- strong_break_exhaustion（D010 文脈）
- pre_range_length（D012a 文脈）

の **相互作用（interaction）** を観測する。

特に、

- D011 で「単純な exhaustion 除外が overfiltering だった理由」
- strong_break が  
  - いつ「毒」になり  
  - いつ「最大利益源」になるのか  

を **構造的に分解できるか**を確認する。

重要：
- 本観測は **filter を作るためではない**
- 勝たせに行かない（現象理解のみ）
- 最適化しない
- verify / forward の符号一致（方向一致）を最重視する

---

## 観測仮説（問い）

strong_break_exhaustion は常に悪なのではなく、

> **「ブレイク前のレンジ蓄積（pre_range_length）」との組み合わせによって  
>  early_loss 優勢にも、利益源にもなり得るのではないか？**

という **条件付き現象**なのではないか。

---

## 前提（母集団）

- 母集団：
  - `results/family_D_momentum/D009/`
  - BUY / SELL 両方
  - 24h（W1_only ではない）
- D009 は observe ID であり、strategy は一切変更しない
- D012e は D009 の結果（trades）に特徴量を **付与して観測**するのみ

---

## 定義（Contract）

### 1) pre_range_length（固定・D012a準拠）

- lookback N = 20
- verify 固定の分位点：
  - q33 = 0.0
  - q66 = 3.0

**カテゴリ定義**

- short_range:
  - pre_range_length <= 0
- mid_range:
  - 1 <= pre_range_length <= 3
- long_range:
  - pre_range_length >= 4

※ 本観測では **short / long を主に使用**する  
※ mid は補助的に参照してよいが、主結論は short vs long で出す

---

### 2) exhaustion_ratio（D010 準拠）

- exhaustion_ratio の定義・計算方法は [[D010_observe_exhaustion_ratios]] と同一
- exhaustion の分類：
  - verify 期間のみで p80（上位20%）を算出し固定
  - weak_break: exhaustion_ratio < p80
  - strong_break: exhaustion_ratio >= p80
- forward では **同じ p80 を使用**（再計算禁止）

重要：
- strong_break を除外しない
- 分類して観測するのみ

---

### 3) early_loss の定義

- [[D007_observe_holding_time]] と同一定義
- early_loss:
  - 0 <= holding_time_min <= 3

---

## 観測セル（最低限）

### 基本セル（必須）

2 × 2 の組み合わせを主観測対象とする：

| pre_range | exhaustion |
|---------|------------|
| short   | weak       |
| short   | strong     |
| long    | weak       |
| long    | strong     |

※ BUY / SELL 別に必ず分けて観測する  
※ all（BUY+SELL）は補助

---

## 観測項目（Metrics）

### 主指標

- early_loss_rate（セル別）
- holding_time 分布（セル別）

### 補助指標

- holding_time の中央値・分散
- exhaustion_ratio の分布（セル内）

### 注意

- PnL は主指標にしない（参考としてのみ）
- 「良い / 悪い」ではなく
  - 偏り
  - 分岐
  - 役割の違い
  を言語化する

---

## 実験設計（Leak回避）

- pre_range_length の閾値は D012a から固定
- exhaustion p80 は verify で固定
- forward では **一切再計算しない**
- 除外フィルターは作らない

---

## 期待される結論パターン（事前固定）

### ケースA：相互作用が見えない

- strong_break は依然として高分散であり、
  pre_range_length を加えても分解できない
- D011 の overfiltering は不可避だった
- 次：
  - D004 / D009 を「高分散前提の戦略」として扱う設計へ

### ケースB：相互作用が明確に見える

例：
- short × strong:
  - early_loss が突出（罠）
- long × strong:
  - early_loss は高くない／生存側の伸びが大きい（利益源）

→ strong_break は
**「レンジ蓄積がある場合にのみ許容される現象」**
である可能性が浮上

次：
- family 分離（F系：range-compression breakout）の検討
- ただし、この時点でも **filter 化はしない**

---

## 出力物（Artifacts）

- verify:
  - セル別 trades / early_loss_rate / holding_time summary（BUY/SELL別）
  - exhaustion p80
- forward:
  - 同一セル定義での集計（符号一致確認）
- 図表：
  - early_loss_rate（セル別バー）
  - holding_time 分布（セル別ヒストグラム）

---

## status / result 記録（完了時に更新）

- status: observed
- result:
  - interaction_effect: mixed
  - key_pattern:
      - buy: strong が short/long とも early_loss_rate 高め（verify/forward一致）
      - sell: long×strong は early_loss_rate 低め（verify/forward一致）
      - sell: short×strong は verify/forward で符号が一致せず
  - affected_metric:
      - early_loss_rate
      - holding_time_distribution
  - thresholds:
      - pre_range_length:
          - q33: 0
          - q66: 3
      - exhaustion_p80:
          - buy: 3.2409795918349475
          - sell: 2.965517241378161
  - notes:
      - dataset_source = D009 (24h, allow_sell=true)
      - No filtering or optimization applied

---

## D012e観測メモ（事実：D009から集計）

### 生成物
- `FX/results/family_D_momentum/D012e/summary_verify.csv`
- `FX/results/family_D_momentum/D012e/summary_forward.csv`
- `FX/results/family_D_momentum/D012e/thresholds.json`
- `FX/results/family_D_momentum/D012e/plots/*.png`

### 閾値（固定）
- pre_range_length（D012a固定）: q33=0.0 / q66=3.0（short<=0 / mid=1..3 / long>=4）
- exhaustion p80（verify固定）:
  - buy: 3.2409795918349475
  - sell: 2.965517241378161
- exhaustion_ratio: break_margin_over_mean_prev_body（fallback=break_margin_ratio）

### early_loss_rate（2×2, short/long × weak/strong）
※ early_loss: 0 <= holding_time_min <= 3

#### BUY
- verify:
  - short/weak 0.0542 (n=295)
  - short/strong 0.1000 (n=90)
  - long/weak 0.0332 (n=241)
  - long/strong 0.1282 (n=39)
- forward:
  - short/weak 0.0620 (n=242)
  - short/strong 0.1695 (n=59)
  - long/weak 0.0520 (n=173)
  - long/strong 0.0952 (n=21)

#### SELL
- verify:
  - short/weak 0.2169 (n=189)
  - short/strong 0.1964 (n=56)
  - long/weak 0.1469 (n=143)
  - long/strong 0.1364 (n=22)
- forward:
  - short/weak 0.1467 (n=259)
  - short/strong 0.2909 (n=55)
  - long/weak 0.1056 (n=180)
  - long/strong 0.0741 (n=27)

### holding_time 中央値（min, short/long × weak/strong）
- BUY: verify は long/strong が高めだが、forward は逆（方向不一致）
- SELL: long/strong は forward で低めだが、verify は差が小さく一貫しない

## 結論（事実）
- BUY は short/long いずれも strong が early_loss_rate 高めで、verify/forward で一致。
- SELL は long×strong が early_loss_rate 低めで一致。short×strong は verify/forward で不一致。
- holding_time 分布の方向性は不安定で、主結論は early_loss_rate に限定する。



status: observed
result:
  interaction_effect: yes
  key_pattern:
    buy:
      - strong_break increases early_loss regardless of pre_range_length
    sell:
      - long_range × strong_break shows lower early_loss_rate
      - short_range × strong_break is unstable (no direction consistency)
  affected_metric:
    - early_loss_rate
  thresholds:
    pre_range_length:
      q33: 0
      q66: 3
    exhaustion_p80:
      buy: 3.2409795918349475
      sell: 2.965517241378161
notes:
  - BUY/SELL show asymmetric interaction with exhaustion
  - holding_time was unstable and not used for main conclusion
  - No filtering or optimization applied



## 終了メモ

D012a → D012e により、
momentum burst における early_loss 優勢は
pre_range_length と exhaustion の相互作用で部分的に分解可能であることが確認された。

特に、
- BUY では strong_break_exhaustion は一貫して early_loss を増加させる
- SELL では long_range に限り strong_break が毒ではない可能性が示唆された

本観測は、family 分離（F系：range-compression breakout）検討の十分条件を満たしたと判断し、
これ以上の observe は行わず、ここでクローズする。
