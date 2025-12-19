---
id: D012a
family: family_D_momentum
type: observe
title: D012a_observe_pre_range_length
base_strategy: "[[D001_m1_momentum_burst_range_only]]"
dataset_source: "[[D009_observe_buysell_side_24h_baseline]]"
timeframe_signal: M1
timeframe_exec: 10s
status: observed
result: pre_range_length_effect_mixed
tags:
  - fx
  - family_D_momentum
  - observe
  - precondition
  - range
---

## 目的

D系の基盤観測（D009：BUY/SELL 両方、24h）を母集団として、
「ブレイク前のレンジ蓄積（pre_range_length）」が、

- early_loss 優勢（D007文脈）
- strong_break_exhaustion（D009〜D011文脈）

の **構造分解軸になり得るか**を観測する。

重要：
- 勝たせに行かない（現象理解）
- 最適化しない（閾値探索しない）
- verify / forward の符号一致（方向の一致）を最重視
- strategy を後から書き換えない
- 本観測は **D004（W1_only / allow_sell=False）を母集団にしない**
  - 母集団は **D009（24h、BUY/SELLあり）** とする

---

## 観測仮説（問い）

同じ（D009母集団の）momentum burst でも、
直前に価格が「閉じていた時間（pre_range_length）」が長い／短いで、

- early_loss 率
- holding_time 分布

に **一貫した偏り（verify / forward で方向一致）が出る**のではないか。

※「伸びるか？」ではなく **構成比・分布が変わるか？** を問う。

---

## 前提（母集団の定義）

本観測の母集団は、以下を満たすトレード集合とする：

- 結果参照元：`results/family_D_momentum/D009/`
- allow_sell: true（BUY/SELL 両方）
- 時間帯：24h（W1_only に限定しない）

※ D009は観測IDであり、strategyそのものを書き換えない。
※ D012a は D009 の結果（trades）に特徴量を「付与」して分析する。

---

## 定義（Contract）

### 1) pre_range_length（M1）

- entry 時刻の直前から遡って計算する（M1）
- ルールは **close 基準で統一**する（ヒゲは無視）
- lookback N は固定（最適化しない）
  - 初期値：N = 20
  - 変更する場合は「理由付きで固定値を変更」するが、探索はしない

**pre_range_length の定義案（close基準・簡易）**

- 直近 N 本の範囲で、
  - 「終値で直近高値更新しない」
  - かつ「終値で直近安値更新しない」
- が連続して成立していた **連続本数**を `pre_range_length` とする

（実装の都合で厳密な算出手順が変わっても、概念は上記で固定）

### 2) 分類（verify 固定）

- verify 期間で `pre_range_length` を集計し、
- **3分位**でカテゴリを固定する（forward では再計算しない）

カテゴリ：
- short_range
- mid_range
- long_range

※ 分位点の具体値（q33, q66）は結果ノートに必ず記録する。

---

## 観測項目（Metrics）

### 主指標（必須）

- early_loss 率  
  - 定義は [[D007_observe_holding_time]] と同一にする（例：holding_time <= 3min）
- holding_time 分布（カテゴリ別、BUY/SELL別も併記）

### 補助指標（参考）

- exhaustion_ratio 分布（D010と同一定義）
- survivor 側の holding_time 分散（カテゴリ別）

### 注意

- PnL は主目的ではないため **参考**としてのみ併記する
- 解釈は「改善/悪化」ではなく **偏り・分岐・構造の有無**で行う
- BUY/SELL を混ぜた集計だけで結論を出さず、BUY/SELL別も必ず見る

---

## 実験設計（Leak回避）

- D009 の trades を母集団とし、
  - 各トレードに `pre_range_length` を付与
  - verify で分類閾値（3分位）を確定
  - forward では同じ閾値で分類して観測する
- 除外フィルターは作らない（observeのみ）

---

## 期待される結論パターン（事前固定）

### ケースA：差が出ない（no_effect）

- pre_range_length は D009母集団において構造分解軸にならない
- 「レンジが長い方が伸びる」という直感は、少なくとも本定義では支持されない
- 次アクション：
  - D012a を observed/no_effect でクローズ
  - 別軸の observe へ（例：レンジ“tightness”など）

### ケースB：明確な差が出る（effect_yes）

- momentum burst は precondition（レンジ蓄積）で性質が分岐する可能性がある
- 次アクション候補：
  - range × exhaustion の相互作用観測（次の observe）
  - family 分離（F系）検討（ただしこの時点ではまだ filter化しない）

---

## 出力物（Artifacts）

- verify:
  - pre_range_length の分位点（q33, q66）
  - カテゴリ別：trades, early_loss_rate, holding_time summary（BUY/SELL別）
- forward:
  - 同じ分位点で分類したカテゴリ別集計（符号一致確認）
- 図表：
  - holding_time ヒストグラム（カテゴリ別）
  - early_loss_rate の棒グラフ（カテゴリ別）
  - 参考：exhaustion_ratio 分布（カテゴリ別）

---

## status / result 記録（完了時に更新）

- status: observed
- result:
  - pre_range_length_effect: mixed
  - affected_metric:
    - early_loss_rate（全体/SELLで方向一致、BUYはverifyが非単調）
    - holding_time_distribution（中央値は短→長で増加傾向）
  - verify_thresholds:
    - q33: 0.0
    - q66: 3.0
  - notes:
    - dataset_source = D009 (24h, allow_sell=true)
    - No filtering or optimization applied

---

## D012a観測メモ（事実：D009から集計）

### 生成物
- `FX/results/family_D_momentum/D012a/summary_verify.csv`
- `FX/results/family_D_momentum/D012a/summary_forward.csv`
- `FX/results/family_D_momentum/D012a/thresholds.json`
- `FX/results/family_D_momentum/D012a/plots/*.png`

### 閾値（verify 固定）
- lookback N = 20
- q33=0.0 / q66=3.0
- 範囲分類:
  - short: pre_range_length <= 0
  - mid: 1 <= pre_range_length <= 3
  - long: pre_range_length >= 4

### early_loss_rate（short → mid → long）
※ early_loss: 0 <= holding_time_min <= 3

- verify（all）: 0.1226 → 0.0866 → 0.0831
- verify（buy）: 0.0649 → 0.0857 → 0.0464（非単調）
- verify（sell）: 0.2112 → 0.0879 → 0.1455（short高 / mid低）

- forward（all）: 0.1270 → 0.0829 → 0.0796
- forward（buy）: 0.0831 → 0.0745 → 0.0564
- forward（sell）: 0.1682 → 0.0901 → 0.1014（short高 / mid低）

### holding_time 中央値（min, all）
- verify: 15.4 → 21.8 → 20.2
- forward: 13.8 → 17.0 → 18.5

## 結論（事実）
- 全体/SELLでは short→long で early_loss_rate が低下する方向が verify/forward で一致。
- BUY は verify が非単調で、方向一致は未確定（forwardは減少傾向）。

## 補足（BUYの非単調）
- verify BUY の early_loss_rate は mid が最大。サンプル数は short=385 / mid=140 / long=280 で、mid が最小。
- forward BUY は short→mid→long の減少方向（0.0831→0.0745→0.0564）なので、BUYは追加観測が必要。
