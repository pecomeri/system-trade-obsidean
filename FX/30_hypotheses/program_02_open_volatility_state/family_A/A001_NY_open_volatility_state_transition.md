---
id: program_02_A001
program: program_02
family: A
hypothesis: A001
type: observation
status: observing
title: NY open volatility state transition
market: FX
instruments: majors
timeframe: M1
anchor_event: ny_open
state_metric: ATR
state_categories: [low, mid, high]
evaluation:
  verify_forward: required
  criterion: sign_consistency_only
constraints:
  - no_optimization
  - no_parameter_tuning
  - no_pnl_evaluation
  - no_direction_decision
  - no_breakout_logic
  - no_exhaustion_concepts
output_dir: results/program_02_open_volatilyty_state/family_A
---




# A001：NYオープン前後のボラティリティ状態遷移の観測

## 目的
NYオープン前後において、価格の方向や勝敗・PnLではなく、  
**ボラティリティ状態（low / mid / high）が遷移しているか**を観測する。

本仮説は戦略構築を目的とせず、  
「状態遷移という構造が存在するかどうか」の確認に限定する。

---

## 観測仮説（問い）
**NYオープン前後は、それ以前とは異なるボラティリティ状態へ遷移しやすいか。**

- 方向（上か下か）は問わない  
- breakout / exhaustion / early_loss の概念は使わない  

---

## 母集団の定義（Contract）
- 基準点：NYオープン時刻を T0 とする  
  （サマータイム等は既存の定義に従う）
- 観測ウィンドウ：
  - T0 前：K_pre 本（固定）
  - T0 後：K_post 本（固定）
- 対象：
  - FX 主要通貨ペア
  - M1
- K_pre / K_post は最適化しない  
  （コード側の定数として固定）

---

## 状態の定義（Contract）
- 指標：ATR（既存定義を流用）
- 分類方法：
  - 相対比較により  
    **low / mid / high** の3カテゴリに分類
- 絶対値の大小や方向は扱わない

---

## 記録項目
- pre_state：low / mid / high
- post_state：low / mid / high
- transition_type：
  - stay（変化なし）
  - up（低→高方向）
  - down（高→低方向）

※ 勝敗・pips・PnL・方向は一切記録しない

---

## 評価（事前固定）
- verify / forward の両方で  
  **状態遷移分布の「符号一致」があるか**のみを見る
- 数値の大小・改善方向は評価しない

### 期待される結論パターン
- **Pattern A**  
  遷移に偏りがあり、verify / forward で一致  
  → 次フェーズ（family_B）へ進む
- **Pattern B**  
  遷移はあるが verify / forward で不一致  
  → dead
- **Pattern C**  
  遷移が見られない  
  → 構造なしとして終了

---

## 禁止事項
- K_pre / K_post の変更
- 状態定義（ATR・カテゴリ境界）の微調整
- 成績（勝率・PnL）を根拠にした解釈
- 他 program（特に program_01 / D系）の知見流用

---

## 実行・出力
- 出力先：`results/program_02_open_volatilyty_state/family_A/`
- 出力形式：
  - CSV（全イベント）
  - summary_md（遷移分布の要約）

## ロジック（program_02_a001_ny_open_vol_transition.py）
- 入力：10秒足を読み込み、M1にリサンプル（open/high/low/close）
- ATR：M1のOHLCからATR（period=14）を算出
- NYオープン時刻：`backtest_core.Config.w2_start` の時刻（UTC 13:30）をT0として使用
- 観測ウィンドウ：
  - T0前K_pre本（30本）
  - T0後K_post本（30本）
  - どちらも連続1分足で埋まっていることを必須条件とする
- 状態分類：
  - verify期間のATR分位（q33/q66）を閾値として固定
  - 各ウィンドウのATR平均をlow/mid/highに分類
- 遷移判定：
  - pre_state → post_state を記録
  - transition_typeは stay / up / down（low < mid < high の順序で判定）
- 検証の確認項目：
  - verify/forwardでup vs downの符号一致のみを確認（数値の大小は評価しない）

---

## status / result
- status: observed
- result:
  - verify:
      - total_events: 260
      - transition_type_counts: stay=194, up=53, down=13
      - sign_overall: +
      - sign_by_pre_state: low=+, mid=+, high=-
  - forward:
      - total_events: 244
      - transition_type_counts: stay=191, up=47, down=6
      - sign_overall: +
      - sign_by_pre_state: low=+, mid=+, high=-
  - decision: sign_consistency=match (overall +, pre_state sign match)
  - notes: NY open=13:30 UTC, K_pre=30, K_post=30, ATR period=14, thresholds q33=0.016643 q66=0.028857, outputs=results/program_02_open_volatilyty_state/family_A

---

## 考察（限定）
- 遷移分布の「符号一致」は overall / pre_state 別ともに verify と forward で一致。
- ここでの確認は「符号一致の有無」だけに限定し、状態定義や閾値の調整は行わない。

## 結論（固定）
- NYオープン前後のボラ状態遷移は、verify / forward で符号一致が確認できた。

