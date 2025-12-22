---
id: D007
family: family_D_momentum
type: observation_variant
strategy: "[[FX/30_hypotheses/program_01_breakout/family_D_momentum/D004_m1_momentum_burst_break_prev]]"
base_strategy: "[[FX/30_hypotheses/program_01_breakout/family_D_momentum/D004_m1_momentum_burst_break_prev]]"
filters:
  - "[[h1_trend_up]]"
regimes:
  - none
timeframe_signal: M1
timeframe_exec: 10s
status: observed
result: early_loss_dominant
tags:
  - fx
  - family_D_momentum
  - observation
  - holding_time
  - time_decay
---

# D007: D004 × Observe Holding Time (Time Decay)

## 目的
- D004（M1 Momentum Burst / Break Prev + Body Quality）を **完全に固定**したまま、
  「エントリー後の時間経過」が期待値に与える影響を観測する。
- 特に、
  - 初動が出なかったトレード
  - 一度伸びたあと停滞したトレード
  が、その後どのような結末になりやすいかを確認する。

> 本ID（D007）は exit 改変のためではなく、  
> **時間経過による期待値の劣化（time decay）が存在するか**を観測する目的で作成する。

---

## 位置づけ
- D007 は **観測用 variant**
- strategy / filter / entry 条件は一切変更しない
- このID自体を最終的に使う想定はない
- 結果は D007a（exit仮説）を切るかどうかの判断材料にする

---

## 使用する戦略
- strategy:
  - `D004_m1_momentum_burst_break_prev`
- エントリー条件・方向判定・執行方法は **D004に完全準拠**
- W1_only / 時間帯制約は設計上の前提とする（D004と同一）

---

## 観測内容（必須）

### 1) 保有時間（holding time）の分布
- 定義：
  - `holding_time = exit_time - entry_time`
- 単位：
  - 秒 / 分（両方出せるなら分を主に使う）

### 2) 保有時間 × 損益の関係
- 例：
  - 0–1分
  - 1–3分
  - 3–5分
  - 5–10分
  - 10分以上
- 各ビンについて：
  - n_trades
  - sum_pnl_pips
  - avg_pnl_pips

### 3) 勝ち／負け別の保有時間分布
- 勝ちトレードの holding_time 分布
- 負けトレードの holding_time 分布

---

## 観測の焦点（問い）
- 「短時間で決着したトレード」と「長引いたトレード」で、
  期待値に明確な差はあるか？
- 一度伸びたあと停滞したトレードは、
  - そのままブレイクするか
  - 時間経過で戻されるか
- ある時間を超えると、
  **平均損益が一貫して悪化する“閾値”**は存在するか？

※ 閾値を“決める”のではなく、“存在するか”を観測する。

---

## 生成物
- results:
  - `FX/results/family_D_momentum/D007/`
- diagnostics:
  - `FX/results/family_D_momentum/D007/diagnostics/`
  - `holding_time_bins.csv`
  - `holding_time_summary.md`
  - （可能なら）簡易ヒストグラム / プロット

---

## 検証観点（このIDでやらないこと）
- SL/TP の変更
- 時間制限 exit の実装
- 勝率・PF の最適化
- 「◯分で切る」ルールを即座に作ること

> D007は **exit設計の前段階**であり、  
> 結論を急がないことが重要。

---

## 次のアクション（D007完了後）
- もし以下が確認できた場合のみ次へ進む：
  - 一定時間を超えると avg_pnl が明確に悪化
  - verify / forward の両方で同傾向
- その場合：
  - D007a: Time-based Exit 仮説（例：◯分以内に初動が出ない場合の扱い）
- 明確な差が出なければ：
  - 「時間経過は主要因ではない」として D007 を stop にする

---

## メモ（後で記入）
- holding_time が短いトレードの傾向：
- 長引いたトレードの共通点：
- time decay の兆候有無：
- 次に進むか／止めるかの判断：

### D007観測メモ（事実：D004結果から集計）
- 生成物（D004）:
  - `FX/results/family_D_momentum/D004/diagnostics/holding_time_bins.csv`
  - `FX/results/family_D_momentum/D004/diagnostics/holding_time_summary.md`
- 入力:
  - in_sample_2024: `FX/results/family_D_momentum/D004/in_sample_2024/trades.csv`
  - forward_2025: `FX/results/family_D_momentum/D004/forward_2025/trades.csv`
- holding_time の定義:
  - holding_time_sec = (exit_time - entry_time).total_seconds()
  - holding_time_min = holding_time_sec / 60.0
- 短時間ビン（0–1, 1–3）と長時間ビン（10–20, 20+）の avg_pnl_pips 差（短-長）:
  - 2024 verify:
    - short: n=30 sum_pnl_pips=-80.0 avg_pnl_pips=-2.67
    - long : n=370 sum_pnl_pips=-224.0 avg_pnl_pips=-0.61
    - diff(short-long)=-2.06
  - 2025 forward:
    - short: n=12 sum_pnl_pips=-54.0 avg_pnl_pips=-4.50
    - long : n=311 sum_pnl_pips=190.0 avg_pnl_pips=0.61
    - diff(short-long)=-5.11




## 観測結果まとめ（Holding Time）

### 事実
- holding_time を 1分刻みで可視化したところ、
  **0–3分の短時間トレードが一貫して大きくマイナス**となることを確認。
- 10–20分帯は avg_pnl の分散が大きく、正負が安定しない「遷移ゾーン」に見える。
- 20分以降は avg_pnl がプラス側に振れやすいが、
  マイナスに大きく振れる点もあり、分散は依然として大きい。

### 解釈
- 「時間が経つほど不利になる（time decay）」というより、
  **最初から不利なトレードが短時間でSLに到達している**
  可能性が高い。
- 20分以降のプラス傾向は、
  **生存者バイアスを含む可能性が高く、
  単純な時間ルール（◯分待つ等）を導く根拠にはならない**。

### 暫定結論
- D004の損失の主因は「長期保有」ではなく、
  **エントリー直後に逆行する“即死ゾーン（0–3分）”**にある。
- 時間を基準に exit を変更する前に、
  **即死トレードの共通点（entry品質・構造）を観測する必要がある**。

### 次の検証候補
- 即死ゾーン（0–3分）と生存ゾーン（20分以上）での
  - burst_strength
  - break_prev の余裕
  - トリガー直後の否定有無
  を比較する観測（D008）を検討する。



![[FX/results/family_D_momentum/D007/スクリーンショット 2025-12-19 17.13.03.png]]

![[FX/results/family_D_momentum/D007/スクリーンショット 2025-12-19 17.13.09.png]]