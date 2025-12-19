---
id: D006
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
  - time_structure
---

# D006: D004 × Observe 24h (No Session Filter)

## 目的
- D004（M1 Momentum Burst / Break Prev + Body Quality）を **完全に固定**したまま、
  時間帯フィルター（W1_only 等）を一切かけずに **24時間フルで観測**する。
- 曜日 × 時間帯（UTC）ごとの
  - エントリー数
  - 損益傾向
を可視化し、**次に立てるべき構造仮説の材料**を得る。

> 本ID（D006）は「勝たせる」ためのものではなく、  
> **どこが悪い／どこがマシかを観測するための中間ステップ**である。

---

## 位置づけ
- D006 は strategy / filter の改良ではなく **観測用 variant**
- このID自体を最終的に使う想定はない
- D006 の結果をもとに、
  - D006a / D006b …（時間帯フィルター仮説）
を切り出すための前処理とする

---

## 使用する戦略
- strategy:
  - `D004_m1_momentum_burst_break_prev`
- エントリー条件・方向判定・執行方法は **D004から一切変更しない**

---

## フィルター条件
- 有効：
  - `[[h1_trend_up]]`
- 無効：
  - W1_only
  - London / NY などの時間帯制約
  - no_weekend_entry 等の時間構造フィルター

> 目的は「制約をかけること」ではなく  
> **制約をかける前の素の分布を見ること**。

---

## 観測内容（必須）
D006では、以下の集計を必ず生成する。

### 1) 曜日 × 時間帯（UTC）のエントリー分布
- 指標：
  - n_entries
  - sum_pnl_pips
  - avg_pnl_pips
- 時刻基準：
  - 原則 `signal_eval_ts`
  - 無ければ `entry_time - 10秒` を近似として使用

### 2) verify / forward の分離
- 2024 verify
- 2025 forward
を同一フォーマットで比較可能にする。

---

## 生成物
- results:
  - `FX/results/family_D_momentum/D006/`
- diagnostics:
  - `FX/results/family_D_momentum/D006/diagnostics/`
  - `entry_by_dow_hour.csv`
  - `entry_by_dow_hour_pivot.csv`
  - `entry_by_dow_hour_summary.md`

---

## 検証観点（このIDで“やらないこと”）
- 勝率・PF・DD を評価しない
- 良さそうな時間帯をその場で除外しない
- strategy を修正しない

> D006は「判断」ではなく「観測」に徹する。

---

## 次のアクション（D006完了後）
- CSVを確認し、
  - verify / forward の両方で
  - 件数が十分あり
  - 一貫して悪い（または良い）
  曜日 × 時間帯を **最大2つまで**選ぶ
- それらを仮説として
  - D006a / D006b（filter_variant）
  を新規作成する

---

## メモ（後で記入）
- 悪化が目立つ曜日 × 時間帯：
- 改善が見られる曜日 × 時間帯：
- 次に切るフィルター案：
