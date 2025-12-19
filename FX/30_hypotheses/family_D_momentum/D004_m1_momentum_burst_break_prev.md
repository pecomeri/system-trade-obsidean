---
id: D004
family: family_D_momentum

strategy: "[[D004_m1_momentum_burst_break_prev]]"
base_strategy: "[[D001_m1_momentum_burst_range_only]]"


filters:
  - "[[W1_only]]"
  - "[[h1_trend_up]]"


regimes:
  - none
# または
# regimes:
#   - [[m1_compression]]

timeframe_signal: M1
timeframe_exec: 10s

status: draft
result: unknown

tags:
  - fx
  - family_D_momentum
---

# D004（family_D_momentum）M1 momentum burst + break prev + body_ratio


## 目的
- 「momentum burst」を **単なる大実体** ではなく、  
  **構造を破壊し、かつ継続しそうな初動** として定義し直す。
- D001で混入していた以下を減らす：
  - 反転の初動
  - レンジ端の拒否足
  - ヒゲ優勢の否定足
- 数値最適化は行わず、**構造条件のみ**で定義を固定する。

## 位置づけ
- family_D_momentum における **新しい strategy**
- D001の「range-only momentum」仮説を一段具体化したもの
- フィルターやレジームではなく、**エントリートリガーの意味が変わる**

> D001: momentum = 相対的に大きな実体  
> D004: momentum = 構造破壊 + 実体品質を伴う初動

## 定義（Contract）

### 1) M1 Momentum Burst（D001を継承）
- body = abs(close - open)
- mean_prev_body = rolling_mean(body.shift(1), N=lookback_bars)
- burst = body > mean_prev_body
- direction:
  - up: close > open
  - down: close < open

※ lookback_bars は既存の固定値を流用（最適化しない）

---

### 2) Break Prev（直近1本の構造破壊）
直近1本の M1 足を **終値で抜けること** を必須とする。

- prev_high = high.shift(1)
- prev_low  = low.shift(1)

条件：
- break_prev_up = close > prev_high
- break_prev_dn = close < prev_low

> ヒゲで一瞬抜けただけの足は対象外  
> 「終値で抜けている」ことが必須

---

### 3) Body Quality（拒否足の除外）
ヒゲ優勢の足（否定された足）を除外する。

- range = high - low
- body_ratio = body / range
- body_quality = (range > 0) and (body_ratio >= 0.5)

補足：
- body_ratio の閾値 0.5 は固定
- range == 0（doji的足）は除外
- 数値最適化は禁止



## D001からの差分（差分はこれだけ）
M1確定足での burst 判定（D001の `body > mean_prev(body)`）は維持しつつ、追加で次を満たした場合のみ burst を採用する。

- prev 抜け（終値ベース）
  - Buy: `close > prev_high`
  - Sell: `close < prev_low`
  - `prev_high/prev_low` は M1 `high/low` の `shift(1)`
- wick優勢の拒否（固定比率・最適化なし）
  - `range = high - low`
  - `body = abs(close-open)`
  - `body_ratio = body / range`
  - 条件: `range > 0` かつ `body_ratio >= 0.5`

実装メモ（lookahead禁止）：
- M1で条件判定 → `shift(1)` で「直近確定M1のみ」採用 → 10秒足へ `ffill` → coreの「次の10秒open」執行。

## 実装参照
- `FX/code/backtest_runner.py`（`entry_mode == "D_m1_momentum"` かつ `momentum_mode == "D004_continuation"`）

## 結果（2024 verify / 2025 forward）
出典：`FX/results/summary_family_D_momentum.csv`

### D001（baseline）
- 2024 verify: sum_pnl_pips=-560.0, trades=485
- 2025 forward: sum_pnl_pips=-198.0, trades=396

### D002（m1_compression追加）
- 2024 verify: sum_pnl_pips=+8.0, trades=19
- 2025 forward: sum_pnl_pips=-18.0, trades=26

### D003（H1 filter OFF）
- 2024 verify: sum_pnl_pips=-1106.0, trades=808
- 2025 forward: sum_pnl_pips=-604.0, trades=859

### D004（break prev + body_ratio 追加）
- 2024 verify: sum_pnl_pips=-480.0, trades=488
- 2025 forward: sum_pnl_pips=-88.0, trades=396

## 比較（事実のみ）
- D004 vs D001（forward）: sum_pnl_pips の差分 +110.0（-88.0 - (-198.0)）, trades の差分 0（396 - 396）
- D004 vs D001（verify）: sum_pnl_pips の差分 +80.0（-480.0 - (-560.0)）, trades の差分 +3（488 - 485）

## チャート（目視リンク）
出力先：`FX/results/family_D_momentum/D004/charts/`

- 全体俯瞰（期間×時間足を選べる / ズーム可）:
  - `FX/results/family_D_momentum/D004/charts/overview_1min_2025-06-16_2025-06-22.html`
- 一覧: `FX/results/family_D_momentum/D004/charts/index.html`
- loss側（pnl_pips_ascでtop10生成）：
  - `FX/results/family_D_momentum/D004/charts/trade_161.html`
  - `FX/results/family_D_momentum/D004/charts/trade_162.html`
  - `FX/results/family_D_momentum/D004/charts/trade_166.html`
  - `FX/results/family_D_momentum/D004/charts/trade_167.html`
  - `FX/results/family_D_momentum/D004/charts/trade_170.html`
  - `FX/results/family_D_momentum/D004/charts/trade_175.html`
  - `FX/results/family_D_momentum/D004/charts/trade_333.html`
  - `FX/results/family_D_momentum/D004/charts/trade_334.html`
  - `FX/results/family_D_momentum/D004/charts/trade_335.html`
  - `FX/results/family_D_momentum/D004/charts/trade_396.html`

## 再現手順
- backtest: `uv run python FX/code/backtest_runner.py --hyp D004 --root FX/code/dukas_out_v2`
- charts: `uv run python FX/code/viz/viz_trades_plotly.py --results_dir FX/results/family_D_momentum/D004 --symbol USDJPY --window_minutes 30 --top_n 10 --sort_by pnl_pips_asc`
