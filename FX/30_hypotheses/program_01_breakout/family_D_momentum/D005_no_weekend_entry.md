---
id: D005
family: family_D_momentum
type: filter_variant
strategy: "[[FX/30_hypotheses/program_01_breakout/family_D_momentum/D004_m1_momentum_burst_break_prev]]"
base_strategy: "[[FX/30_hypotheses/program_01_breakout/family_D_momentum/D004_m1_momentum_burst_break_prev]]"
filters:
  - "[[W1_only]]"
  - "[[h1_trend_up]]"
  - "[[no_weekend_entry]]"
regimes:
  - none
timeframe_signal: M1
timeframe_exec: 10s
status: stop
result: dead
tags:
  - fx
  - family_D_momentum
  - filter_variant
  - week_transition
---
# D005: D004 × No Weekend Entry

## 目的
- D004（M1 Momentum Burst / Break Prev + Body Quality）を **固定（凍結）** したうえで、
  「週跨ぎ近辺（週末クローズ／週明けオープン）での失敗」を減らせるかを検証する。
- 週跨ぎは市場参加者構造が断絶しやすく、D004の想定する「初動の継続」が崩れやすい。
- strategy は一切変更せず、**適用可否（filter）だけ**で切り分ける。

---

## 背景（観測）
- チャート目視で「週跨ぎで負けている」ケースが目立つ。
- 週末前後は
  - 流動性断絶
  - ポジション調整
  - ギャップ／初動フェイク
  が起こりやすく、短期モメンタム戦略に不利な可能性がある。

---

## 変更点（D004からの差分）
- 追加するのは filter のみ：
  - `[[no_weekend_entry]]`
- それ以外（D004のトリガー／方向／執行／W1／H1）は **完全に同一**。

---

## filter 定義（概要）
- `[[no_weekend_entry]]` に従い、
  - **金曜後半は新規エントリー禁止**
  - **月曜序盤は新規エントリー禁止**
- 「週跨ぎを跨いだ既存ポジションの扱い（強制クローズなど）」は今回は触らない。
  - ※これは exit 変更となり戦略改変に近くなるため。

---

## 仮説
- D004の負けの一部は、週末の構造断絶に起因している。
- 週跨ぎ近辺の新規エントリーを止めることで、
  - trades を極端に減らさずに
  - sum_pnl のマイナスが縮む
  可能性がある。

---

## 検証観点（Phase 2：構造仮説）
※勝率・PF・DDは見ない

評価ポイント：
1. D004比で sum_pnl が改善するか（verify / forward で同方向か）
2. trades が減りすぎていないか
3. 週跨ぎ近辺の loss が目視でも減っているか
4. diagnostics で「金曜後半／月曜序盤の entry が消えている」ことを確認

---

## 生成物
- results:
  - `FX/results/family_D_momentum/D005/`
- charts:
  - `FX/results/family_D_momentum/D005/charts/`
- diagnostics（導入済みなら）:
  - `FX/results/family_D_momentum/D005/diagnostics/`
  - `CHARTS.md`（チャートリンク集）

---

## 実行メモ
- backtest:
  - `python FX/code/backtest_runner.py --family D_momentum --id D005`
- postprocess（導入済みなら）:
  - `python -m FX.code.postprocess.diagnostics --run FX/results/family_D_momentum/D005`

---

## 結果記入欄（後で追記）
- 2024 verify:
  - sum_pnl_pips=-480.0
  - trades=488
- 2025 forward:
  - sum_pnl_pips=-88.0
  - trades=396
- D004との差分（forward）:
  - 0.0 pips / 0 trades（D004と同値）
- D004との差分（verify）:
  - 0.0 pips / 0 trades（D004と同値）
- チャート所感（週跨ぎ由来の負けが減ったか）:
  - `W1_only (08:00–11:00 UTC)` のため、禁止帯（Fri>=20:00 / Mon<02:00）に該当する signal_ts が元々 0 件で、見た目の変化は確認できなかった。
- 自己チェック（必須）:
  - D004の結果が変わっていない確認：D005実行の前後で `FX/results/family_D_momentum/D004/{in_sample_2024,forward_2025}/{monthly.csv,monthly_by_session.csv}` の `sha256` が一致（diffなし）。
  - 禁止時間帯エントリーが減っている確認（ts基準）：`signal_eval_ts ~= entry_time - 10秒` でカウントし、D004=0件 / D005=0件。
- 次のアクション:
  - このフィルタ単体では効果が観測できない（W1_onlyと禁止帯が重ならない）ため、実験設計を見直して別IDで扱う（※D004は凍結のまま）。
