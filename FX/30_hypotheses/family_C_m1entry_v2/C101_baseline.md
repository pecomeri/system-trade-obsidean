---
regime: w1_only + h1_trend_aligned
strategy: m1_close_trigger + 10s_execution + m1_candle_bias
result: dead
---

baseline（C101 / family_C_m1entry_v2）：
- エントリー判断は **M1の確定足**（終値ベース）
- 10秒足は **執行補助のみ**（次バー執行のための足）
- 時間帯は **W1 のみ**（`only_session=W1` を基盤に組み込む）
- H1フィルタは **事故抑制目的でON**（`use_h1_trend_filter=True`）
- A001/B系とは構造が異なる（A/Bは10秒足ブレイク起点で、C101はM1確定起点）

検証：
- 2024 verify / 2025 forward
- 出力: `results/family_C_m1entry_v2/C101/`

理由：2025 forward が負（`sum_pnl_pips=-324`）で、一貫してマイナス
