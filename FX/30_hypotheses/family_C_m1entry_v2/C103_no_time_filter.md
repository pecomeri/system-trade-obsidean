---
regime: w1_only + h1_trend_aligned (no_time_filter)
strategy: m1_close_trigger + 10s_execution + m1_candle_bias
result: dead
---

差分（C101→C103）：
- `use_time_filter=False` のみ（差分1つ、数値変更なし）

検証：
- 2024 verify / 2025 forward
- 出力: `results/family_C_m1entry_v2/C103/`

理由：2025 forward が負（`sum_pnl_pips=-324`）で、C101から改善なし


