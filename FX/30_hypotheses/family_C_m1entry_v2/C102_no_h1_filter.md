---
regime: w1_only
strategy: m1_close_trigger + 10s_execution + m1_candle_bias
result: conditional
---

差分（C101→C102）：
- `use_h1_trend_filter=False` のみ（差分1つ、数値変更なし）

検証：
- 2024 verify / 2025 forward
- 出力: `results/family_C_m1entry_v2/C102/`

理由：2025 forward が C101 より改善（`-324 → -254`）
