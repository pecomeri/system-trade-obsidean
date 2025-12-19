---
id: h1_trend_down
type: filter
status: active
---

# h1_trend_down

このノートは `FX/27_filters/h1_trend_down.md` の参照用ミラーです（正本はそちら）。

- 正本: `[[FX/27_filters/h1_trend_down]]`

## ロジック（Contract）

```text
# 10秒closeからH1の終値系列を作る
h1_close = resample_1H_last(close_bid)

# H1終値のEMAを計算
h1_ema = EMA(h1_close, period=h1_ema_period)

# EMAの過去値（傾き判定用）
ema_past = shift(h1_ema, h1_ema_slope_bars)

# downtrend条件（下降のみ）
ok = (h1_close < h1_ema) and (h1_ema < ema_past)

# “確定足のみ”にするため1本遅らせてから、10秒足へ前方埋め
ok_closed = shift(ok, 1)
h1_downtrend_10s = ffill_to_10s(ok_closed)
```

