---
id: D001_m1_momentum_burst_range_only
type: strategy
status: draft
---

# D001 M1 momentum burst（price range only）

## 目的

- family_D_momentum / D001 baseline の「エントリートリガー」を、擬似コード（if条件）で固定する。
- level breakout は使わない（C系の条件を持ち込まない）。
- 数値最適化は禁止のため、閾値探索はしない。

## 入力

- 10秒足（`df10`）から作る M1 確定足（open/close）

## momentum burst（M1確定足）

定義：
- 当該 M1 の **実体レンジ**（`abs(close-open)`）が、
  直近 N 本の平均実体レンジより大きい（加速）こと
- 方向は当該 M1 足の方向（陽線なら Buy、陰線なら Sell）

※ N は新規に作らず、既存の `lookback_bars` を流用する（最適化しない）。

```text
# build M1 bars from 10s
o = resample_1min_open(df10)
c = resample_1min_close(df10)

body = abs(c - o)
mean_prev = rolling_mean(body.shift(1), N=lookback_bars)  # N is existing fixed parameter

burst = (body > mean_prev)
burst_up = burst and (c > o)
burst_dn = burst and (c < o)

# use last closed M1 only (no lookahead)
burst_up_closed = shift1(burst_up)
burst_dn_closed = shift1(burst_dn)
```

## 10秒足は執行補助のみ

```text
# align to 10s index (execution helper)
sig_up_10s = ffill_to_10s(burst_up_closed)
sig_dn_10s = ffill_to_10s(burst_dn_closed)

# entry execution is handled by core: next 10s open (market-like)
```

## 参照レジーム

- `FX/10_regimes/D001_W1_only_h1_uptrend.md`

