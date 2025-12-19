---
id: D004
family: family_D_momentum
type: strategy
strategy: D004_m1_momentum_burst_break_prev
base_strategy: D001_m1_momentum_burst_range_only
filters:
  - W1_only
  - H1_uptrend_on
regimes:
  - none
timeframe_signal: M1
timeframe_exec: 10s
status: draft
result: unknown
tags:
  - fx
  - momentum
  - m1
  - family_D_momentum
---

# D004: M1 Momentum Burst (Break Prev + Body Quality)

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

---

## 入力データ
- 入力: 10秒足 (`df10`)
- 10秒足から M1 確定足を生成
  - open: 分の最初の10秒open
  - close: 分の最後の10秒close
  - high / low: 分内の高値 / 安値

使用カラム（M1）:
- open, close, high, low

---

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

---

### 4) 最終シグナル（M1確定足）
```text
burst_up =
    burst
    and (close > open)
    and break_prev_up
    and body_quality

burst_dn =
    burst
    and (close < open)
    and break_prev_dn
    and body_quality
