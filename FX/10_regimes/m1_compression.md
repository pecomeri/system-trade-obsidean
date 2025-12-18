---
id: m1_compression
type: regime
status: draft
---

# m1_compression（M1圧縮レジーム）

## 目的

- family_D_momentum の D002 で使う「圧縮（compression）局面」を、価格レンジのみ（M1のOHLC）で定義する。
- 「レンジ相場一般」ではなく、圧縮→解放（burst）に繋がりやすい局面に限定するためのゲート。

## 入力

- M1 確定足（10秒足 `df10` から集約した open/close を使用）
- 実装上の窓幅は `lookback_bars` を流用し、数値最適化はしない（固定）。

## Contract（Yes/No）

このプロジェクトでは、圧縮判定の “レンジ” を **M1 実体レンジ**（`abs(close-open)`）に固定する。

- Yes（compression）
  - 直近 K 本の M1 実体レンジが、過去 N 本の M1 実体レンジ中央値より小さい状態が連続している
- No
  - 上記以外

パラメータ（固定）：
- `N = lookback_bars`（既存の固定値を流用）
- `K = floor(lookback_bars / 2)`（既存値から派生して固定。例：lookback=6ならK=3）

擬似コード（実装と一致させる）：

```text
body = abs(m1_close - m1_open)
median_prev = rolling_median(body.shift(1), window=N)
is_small = (body < median_prev)
compression = (rolling_sum(is_small, window=K) == K)

# D002では「直近の確定M1」で判定するため shift(1) して使う
compression_closed = shift1(compression)
```

## 実装参照

- `FX/code/backtest_runner.py`
  - `HypConfig.regime_mode == "m1_compression"` のとき、D001シグナル（momentum burst）に対して `compression_closed` でゲートする。

