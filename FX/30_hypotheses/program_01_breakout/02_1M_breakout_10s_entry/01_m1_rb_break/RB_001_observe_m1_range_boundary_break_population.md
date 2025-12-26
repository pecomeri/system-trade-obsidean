---
id: RB_001
program: program_01_breakout
type: observe
title: RB_001_observe_m1_range_boundary_break_population

base_population: none
depends_on: none

timeframe_signal: M1
timeframe_exec: none（10s は使用しない）

status: draft
result: TBD

tags:
  - fx
  - program_01
  - breakout
  - population
  - observe
  - structure
---

## 目的

本観測の目的は、

**D009 とは独立に定義した M1 breakout 母集団（RB_001）について、  
その「出現構造」と「事後状態分布」を固定すること**

である。

ここでの焦点は：

- 勝たせることではない
- breakout を戦略化することではない
- **「M1 breakout とは、統計的にどういう集合か」を地図として残すこと**

である。

---

## 前提（Absolute Rules）

以下は本 observe における絶対条件である。

- 勝たせに行かない
- 最適化しない  
  （閾値・本数・TP/SL・lookback 等の数値探索は禁止）
- フィルター禁止  
  （entry 可否を変え、母集団を削る条件は禁止）
- 観測は entry 後ラベル付けのみ
- D009 母集団固定は不可侵
- D系で用いた exhaustion / early_loss / tail 概念は使用しない

---

## 母集団定義（Population Contract）

### 対象データ

- timeframe: M1
- symbols: backtest universe 全体
- session filter: なし（24h）
- direction: BUY / SELL 両方向

---

### 直前レンジ定義（Pre-Break Range）

- reference window: 直近 N 本の M1
- range_high: max(high)
- range_low: min(low)

※ N は既存検証基盤（FX/code）で固定されている値を使用し、  
　本 observe では変更・探索を行わない。

---

### Breakout 定義（RB_001）

以下の条件を満たした M1 を **RB_001 breakout** と定義する。

- Up Break:
  - close > range_high

- Down Break:
  - close < range_low

上記以外の条件は一切追加しない。

---

## 観測起点（Observation Anchor）

- 観測起点は breakout が確定した M1 の close 時点とする。
- breakout 発生前の状態は観測対象外とする。

---

## 観測仮説（問い）

- RB_001 で定義される M1 breakout は、
  出現頻度・時間帯分布に偏りを持つか。
- breakout 後の挙動は、
  継続・停滞・反転のいずれに多く分布するか。
- 上記の分布構造は verify / forward で符号一致するか。

---

## 観測項目（Metrics）

### 観測1：出現構造（Occurrence Structure）

- 件数
- 時間帯分布（hour, weekday）
- BUY / SELL 比率

目的：
- M1 breakout が「いつ・どの程度」起きる集合かを固定する。

---

### 観測2：事後状態分布（Post-Break State）

※ 母集団は変更しない（ラベル付けのみ）

- continuation:
  - breakout 方向への更新が一定期間内に発生
- stall:
  - 高安更新が発生しない
- reversal:
  - 直前レンジ内への明確な回帰

※ 判定ウィンドウ・定義は RB_002 で固定し、
　本ファイルでは数値最適化を行わない。

---

## 明示的に扱わないもの（Out of Scope）

- entry / exit
- TP / SL
- 勝率・PnL・期待値
- breakout 強度
- exhaustion / early_loss / tail / failure mode
- フィルター設計

---

## 出力物（Artifacts）

- `results/program_01_breakout/RB_001/`
  - summary_verify.csv
  - summary_forward.csv
  - counts_by_hour.csv
  - counts_by_weekday.csv
  - README.md（定義・集計手順・リーク回避）

---

## 期待される結論パターン（事前固定）

### ケースA：継続は少数派

- M1 breakout 全体は、
  継続よりも stall / reversal が多数派。
- breakout は「構造的に脆い」集合である。

### ケースB：継続が一定数存在

- breakout 全体像の中に、
  条件を切らずとも残る継続構造が存在。
- D009 はこの部分集合に位置する可能性。

※ いずれのケースでも、
　本 observe は戦略判断を行わず、
　構造の固定のみを行う。
