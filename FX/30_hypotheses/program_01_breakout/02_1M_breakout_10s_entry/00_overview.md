# program_01_breakout｜02_1M_breakout_10s_entry｜Overview

## 目的（This Folder Exists For）

本フォルダは、**program_01（breakout 世界観）** において  
**M1 を主語とした breakout 母集団の定義と、その性質の観測**を行うための探索領域である。

ここでの目的は：

- 「breakout で勝つ方法」を作ることではない
- 既存 strategy（D系等）を改良することではない
- **M1 breakout 全体像の構造（地図）を得ること**

である。

---

## 基本方針（最上位憲章の再掲）

本フォルダ配下では、以下を絶対条件とする。

- 勝たせに行かない
- 最適化しない  
  （閾値・本数・TP/SL・lookback 等の数値調整は禁止）
- フィルター禁止  
  （entry 可否を変え、母集団を変える条件追加は禁止）
- 観測（観測軸）は必須  
  （entry 後のラベル付けのみ。母集団・PnL分布は変えない）
- verify / forward で「符号一致」しない構造は不採用
- 既存 D 系（特に D009 母集団固定）は不可侵

---

## 本フォルダで扱う breakout の定義

- breakout の主語は **M1**
- breakout は **価格挙動で判定可能な Contract** によって定義する
- 流動性・意図・SMC 用語などの解釈的概念は使用しない
- breakout 後の挙動は「状態」として観測する（継続 / 失速 / 反転 等）

---

## フォルダ構成の意味

### 探索対象（breakout 母集団の定義空間）

以下は **program_01 の探索対象そのもの**であり、  
「breakout とは何か」を測量するための候補群である。

```text
01_m1_rb_break
02_m1_body_expansion_break
03_m1_compression_outside_close
04_m1_highlow_sweep_close
05_m1_timebox_break
