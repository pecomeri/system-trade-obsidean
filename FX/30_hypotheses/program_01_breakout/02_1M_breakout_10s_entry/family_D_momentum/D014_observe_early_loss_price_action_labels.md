# D014｜Observe Early Loss Classifiability (D009 fixed)

## Purpose
D009 母集団を一切変更せず、
early_loss が entry 後の価格挙動ラベルによって
**分類可能な対象かどうか**を観測する。

本仮説は early_loss を減らすこと、
戦略改善を目的としない。

---

## Fixed Constraints
- D009母集団固定
- entry / exit / SL / TP / 時間帯 / direction 不変
- フィルター禁止
- 閾値最適化禁止
- 評価は verify / forward の符号一致のみ

---

## Observation Axes (Labels)

### L1: Immediate Directional Alignment
- align / counter / flat

### L2: Initial MAE Stage
- small / medium / large

### L3: Early Retracement Presence
- with_retrace / no_retrace

### L4: Immediate Volatility Expansion
- expanded / not_expanded

### L5: Time-to-Failure Stage
- very_fast / fast / slow

※ すべて entry 後確定ラベル
※ 既存スケール・定義を流用する

---

## Evaluation Rule
- early_loss 比率の大小ではなく **符号一致のみ**
- verify / forward 両方で一致した場合のみ A 判定
- A/B/C 判定を軸ごとに独立して行う

---

## Expected Outcomes
- A: 分類可能性あり（構造存在）
- B: 期間依存（偶然）
- C: 分類不能（early_loss は一様）

---

## Status
- type: observation
- status: planned
