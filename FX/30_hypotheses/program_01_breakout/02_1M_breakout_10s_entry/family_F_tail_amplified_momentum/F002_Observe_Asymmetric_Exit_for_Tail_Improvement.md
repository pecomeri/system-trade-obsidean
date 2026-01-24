---
id: F002
program: program_01_breakout
family: family_F_tail_amplified_momentum
type: observe
status: observed

result:
  right_tail_preserved: mixed
  left_tail_reduced: mixed
  tail_asymmetry: weak
  stability: unstable

tags:
  - fx
  - momentum_breakout
  - tail_design
  - asymmetric_exit
  - observe
---

# F002｜Observe Asymmetric Exit for Tail Improvement
(family_F_tail_amplified_momentum)

## 目的

本仮説は、**TP 拡張で確認された「左右同時の尾部拡張」**に対し、  
**exit に非対称性を持たせることで、  
右尾（利益側）を維持したまま左尾（損失側）を抑制できるか**を観測する。

本検証は以下を目的としない。

- 勝率改善
- 平均PnLの最大化
- 即運用への接続

---

## 位置づけ

- program：momentum breakout（不変）
- family：F系（tail amplified）
- F001（TP 拡張）を前提とする
- D系・F001 の結論を上書きしない

---

## 前提（固定条件）

- entry / timing / filters：D系・F001と完全に同一
- SL：固定（D系と同一）
- TP：F001 と同一（baseline / tp_x2 / tp_x3）
- **exit のみ非対称化**
- 母集団生成ロジックは変更しない

---

## 非対称 exit の定義（Contract）

以下の **最小・固定パターンのみ**を検証対象とする。

### パターンA：時間ベース損切り（左尾抑制）

- TP 未到達かつ
- 保有時間が T を超過した場合
- 強制 exit

※ T は **少数固定値のみ**  
（例：30分 / 60分）

---

### パターンB：段階的損切り（左尾抑制）

- 含み損が一定水準に到達した場合
- SL 到達を待たずに exit

※ 水準は SL 比率で固定  
（例：0.5×SL）

---

※ どちらも **最適化・探索は禁止**  
※ baseline / F001（非対称なし）と並列比較する

---

## 仮説（問い）

- 非対称 exit により、
  - 左尾（bottom tail）の寄与は減少するか
  - 右尾（top tail）の寄与は維持されるか
- 尾部の **左右非対称性**は生まれるか
- verify / forward で符号一致するか

---

## 観測指標（見るもの）

- tail share（上位 / 下位 1% / 5% / 10%）
- 右尾と左尾の寄与バランス
- 分布の歪み（定性的）
- verify / forward の符号一致

---

## 見ないもの

- 勝率
- 平均PnL
- 最大連敗
- 短期収益安定性

---

## 期待される結論パターン（事前固定）

### ケースA：右尾維持・左尾縮小
- 非対称 exit が有効
- 尾部設計に進む資格あり

### ケースB：左右ともに縮小
- 尾部自体が削られる
- exit 非対称化は不適

### ケースC：効果なし／不安定
- verify / forward 不一致
- exit 非対称化は構造的に効かない

---

## 禁止事項

- exit 条件の最適化
- entry 条件の変更
- 結果を即運用ルールに接続すること

---

## 結果まとめ（事実）

- baseline / tp_x2 / tp_x3 の各系列で、tail share（上位/下位 1%/5%/10%）を verify/forward 別・BUY/SELL 別に比較。
- パターンA（time_30m）は top/bottom とも変化が混在し、forward 側では縮小が優勢。
- パターンA（time_60m）は top/bottom とも縮小が優勢で、右尾も同時に弱まる傾向。
- パターンB（step_sl_0p5）は bottom の縮小が多く観測される一方、top は forward 側で維持〜拡大が多いが verify/forward の符号一致は限定的。

## 考察（限定・設計非接続）

- 時間ベース exit は「保有時間の上限」により尾部全体が圧縮されやすく、左右非対称性は生まれにくい。
- 段階的損切りは左尾を削る方向に働くが、右尾維持の安定性は期間依存で揺れるため、効果は限定的と扱う。
- tail share は総損益が小さい場合に値が振れやすい。ここでの変化は分布形の観測に留め、運用判断へ直結しない。
