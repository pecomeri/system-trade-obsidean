---
id: F001
program: program_01_breakout
family: family_F_tail_amplified_momentum
type: observe
status: observed

result:
  tail_amplification_effect: confirmed
  right_tail_change: increased
  left_tail_change: increased
  tail_dependency_change: strengthened
  stability: low

tags:
  - fx
  - momentum_breakout
  - tail_design
  - tp_extension
  - observe
---

# F001｜Observe TP Extension Effect on Tail Distribution

## 目的

本仮説は、**momentum breakout 戦略（program_01）において  
TP を拡張した場合に、損益分布の「右尾（利益側）」が  
どのように変形するか**を観測する。

D系（family_D_momentum）により、以下がすでに確認されている。

- 戦略は高分散であり、尾部依存が強い
- 平均的なトレードに構造的優位はほぼ存在しない
- 利益はごく一部のトレードが担っている

本仮説では  
**TP 拡張が「尾部を意図的に太らせる方向に機能するか」**のみを問う。  
勝率改善・最適化・即運用への接続は目的としない。

---

## 位置づけ（重要）

- program：momentum breakout（変更なし）
- family：**F系（tail amplified）**
- D系は「観測の基準」として参照する
- D系の結論を上書き・修正しない

---

## 前提（母集団・固定条件）

- entry / timing / filters：D系と完全に同一
- SL：D系と同一（固定）
- TP：**拡張**
- 母集団生成ロジックは変更しない
- 時間帯・曜日・レジームによる除外は行わない

※ TP を変更するため **別 family** として扱う

---

## 仮説（問い）

- TP を拡張すると、
  - 右尾（上位トレード）の損益寄与は増加するか
  - 左尾（損失側）の拡張を上回るか
- tail dependency は
  - 維持されるか
  - さらに強まるか
  - 不安定化するか

---

## 定義（Contract）

### TP 拡張パターン

- baseline：D系と同一 TP
- extended：
  - TP = baseline × k
  - k は **少数固定値のみ**

例：
- k = 2.0
- k = 3.0

※ k の探索・最適化は禁止  
※ 比較対象は baseline vs extended のみ

---

### 観測指標（見るもの）

- tail share（上位 / 下位 1% / 5% / 10%）
- 右尾と左尾の寄与バランス
- 分布の集中度・歪み（定性的）
- verify / forward の符号一致

---

### 見ないもの（明示）

- 勝率
- 平均PnL
- 最大連敗
- 短期の収益安定性

---

## 観測方法

- D系と同一期間・同一分割で
  - verify
  - forward
  を分離して集計
- baseline と extended を **並列比較**
- 分布形の変化のみを記録する

---

## 期待される結論パターン（事前固定）

### ケースA：右尾が明確に太る
- 上位 tail の寄与が増加
- 左尾の悪化を上回る
- 尾部増幅という設計仮説が支持される

### ケースB：左右ともに拡張
- 分布がさらに尖る
- 安定性は低下
- 尾部設計の難易度が高いことを示す

### ケースC：効果が弱い／不安定
- verify / forward で符号不一致
- TP 拡張が構造的優位に繋がらない可能性

---

## 禁止事項

- TP 値の最適化・探索
- SL の同時変更
- entry 条件の変更
- 観測結果を即運用に接続すること

---

## 結果まとめ（事実）

- TP 拡張でトレード件数が減少（verify: buy 805→586→475 / sell 507→359→323、forward: buy 590→441→364 / sell 639→461→384）。
- top 1/5/10% share は tp_x2 / tp_x3 で増加方向（絶対値が拡大）し、右尾寄与が拡張（verify/forward で方向一致）。
- bottom 1/5/10% share も拡大し、左尾の寄与も同時に増加。
- verify では total pnl が小さいサイドがあり、tail share の値が極端に振れる（tp_x2/tp_x3 で顕著）。

## 考察（限定・設計非接続）

- TP 拡張は右尾の寄与を伸ばすが、左尾も同時に拡張し、左右バランスの改善は確認できない。
- 尾部依存は強まるが、verify/forward で振れ幅が大きく安定性は低い。
- 本観測は分布形の理解に限定し、D系の結論や即運用判断には接続しない。
