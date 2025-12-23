---
id: D013
family: family_D_momentum
type: observe
title: D013_observe_tail_contribution_and_failure_modes

base_strategy: "[[D001_m1_momentum_burst_range_only]]"
dataset_source: "[[D009_observe_sell_side]]"
depends_on:
  - "[[D007_observe_holding_time]]"
  - "[[D012a_observe_pre_range_length]]"
  - "[[D012e_observe_pre_range_length_x_exhaustion_interaction]]"
boundary_reference:
  - "[[F001_range_compression_release（仮）]]"

timeframe_signal: M1
timeframe_exec: 10s

status: observed
result: tail_dependency_high

tags:
  - fx
  - family_D_momentum
  - observe
  - high_variance
  - tail
  - failure_mode
  - meta
---

## 目的

D系（family_D_momentum）が **高分散である**という性質を、
「構造（分布の形）」として固定し、以後の設計（合成・サイズ・役割分担）に利用できる形で残す。

本観測の焦点は「勝たせる」ではなく、以下の3点を **verify/forward 一致**で確認すること：

1) D系の損益が **尾っぽ（上位少数トレード）にどれだけ依存**しているか  
2) early_loss（即死）と「大勝ち」が **同じ領域から同居して出る**構造かどうか  
3) 負けの形（failure modes）が  
   - 早期損失（early_loss）  
   - 遅延損失（長く持って負け）  
   のどちらに偏っているか（BUY/SELL別）

重要：
- D系は filter で整える対象ではない（D011の再発を避ける）
- 本観測は D系の「扱い方」を決めるための地層である
- F001 は境界参照であり、D系内にレンジ解放ロジックを混ぜない

---

## 前提（母集団）

- 母集団：
  - `results/family_D_momentum/D009/`
  - BUY / SELL 両方
  - 24h（W1_only ではない）

※ D004（W1_only / allow_sell=False）は母集団にしない  
※ D013 は D009 の結果（trades）に対して集計・観測するのみ（既存結果は変更しない）

---

## 観測仮説（問い）

- D系の期待値（または総損益）は、
  上位少数（尾っぽ）への依存度が高いのではないか。
- early_loss を減らそうとする設計（filter化）は、
  尾っぽ（最大利益源）も同時に削ってしまうのではないか。
- 負けの主因は、D007同様に early_loss に偏っているのではないか（BUY/SELL別）。

---

## 定義（Contract）

### 1) 損益列

- D009 trades が採用している損益列（例：pnl_pips 等）を使用する。
- 複数候補がある場合は「最も基本となる損益列」を自動選択し、
  選択した列名を README / thresholds.json に記録する。

### 2) early_loss

- [[D007_observe_holding_time]] と同一定義
- early_loss: 0 <= holding_time_min <= 3

---

## 観測項目（Metrics）

### 観測1：Tail share（尾っぽ寄与）

グループ別に、損益寄与の集中度を測る。

グループ：
- verify / forward
- all / BUY / SELL

算出：
- 上位 1%, 5%, 10%, 20% の累積損益 / 総損益（寄与率）
- 下位 1%, 5%, 10%, 20% の累積損益 / 総損益（損失側寄与）

目的：
- D系が「尾っぽ依存」かどうかを構造として固定する。

---

### 観測2：Tail と early_loss の同居（条件付き観測）

※ 観測1で尾っぽ依存が強い場合に実施する（追加observeとしても良い）

- pre_range_length（D012a固定：short/mid/long）
- exhaustion（D012eの p80：BUY/SELL別、verify固定）

上記のセルに対して、
- 大勝ち（上位1% or 5%）の出現率
- early_loss_rate

を併記し、
「危険ゾーン＝利益ゾーン」の同居構造があるかを観測する。

---

### 観測3：Failure modes（負けの形）

- 負けトレードを
  - early_loss（<=3min）
  - non-early loss（>3min）
  に分け、BUY/SELL別に比率・損失寄与を出す。

目的：
- D系の損失の主因がどの形に偏っているかを固定し、
  今後の「扱い方」の議論に接続する。

---

## 出力物（Artifacts）

- `results/family_D_momentum/D013/`
  - summary_verify.csv
  - summary_forward.csv
  - plots/*.png
  - thresholds.json（損益列名、early_loss定義など）
  - README.md（定義・手順・リーク回避・採用列名）

---

## 期待される結論パターン（事前固定）

### ケースA：尾っぽ依存が強い

- D系は本質的に「高分散・尾っぽ依存」。
- filter化は overfiltering を再生産しやすい。
- 次：
  - 合成・サイズ・役割設計を本丸として進める
  - F001は「中分散側の役割」を担う可能性を検討

### ケースB：尾っぽ依存が弱い

- D系は「薄い優位が広く存在」している可能性。
- 次：
  - 分布の形（時間帯、方向、レンジ前提など）を別軸で分解する observe を検討

---

## 結果概要（Result Summary）

- tail share（総損益は verify/forward ともマイナスのため、寄与率は符号反転で解釈）  
  - all: top10= -1.30（verify）/ -1.02（forward）、top20= -2.59 / -2.04  
  - all: bottom10= 1.08（verify）/ 0.85（forward）、bottom20= 2.16 / 1.70  
  - BUY は verify の top10 寄与率が大きく、SELL は forward で top10 寄与率が大きい
- failure modes（件数比率と損益寄与）  
  - early_loss_rate（all）: 0.103（verify）/ 0.104（forward）  
  - early_loss_rate（BUY）: 0.062 / 0.073、（SELL）: 0.168 / 0.133  
  - early_loss_pnl_share（BUY）: 0.336 / 0.250、（SELL）: 0.496 / 0.606

## 考察（Discussion）

- tail share の絶対値が 1 を超える範囲が複数あり、D系の損益分布は尾部の寄与が大きい構造で観測された。  
- early_loss の件数比率は小さい一方で、SELL 側の損益寄与が大きく、early_loss を抑制する設計は尾部構造そのものを変えるリスクがある。  
- BUY と SELL で tail share と failure mode の偏りが異なるため、D系は「方向別に異なる分布形」を持つ前提で扱う必要がある。  
- F001（range-compression release）との境界は「尾部依存度の差」で整理できる可能性があり、D系は尾部寄与の大きい側として性質を固定するのが妥当。  

---

## status / result 記録（完了時に更新）

- status: observed
- result:
  - tail_dependency: high
  - tail_share_summary:
      - top_10pct: all=-1.30（verify）/ -1.02（forward）
      - top_20pct: all=-2.59（verify）/ -2.04（forward）
  - loss_tail_summary:
      - bottom_10pct: all=1.08（verify）/ 0.85（forward）
      - bottom_20pct: all=2.16（verify）/ 1.70（forward）
  - failure_modes:
      - dominant_failure_mode: mixed（BUYは非early寄り / SELLはearly寄り）
  - notes:
      - dataset_source = D009 (24h, allow_sell=true)
      - No filtering or optimization applied
