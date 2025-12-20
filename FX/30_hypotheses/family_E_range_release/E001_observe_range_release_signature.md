---
id: E001
family: family_E_range_release
type: observe
title: E001_observe_range_release_signature

dataset_source: "[[D009_observe_sell_side]]"
depends_on:
  - "[[D012a_observe_pre_range_length]]"
  - "[[D012e_observe_pre_range_length_x_exhaustion_interaction]]"
  - "[[D013_observe_tail_contribution_and_failure_modes]]"

timeframe_signal: M1
timeframe_exec: 10s

status: draft
result: unknown

tags:
  - fx
  - family_E_range_release
  - observe
  - range
  - compression
  - release
  - signature
---

## 目的

D系の研究で見えた「range-release（条件付きで分散が下がる可能性）」を、
D系の strategy / filter に混ぜずに、E系として独立に扱えるようにする。

本観測（E001）は **勝たせに行かない**。
最初にやるのは「レンジ解放の署名（signature）」を定義して固定すること。

- 何をレンジと呼ぶか
- 何を解放と呼ぶか
- 解放後の価格応答をどう分類するか

を verify / forward の方向一致で観測する。

---

## 前提（母集団）

- 母集団は D009（24h、BUY/SELLあり）を使用する：
  - `results/family_D_momentum/D009/`
- E001は D009 の trades に特徴量を付与して観測する（既存結果は変更しない）
- 重要：
  - E系はD系の改良ではない
  - D系を整えるためにE系を作らない（役割の分離）

---

## 観測仮説（問い）

range-release は単なる momentum burst ではなく、

- レンジ状態（蓄積の質）
- 解放の仕方（抜けの質）
- 解放後の応答（継続 / 否定）

の組み合わせで特徴づけられるはずである。

特に SELL 側で、レンジ蓄積がある条件下では
強い抜け（exhaustion相当）が「毒ではない」可能性が示唆されている（D012e）。
この構造を E系として言語化・固定する。

---

## 定義（Contract）

### 1) レンジ蓄積（Range accumulation）

レンジは複数の観測軸で表現し、単一の閾値で断定しない（observeのみ）。

#### (A) pre_range_length
- D012aの定義と同様（M1, lookback N=20, close基準）
- まずはカテゴリは D012a固定を参照：
  - short <= 0
  - mid = 1..3
  - long >= 4

#### (B) range_tightness（締まり）
- `range_width = (high_max - low_min)`（直近N本）
- `range_tightness = range_width / (mean_body_prevN + eps)` のような正規化指標
- tight / normal / wide は verify の分位点で固定（forwardで再計算しない）

#### (C) probe_count（レンジ内のヒゲ試し）
- 直近N本で、レンジ外へのヒゲ（wick only）が発生した回数をカウント
- “clean_range / messy_range” を verify の分位点で固定して分類（observeのみ）

---

### 2) 解放（Release）の品質

D010/D012eの exhaustion_ratio と整合させる。

- exhaustion_ratio:
  - break_margin_over_mean_prev_body（fallback=break_margin_ratio）
- strong/weak は verify p80固定（BUY/SELL別でも可、ただし再計算禁止）

---

### 3) 解放後の応答（Post-release response）

「勝てるか」を判定しない。**解放後の形**だけを分類する。

- next_bar_confirm（次足確認）：
  - entry方向に次のM1終値が進むか（Yes/No）
- quick_reject（早期否定）：
  - entry方向と逆に、次の1本または2本で否定的な終値になるか（Yes/No）
- hold_time は参考（D013で不安定になり得るため主結論にはしない）

※ “次の1本/2本” の本数は固定（最適化しない）

---

## 観測項目（Metrics）

- verify / forward × BUY/SELL別で、以下を出す
  - pre_range_lengthカテゴリ別の
    - next_bar_confirm率
    - quick_reject率
    - early_loss_rate（参考）
  - range_tightness（tight/normal/wide）別の同指標
  - probe_count（clean/messy）別の同指標
  - exhaustion（strong/weak）別の同指標
- 2軸のクロス（最小限）：
  - long_range × strong_break のセル（BUY/SELL別）
  - tight_range × strong_break のセル（BUY/SELL別）

主目的：
- “range-release 署名”として、方向一致のある偏りを見つけて固定する

---

## 出力物（Artifacts）

- `results/family_E_range_release/E001/`
  - summary_verify.csv
  - summary_forward.csv
  - thresholds.json（分位点、N、定義、使用列名）
  - plots/*.png
  - README.md（定義とリーク回避）
- E001.md 自体に「結果概要」「考察」「status/result」を追記する

---

## 期待される結論パターン（事前固定）

### ケースA：署名が固まらない
- range_tightness / probe_count などを足しても方向一致が弱い
- → range-release は定義の問題か、D系のノイズに吸収される
- 次：定義の見直しをobserveとして継続（strategy化しない）

### ケースB：署名が固まる
例：
- tight_range かつ messy が少ない
- long_range かつ strong_break
- で next_bar_confirmが高く quick_rejectが低い（SELL側で顕著）
- → E系として独立に扱える “候補形” が成立
- 次：E002（observe）で「候補形の再現性」を別期間・別通貨で確認

---

## status / result 記録（完了時に更新）

- status: observed
- result:
  - signature_found: yes / no
  - key_pattern:
      - sell_long_strong:
      - sell_tight_strong:
  - notes:
      - dataset_source = D009 (24h, allow_sell=true)
      - No filtering or optimization applied
