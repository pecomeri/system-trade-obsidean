---
id: E002
family: family_E_range_release
type: observe
title: E002_observe_post_release_response_distribution

dataset_source: "[[D009_observe_sell_side]]"
depends_on:
  - "[[E001_observe_range_release_signature]]"
  - "[[D013_observe_tail_contribution_and_failure_modes]]"

timeframe_signal: M1
timeframe_exec: 10s

status: observed
result: response_signature_not_found

tags:
  - fx
  - family_E_range_release
  - observe
  - range
  - release
  - response
  - distribution
---

## 目的

E001 において、range-release の署名が
confirm / reject（Yes/No）指標では安定しなかった理由を切り分ける。

E002 では **range 側の定義は増やさず**、
解放後の「応答の測り方」を **連続値の分布**に置き換えることで、

- 署名が本当に存在しないのか
- 応答指標が粗すぎて見えていなかっただけなのか

を判定する。

重要：
- 勝たせに行かない
- exit最適化をしない
- TP/SL を変更しない
- 未来情報で条件を作らない

---

## 前提（母集団）

- 母集団は E001 と同一：
  - `results/family_D_momentum/D009/`
  - BUY / SELL 両方
  - 24h
- E002 は D009 の trades に特徴量を付与して観測するのみ
- E001で定義した range 指標（pre_range_length / tightness / probe / exhaustion）は
  **そのまま再利用**する（再定義しない）

---

## 観測仮説（問い）

range-release が実在するなら、

- 解放後の **ごく短期（次の数本）** において
- confirm / reject の2値では捉えきれない
  **価格応答の分布差**が現れるはずである。

---

## 定義（Contract）

### 1) 観測ウィンドウ（固定）

- entry 後の **次の K 本（M1）**
- K は固定（推奨：K = 3 または 5）
- 最適化は禁止（Kを振らない）

---

### 2) 応答指標（連続値）

entry 価格を基準に、次の K 本での：

- `max_favorable_excursion_pips (MFE_K)`
  - entry方向に進んだ最大幅
- `max_adverse_excursion_pips (MAE_K)`
  - entry方向と逆に進んだ最大幅

※ 方向は BUY / SELL で正規化する  
※ pips 単位（D009で使っているスケールに合わせる）

---

### 3) 参照する range / release ラベル

- pre_range_length（D012a固定カテゴリ）
- range_tightness（E001で確定した分位点）
- probe_count（E001で確定した分位点）
- exhaustion（p80, verify固定）

※ E002では **新しいラベルを追加しない**

---

## 観測項目（Metrics）

verify / forward × BUY/SELL別に、以下を出す。

- MFE_K の分布
  - median / p75 / p90
- MAE_K の分布
  - median / p75 / p90
- 簡易な分離指標（参考）：
  - `MFE_K - MAE_K` の分布（符号を見る）

比較軸（最小限）：
- exhaustion: strong / weak
- pre_range_length: long / short
- （SELL側を主、BUY側は参考）

目的：
- 方向一致する「分布のズレ」があるかを確認する
- 有無を言語化する（勝率化しない）

---

## 出力物（Artifacts）

- `results/family_E_range_release/E002/`
  - summary_verify.csv
  - summary_forward.csv
  - thresholds.json（K、pipsスケール、参照列名）
  - plots/*.png（箱ひげ or ECDF）
  - README.md（定義とリーク回避）
- E002.md に「結果概要」「考察」「status/result」を追記する

---

## 期待される結論パターン（事前固定）

### ケースA：分布差が見える
- SELL 側の特定条件（例：long_range × strong）で
  - MFE_K が高め
  - MAE_K が低め
- verify / forward で方向一致
→ range-release の署名が **連続値では存在**

次：
- E003（observe）：別通貨・別期間で再現性確認

---

### ケースB：分布差が見えない
- MFE_K / MAE_K ともに方向一致が弱い
→ range-release は現定義では実在しない可能性が高い

次：
- E系を一旦止める
- D系＋合成設計に集中する

---

## 結果概要（Result Summary）

- MFE_K / MAE_K の分布は、主要ラベル（pre_range / tightness / probe / exhaustion）で **verify/forward の方向一致が弱い**。  
- SELL側の pre_range/tightness は、verify と forward で MFE_K の順位が入れ替わり、署名として固定できない。  
- BUY側の exhaustion strong は MFE_K が高めになる一方、MAE_K も高めで、分離指標（MFE_K - MAE_K）は一貫して改善しない。  
- 全体として、MFE_K/MAE_K の差は **side（BUY/SELL）の違いが支配的**で、ラベルによる明確な分布分離は観測されなかった。  

## 考察（Discussion）

- confirm/reject の2値（E001）と同様に、連続値分布でも **方向一致が弱い**ため、署名は現定義では固定されない。  
- 連続値で見ても差が出ないことは、E001で見えなかった理由が「指標の粗さ」ではない可能性を示す。  
- BUY は MAE_K が MFE_K を上回る構造が多く、SELL は条件によって入れ替わるため、**E系をD系と混ぜない理由**（分布構造の差）が強化される。  
- 次に進むか止めるかは、E003で別通貨・別期間の再現性を確認するか、E系の観測を一旦停止するかの判断材料として扱う。  

## status / result 記録（完了時に更新）

- status: observed
- result:
  - response_signature_found: no
  - key_pattern:
      - buy_exhaustion_strong: MFE_K/MAE_K の双方が高め（verify/forward一致）
  - notes:
      - dataset_source = D009 (24h, allow_sell=true)
      - Response measured as continuous distribution (MFE/MAE)
      - No filtering or optimization applied
