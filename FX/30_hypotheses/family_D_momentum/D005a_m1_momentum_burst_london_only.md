
# D005a: D004 × London Session Only

## 目的
- D004（M1 Momentum Burst / Break Prev + Body Quality）が  
  **どの時間帯で最も成立しやすいか**を切り分ける。
- 特に「流動性の立ち上がり」が起きやすい  
  **London セッション単体**での挙動を確認する。
- strategy そのものは一切変更せず、  
  **時間帯という構造仮説のみ**を検証する。

---

## 位置づけ
- D004 の **filter variant**
- D005 系はすべて
  - *D004 × 市場構造*  
  の検証を目的とする
- 本 variant は
  > 「momentum burst は London で最も機能するのではないか？」  
  という仮説を検証する

---

## フィルター定義

### W1_London_only
- London セッションの時間帯のみを対象とする
- 具体的な時刻定義は `[[W1_London_only]]` ノートに従う
- NY open / NY 午後 / アジア時間は除外

### H1_uptrend
- 上位足（H1）が uptrend と判定されている場合のみ通す
- 判定ロジックは core の `compute_h1_uptrend` をそのまま利用
- 定義の詳細は `[[H1_uptrend]]` を参照

---

## 戦略定義（変更なし）
- 使用 strategy:
  - `D004_m1_momentum_burst_break_prev`
- エントリートリガー・方向判定・執行方法は  
  **すべて D004 に完全準拠**
- 本 variant では
  - 条件の追加
  - 閾値の変更
  - 判定順序の変更  
  は一切行わない

---

## 仮説
- London セッションは
  - 流動性が急増しやすい
  - 欧州勢のポジション構築が始まりやすい
- そのため
  - M1 の構造破壊を伴う momentum burst が  
    **継続に繋がりやすい**可能性がある
- 逆に
  - アジア時間由来のノイズ
  - NY 午後の一方向性の弱さ  
  を除外できると期待する

---

## 検証観点（Phase 2）
※ strategy の良し悪しは再評価しない

評価ポイント：
1. sum_pnl が D004 より改善しているか
2. trades が極端に減っていないか
3. verify / forward が同じ方向を向いているか
4. 負けトレードが
   - レンジ化
   - 行って来い
   によるものが減っているか（チャート目視）

見ないもの：
- 勝率
- PF
- 最大DD
- 「勝てそうかどうか」

---

## 想定される結果パターン
- ✔ London only で明確にマイナスが減る  
  → 時間帯は有効な構造仮説
- △ 改善するが trades が激減  
  → 条件が厳しすぎる可能性
- ✖ ほぼ変化なし  
  → 時間帯は主要因ではない

---

## 生成物
- results:
  - `FX/results/family_D_momentum/D005a/`
- charts:
  - `FX/results/family_D_momentum/D005a/charts/`
- diagnostics:
  - `FX/results/family_D_momentum/D005a/diagnostics/`
  （D004 と同様の後処理を適用）

---

## 実行メモ
- backtest:
  - `python FX/code/backtest_runner.py --family D_momentum --id D005a`
- postprocess（導入済みの場合）:
  - `python -m FX.code.postprocess.diagnostics --run FX/results/family_D_momentum/D005a`

---

## 判断メモ（後で記入）
- D004 比での sum_pnl 差分：
- trades の変化：
- チャートの納得感（主観）：
- 次のアクション：
  - D005b（London + NY open）へ進む
  - D006（H1の質）へ進む
  - D004 を基点に別構造を試す
