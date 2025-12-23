# A案｜Binance現物 板の歪み × maker（ideas）

## 背景・前提
- 個人でも戦える準HFT領域として  
  **「秒〜15秒 × 板の歪み観測 × maker限定」** を検討
- ミリ秒椅子取りはやらない
- バックテストで勝たせに行かない
- フォワード前提、事故耐性と運用力で勝負

---

## 対象市場
- Binance 現物
- 想定ペア：SOL/USDT（初期）
- ポジションは短時間（数秒〜十数秒）

---

## 戦略の核（A案）
- 板の「歪み」が出た瞬間のみ参加
- maker 指値のみ（taker禁止）
- 刺さらなければ追わない
- 期待が崩れたら即撤退（小損）

### 歪みの要素（候補）
- Top-of-book imbalance（上位N段のbid/ask量差）
- Liquidity shock（片側板の急減）
- Trade flow bias（短時間の約定方向偏り）

---

## バックテストについての結論
- ❌ 約定PnLを再現するバックテストは不可
  - キュー順位が分からない
  - maker約定の再現不能
- ⚠️ 疑似検証は可能
  - 歪み発生頻度
  - 歪み後の方向一致率
- ✅ 本命はフォワード（極小サイズ・半自動）

---

## データの役割整理

| レイヤー | 目的 | データ |
|---|---|---|
| 戦略本体 | 歪み検出・執行 | WS板・WS trade |
| 生存確認 | 約定率・事故率 | フォワード |
| 事前ふるい | 地合い・時間帯 | 1分足kline |
| 事後分析 | 条件ラベル | kline + WSログ |

---

## 現在の実装状況（Python）
- Binance Spot orderbook diff を
  - snapshot + diff で正しく同期
  - out-of-sequence を即例外
  - JSONLで日別保存
- 板取得コードは **A案に十分な完成度**

---

## A案用に最低限追加すべきもの

### 1. event time（取引所時刻）
- diff event の `E` を必ず保存
- WS遅延・環境比較（EC2東京 vs Beeks）に必須

### 2. 板メトリクスの集約ログ
- 生板はそのまま保存（事故調査用）
- 別途、以下を1イベント1行で保存
  - top N imbalance（例：N=5）
  - 上位N段のbid/ask合計量

### 3. trade WSの追加
- `{symbol}@trade` を購読
- 保存項目は最小限
  - event time
  - price
  - qty
  - is_buyer_maker

---

## やらないこと（重要）
- 板の完全再構築バックテスト
- 約定PnLの再現
- 数値最適化
- 早期のRust移行

---

## 次のアクション
1. Pythonコードに event time 保存を追加
2. top5 imbalance の metrics.jsonl を出力
3. trade WS を追加
4. EC2東京 / Beeks ロンドンで同時稼働
5. p95/p99遅延と歪み分布を比較

→ A案が「現実的か」をデータで判断
