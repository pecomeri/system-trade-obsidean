# Protocol Review Context (FX) - pre-commit

## 目的（何を判断するか）
- TODO: このサイクル（2025-12）で最終的に判断することを1つに絞る

## データ分割（in-sample / verify / forward の扱い）
- in-sample: TODO:（例：設計とデバッグ用の期間）
- verify: TODO:（例：2024年。比較・選抜のために何回でも見る）
- forward: TODO:（例：2025年。freeze済み実験IDのみ、最後に一度だけ見る）
- 重要：forwardは“結果を見る行為”そのものがコスト（スヌーピング）になるため、ゲートを設ける

## 変更ルール
- 差分は必ず1つ（change_one_thing を実験IDに紐づけて固定）
- 数値最適化は禁止（SL/TP/EMA/Lookback 等を動かさない）
- Yes/No定義は曖昧さを排し、擬似コード（チェック可能な形）まで落とす
- TODO: 実験IDの命名規約（例：family + 連番 + 変更点の短縮）

## 合否指標（最低3つ）
- TODO: 最大DD上限（例：月次/日次/トレード単位のどれで測るか）
- TODO: trade数下限（サンプル不足の拒否）
- TODO: 月次安定（偏りの定義：例：上位月寄与率、負け月の連続性）

## いつforwardを見るか（freeze済みのみ、1回だけ）
- forwardを見る前に：experiment_registry.csv で status=frozen になっていること
- forwardを見るのは：TODO: いつ/誰が/どのコマンドで

## “見たら終了”ルール
- forwardを見た experiment_id は forward_viewed_at を記録し、以後そのIDでは仕様変更しない
- TODO: 例外（本当に壊れていた等）を認めるかどうか
