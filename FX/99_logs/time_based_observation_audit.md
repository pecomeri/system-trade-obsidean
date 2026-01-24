# Time-based observation audit (program_01_breakout / D系中心)

## 概要
- D系で「時間別観測」として明記されたものは D006（曜日×時間帯の分布）が実施済み。
- D系で時間フィルターを使った検証は D005（週末近辺の除外）が実施済みだが、W1_only と重なり実質的な差分なし。
- D009 は「曜日×時間帯分布」を必要なら作ると書かれているが、時間分布の結果は記録なし。
- A/B/C 系では W1/W2 セッションや time_filter の有無を使った差分検証が複数ある（時間フィルターの検証履歴あり）。
- RB_001 は hour/weekday 分布を観測する設計だが status=draft。
- holding_time（保有時間）は本棚卸しの対象外（カレンダー時間のみ対象）。

## 時間をフィルターとして使った検証

| file | 観測の種類 | 母集団（D009?） | フィルター/ラベル | verify/forward | 結論の有無 |
| --- | --- | --- | --- | --- | --- |
| FX/30_hypotheses/program_01_breakout/02_1M_breakout_10s_entry/family_D_momentum/D005_no_weekend_entry.md | 週末近辺（金曜後半/月曜序盤） | no（D004母集団） | filter（no_weekend_entry） | yes | 差分なし（禁止帯エントリー0件と記載） |
| FX/30_hypotheses/program_01_breakout/01_10s_breakout/family_A_breakout/A002_W2_only.md | セッション W2 のみ | n/a（非D系） | filter（only_session=W2） | yes | result=dead（forward sum_pnl_pips=-412 と記載） |
| FX/30_hypotheses/program_01_breakout/01_10s_breakout/family_A_breakout/A005：時間帯なし（ロンドンフィルターの価値を測る）.md | 時間帯制限を外す | n/a（非D系） | filter（use_time_filter=False） | yes | result=dead（forward sum_pnl_pips=-1040 と記載） |
| FX/30_hypotheses/program_01_breakout/01_10s_breakout/family_B_failedbreakout/B002_W1_only.md | セッション W1 のみ | n/a（非D系） | filter（only_session=W1） | yes | result=conditional（forward sum_pnl_pips=+190 と記載） |
| FX/30_hypotheses/program_01_breakout/01_10s_breakout/family_B_failedbreakout/B005_no_time_filter.md | 時間帯制限を外す | n/a（非D系） | filter（use_time_filter=False） | yes | result=dead（forward sum_pnl_pips=-712 と記載） |
| FX/30_hypotheses/program_01_breakout/02_1M_breakout_10s_entry/family_C_m1entry/C002_W1_only.md | セッション W1 のみ | n/a（非D系） | filter（only_session=W1） | yes | result=conditional（forward -462→-324 と記載） |

## 時間を観測ラベルとして付与した検証

| file | 観測の種類 | 母集団（D009?） | フィルター/ラベル | verify/forward | 結論の有無 |
| --- | --- | --- | --- | --- | --- |
| FX/30_hypotheses/program_01_breakout/02_1M_breakout_10s_entry/family_D_momentum/D006_observe_24h.md | 曜日×時間帯（UTC hour） | no（D004母集団） | labelのみ（24h観測） | yes | no_clear_time_effect（明確な悪い塊なし） |
| FX/30_hypotheses/program_01_breakout/02_1M_breakout_10s_entry/family_D_momentum/D009_observe_buysell_side_24h_baseline.md | 曜日×時間帯（必要なら） | yes（D009） | labelのみ（記載のみ） | yes（time分布は未記録） | 結論なし（時間分布は未記録） |
| FX/30_hypotheses/program_01_breakout/01_10s_breakout/family_A_breakout/A001：ベースライン（板フィルターなし）.md | セッション（W1/W2） | n/a（非D系） | labelのみ（session集計） | yes | W2 偏りあり、W1 一貫性なしと記載 |
| FX/30_hypotheses/program_01_breakout/02_1M_breakout_10s_entry/01_m1_rb_break/RB_001_observe_m1_range_boundary_break_population.md | hour/weekday 分布 | n/a（非D系） | labelのみ（observe設計） | yes（予定） | status=draft / result=TBD |

## 既存観測で言えること / 言えないこと

言えること（記録に基づく事実）
- D006 で曜日×時間帯分布を観測済みだが、verify/forward で一貫した「明確に悪い時間帯の塊」は確認されていない。
- D005 の週末近辺フィルターは、W1_only と重なるため差分が出ていない。
- A/B/C 系では W1/W2 などセッション単位のフィルター差分検証が複数実施されている。

言えないこと（記録にない/断定不可）
- D009 母集団に対する曜日×時間帯分布の具体的な結論（D009 では記録なし）。
- 月内位置（初旬/末尾）や週内位置（週前半/後半）による差異の有無。
- A/B/C 系の時間帯結論を D 系に一般化できるかどうか。

## 未観測の時間軸（記録に見当たらないもの）

- 月内位置（例：月初/中/末）
- 週内位置（週前半/後半）
- 祝日・連休・月末週などのカレンダーイベント軸
- D009 を対象とした曜日×時間帯分布の明示的な結果記録
