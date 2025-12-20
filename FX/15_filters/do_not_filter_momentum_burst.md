## 禁止フィルター：momentum burst 系

以下の条件は、
early_loss を減らす効果はあるが、
同時に最大利益源を削るため、
filterとしては採用しない。

### 禁止例

- pre_range_length <= 0 の一律除外
- exhaustion_ratio 上位20%の一律除外
- strong_break × short_range の全面除外

### 根拠

- [[D011a_exhaustion_ratio_buy]]D011: exhaustion 除外は overfiltering
- [[D012a_observe_pre_range_length]]D012a: short_range は悪いが、勝ちも含む
- [[D012e_observe_pre_range_length_x_exhaustion_interaction]]D012e:
  - BUY: strong_break は一貫して危険
  - SELL: long_range×strong は利益源になり得る

結論：
momentum burst 系は
「切る戦略」ではなく
「高分散を前提に扱う戦略」である。
