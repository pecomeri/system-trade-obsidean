---
topic: family_BC_results_summary
data: results/summary_family_BC.csv
periods: 2024_verify + 2025_forward
---

## まとめ（B系 / C系）

前提：
- データ: Dukascopy（`FX/code/dukas_out_v2`）
- 2024 = verify / 2025 = forward
- 数値パラメータ（SL/TP/EMA/Lookback等）は未変更

### 結果サマリ（summary_family_BC.csv）

```dataviewjs
const path = "results/summary_family_BC.csv";
const data = await dv.io.csv(path);
const rows = (data && typeof data.array === "function")
  ? data.array()
  : (Array.isArray(data) ? data : (data?.values ?? []));

function num(x) {
  const v = Number(x);
  return Number.isFinite(v) ? v : 0;
}

const out = rows.map(r => ({
  family: r.family,
  hyp: r.hyp,
  delta: r.delta,
  verify_sum_pnl_pips: num(r.verify_sum_pnl_pips),
  verify_trades: num(r.verify_trades),
  forward_sum_pnl_pips: num(r.forward_sum_pnl_pips),
  forward_trades: num(r.forward_trades),
  judge: r.judge,
}));

dv.table(
  ["family","hyp","delta","verify_sum_pnl_pips","verify_trades","forward_sum_pnl_pips","forward_trades","judge"],
  out.map(r => [
    r.family, r.hyp, r.delta,
    r.verify_sum_pnl_pips.toFixed(1), r.verify_trades,
    r.forward_sum_pnl_pips.toFixed(1), r.forward_trades,
    r.judge
  ])
);
```

## 考察（結論だけ）

### family_B_failedbreakout
- **最有力は B002（W1 only）**：2025 forward が `+190 pips / 146 trades` で、このfamily内では唯一プラス。
- B001（baseline）は 2025 forward が `-2 pips / 273 trades` とほぼフラットで、「構造としては成立しうるが優位性が薄い」状態。
- 事故抑制の寄与が強い：
  - `use_time_filter=False`（B005）で大幅悪化 → 時間帯制限は必要そう
  - `use_h1_trend_filter=False`（B003）で悪化 → H1フィルターは必要そう
  - `max_losing_streak=999`（B004）でも悪化 → 日次停止はリスク抑制として有効そう

### family_C_m1entry
- baseline（C001）は 2025 forward が `-462 pips / 550 trades` で負。
- 改善方向は C002（W1 only）だが、2025 forward は `-324 pips / 358 trades` と依然マイナス。
- 事故抑制の寄与が強い：
  - `use_time_filter=False`（C005）で大幅悪化
  - `use_h1_trend_filter=False`（C003）で悪化
  - `max_losing_streak=999`（C004）で悪化

## Web版で「次」を決めるための提案（発明ではなく、判定のため）

- **B系を継続候補にするなら B002（W1 only）**を「conditional採用」し、次は「月次の利益が少数月に偏っていないか」「負け月の形」を確認するのが妥当。
- **C系は現状は見送り寄り**（W1にしてもマイナスが残る）で、次をやるなら「C系を継続する理由（構造理解として何を確かめたいか）」を先に明確化した方が良い。

