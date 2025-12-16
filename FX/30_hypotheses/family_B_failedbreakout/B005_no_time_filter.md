---
regime: h1_trend_aligned
strategy: failed_breakout_reversal + 10s_execution
result: dead
---

差分（B001→B005）：
- `use_time_filter=False`（時間帯制限OFF）だけ

検証：
- 2024 verify / 2025 forward
- 出力: `results/family_B_failedbreakout/B005/`

（実行後に、DataviewでCSV参照＋理由1行を追記）

## 結果（DataviewでCSV参照）

```dataviewjs
const paths = {
  verify_m: "results/family_B_failedbreakout/B005/in_sample_2024/monthly.csv",
  forward_m: "results/family_B_failedbreakout/B005/forward_2025/monthly.csv",
};

async function csvRows(path) {
  const data = await dv.io.csv(path);
  return (data && typeof data.array === "function")
    ? data.array()
    : (Array.isArray(data) ? data : (data?.values ?? []));
}

function num(x) {
  const v = Number(x);
  return Number.isFinite(v) ? v : 0;
}

function summarizeMonthly(rows) {
  const sumPnl = rows.reduce((a, r) => a + num(r.sum_pnl_pips), 0);
  const trades = rows.reduce((a, r) => a + num(r.trades), 0);
  return { sumPnl, trades };
}

const v = summarizeMonthly(await csvRows(paths.verify_m));
const f = summarizeMonthly(await csvRows(paths.forward_m));

dv.table(
  ["period", "sum_pnl_pips", "trades"],
  [
    ["2024 verify", v.sumPnl.toFixed(1), v.trades],
    ["2025 forward", f.sumPnl.toFixed(1), f.trades],
  ]
);
```

理由：時間帯制限を外すとフォワードが大幅悪化（`sum_pnl_pips=-712`）し、事故抑制として時間帯が必要そう
