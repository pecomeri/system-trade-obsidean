---
regime: london_active + h1_trend_aligned
strategy: failed_breakout_reversal + 10s_execution
result: dead
---

差分（B001→B004）：
- `max_losing_streak=999`（日次停止を無効化）だけ

検証：
- 2024 verify / 2025 forward
- 出力: `results/family_B_failedbreakout/B004/`

（実行後に、DataviewでCSV参照＋理由1行を追記）

## 結果（DataviewでCSV参照）

```dataviewjs
const paths = {
  verify_m: "results/family_B_failedbreakout/B004/in_sample_2024/monthly.csv",
  forward_m: "results/family_B_failedbreakout/B004/forward_2025/monthly.csv",
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

理由：日次停止を無効化するとフォワードが baseline（B001）より悪化（`-2 → -82`）したため
