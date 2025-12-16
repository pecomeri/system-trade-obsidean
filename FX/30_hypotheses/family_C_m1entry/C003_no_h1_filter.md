---
regime: london_active
strategy: topdown_trend_trigger + m1_signal_to_10s_entry
result: dead
---

差分（C001→C003）：
- `use_h1_trend_filter=False` のみ（差分1つ、数値パラメータ変更なし）

検証：
- 2024 verify / 2025 forward
- 出力: `results/family_C_m1entry/C003/`

## 結果（DataviewでCSV参照）

```dataviewjs
const paths = {
  verify_m: "results/family_C_m1entry/C003/in_sample_2024/monthly.csv",
  forward_m: "results/family_C_m1entry/C003/forward_2025/monthly.csv",
  forward_ms: "results/family_C_m1entry/C003/forward_2025/monthly_by_session.csv",
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
  const wins = rows.reduce((a, r) => a + num(r.wins), 0);
  const winrate = trades > 0 ? wins / trades : 0;
  return { sumPnl, trades, wins, winrate };
}

function summarizeBySession(rows) {
  const by = new Map();
  for (const r of rows) {
    const k = String(r.session ?? "NA");
    const cur = by.get(k) ?? { trades: 0, wins: 0, losses: 0, sumPnl: 0 };
    cur.trades += num(r.trades);
    cur.wins += num(r.wins);
    cur.losses += num(r.losses);
    cur.sumPnl += num(r.sum_pnl_pips);
    by.set(k, cur);
  }
  return [...by.entries()].sort((a, b) => a[0].localeCompare(b[0]));
}

const verifyM = await csvRows(paths.verify_m);
const forwardM = await csvRows(paths.forward_m);
const forwardMS = await csvRows(paths.forward_ms);

const v = summarizeMonthly(verifyM);
const f = summarizeMonthly(forwardM);

dv.header(3, "集計（2024 verify / 2025 forward）");
dv.table(
  ["period", "sum_pnl_pips", "trades", "wins", "winrate"],
  [
    ["2024 verify", v.sumPnl.toFixed(1), v.trades, v.wins, v.winrate.toFixed(3)],
    ["2025 forward", f.sumPnl.toFixed(1), f.trades, f.wins, f.winrate.toFixed(3)],
  ]
);

dv.header(3, "2025 forward：セッション別合計");
dv.table(
  ["session", "trades", "wins", "losses", "sum_pnl_pips"],
  summarizeBySession(forwardMS).map(([k, s]) => [k, s.trades, s.wins, s.losses, s.sumPnl.toFixed(1)])
);
```

理由：H1フィルターを外すとフォワードが baseline（C001）より明確に悪化（`-384 → -672`）したため
