---
regime: london_active + h1_trend_aligned + session_W2_only
strategy: topdown_trend_trigger + m1_signal_to_10s_entry
result: dead
---

差分（A001→A002）：
- エントリー対象を `W2` セッションのみに限定する（執行条件のみ）

検証手順：
- 2024年を検証（in-sample 相当）
- 2025年をフォワード（out-of-sample）
- 出力は `results/family_A/A002/` に保存

理由：2025フォワードの合計PnLが負（`sum_pnl_pips=-412`）で、改善が確認できず

## 根拠（DataviewでCSV参照）

前提：`results/family_A/A002/` 配下に runner 出力のCSVがあること。

```dataviewjs
const paths = {
  verify: "results/family_A/A002/in_sample_2024/monthly.csv",
  forward: "results/family_A/A002/forward_2025/monthly.csv",
};

async function loadMonthly(path) {
  const data = await dv.io.csv(path);
  const rows = (data && typeof data.array === "function")
    ? data.array()
    : (Array.isArray(data) ? data : (data?.values ?? []));
  return rows.map(r => ({
    month: r.month,
    trades: Number(r.trades ?? 0),
    wins: Number(r.wins ?? 0),
    losses: Number(r.losses ?? 0),
    sum_pnl_pips: Number(r.sum_pnl_pips ?? 0),
    avg_pnl_pips: Number(r.avg_pnl_pips ?? 0),
    winrate: Number(r.winrate ?? 0),
  }));
}

function summarize(rows) {
  const sumPnl = rows.reduce((a, r) => a + r.sum_pnl_pips, 0);
  const trades = rows.reduce((a, r) => a + r.trades, 0);
  const wins = rows.reduce((a, r) => a + r.wins, 0);
  const winrate = trades > 0 ? wins / trades : 0;
  return { sumPnl, trades, wins, winrate };
}

const verify = await loadMonthly(paths.verify);
const forward = await loadMonthly(paths.forward);

const v = summarize(verify);
const f = summarize(forward);

dv.header(3, "集計（2024 verify / 2025 forward）");
dv.table(
  ["period", "sum_pnl_pips", "trades", "wins", "winrate"],
  [
    ["2024 verify", v.sumPnl.toFixed(1), v.trades, v.wins, v.winrate.toFixed(3)],
    ["2025 forward", f.sumPnl.toFixed(1), f.trades, f.wins, f.winrate.toFixed(3)],
  ]
);

dv.header(3, "2025 forward 月次（参考）");
dv.table(
  ["month", "trades", "sum_pnl_pips", "avg_pnl_pips", "winrate"],
  forward.map(r => [r.month, r.trades, r.sum_pnl_pips.toFixed(1), r.avg_pnl_pips.toFixed(3), r.winrate.toFixed(3)])
);
```
