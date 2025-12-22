---
regime: london_active + h1_trend_aligned + session_W1_only
strategy: failed_breakout_reversal + 10s_execution
result: conditional
---

差分（B001→B002）：
- `only_session=W1` のみ（差分1つ、数値パラメータ変更なし）

検証：
- 2024 verify / 2025 forward
- 出力: `results/family_B_failedbreakout/B002/`

## 結果（DataviewでCSV参照）

```dataviewjs
const paths = {
  verify_m: "results/family_B_failedbreakout/B002/in_sample_2024/monthly.csv",
  forward_m: "results/family_B_failedbreakout/B002/forward_2025/monthly.csv",
  forward_ms: "results/family_B_failedbreakout/B002/forward_2025/monthly_by_session.csv",
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

理由：フォワード（2025）がプラス（`sum_pnl_pips=+190`）で、baseline（B001）より改善

## 月次PnL分布の読み（B002 / 2025 forward）

見たいこと：
- 利益が特定の月に偏っていないか（少数の月だけで総利益を作っていないか）
- 負け月の形が再現的か（毎回似た負け方なのか、単発の事故なのか）

### 月次PnLテーブル（CSV参照）

```dataviewjs
const path = "results/family_B_failedbreakout/B002/forward_2025/monthly.csv";
const data = await dv.io.csv(path);
const rows = (data && typeof data.array === "function")
  ? data.array()
  : (Array.isArray(data) ? data : (data?.values ?? []));

function num(x) {
  const v = Number(x);
  return Number.isFinite(v) ? v : 0;
}

const m = rows.map(r => ({
  month: r.month,
  trades: num(r.trades),
  sum_pnl_pips: num(r.sum_pnl_pips),
  avg_pnl_pips: num(r.avg_pnl_pips),
  winrate: num(r.winrate),
})).sort((a,b) => String(a.month).localeCompare(String(b.month)));

dv.table(
  ["month","trades","sum_pnl_pips","avg_pnl_pips","winrate"],
  m.map(r => [r.month, r.trades, r.sum_pnl_pips.toFixed(1), r.avg_pnl_pips.toFixed(3), r.winrate.toFixed(3)])
);
```

### 偏りチェック（上位月の寄与 / 負け月の並び）

```dataviewjs
const path = "results/family_B_failedbreakout/B002/forward_2025/monthly.csv";
const data = await dv.io.csv(path);
const rows = (data && typeof data.array === "function")
  ? data.array()
  : (Array.isArray(data) ? data : (data?.values ?? []));

function num(x) {
  const v = Number(x);
  return Number.isFinite(v) ? v : 0;
}

const m = rows.map(r => ({ month: r.month, pnl: num(r.sum_pnl_pips), trades: num(r.trades) }));
const total = m.reduce((a, r) => a + r.pnl, 0);
const pos = m.filter(r => r.pnl > 0).sort((a,b) => b.pnl - a.pnl);
const neg = m.filter(r => r.pnl < 0).sort((a,b) => a.pnl - b.pnl);

function topShare(k) {
  const s = pos.slice(0, k).reduce((a, r) => a + r.pnl, 0);
  return total !== 0 ? s / total : 0;
}

dv.header(3, "上位月寄与（pnlがプラスの月のみ）");
dv.table(
  ["metric", "value"],
  [
    ["total_pnl_pips", total.toFixed(1)],
    ["positive_months", String(pos.length)],
    ["negative_months", String(neg.length)],
    ["top1_share_of_total", topShare(1).toFixed(3)],
    ["top2_share_of_total", topShare(2).toFixed(3)],
    ["top3_share_of_total", topShare(3).toFixed(3)],
  ]
);

dv.header(3, "ワースト月（負け月の形）");
dv.table(
  ["rank","month","sum_pnl_pips","trades"],
  neg.slice(0, 6).map((r, i) => [i+1, r.month, r.pnl.toFixed(1), r.trades])
);
```

読み方（目安）：
- `top1/top2/top3_share_of_total` が 1.0 に近いほど「特定の月に偏る」。
- ワースト月が少数で、他が小さな負け/勝ちなら「事故月に依存」。
- ワースト月が複数あり、毎回似た規模で負けるなら「負け月の形が再現的（＝構造的な欠陥が残っている）」可能性が高い。
