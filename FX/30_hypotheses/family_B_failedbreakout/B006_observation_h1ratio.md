---
regime: observation_only
strategy: none
result: informational
---

B006（観測仮説）：
- 対象: B002（family_B_failedbreakout / W1 only）
- 仮説: 勝ち月と負け月で「H1 uptrend（既存定義）の true 時間割合」が違うのではないか
- 注意: **勝たせるための変更は一切なし**（観測のみ）

出力先：
- `results/family_B_failedbreakout/B006_observation/h1_uptrend_monthly_ratio_2025.csv`
- `results/family_B_failedbreakout/B006_observation/merged_b002_pnl_vs_h1ratio_2025.csv`
- `results/family_B_failedbreakout/B006_observation/summary.txt`
- `results/family_B_failedbreakout/B006_observation/config.json`

## 集計（summary.txt）

```dataviewjs
const p = "results/family_B_failedbreakout/B006_observation/summary.txt";
dv.paragraph(await dv.io.load(p));
```

## 月次テーブル（PnL vs H1 ratio）

```dataviewjs
const p = "results/family_B_failedbreakout/B006_observation/merged_b002_pnl_vs_h1ratio_2025.csv";
const data = await dv.io.csv(p);
const rows = (data && typeof data.array === "function")
  ? data.array()
  : (Array.isArray(data) ? data : (data?.values ?? []));

function num(x) {
  const v = Number(x);
  return Number.isFinite(v) ? v : 0;
}

rows.sort((a,b) => String(a.month).localeCompare(String(b.month)));
dv.table(
  ["month","sum_pnl_pips","trades","h1_uptrend_ratio"],
  rows.map(r => [r.month, num(r.sum_pnl_pips).toFixed(1), num(r.trades), num(r.h1_uptrend_ratio).toFixed(3)])
);
```

