#!/usr/bin/env python3
"""
Postprocess: entry distribution by weekday x hour (UTC).

Scope:
  - Aggregation only (no strategy logic changes).
  - Reads either diagnostics/diag_trades.csv (preferred) or trades.csv.
  - Uses signal_eval_ts if present; otherwise approximates:
      signal_eval_ts ~= entry_time - 10 seconds
    (because core uses entry_on_next_open=ts_next).

Example:
  uv run python FX/code/postprocess/entry_time_analysis.py \
    --hyp_dir FX/results/family_D_momentum/D004
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Inputs:
    source_csv: Path
    source_kind: str  # "diag_trades" | "trades"
    used_signal_eval: str  # "signal_eval_ts" | "entry_time_minus_10s"


def _read_trades_like_csv(hyp_dir: Path, period_dir: Path) -> tuple["object", Inputs]:
    import pandas as pd

    # Priority: diagnostics/diag_trades.csv (per-hyp or per-period if present)
    candidates = [
        hyp_dir / "diagnostics" / "diag_trades.csv",
        period_dir / "diagnostics" / "diag_trades.csv",
        period_dir / "trades.csv",
        hyp_dir / "trades.csv",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError(f"No diag_trades.csv or trades.csv found under: {hyp_dir}")

    df = pd.read_csv(src)
    if df.empty:
        return df, Inputs(source_csv=src, source_kind="unknown", used_signal_eval="unknown")

    cols = set(df.columns)
    if "signal_eval_ts" in cols:
        used = "signal_eval_ts"
    else:
        used = "entry_time_minus_10s"

    kind = "diag_trades" if src.name == "diag_trades.csv" else "trades"
    return df, Inputs(source_csv=src, source_kind=kind, used_signal_eval=used)


def _compute_entry_dist(df, period: str, inputs: Inputs):
    import pandas as pd

    if df.empty:
        return pd.DataFrame(
            columns=["period", "dow", "hour_utc", "n_entries", "sum_pnl_pips", "avg_pnl_pips"]
        ), inputs

    # Normalize column names across possible sources
    col_entry = "entry_time" if "entry_time" in df.columns else None
    if col_entry is None:
        raise ValueError(f"{inputs.source_csv} missing entry_time")

    col_pnl = None
    for c in ("pnl_pips", "pnl", "pnl_pips_sum"):
        if c in df.columns:
            col_pnl = c
            break
    if col_pnl is None:
        raise ValueError(f"{inputs.source_csv} missing pnl column (expected pnl_pips/pnl)")

    d = df.copy()
    d[col_entry] = pd.to_datetime(d[col_entry], utc=True, errors="coerce")
    d = d.dropna(subset=[col_entry])
    d[col_pnl] = pd.to_numeric(d[col_pnl], errors="coerce")

    if inputs.used_signal_eval == "signal_eval_ts":
        d["signal_eval_ts"] = pd.to_datetime(d["signal_eval_ts"], utc=True, errors="coerce")
        d = d.dropna(subset=["signal_eval_ts"])
    else:
        d["signal_eval_ts"] = d[col_entry] - pd.Timedelta(seconds=10)

    ts = d["signal_eval_ts"]
    d["dow"] = ts.dt.day_name().str.slice(0, 3)  # Mon..Sun
    d["hour_utc"] = ts.dt.hour.astype(int)

    out = (
        d.groupby(["dow", "hour_utc"], as_index=False)
        .agg(
            n_entries=(col_pnl, "count"),
            sum_pnl_pips=(col_pnl, "sum"),
            avg_pnl_pips=(col_pnl, "mean"),
        )
        .assign(period=period)
        .loc[:, ["period", "dow", "hour_utc", "n_entries", "sum_pnl_pips", "avg_pnl_pips"]]
    )
    return out, inputs


def _pivot_n_entries(long_df):
    import pandas as pd

    if long_df.empty:
        return pd.DataFrame()

    piv = long_df.pivot_table(
        index=["period", "dow"],
        columns="hour_utc",
        values="n_entries",
        aggfunc="sum",
        fill_value=0,
    )
    piv.columns = [f"h{int(c):02d}" for c in piv.columns]
    piv = piv.reset_index()
    return piv


def _read_hyp_config(hyp_dir: Path) -> dict:
    """
    Best-effort read of `{hyp_dir}/config.json` produced by backtest_runner.
    Used only to make the summary.md factual (what filters were on/off).
    """
    import json

    p = hyp_dir / "config.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _write_md(out_dir: Path, *, hyp_dir: Path, rows, piv, meta: dict):
    import pandas as pd

    lines = []
    lines.append("# entry_by_dow_hour（集計）")
    lines.append("")
    lines.append("## 前提")
    lines.append(f"- hyp_dir: `{hyp_dir}`")
    lines.append("- 集計時刻（periodごと）:")
    for period, info in meta["inputs"].items():
        used = info.get("used_signal_eval", "unknown")
        lines.append(f"  - {period}: `{used}`")
        if used == "entry_time_minus_10s":
            lines.append("    - `signal_eval_ts ~= entry_time - 10秒`（coreの entry_on_next_open=ts_next に基づく近似）")
    lines.append("- timezone: UTC（timestampは tz-aware として処理）")
    lines.append("")
    cfg = meta.get("config", {})
    diff = cfg.get("diff", {}) if isinstance(cfg, dict) else {}
    if isinstance(diff, dict) and diff:
        only_session = diff.get("only_session", "(unknown)")
        use_time_filter = diff.get("use_time_filter", "(unknown)")
        use_h1 = diff.get("use_h1_trend_filter", "(unknown)")
        no_weekend = diff.get("no_weekend_entry", "(unknown)")
        lines.append("## 設定メモ（runner config.json の diff）")
        lines.append(f"- only_session={only_session}")
        lines.append(f"- use_time_filter={use_time_filter}")
        lines.append(f"- use_h1_trend_filter={use_h1}")
        lines.append(f"- no_weekend_entry={no_weekend}")
        lines.append("")
    lines.append("## 入力ファイル")
    for period, info in meta["inputs"].items():
        lines.append(f"- {period}: `{info['source_csv']}`（kind={info['source_kind']}）")
    lines.append("")

    if rows.empty:
        lines.append("## Top10")
        lines.append("- （データなし）")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "entry_by_dow_hour_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    # Top 10 per period by n_entries
    lines.append("## Top10（n_entriesが多い順）")
    df = rows.copy()
    df = df.sort_values(["period", "n_entries"], ascending=[True, False])
    for period, grp in df.groupby("period"):
        lines.append(f"### {period}")
        top = grp.head(10)
        for _, r in top.iterrows():
            lines.append(
                f"- {r['dow']} h{int(r['hour_utc']):02d}: n={int(r['n_entries'])} sum_pnl_pips={float(r['sum_pnl_pips']):.1f} avg_pnl_pips={float(r['avg_pnl_pips']):.2f}"
            )
    lines.append("")

    # Worst avg_pnl (fact only; fixed rule: consider only cells with n_entries >= median)
    lines.append("## Top10（avg_pnl_pips が悪い順 / n_entries >= median(n_entries) のセルのみ）")
    for period, grp in rows.groupby("period"):
        if grp.empty:
            continue
        med = float(grp["n_entries"].median())
        g2 = grp[grp["n_entries"] >= med].copy()
        g2 = g2.sort_values(["avg_pnl_pips", "n_entries"], ascending=[True, False]).head(10)
        lines.append(f"### {period} (median_n_entries={med:.1f})")
        for _, r in g2.iterrows():
            lines.append(
                f"- {r['dow']} h{int(r['hour_utc']):02d}: n={int(r['n_entries'])} sum_pnl_pips={float(r['sum_pnl_pips']):.1f} avg_pnl_pips={float(r['avg_pnl_pips']):.2f}"
            )
    lines.append("")

    lines.append("## 参考（Fri>=20:00 / Mon<02:00 のエントリー数）")
    for period, counts in meta.get("forbidden_counts", {}).items():
        lines.append(f"- {period}: forbidden_window_entries={counts}")
    lines.append("")

    lines.append("## 実行例")
    lines.append(f"- `uv run python FX/code/postprocess/entry_time_analysis.py --hyp_dir {hyp_dir}`")
    lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "entry_by_dow_hour_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hyp_dir", required=True, help="e.g. FX/results/family_D_momentum/D004")
    args = p.parse_args()

    import pandas as pd

    hyp_dir = Path(args.hyp_dir)
    if not hyp_dir.exists():
        raise FileNotFoundError(str(hyp_dir))

    periods = []
    for name in ("in_sample_2024", "forward_2025"):
        d = hyp_dir / name
        if d.exists():
            periods.append((name, d))
    if not periods:
        periods = [("all", hyp_dir)]

    rows_all = []
    inputs_meta = {}
    forbidden_counts = {}

    for period, period_dir in periods:
        df, inputs = _read_trades_like_csv(hyp_dir, period_dir)
        rows, inputs2 = _compute_entry_dist(df, period=period, inputs=inputs)
        rows_all.append(rows)
        inputs_meta[period] = {
            "source_csv": str(inputs2.source_csv),
            "source_kind": inputs2.source_kind,
            "used_signal_eval": inputs2.used_signal_eval,
        }

        # forbidden window count (ts basis)
        if not df.empty:
            col_entry = "entry_time"
            d = df.copy()
            d[col_entry] = pd.to_datetime(d[col_entry], utc=True, errors="coerce")
            d = d.dropna(subset=[col_entry])
            if inputs2.used_signal_eval == "signal_eval_ts" and "signal_eval_ts" in d.columns:
                sig = pd.to_datetime(d["signal_eval_ts"], utc=True, errors="coerce")
            else:
                sig = d[col_entry] - pd.Timedelta(seconds=10)

            dow = sig.dt.dayofweek
            t = sig.dt.time
            from datetime import time as dtime
            fri = (dow == 4) & (t >= dtime(20, 0))
            mon = (dow == 0) & (t < dtime(2, 0))
            forbidden_counts[period] = int((fri | mon).sum())
        else:
            forbidden_counts[period] = 0

    rows = pd.concat(rows_all, ignore_index=True) if rows_all else pd.DataFrame()
    piv = _pivot_n_entries(rows)

    out_dir = hyp_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "entry_by_dow_hour.csv").write_text(rows.to_csv(index=False), encoding="utf-8")
    (out_dir / "entry_by_dow_hour_pivot.csv").write_text(piv.to_csv(index=False), encoding="utf-8")

    cfg = _read_hyp_config(hyp_dir)
    meta = {
        "inputs": inputs_meta,
        "forbidden_counts": forbidden_counts,
        "config": cfg,
    }
    _write_md(out_dir, hyp_dir=hyp_dir, rows=rows, piv=piv, meta=meta)

    print(f"[entry_time_analysis] wrote: {out_dir / 'entry_by_dow_hour.csv'}", flush=True)
    print(f"[entry_time_analysis] wrote: {out_dir / 'entry_by_dow_hour_pivot.csv'}", flush=True)
    print(f"[entry_time_analysis] wrote: {out_dir / 'entry_by_dow_hour_summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
