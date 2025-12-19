#!/usr/bin/env python3
"""
Postprocess: holding time distribution and PnL by holding-time bins.

Scope:
  - Aggregation only (no strategy logic changes).
  - Reads either diagnostics/diag_trades.csv (preferred) or trades.csv.
  - Works per period if folders exist: in_sample_2024 / forward_2025.

Fixed bins (minutes; non-optimized):
  - 0–1, 1–3, 3–5, 5–10, 10–20, 20+

Example:
  uv run python FX/code/postprocess/holding_time_analysis.py \
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
    entry_col: str
    exit_col: str
    pnl_col: str


BIN_EDGES_MIN = [0.0, 1.0, 3.0, 5.0, 10.0, 20.0, float("inf")]
BIN_LABELS = ["0-1", "1-3", "3-5", "5-10", "10-20", "20+"]


def _read_trades_like_csv(hyp_dir: Path, period_dir: Path) -> tuple["object", Inputs]:
    import pandas as pd

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
        return df, Inputs(source_csv=src, source_kind="unknown", entry_col="entry_time", exit_col="exit_time", pnl_col="pnl_pips")

    cols = set(df.columns)
    entry_col = "entry_time" if "entry_time" in cols else None
    exit_col = "exit_time" if "exit_time" in cols else None
    if entry_col is None or exit_col is None:
        raise ValueError(f"{src} missing entry/exit time columns (need entry_time and exit_time)")

    pnl_col = None
    for c in ("pnl_pips", "pnl", "pnl_ticks"):
        if c in cols:
            pnl_col = c
            break
    if pnl_col is None:
        raise ValueError(f"{src} missing pnl column (expected pnl_pips/pnl/pnl_ticks)")

    kind = "diag_trades" if src.name == "diag_trades.csv" else "trades"
    return df, Inputs(source_csv=src, source_kind=kind, entry_col=entry_col, exit_col=exit_col, pnl_col=pnl_col)


def _compute_bins(df, *, period: str, inputs: Inputs):
    import numpy as np
    import pandas as pd

    if df.empty:
        out = pd.DataFrame(
            columns=[
                "period",
                "bin_label",
                "n_trades",
                "sum_pnl_pips",
                "avg_pnl_pips",
                "win_rate",
                "n_win",
                "n_loss",
                "sum_pnl_win",
                "avg_pnl_win",
                "sum_pnl_loss",
                "avg_pnl_loss",
            ]
        )
        return out

    d = df.copy()
    d[inputs.entry_col] = pd.to_datetime(d[inputs.entry_col], utc=True, errors="coerce")
    d[inputs.exit_col] = pd.to_datetime(d[inputs.exit_col], utc=True, errors="coerce")
    d = d.dropna(subset=[inputs.entry_col, inputs.exit_col])

    d[inputs.pnl_col] = pd.to_numeric(d[inputs.pnl_col], errors="coerce")
    d = d.dropna(subset=[inputs.pnl_col])

    holding_sec = (d[inputs.exit_col] - d[inputs.entry_col]).dt.total_seconds()
    d["holding_time_sec"] = holding_sec
    d = d[(d["holding_time_sec"] >= 0) & (d["holding_time_sec"].notna())]
    d["holding_time_min"] = d["holding_time_sec"] / 60.0

    d["is_win"] = d[inputs.pnl_col] > 0
    d["is_loss"] = ~d["is_win"]

    # Fixed bins; left-closed / right-open (except +inf).
    d["bin_label"] = pd.cut(
        d["holding_time_min"].astype(float),
        bins=BIN_EDGES_MIN,
        labels=BIN_LABELS,
        right=False,
        include_lowest=True,
    ).astype(str)

    # Guard: if pandas renders NaN bin as "nan"
    d = d[d["bin_label"].isin(BIN_LABELS)]

    def _agg(g):
        pnl = g[inputs.pnl_col].to_numpy(dtype=float)
        n = int(len(pnl))
        n_win = int(np.sum(pnl > 0))
        n_loss = n - n_win
        sum_pnl = float(np.sum(pnl)) if n else 0.0
        avg_pnl = float(np.mean(pnl)) if n else float("nan")
        win_rate = float(n_win / n) if n else float("nan")

        pnl_win = pnl[pnl > 0]
        pnl_loss = pnl[pnl <= 0]
        sum_win = float(np.sum(pnl_win)) if len(pnl_win) else 0.0
        sum_loss = float(np.sum(pnl_loss)) if len(pnl_loss) else 0.0
        avg_win = float(np.mean(pnl_win)) if len(pnl_win) else float("nan")
        avg_loss = float(np.mean(pnl_loss)) if len(pnl_loss) else float("nan")
        return pd.Series(
            {
                "n_trades": n,
                "sum_pnl_pips": sum_pnl,
                "avg_pnl_pips": avg_pnl,
                "win_rate": win_rate,
                "n_win": n_win,
                "n_loss": n_loss,
                "sum_pnl_win": sum_win,
                "avg_pnl_win": avg_win,
                "sum_pnl_loss": sum_loss,
                "avg_pnl_loss": avg_loss,
            }
        )

    out = d.groupby("bin_label", as_index=False).apply(_agg, include_groups=False)
    out.insert(0, "period", period)

    # Normalize dtypes (keep integers as integers for stable CSV)
    for c in ("n_trades", "n_win", "n_loss"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    # Keep bin order stable
    out["bin_label"] = pd.Categorical(out["bin_label"], categories=BIN_LABELS, ordered=True)
    out = out.sort_values(["period", "bin_label"]).reset_index(drop=True)
    out["bin_label"] = out["bin_label"].astype(str)
    return out


def _compute_by_minute(df, *, period: str, inputs: Inputs, max_minutes: int = 60):
    import numpy as np
    import pandas as pd

    if df.empty:
        return pd.DataFrame(columns=["period", "minute", "n_trades", "sum_pnl_pips", "avg_pnl_pips"])

    d = df.copy()
    d[inputs.entry_col] = pd.to_datetime(d[inputs.entry_col], utc=True, errors="coerce")
    d[inputs.exit_col] = pd.to_datetime(d[inputs.exit_col], utc=True, errors="coerce")
    d = d.dropna(subset=[inputs.entry_col, inputs.exit_col])

    d[inputs.pnl_col] = pd.to_numeric(d[inputs.pnl_col], errors="coerce")
    d = d.dropna(subset=[inputs.pnl_col])

    holding_sec = (d[inputs.exit_col] - d[inputs.entry_col]).dt.total_seconds()
    d = d[(holding_sec >= 0) & holding_sec.notna()]
    holding_min = holding_sec / 60.0

    minute = np.floor(holding_min).astype(int)
    minute = np.clip(minute, 0, int(max_minutes))
    d["minute"] = minute

    out = (
        d.groupby("minute", as_index=False)
        .agg(
            n_trades=(inputs.pnl_col, "count"),
            sum_pnl_pips=(inputs.pnl_col, "sum"),
            avg_pnl_pips=(inputs.pnl_col, "mean"),
        )
        .assign(period=period)
        .loc[:, ["period", "minute", "n_trades", "sum_pnl_pips", "avg_pnl_pips"]]
        .sort_values(["period", "minute"])
    )
    return out


def _write_summary_md(out_dir: Path, *, hyp_dir: Path, inputs_meta: dict, bins_df, by_min_df, max_minutes: int) -> None:
    lines: list[str] = []
    lines.append("# holding_time（集計）")
    lines.append("")
    lines.append("## 前提")
    lines.append(f"- hyp_dir: `{hyp_dir}`")
    lines.append("- holding_time_sec = (exit_time - entry_time).total_seconds()")
    lines.append("- holding_time_min = holding_time_sec / 60.0")
    lines.append("- win: pnl_pips > 0 / loss: pnl_pips <= 0")
    lines.append("")
    lines.append("## 入力ファイル")
    for period, info in inputs_meta.items():
        lines.append(f"- {period}: `{info['source_csv']}`（kind={info['source_kind']} cols: entry={info['entry_col']} exit={info['exit_col']} pnl={info['pnl_col']}）")
    lines.append("")
    lines.append("## ビン（固定）")
    lines.append("- 0–1分 / 1–3分 / 3–5分 / 5–10分 / 10–20分 / 20分以上")
    lines.append("")

    if bins_df.empty:
        lines.append("## 集計結果")
        lines.append("- （データなし）")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "holding_time_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    # Top/bottom bins by avg_pnl_pips (fact only)
    lines.append("## Top/Bot（avg_pnl_pips）")
    for period, grp in bins_df.groupby("period"):
        g2 = grp[grp["n_trades"] > 0].copy()
        if g2.empty:
            continue
        best = g2.sort_values(["avg_pnl_pips", "n_trades"], ascending=[False, False]).head(3)
        worst = g2.sort_values(["avg_pnl_pips", "n_trades"], ascending=[True, False]).head(3)

        lines.append(f"### {period}")
        lines.append("- best (top3):")
        for _, r in best.iterrows():
            lines.append(f"  - {r['bin_label']}: n={int(r['n_trades'])} avg_pnl_pips={float(r['avg_pnl_pips']):.2f} sum_pnl_pips={float(r['sum_pnl_pips']):.1f} win_rate={float(r['win_rate']):.2f}")
        lines.append("- worst (bottom3):")
        for _, r in worst.iterrows():
            lines.append(f"  - {r['bin_label']}: n={int(r['n_trades'])} avg_pnl_pips={float(r['avg_pnl_pips']):.2f} sum_pnl_pips={float(r['sum_pnl_pips']):.1f} win_rate={float(r['win_rate']):.2f}")
    lines.append("")

    if not by_min_df.empty:
        lines.append("## 参考（holding_time_min を floor(min) で丸めた集計）")
        lines.append(f"- 0..{max_minutes} 分にクリップし、minuteごとの avg_pnl_pips を出力")
        lines.append(f"- csv: `{out_dir / 'holding_time_by_minute.csv'}`")
        lines.append("")

    lines.append("## 実行例")
    lines.append(f"- `uv run python FX/code/postprocess/holding_time_analysis.py --hyp_dir {hyp_dir}`")
    lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "holding_time_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hyp_dir", required=True, help="e.g. FX/results/family_D_momentum/D004")
    p.add_argument("--max_minutes", type=int, default=60, help="Max minute bucket for by-minute output (optional).")
    args = p.parse_args()

    import pandas as pd

    hyp_dir = Path(args.hyp_dir)
    if not hyp_dir.exists():
        raise FileNotFoundError(str(hyp_dir))

    periods: list[tuple[str, Path]] = []
    for name in ("in_sample_2024", "forward_2025"):
        d = hyp_dir / name
        if d.exists():
            periods.append((name, d))
    if not periods:
        periods = [("all", hyp_dir)]

    inputs_meta: dict[str, dict] = {}
    bins_all = []
    by_min_all = []

    for period, period_dir in periods:
        df, inputs = _read_trades_like_csv(hyp_dir, period_dir)
        inputs_meta[period] = {
            "source_csv": str(inputs.source_csv),
            "source_kind": inputs.source_kind,
            "entry_col": inputs.entry_col,
            "exit_col": inputs.exit_col,
            "pnl_col": inputs.pnl_col,
        }

        bins_df = _compute_bins(df, period=period, inputs=inputs)
        bins_all.append(bins_df)

        by_min_df = _compute_by_minute(df, period=period, inputs=inputs, max_minutes=int(args.max_minutes))
        by_min_all.append(by_min_df)

    bins_out = pd.concat(bins_all, ignore_index=True) if bins_all else pd.DataFrame()
    by_min_out = pd.concat(by_min_all, ignore_index=True) if by_min_all else pd.DataFrame()

    out_dir = hyp_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "holding_time_bins.csv").write_text(bins_out.to_csv(index=False), encoding="utf-8")
    if not by_min_out.empty:
        (out_dir / "holding_time_by_minute.csv").write_text(by_min_out.to_csv(index=False), encoding="utf-8")

    _write_summary_md(out_dir, hyp_dir=hyp_dir, inputs_meta=inputs_meta, bins_df=bins_out, by_min_df=by_min_out, max_minutes=int(args.max_minutes))

    print(f"[holding_time_analysis] wrote: {out_dir / 'holding_time_bins.csv'}", flush=True)
    print(f"[holding_time_analysis] wrote: {out_dir / 'holding_time_summary.md'}", flush=True)
    if not by_min_out.empty:
        print(f"[holding_time_analysis] wrote: {out_dir / 'holding_time_by_minute.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
