#!/usr/bin/env python3
"""
Observe D013: tail contribution and failure modes on D009 trades.

Definitions (fixed; no optimization):
  - early_loss: holding_time_min in [0, 3]
  - tail share: contribution of top/bottom 1/5/10/20% trades by pnl

Outputs (overwritten):
  - results/family_D_momentum/D013/summary_verify.csv
  - results/family_D_momentum/D013/summary_forward.csv
  - results/family_D_momentum/D013/thresholds.json
  - results/family_D_momentum/D013/plots/*.png
  - results/family_D_momentum/D013/README.md
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PeriodInputs:
    period: str
    trades_csv: Path
    core_config_json: Path


TAIL_PCTS = [1, 5, 10, 20]
EARLY_LOSS_MAX_MIN = 3.0


def _read_period_inputs(d009_dir: Path) -> list[PeriodInputs]:
    out: list[PeriodInputs] = []
    for name in ("in_sample_2024", "forward_2025"):
        d = d009_dir / name
        if not d.exists():
            continue
        trades = d / "trades.csv"
        core_cfg = d / "core_config.json"
        if trades.exists() and core_cfg.exists():
            out.append(PeriodInputs(period=name, trades_csv=trades, core_config_json=core_cfg))
    if out:
        return out

    # Fallback: use top-level trades.csv if period dirs not found
    trades = d009_dir / "trades.csv"
    core_cfg = d009_dir / "core_config.json"
    if trades.exists() and core_cfg.exists():
        return [PeriodInputs(period="all", trades_csv=trades, core_config_json=core_cfg)]

    raise FileNotFoundError(f"No trades.csv/core_config.json found under: {d009_dir}")


def _select_pnl_column(cols: set[str]) -> str:
    for c in ("pnl_pips", "pnl", "pnl_ticks"):
        if c in cols:
            return c
    raise ValueError(f"No pnl column found (expected pnl_pips/pnl/pnl_ticks). cols={sorted(cols)}")


def _compute_holding_time_min(trades):
    import pandas as pd

    if "entry_time" in trades.columns and "exit_time" in trades.columns:
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce")
        holding_sec = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds()
        trades["holding_time_min"] = holding_sec / 60.0
        return trades, "entry_exit"

    for col in ("holding_time_min", "holding_secs", "holding_time_sec"):
        if col in trades.columns:
            series = pd.to_numeric(trades[col], errors="coerce")
            if col == "holding_time_min":
                trades["holding_time_min"] = series
                return trades, col
            trades["holding_time_min"] = series / 60.0
            return trades, col

    raise ValueError("No holding time columns found (need entry_time/exit_time or holding_time_min/holding_secs).")


def _tail_shares(pnl, *, pcts: list[int]):
    import numpy as np

    pnl = pnl.astype(float)
    n = len(pnl)
    total = float(np.sum(pnl)) if n else float("nan")
    if n == 0 or total == 0.0 or not np.isfinite(total):
        return {f"top_{p}_share": float("nan") for p in pcts} | {f"bottom_{p}_share": float("nan") for p in pcts}

    pnl_sorted = np.sort(pnl)[::-1]
    pnl_sorted_low = pnl_sorted[::-1]

    out = {}
    for p in pcts:
        k = max(1, int(math.ceil(n * (p / 100.0))))
        out[f"top_{p}_share"] = float(np.sum(pnl_sorted[:k]) / total)
        out[f"bottom_{p}_share"] = float(np.sum(pnl_sorted_low[:k]) / total)
    return out


def _failure_modes(pnl, early_mask):
    import numpy as np

    pnl = pnl.astype(float)
    n = len(pnl)
    total = float(np.sum(pnl)) if n else float("nan")
    early_n = int(np.sum(early_mask)) if n else 0
    non_early_n = int(n - early_n) if n else 0

    early_pnl = float(np.sum(pnl[early_mask])) if n else float("nan")
    non_early_pnl = float(np.sum(pnl[~early_mask])) if n else float("nan")

    if n == 0 or total == 0.0 or not np.isfinite(total):
        early_share = float("nan")
        non_early_share = float("nan")
    else:
        early_share = float(early_pnl / total)
        non_early_share = float(non_early_pnl / total)

    return {
        "early_loss_trades": early_n,
        "non_early_trades": non_early_n,
        "early_loss_rate": float(early_n / n) if n else float("nan"),
        "non_early_rate": float(non_early_n / n) if n else float("nan"),
        "early_loss_pnl": early_pnl,
        "non_early_pnl": non_early_pnl,
        "early_loss_pnl_share": early_share,
        "non_early_pnl_share": non_early_share,
    }


def _summarize(df, *, period_label: str, pnl_col: str):
    import pandas as pd

    if df.empty:
        return pd.DataFrame()

    base = df.copy()
    base["side"] = base["side"].astype(str).str.lower()
    all_side = base.copy()
    all_side["side"] = "all"
    merged = pd.concat([base, all_side], ignore_index=True)

    rows = []
    for side in ("all", "buy", "sell"):
        g = merged[merged["side"] == side]
        pnl = g[pnl_col].astype(float)
        early_mask = g["early_loss"].astype(bool).to_numpy()
        tail = _tail_shares(pnl, pcts=TAIL_PCTS)
        fail = _failure_modes(pnl.to_numpy(), early_mask)
        row = {
            "period": period_label,
            "side": side,
            "n_trades": int(len(g)),
            "total_pnl": float(pnl.sum()) if len(g) else float("nan"),
        }
        row.update(tail)
        row.update(fail)
        rows.append(row)

    return pd.DataFrame(rows)


def _plot_tail_shares(summary_df, *, period_label: str, side: str, out_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row = summary_df[(summary_df["period"] == period_label) & (summary_df["side"] == side)]
    if row.empty:
        return

    r = row.iloc[0]
    labels = [f"top_{p}%" for p in TAIL_PCTS] + [f"bottom_{p}%" for p in TAIL_PCTS]
    values = [r[f"top_{p}_share"] for p in TAIL_PCTS] + [r[f"bottom_{p}_share"] for p in TAIL_PCTS]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, values, color=["#4C78A8"] * len(TAIL_PCTS) + ["#E45756"] * len(TAIL_PCTS))
    ax.axhline(0.0, color="#444", linewidth=0.8)
    ax.set_title(f"Tail share ({period_label}, {side})")
    ax.set_ylabel("share of total pnl")
    ax.set_xlabel("percentile group")
    fig.tight_layout()
    out_path = out_dir / f"tail_share_{period_label}_{side}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_failure_modes(summary_df, *, period_label: str, side: str, out_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row = summary_df[(summary_df["period"] == period_label) & (summary_df["side"] == side)]
    if row.empty:
        return

    r = row.iloc[0]
    labels = ["early_loss_rate", "early_loss_pnl_share", "non_early_pnl_share"]
    values = [r["early_loss_rate"], r["early_loss_pnl_share"], r["non_early_pnl_share"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["#F58518", "#E45756", "#54A24B"])
    ax.axhline(0.0, color="#444", linewidth=0.8)
    ax.set_title(f"Failure modes ({period_label}, {side})")
    ax.set_ylabel("ratio")
    fig.tight_layout()
    out_path = out_dir / f"failure_modes_{period_label}_{side}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_thresholds(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "thresholds.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_readme(out_dir: Path, *, meta: dict, pnl_col: str, d009_dir: Path, data_root: Path | None) -> None:
    lines: list[str] = []
    lines.append("# D013 tail contribution / failure modes (observe)")
    lines.append("")
    lines.append("## Purpose")
    lines.append("- Observe tail share and failure modes on D009 trades.")
    lines.append("- No filtering; aggregation only.")
    lines.append("")
    lines.append("## Definitions")
    lines.append(f"- early_loss: holding_time_min in [0, {EARLY_LOSS_MAX_MIN}]")
    lines.append(f"- pnl column: {pnl_col}")
    lines.append("- tail share: top/bottom 1/5/10/20% contribution to total pnl.")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- d009_dir: `{d009_dir}`")
    if data_root is not None:
        lines.append(f"- data_root (override): `{data_root}`")
    for period, info in meta.items():
        lines.append(f"- {period}: trades_csv=`{info['trades_csv']}` core_config=`{info['core_config_json']}`")
        lines.append(
            f"  raw={info['n_trades_raw']} time_valid={info['n_time_valid']} side_valid={info['n_side_valid']} "
            f"pnl_valid={info['n_pnl_valid']} pnl_missing={info['n_pnl_missing']}"
        )
    lines.append("")
    lines.append("## Outputs")
    lines.append("- summary_verify.csv / summary_forward.csv")
    lines.append("- thresholds.json")
    lines.append("- plots/*.png")
    lines.append("- README.md")
    lines.append("")
    lines.append("## Usage")
    lines.append("```bash")
    lines.append("python FX/code/observe_D013_tail_contribution.py \\")
    lines.append("  --d009_dir FX/results/family_D_momentum/D009 \\")
    lines.append("  --out_dir FX/results/family_D_momentum/D013")
    lines.append("```")
    lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d009_dir", type=Path, default=Path("FX/results/family_D_momentum/D009"))
    parser.add_argument("--out_dir", type=Path, default=Path("FX/results/family_D_momentum/D013"))
    parser.add_argument("--data_root", type=Path, default=None)
    args = parser.parse_args()

    d009_dir = args.d009_dir
    out_dir = args.out_dir
    data_root = args.data_root

    periods = _read_period_inputs(d009_dir)
    period_map = {"in_sample_2024": "verify", "forward_2025": "forward", "all": "all"}

    period_frames = {}
    period_meta = {}
    pnl_col = None

    for p in periods:
        import pandas as pd

        trades_raw = pd.read_csv(p.trades_csv)
        n_raw = int(len(trades_raw))

        if trades_raw.empty:
            period_frames[p.period] = trades_raw
            period_meta[p.period] = {
                "trades_csv": str(p.trades_csv),
                "core_config_json": str(p.core_config_json),
                "n_trades_raw": n_raw,
                "n_time_valid": 0,
                "n_side_valid": 0,
                "n_pnl_valid": 0,
                "n_pnl_missing": 0,
            }
            continue

        if pnl_col is None:
            pnl_col = _select_pnl_column(set(trades_raw.columns))

        trades = trades_raw.copy()
        trades, time_src = _compute_holding_time_min(trades)
        time_valid = trades["holding_time_min"].notna() & (trades["holding_time_min"] >= 0)
        trades = trades[time_valid].copy()
        n_time_valid = int(len(trades))

        trades["side"] = trades["side"].astype(str).str.lower()
        side_valid = trades["side"].isin(["buy", "sell"])
        trades = trades[side_valid].copy()
        n_side_valid = int(len(trades))

        trades[pnl_col] = pd.to_numeric(trades[pnl_col], errors="coerce")
        pnl_valid = trades[pnl_col].notna()
        n_pnl_valid = int(pnl_valid.sum())
        n_pnl_missing = int(len(trades) - n_pnl_valid)
        trades = trades[pnl_valid].copy()

        trades["early_loss"] = (trades["holding_time_min"] >= 0.0) & (trades["holding_time_min"] <= EARLY_LOSS_MAX_MIN)

        period_frames[p.period] = trades
        period_meta[p.period] = {
            "trades_csv": str(p.trades_csv),
            "core_config_json": str(p.core_config_json),
            "n_trades_raw": n_raw,
            "n_time_valid": n_time_valid,
            "n_side_valid": n_side_valid,
            "n_pnl_valid": n_pnl_valid,
            "n_pnl_missing": n_pnl_missing,
            "holding_time_source": time_src,
        }

    if pnl_col is None:
        raise ValueError("No pnl column detected in trades.")

    summaries = {}
    for key, df in period_frames.items():
        period_label = period_map.get(key, key)
        summaries[key] = _summarize(df, period_label=period_label, pnl_col=pnl_col)

    out_dir.mkdir(parents=True, exist_ok=True)
    if "in_sample_2024" in summaries:
        summaries["in_sample_2024"].to_csv(out_dir / "summary_verify.csv", index=False)
    if "forward_2025" in summaries:
        summaries["forward_2025"].to_csv(out_dir / "summary_forward.csv", index=False)
    if "all" in summaries:
        summaries["all"].to_csv(out_dir / "summary_all.csv", index=False)

    thresholds_payload = {
        "early_loss_definition": f"holding_time_min in [0, {EARLY_LOSS_MAX_MIN}] (D007)",
        "tail_percentiles": TAIL_PCTS,
        "pnl_column": pnl_col,
        "period_map": period_map,
        "meta": period_meta,
    }
    _write_thresholds(out_dir, thresholds_payload)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for key, summary in summaries.items():
        if summary.empty:
            continue
        period_label = period_map.get(key, key)
        for side in ("all", "buy", "sell"):
            _plot_tail_shares(summary, period_label=period_label, side=side, out_dir=plots_dir)
            _plot_failure_modes(summary, period_label=period_label, side=side, out_dir=plots_dir)

    _write_readme(out_dir, meta=period_meta, pnl_col=pnl_col, d009_dir=d009_dir, data_root=data_root)
    print(f"[D013] done: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
