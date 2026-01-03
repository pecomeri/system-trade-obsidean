#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_core as core


EARLY_LOSS_MAX_MIN = 3.0
LABEL_WINDOW_MIN = 3
TAIL_PCTS = [1, 5, 10, 20]


@dataclass(frozen=True)
class PeriodSpec:
    label: str
    from_month: str
    to_month: str
    trades_csv: Path


def _load_d009_config() -> dict:
    path = Path("FX/results/family_D_momentum/D009/config.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_core_config() -> dict:
    path = Path("FX/results/family_D_momentum/D009/core_config.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_trades_csv(d009_dir: Path, *, run_dir: str | None, fallback_dir: str) -> Path:
    candidate = d009_dir / fallback_dir / "trades.csv"
    if candidate.exists():
        return candidate
    if run_dir:
        run_path = Path(run_dir) / "trades.csv"
        if run_path.exists():
            return run_path
    candidate = d009_dir / "trades.csv"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"trades.csv not found (checked {fallback_dir}, run_dir, and top-level under {d009_dir})")


def _compute_holding_time_min(trades: pd.DataFrame) -> pd.DataFrame:
    if "holding_time_min" in trades.columns:
        trades["holding_time_min"] = pd.to_numeric(trades["holding_time_min"], errors="coerce")
        return trades
    if "holding_secs" in trades.columns:
        trades["holding_time_min"] = pd.to_numeric(trades["holding_secs"], errors="coerce") / 60.0
        return trades
    if "entry_time" in trades.columns and "exit_time" in trades.columns:
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce")
        holding_sec = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds()
        trades["holding_time_min"] = holding_sec / 60.0
        return trades
    raise ValueError("No holding time columns found (need holding_time_min, holding_secs, or entry_time/exit_time).")


def _early_retrace_presence(
    df_win: pd.DataFrame,
    *,
    side: str,
    entry_price: float,
) -> str:
    if df_win.empty:
        return "no_retrace"
    if side == "buy":
        favorable = (df_win["high_bid"] > entry_price).any()
    else:
        favorable = (df_win["low_ask"] < entry_price).any()
    return "with_retrace" if favorable else "no_retrace"


def _label_early_retrace(trades: pd.DataFrame, df10: pd.DataFrame) -> list[str]:
    labels: list[str] = []
    window_delta = pd.Timedelta(minutes=LABEL_WINDOW_MIN)

    for row in trades.itertuples(index=False):
        entry_time = pd.Timestamp(row.entry_time)
        if entry_time.tzinfo is None:
            entry_time = entry_time.tz_localize("UTC")
        side = str(row.side).lower()
        entry_price = float(row.entry_price)

        df_win = df10.loc[entry_time : entry_time + window_delta]
        labels.append(_early_retrace_presence(df_win, side=side, entry_price=entry_price))

    return labels


def _select_pnl_column(cols: set[str]) -> str:
    for c in ("pnl_pips", "pnl", "pnl_ticks"):
        if c in cols:
            return c
    raise ValueError(f"No pnl column found (expected pnl_pips/pnl/pnl_ticks). cols={sorted(cols)}")


def _tail_shares(pnl: pd.Series, *, pcts: list[int]) -> dict[str, float]:
    pnl = pnl.astype(float)
    n = len(pnl)
    total = float(np.sum(pnl)) if n else float("nan")
    if n == 0 or total == 0.0 or not np.isfinite(total):
        return {f"top_{p}_share": float("nan") for p in pcts} | {f"bottom_{p}_share": float("nan") for p in pcts}

    pnl_sorted = np.sort(pnl)[::-1]
    pnl_sorted_low = pnl_sorted[::-1]

    out: dict[str, float] = {}
    for p in pcts:
        k = max(1, int(math.ceil(n * (p / 100.0))))
        out[f"top_{p}_share"] = float(np.sum(pnl_sorted[:k]) / total)
        out[f"bottom_{p}_share"] = float(np.sum(pnl_sorted_low[:k]) / total)
    return out


def _early_loss_summary(trades: pd.DataFrame, *, period: str, population: str) -> pd.DataFrame:
    base = trades.copy()
    base["side"] = base["side"].astype(str).str.lower()
    rows = []
    for side in ("buy", "sell"):
        g = base[base["side"] == side]
        rate = float(g["early_loss"].mean()) if len(g) else float("nan")
        rows.append(
            {
                "period": period,
                "population": population,
                "side": side,
                "early_loss_rate": rate,
            }
        )
    return pd.DataFrame(rows)


def _tail_summary(trades: pd.DataFrame, *, period: str, population: str, pnl_col: str) -> pd.DataFrame:
    base = trades.copy()
    base["side"] = base["side"].astype(str).str.lower()
    rows = []
    for side in ("buy", "sell"):
        g = base[base["side"] == side]
        tail = _tail_shares(g[pnl_col], pcts=TAIL_PCTS)
        row = {
            "period": period,
            "population": population,
            "side": side,
        }
        row.update(tail)
        rows.append(row)
    return pd.DataFrame(rows)


def _write_readme(out_dir: Path, *, notes: list[str]) -> None:
    lines = [
        "# D015: Distribution change by no_retrace exclusion (observe)",
        "",
        "## Definitions",
        "- baseline: results/family_D_momentum/D009 (24h, BUY/SELL)",
        "- modified: baseline excluding early_retrace_presence = no_retrace",
        "- early_loss: holding_time_min <= 3.0",
        "- early_retrace_presence: same definition as D014",
        "- tail share: top/bottom 1/5/10/20% contribution to total pnl (D013)",
        "",
        "## Notes",
        "- No entry/exit/SL/TP/time/direction changes.",
        "- No filtering optimization; observation only.",
        "- Interpretation: use tail share shifts to read concentration/skew; do not treat as performance improvement.",
        "- Tail share ratios can flip sign or inflate when total pnl is small; read directionally.",
        "",
        "## Distribution shape summary",
        *notes,
        "",
        "## Outputs",
        "- summary_verify.csv",
        "- summary_forward.csv",
        "- tail_compare.csv",
        "- README.md",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_tail_note(row_base: pd.Series, row_mod: pd.Series, *, label: str) -> str:
    def fmt(key: str) -> str:
        b = row_base.get(key)
        m = row_mod.get(key)
        if b is None or m is None or not np.isfinite(b) or not np.isfinite(m):
            return f"{key}=n/a"
        return f"{key}={b:.3f}->{m:.3f}"

    keys = ["top_10_share", "bottom_10_share", "top_20_share", "bottom_20_share"]
    return f"- {label}: " + ", ".join(fmt(k) for k in keys)


def main() -> None:
    cfg = _load_d009_config()
    core_cfg = _load_core_config()

    data_root = cfg["root"]
    symbol = cfg["symbol"]
    spread_pips = float(core_cfg.get("spread_pips", cfg.get("spread_pips", 1.0)))
    pip_size = float(core_cfg.get("pip_size", 0.01))

    d009_dir = Path("FX/results/family_D_momentum/D009")
    specs = [
        PeriodSpec(
            label="verify",
            from_month=cfg["verify"]["from_month"],
            to_month=cfg["verify"]["to_month"],
            trades_csv=_resolve_trades_csv(d009_dir, run_dir=cfg["verify"].get("run_dir"), fallback_dir="in_sample_2024"),
        ),
        PeriodSpec(
            label="forward",
            from_month=cfg["forward"]["from_month"],
            to_month=cfg["forward"]["to_month"],
            trades_csv=_resolve_trades_csv(d009_dir, run_dir=cfg["forward"].get("run_dir"), fallback_dir="forward_2025"),
        ),
    ]

    out_dir = Path("FX/results/family_D_momentum/D015")
    out_dir.mkdir(parents=True, exist_ok=True)

    tail_rows = []
    notes: list[str] = []

    for spec in specs:
        cfg_bt = core.Config(
            root=Path(data_root),
            symbol=symbol,
            from_month=spec.from_month,
            to_month=spec.to_month,
            spread_pips=spread_pips,
            pip_size=pip_size,
        )
        df10 = core.load_parquet_10s_bid(cfg_bt)
        df10 = core.add_synthetic_bidask(df10, cfg_bt)

        trades = pd.read_csv(spec.trades_csv)
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
        trades = _compute_holding_time_min(trades)
        trades["early_loss"] = trades["holding_time_min"] <= EARLY_LOSS_MAX_MIN
        trades["early_retrace_presence"] = _label_early_retrace(trades, df10)

        pnl_col = _select_pnl_column(set(trades.columns))

        baseline = trades.copy()
        modified = trades[trades["early_retrace_presence"] != "no_retrace"].copy()

        summary = pd.concat(
            [
                _early_loss_summary(baseline, period=spec.label, population="baseline"),
                _early_loss_summary(modified, period=spec.label, population="modified"),
            ],
            ignore_index=True,
        )
        summary_path = out_dir / f"summary_{spec.label}.csv"
        summary.to_csv(summary_path, index=False)

        tail_base = _tail_summary(baseline, period=spec.label, population="baseline", pnl_col=pnl_col)
        tail_mod = _tail_summary(modified, period=spec.label, population="modified", pnl_col=pnl_col)
        tail_rows.append(tail_base)
        tail_rows.append(tail_mod)

        for side in ("buy", "sell"):
            row_base = tail_base[tail_base["side"] == side].iloc[0]
            row_mod = tail_mod[tail_mod["side"] == side].iloc[0]
            notes.append(_format_tail_note(row_base, row_mod, label=f"{spec.label}/{side}"))

    tail_compare = pd.concat(tail_rows, ignore_index=True)
    tail_compare.to_csv(out_dir / "tail_compare.csv", index=False)

    _write_readme(out_dir, notes=notes)
    print(f"[ok] outputs: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
