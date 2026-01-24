#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_core as core


TAIL_PCTS = [1, 5, 10]
TP_MULTIPLIERS = [
    ("baseline", 1.0),
    ("tp_x2", 2.0),
    ("tp_x3", 3.0),
]


@dataclass(frozen=True)
class PeriodSpec:
    label: str
    from_month: str
    to_month: str


def _load_d009_config() -> dict:
    path = Path("FX/results/family_D_momentum/D009/config.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_core_config() -> dict:
    path = Path("FX/results/family_D_momentum/D009/core_config.json")
    return json.loads(path.read_text(encoding="utf-8"))


@contextlib.contextmanager
def _temporary_attr(obj, name: str, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


@contextlib.contextmanager
def _patch_d004_momentum_signals():
    def _m1_momentum_burst_signals(df10: pd.DataFrame, core_cfg: core.Config):  # noqa: ANN001
        o = df10["open_bid"].resample("1min", label="right", closed="right").first()
        c = df10["close_bid"].resample("1min", label="right", closed="right").last()
        m1 = pd.DataFrame({"open": o, "close": c}).dropna()

        body = (m1["close"] - m1["open"]).abs()
        mean_prev = body.shift(1).rolling(core_cfg.lookback_bars).mean()
        burst = (body > mean_prev)
        burst_up = (m1["close"] > m1["open"]) & burst
        burst_dn = (m1["close"] < m1["open"]) & burst

        h = df10["high_bid"].resample("1min", label="right", closed="right").max()
        l = df10["low_bid"].resample("1min", label="right", closed="right").min()
        hl = pd.DataFrame({"high": h, "low": l}).reindex(m1.index)

        prev_high = hl["high"].shift(1)
        prev_low = hl["low"].shift(1)

        rng = (hl["high"] - hl["low"])
        body_ratio = body / rng
        cond_body_ratio = (rng > 0) & (body_ratio >= 0.5)

        cond_break_prev_buy = (m1["close"] > prev_high)
        cond_break_prev_sell = (m1["close"] < prev_low)

        burst_up = burst_up & cond_break_prev_buy & cond_body_ratio
        burst_dn = burst_dn & cond_break_prev_sell & cond_body_ratio

        burst_up_closed = burst_up.shift(1).fillna(False)
        burst_dn_closed = burst_dn.shift(1).fillna(False)

        sig_up_10s = burst_up_closed.reindex(df10.index, method="ffill").fillna(False).to_numpy(dtype=bool)
        sig_dn_10s = burst_dn_closed.reindex(df10.index, method="ffill").fillna(False).to_numpy(dtype=bool)
        return sig_up_10s, sig_dn_10s

    def _m1_momentum_burst_up(df10, core_cfg):  # noqa: ANN001
        up, _ = _m1_momentum_burst_signals(df10, core_cfg)
        return up

    def _m1_momentum_burst_down(df10, core_cfg):  # noqa: ANN001
        _, dn = _m1_momentum_burst_signals(df10, core_cfg)
        return dn

    with contextlib.ExitStack() as stack:
        stack.enter_context(_temporary_attr(core, "high_breakout_10s", _m1_momentum_burst_up))
        stack.enter_context(_temporary_attr(core, "low_breakout_10s", _m1_momentum_burst_down))
        yield


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


def _summary_basic(trades: pd.DataFrame, *, period: str, population: str, pnl_col: str) -> pd.DataFrame:
    base = trades.copy()
    base["side"] = base["side"].astype(str).str.lower()
    rows = []
    for side in ("buy", "sell"):
        g = base[base["side"] == side]
        rows.append(
            {
                "period": period,
                "population": population,
                "side": side,
                "n_trades": int(len(g)),
                "total_pnl_pips": float(g[pnl_col].sum()) if len(g) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _write_readme(out_dir: Path) -> None:
    lines = [
        "# F001: TP extension tail effect (observe)",
        "",
        "## Definitions",
        "- baseline: D009 entry/timing/filters, SL fixed (stop_loss_pips=10), TP=baseline",
        "- extended: TP=baseline*2 (rr=2.4), TP=baseline*3 (rr=3.6)",
        "- entry: D_m1_momentum (D004_continuation), same as D009",
        "- periods: verify=2024-01..2024-12, forward=2025-01..2025-12",
        "- tail share: top/bottom 1/5/10% contribution to total pnl (by side)",
        "",
        "## Notes",
        "- Only TP is changed; SL/entry/timing/filters are fixed.",
        "- No optimization or parameter search.",
        "- Tail shares can flip or inflate when total pnl is near zero; interpret directionally.",
        "",
        "## Outputs",
        "- summary_verify.csv",
        "- summary_forward.csv",
        "- tail_compare.csv",
        "- README.md",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_variant(
    df10: pd.DataFrame,
    base_cfg: core.Config,
    *,
    rr: float,
    run_tag: str,
) -> tuple[Path, pd.DataFrame]:
    cfg = replace(base_cfg, rr=float(rr), run_tag=run_tag)
    run_dir = core.make_run_dir(cfg)

    (run_dir / "config.json").write_text(
        json.dumps({**asdict(cfg), "root": str(cfg.root), "results_root": str(cfg.results_root)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    runlog_path = run_dir / "runlog.jsonl"

    trades, monthly, monthly_session = core.backtest(df10, cfg, runlog_path, run_dir)

    if trades is not None and trades.empty and len(trades.columns) == 0:
        trades = pd.DataFrame(
            columns=[
                "side",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "pnl_pips",
                "exit_reason",
                "session",
            ]
        )
    if monthly is not None and monthly.empty and len(monthly.columns) == 0:
        monthly = pd.DataFrame(
            columns=[
                "month",
                "trades",
                "wins",
                "losses",
                "sum_pnl_pips",
                "avg_pnl_pips",
                "winrate",
            ]
        )
    if monthly_session is not None and monthly_session.empty and len(monthly_session.columns) == 0:
        monthly_session = pd.DataFrame(
            columns=[
                "month",
                "session",
                "trades",
                "wins",
                "losses",
                "sum_pnl_pips",
                "avg_pnl_pips",
                "winrate",
            ]
        )

    trades.to_csv(run_dir / "trades.csv", index=False)
    monthly.to_csv(run_dir / "monthly.csv", index=False)
    monthly_session.to_csv(run_dir / "monthly_by_session.csv", index=False)
    core.sanity_check(run_dir)
    return run_dir, trades


def main() -> None:
    cfg = _load_d009_config()
    core_cfg = _load_core_config()

    data_root = cfg["root"]
    symbol = cfg["symbol"]
    spread_pips = float(core_cfg.get("spread_pips", cfg.get("spread_pips", 1.0)))
    stop_loss_pips = float(core_cfg.get("stop_loss_pips", 10.0))
    baseline_rr = float(core_cfg.get("rr", 1.2))

    specs = [
        PeriodSpec(
            label="verify",
            from_month=cfg["verify"]["from_month"],
            to_month=cfg["verify"]["to_month"],
        ),
        PeriodSpec(
            label="forward",
            from_month=cfg["forward"]["from_month"],
            to_month=cfg["forward"]["to_month"],
        ),
    ]

    out_dir = Path("FX/results/family_F_tail_amplified_momentum/F001")
    out_dir.mkdir(parents=True, exist_ok=True)

    tail_rows = []

    for spec in specs:
        base_cfg = core.Config(
            root=Path(data_root),
            symbol=str(symbol).upper(),
            from_month=str(spec.from_month),
            to_month=str(spec.to_month),
            spread_pips=spread_pips,
            run_tag="f001",
        )
        base_cfg = replace(
            base_cfg,
            use_h1_trend_filter=True,
            use_time_filter=False,
            allow_sell=True,
            stop_loss_pips=stop_loss_pips,
        )

        df_bid = core.load_parquet_10s_bid(base_cfg)
        df10 = core.add_synthetic_bidask(df_bid, base_cfg)

        summary_rows = []

        with _patch_d004_momentum_signals():
            for label, mult in TP_MULTIPLIERS:
                rr = baseline_rr * float(mult)
                run_tag = f"f001_{label}"
                _, trades = _run_variant(df10, base_cfg, rr=rr, run_tag=run_tag)

                pnl_col = _select_pnl_column(set(trades.columns))
                summary_rows.append(_summary_basic(trades, period=spec.label, population=label, pnl_col=pnl_col))
                tail_rows.append(_tail_summary(trades, period=spec.label, population=label, pnl_col=pnl_col))

        summary = pd.concat(summary_rows, ignore_index=True)
        summary_path = out_dir / f"summary_{spec.label}.csv"
        summary.to_csv(summary_path, index=False)

    tail_compare = pd.concat(tail_rows, ignore_index=True)
    tail_compare.to_csv(out_dir / "tail_compare.csv", index=False)

    _write_readme(out_dir)
    print(f"[ok] outputs: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
