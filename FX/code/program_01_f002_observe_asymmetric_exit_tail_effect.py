#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import json
import math
import time
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

TIME_EXIT_MINUTES = [30, 60]
STEP_SL_FRAC = 0.5


@dataclass(frozen=True)
class PeriodSpec:
    label: str
    from_month: str
    to_month: str


@dataclass(frozen=True)
class ExitSpec:
    label: str
    time_exit_min: float | None
    step_sl_frac: float | None


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
        "# F002: asymmetric exit tail effect (observe)",
        "",
        "## Definitions",
        "- baseline: D009 entry/timing/filters, SL fixed (stop_loss_pips=10), TP=baseline (rr=1.2)",
        "- TP variants: baseline, tp_x2 (rr=2.4), tp_x3 (rr=3.6)",
        "- pattern_A: time exit if TP not reached and holding_time > T (T=30m, 60m)",
        "- pattern_B: step SL exit if unrealized loss > 0.5 * SL (0.5xSL)",
        "- entry: D_m1_momentum (D004_continuation), same as D009",
        "- periods: verify=2024-01..2024-12, forward=2025-01..2025-12",
        "- tail share: top/bottom 1/5/10% contribution to total pnl (by side)",
        "",
        "## Notes",
        "- Only exit is changed; SL/TP/entry/timing/filters are fixed.",
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


def _resolve_exit_price_buy(
    *,
    pos: dict,
    i: int,
    ts: pd.Timestamp,
    cfg: core.Config,
    low_bid: np.ndarray,
    high_bid: np.ndarray,
    close_bid: np.ndarray,
    time_exit_min: float | None,
    step_sl_frac: float | None,
) -> tuple[float | None, str | None]:
    hit_sl = low_bid[i] <= pos["sl"]
    hit_tp = high_bid[i] >= pos["tp"]
    step_price = None
    hit_step = False
    if step_sl_frac is not None:
        step_price = pos["entry_price"] - (cfg.stop_loss_pips * step_sl_frac) * cfg.pip_size
        hit_step = low_bid[i] <= step_price

    if hit_tp and (hit_sl or hit_step):
        if cfg.sl_first_if_both:
            if hit_step and step_price is not None:
                return step_price, "sl_step_both"
            return pos["sl"], "sl_both"
        return pos["tp"], "tp_both"
    if hit_step and step_price is not None:
        return step_price, "sl_step"
    if hit_sl:
        return pos["sl"], "sl"
    if hit_tp:
        return pos["tp"], "tp"

    if time_exit_min is not None:
        holding_min = (ts - pos["entry_time"]).total_seconds() / 60.0
        if holding_min > time_exit_min:
            return close_bid[i], f"time_exit_{int(time_exit_min)}m"
    return None, None


def _resolve_exit_price_sell(
    *,
    pos: dict,
    i: int,
    ts: pd.Timestamp,
    cfg: core.Config,
    high_ask: np.ndarray,
    low_ask: np.ndarray,
    close_ask: np.ndarray,
    time_exit_min: float | None,
    step_sl_frac: float | None,
) -> tuple[float | None, str | None]:
    hit_sl = high_ask[i] >= pos["sl"]
    hit_tp = low_ask[i] <= pos["tp"]
    step_price = None
    hit_step = False
    if step_sl_frac is not None:
        step_price = pos["entry_price"] + (cfg.stop_loss_pips * step_sl_frac) * cfg.pip_size
        hit_step = high_ask[i] >= step_price

    if hit_tp and (hit_sl or hit_step):
        if cfg.sl_first_if_both:
            if hit_step and step_price is not None:
                return step_price, "sl_step_both"
            return pos["sl"], "sl_both"
        return pos["tp"], "tp_both"
    if hit_step and step_price is not None:
        return step_price, "sl_step"
    if hit_sl:
        return pos["sl"], "sl"
    if hit_tp:
        return pos["tp"], "tp"

    if time_exit_min is not None:
        holding_min = (ts - pos["entry_time"]).total_seconds() / 60.0
        if holding_min > time_exit_min:
            return close_ask[i], f"time_exit_{int(time_exit_min)}m"
    return None, None


def _backtest_asymmetric_exit(
    df10: pd.DataFrame,
    cfg: core.Config,
    *,
    time_exit_min: float | None,
    step_sl_frac: float | None,
    runlog_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = df10.index
    n_bars = len(idx)

    print(f"[bt] preparing signals... bars={n_bars:,} only_session={cfg.only_session}", flush=True)

    time_ok = np.array([(not cfg.use_time_filter) or core.is_trading_time(ts, cfg) for ts in idx], dtype=bool)
    bias = core.compute_bias_htf(df10, cfg)
    h1_up = core.compute_h1_uptrend(df10, cfg) if cfg.use_h1_trend_filter else np.ones(n_bars, dtype=bool)
    h1_dn = core.compute_h1_downtrend(df10, cfg) if cfg.use_h1_trend_filter else np.ones(n_bars, dtype=bool)
    brk_hi = core.high_breakout_10s(df10, cfg)
    brk_lo = core.low_breakout_10s(df10, cfg)

    open_ask = df10["open_ask"].to_numpy()
    open_bid = df10["open_bid"].to_numpy()
    high_bid = df10["high_bid"].to_numpy()
    low_bid = df10["low_bid"].to_numpy()
    high_ask = df10["high_ask"].to_numpy()
    low_ask = df10["low_ask"].to_numpy()
    close_ask = df10["close_ask"].to_numpy()
    close_bid = df10["close_bid"].to_numpy()

    pos = None
    losing_streak = 0
    trading_stopped = False
    last_trade_date = idx[0].date()

    trades = []
    sess_arr = np.array([core.session_label(ts, cfg) for ts in idx])

    t0 = time.time()
    last_print = t0

    with runlog_path.open("a", encoding="utf-8") as lg:
        lg.write(json.dumps({"event": "start_backtest", "bars": n_bars, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")

        for i in range(n_bars - 1):
            ts = idx[i]
            ts_next = idx[i + 1]
            today = ts.date()

            if today > last_trade_date:
                losing_streak = 0
                trading_stopped = False
                last_trade_date = today

            if trading_stopped:
                continue

            if pos is not None:
                if pos["side"] == "buy":
                    exit_price, reason = _resolve_exit_price_buy(
                        pos=pos,
                        i=i,
                        ts=ts,
                        cfg=cfg,
                        low_bid=low_bid,
                        high_bid=high_bid,
                        close_bid=close_bid,
                        time_exit_min=time_exit_min,
                        step_sl_frac=step_sl_frac,
                    )

                    if exit_price is not None:
                        pnl_pips = (exit_price - pos["entry_price"]) / cfg.pip_size
                        trades.append(
                            {
                                "side": "buy",
                                "entry_time": pos["entry_time"],
                                "exit_time": ts,
                                "entry_price": pos["entry_price"],
                                "exit_price": exit_price,
                                "pnl_pips": pnl_pips,
                                "exit_reason": reason,
                                "session": core.session_label(pos["entry_time"], cfg),
                            }
                        )
                        losing_streak = losing_streak + 1 if pnl_pips < 0 else 0
                        if losing_streak >= cfg.max_losing_streak:
                            trading_stopped = True
                            lg.write(json.dumps({"event": "daily_stop", "ts": str(ts), "losing_streak": losing_streak}) + "\n")
                        pos = None

                else:
                    exit_price, reason = _resolve_exit_price_sell(
                        pos=pos,
                        i=i,
                        ts=ts,
                        cfg=cfg,
                        high_ask=high_ask,
                        low_ask=low_ask,
                        close_ask=close_ask,
                        time_exit_min=time_exit_min,
                        step_sl_frac=step_sl_frac,
                    )

                    if exit_price is not None:
                        pnl_pips = (pos["entry_price"] - exit_price) / cfg.pip_size
                        trades.append(
                            {
                                "side": "sell",
                                "entry_time": pos["entry_time"],
                                "exit_time": ts,
                                "entry_price": pos["entry_price"],
                                "exit_price": exit_price,
                                "pnl_pips": pnl_pips,
                                "exit_reason": reason,
                                "session": core.session_label(pos["entry_time"], cfg),
                            }
                        )
                        losing_streak = losing_streak + 1 if pnl_pips < 0 else 0
                        if losing_streak >= cfg.max_losing_streak:
                            trading_stopped = True
                            lg.write(json.dumps({"event": "daily_stop", "ts": str(ts), "losing_streak": losing_streak}) + "\n")
                        pos = None

            if pos is not None:
                continue
            if not time_ok[i]:
                continue

            if cfg.only_session is not None:
                if sess_arr[i] != cfg.only_session:
                    continue

            b = int(bias[i])
            if b == 0:
                continue

            if b == 1 and cfg.use_h1_trend_filter and not bool(h1_up[i]):
                continue

            if b == -1 and cfg.use_h1_trend_filter and not bool(h1_dn[i]):
                continue

            if b == 1 and cfg.allow_buy and bool(brk_hi[i]):
                entry_time = ts_next if cfg.entry_on_next_open else ts
                entry_price = open_ask[i + 1] if cfg.entry_on_next_open else close_ask[i]
                sl = entry_price - cfg.stop_loss_pips * cfg.pip_size
                tp = entry_price + (cfg.stop_loss_pips * cfg.rr) * cfg.pip_size
                pos = {"side": "buy", "entry_time": entry_time, "entry_price": entry_price, "sl": sl, "tp": tp}

            if b == -1 and cfg.allow_sell and bool(brk_lo[i]):
                entry_time = ts_next if cfg.entry_on_next_open else ts
                entry_price = open_bid[i + 1] if cfg.entry_on_next_open else close_bid[i]
                sl = entry_price + cfg.stop_loss_pips * cfg.pip_size
                tp = entry_price - (cfg.stop_loss_pips * cfg.rr) * cfg.pip_size
                pos = {"side": "sell", "entry_time": entry_time, "entry_price": entry_price, "sl": sl, "tp": tp}

            now = time.time()
            if now - last_print >= cfg.log_every_sec:
                elapsed = now - t0
                pct = (i / max(1, n_bars - 1)) * 100.0
                rate = i / elapsed if elapsed > 0 else 0.0
                print(f"[bt] {pct:5.1f}% ({i:,}/{n_bars:,}) rate={rate:,.0f} bars/s trades={len(trades):,} elapsed={elapsed:.1f}s", flush=True)
                last_print = now

        lg.write(json.dumps({"event": "end_backtest", "trades": len(trades), "utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}) + "\n")

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, pd.DataFrame(), pd.DataFrame()

    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True)
    trades_df["month"] = trades_df["entry_time"].dt.to_period("M").astype(str)

    monthly_df = trades_df.groupby("month").agg(
        trades=("pnl_pips", "count"),
        wins=("pnl_pips", lambda s: int((s > 0).sum())),
        losses=("pnl_pips", lambda s: int((s < 0).sum())),
        sum_pnl_pips=("pnl_pips", "sum"),
        avg_pnl_pips=("pnl_pips", "mean"),
    ).reset_index()
    monthly_df["winrate"] = monthly_df["wins"] / monthly_df["trades"]

    monthly_by_session = trades_df.groupby(["month", "session"]).agg(
        trades=("pnl_pips", "count"),
        wins=("pnl_pips", lambda s: int((s > 0).sum())),
        losses=("pnl_pips", lambda s: int((s < 0).sum())),
        sum_pnl_pips=("pnl_pips", "sum"),
        avg_pnl_pips=("pnl_pips", "mean"),
    ).reset_index()
    monthly_by_session["winrate"] = monthly_by_session["wins"] / monthly_by_session["trades"]

    return trades_df, monthly_df, monthly_by_session


def _run_variant(
    df10: pd.DataFrame,
    base_cfg: core.Config,
    *,
    rr: float,
    run_tag: str,
    exit_spec: ExitSpec,
) -> tuple[Path, pd.DataFrame]:
    cfg = replace(base_cfg, rr=float(rr), run_tag=run_tag)
    run_dir = core.make_run_dir(cfg)

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                **asdict(cfg),
                "root": str(cfg.root),
                "results_root": str(cfg.results_root),
                "exit_policy": {
                    "label": exit_spec.label,
                    "time_exit_min": exit_spec.time_exit_min,
                    "step_sl_frac": exit_spec.step_sl_frac,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    runlog_path = run_dir / "runlog.jsonl"

    trades, monthly, monthly_session = _backtest_asymmetric_exit(
        df10,
        cfg,
        time_exit_min=exit_spec.time_exit_min,
        step_sl_frac=exit_spec.step_sl_frac,
        runlog_path=runlog_path,
    )

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


def _population_label(tp_label: str, exit_label: str) -> str:
    if exit_label == "baseline":
        return tp_label
    return f"{tp_label}_{exit_label}"


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

    exit_specs = [
        ExitSpec(label="baseline", time_exit_min=None, step_sl_frac=None),
        ExitSpec(label="time_30m", time_exit_min=TIME_EXIT_MINUTES[0], step_sl_frac=None),
        ExitSpec(label="time_60m", time_exit_min=TIME_EXIT_MINUTES[1], step_sl_frac=None),
        ExitSpec(label="step_sl_0p5", time_exit_min=None, step_sl_frac=STEP_SL_FRAC),
    ]

    out_dir = Path("FX/results/family_F_tail_amplified_momentum/F002")
    out_dir.mkdir(parents=True, exist_ok=True)

    tail_rows = []

    for spec in specs:
        base_cfg = core.Config(
            root=Path(data_root),
            symbol=str(symbol).upper(),
            from_month=str(spec.from_month),
            to_month=str(spec.to_month),
            spread_pips=spread_pips,
            run_tag="f002",
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
            for exit_spec in exit_specs:
                for tp_label, mult in TP_MULTIPLIERS:
                    rr = baseline_rr * float(mult)
                    run_tag = f"f002_{tp_label}_{exit_spec.label}"
                    _, trades = _run_variant(
                        df10,
                        base_cfg,
                        rr=rr,
                        run_tag=run_tag,
                        exit_spec=exit_spec,
                    )

                    pnl_col = _select_pnl_column(set(trades.columns))
                    population = _population_label(tp_label, exit_spec.label)
                    summary_rows.append(_summary_basic(trades, period=spec.label, population=population, pnl_col=pnl_col))
                    tail_rows.append(_tail_summary(trades, period=spec.label, population=population, pnl_col=pnl_col))

        summary = pd.concat(summary_rows, ignore_index=True)
        summary_path = out_dir / f"summary_{spec.label}.csv"
        summary.to_csv(summary_path, index=False)

    tail_compare = pd.concat(tail_rows, ignore_index=True)
    tail_compare.to_csv(out_dir / "tail_compare.csv", index=False)

    _write_readme(out_dir)
    print(f"[ok] outputs: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
