#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path


def _results_root() -> Path:
    """
    Prefer `FX/results` (moved location) and fall back to repo-root `results`.
    """
    return Path("FX/results") if Path("FX/results").exists() else Path("results")


@dataclass(frozen=True)
class HypConfig:
    family: str
    hyp: str
    symbol: str
    root: str
    only_session: str | None
    verify_from_month: str
    verify_to_month: str
    forward_from_month: str
    forward_to_month: str
    spread_pips: float = 1.0
    use_h1_trend_filter: bool | None = None
    use_time_filter: bool | None = None
    disable_m1_bias_filter: bool = False
    entry_mode: str = "A_10s_breakout"  # "A_10s_breakout" | "B_failed_breakout" | "C_m1_close" | "D_m1_momentum"
    bias_mode: str = "default"          # "default" | "m1_candle"
    max_losing_streak: int | None = None
    regime_mode: str | None = None      # e.g. "m1_compression"
    dump_trades: bool = False
    momentum_mode: str | None = None    # e.g. "D004_continuation"
    no_weekend_entry: bool = False      # D005: block new entries near week transition
    allow_buy: bool | None = None
    allow_sell: bool | None = None
    exhaustion_filter_enabled: bool = False
    exhaustion_apply_side: str | None = None   # "buy" or "sell"
    exhaustion_feature: str | None = None      # "break_margin_over_mean_prev_body" | "break_margin_ratio"
    exhaustion_threshold: float | None = None
    exhaustion_nan_policy: str = "pass"        # "pass" (do not exclude NaNs)


def _run_core(cmd: list[str]) -> Path:
    print("[runner] exec:", " ".join(cmd), flush=True)

    run_dir: Path | None = None
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        if "[run] output dir:" in line:
            _, rhs = line.split("[run] output dir:", 1)
            run_dir = Path(rhs.strip())
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"backtest_core failed (code={proc.returncode})")
    if run_dir is None:
        raise RuntimeError("Could not detect run_dir from backtest_core output.")
    return run_dir


@contextlib.contextmanager
def _temporary_attr(obj, name: str, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def _run_core_inprocess(cfg: HypConfig, *, from_month: str, to_month: str, run_tag: str) -> Path:
    import numpy as np
    import pandas as pd

    import backtest_core as bc

    core_cfg = bc.Config(
        root=Path(cfg.root),
        symbol=str(cfg.symbol).upper(),
        from_month=str(from_month),
        to_month=str(to_month),
        only_session=cfg.only_session,
        spread_pips=float(cfg.spread_pips),
        run_tag=str(run_tag),
    )
    if cfg.use_h1_trend_filter is not None:
        core_cfg = replace(core_cfg, use_h1_trend_filter=bool(cfg.use_h1_trend_filter))
    if cfg.use_time_filter is not None:
        core_cfg = replace(core_cfg, use_time_filter=bool(cfg.use_time_filter))
    if cfg.max_losing_streak is not None:
        core_cfg = replace(core_cfg, max_losing_streak=int(cfg.max_losing_streak))
    if cfg.allow_buy is not None:
        core_cfg = replace(core_cfg, allow_buy=bool(cfg.allow_buy))
    if cfg.allow_sell is not None:
        core_cfg = replace(core_cfg, allow_sell=bool(cfg.allow_sell))

    patches: list[contextlib.AbstractContextManager] = []
    if cfg.disable_m1_bias_filter:
        def _bias_all_one(df10, _cfg):  # noqa: ANN001
            return np.ones(len(df10), dtype=np.int8)
        patches.append(_temporary_attr(bc, "compute_bias_htf", _bias_all_one))

    if cfg.bias_mode == "m1_candle":
        def _bias_m1_candle(df10, _cfg):  # noqa: ANN001
            o = df10["open_bid"].resample("1min", label="right", closed="right").first()
            c = df10["close_bid"].resample("1min", label="right", closed="right").last()
            bias_m1 = (c > o).astype(np.int8)
            # use last closed M1
            bias_closed = bias_m1.shift(1).fillna(0).astype(np.int8)
            return bias_closed.reindex(df10.index, method="ffill").fillna(0).to_numpy(dtype=np.int8)
        patches.append(_temporary_attr(bc, "compute_bias_htf", _bias_m1_candle))

    if cfg.entry_mode == "B_failed_breakout":
        def _failed_breakout_long(df10, core_cfg):  # noqa: ANN001
            prev_low = df10["low_bid"].shift(1).rolling(core_cfg.lookback_bars).min()
            breakout_down = (df10["close_bid"] < prev_low)
            # failure = next bar closes back above the breakout reference (2-step structure)
            fail = breakout_down.shift(1) & (df10["close_bid"] > prev_low.shift(1))
            return fail.fillna(False).to_numpy(dtype=bool)
        patches.append(_temporary_attr(bc, "high_breakout_10s", _failed_breakout_long))

    if cfg.entry_mode == "C_m1_close":
        def _m1_close_breakout(df10, core_cfg):  # noqa: ANN001
            hi = df10["high_bid"].resample("1min", label="right", closed="right").max()
            cl = df10["close_bid"].resample("1min", label="right", closed="right").last()
            prev_hi = hi.shift(1).rolling(core_cfg.lookback_bars).max()
            trig = (cl > prev_hi).fillna(False)
            # trigger only after the M1 candle is closed (no lookahead)
            trig_closed = trig.astype(bool)
            return trig_closed.reindex(df10.index, method="ffill").fillna(False).to_numpy(dtype=bool)
        patches.append(_temporary_attr(bc, "high_breakout_10s", _m1_close_breakout))

    if cfg.entry_mode == "D_m1_momentum":
        def _m1_momentum_burst_signals(df10, core_cfg):  # noqa: ANN001
            # M1 confirmed candle is the primary trigger; 10s is execution helper only.
            o = df10["open_bid"].resample("1min", label="right", closed="right").first()
            c = df10["close_bid"].resample("1min", label="right", closed="right").last()
            m1 = pd.DataFrame({"open": o, "close": c}).dropna()

            # momentum burst (price-only): body_range > mean(body_range) over last N bars
            # N is taken from existing core config (lookback_bars) to avoid introducing a tuned number.
            body = (m1["close"] - m1["open"]).abs()
            mean_prev = body.shift(1).rolling(core_cfg.lookback_bars).mean()
            burst = (body > mean_prev)
            burst_up = (m1["close"] > m1["open"]) & burst
            burst_dn = (m1["close"] < m1["open"]) & burst

            if cfg.momentum_mode == "D004_continuation":
                # D004 contract (structure constraints; price-only; fixed threshold):
                #   1) Buy: close > prev_high, Sell: close < prev_low (prev is shift(1) of confirmed M1 high/low)
                #   2) Reject wick-dominant candles: range>0 and body_ratio>=0.5 (0.5 is fixed; no tuning)
                # Notes (alignment / no-lookahead):
                #   - conditions are evaluated on confirmed M1 bars, then shift(1) is applied below
                #     so that only last-closed M1 can trigger a 10s execution.
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

            if cfg.exhaustion_filter_enabled:
                if cfg.exhaustion_feature not in ("break_margin_over_mean_prev_body", "break_margin_ratio"):
                    raise ValueError(f"Unsupported exhaustion_feature: {cfg.exhaustion_feature}")
                if cfg.exhaustion_threshold is None or not np.isfinite(float(cfg.exhaustion_threshold)):
                    raise ValueError(f"Invalid exhaustion_threshold: {cfg.exhaustion_threshold}")
                if cfg.exhaustion_apply_side not in ("buy", "sell"):
                    raise ValueError(f"Invalid exhaustion_apply_side: {cfg.exhaustion_apply_side}")
                if cfg.exhaustion_nan_policy != "pass":
                    raise ValueError(f"Unsupported exhaustion_nan_policy: {cfg.exhaustion_nan_policy}")

                # Compute per-M1 exhaustion feature (price-only; no optimization).
                # Note: this is evaluated on confirmed M1 bars, before shift(1) below.
                h = df10["high_bid"].resample("1min", label="right", closed="right").max()
                l = df10["low_bid"].resample("1min", label="right", closed="right").min()
                hl = pd.DataFrame({"high": h, "low": l}).reindex(m1.index)

                prev_high = hl["high"].shift(1)
                prev_low = hl["low"].shift(1)
                body_f = body.astype(float)
                mean_prev_f = mean_prev.astype(float)

                break_margin_buy = (m1["close"] - prev_high).astype(float)
                break_margin_sell = (prev_low - m1["close"]).astype(float)

                if cfg.exhaustion_feature == "break_margin_over_mean_prev_body":
                    feat_buy = np.where(mean_prev_f > 0, break_margin_buy / mean_prev_f, np.nan)
                    feat_sell = np.where(mean_prev_f > 0, break_margin_sell / mean_prev_f, np.nan)
                else:
                    feat_buy = np.where(body_f > 0, break_margin_buy / body_f, np.nan)
                    feat_sell = np.where(body_f > 0, break_margin_sell / body_f, np.nan)

                thr = float(cfg.exhaustion_threshold)
                exc_buy = feat_buy > thr
                exc_sell = feat_sell > thr

                # nan_policy="pass": NaN => not excluded
                exc_buy = np.where(np.isnan(feat_buy), False, exc_buy)
                exc_sell = np.where(np.isnan(feat_sell), False, exc_sell)

                if cfg.exhaustion_apply_side == "buy":
                    burst_up = burst_up & (~pd.Series(exc_buy, index=m1.index))
                else:
                    burst_dn = burst_dn & (~pd.Series(exc_sell, index=m1.index))

            # use last closed M1 only (no lookahead)
            burst_up_closed = burst_up.shift(1).fillna(False)
            burst_dn_closed = burst_dn.shift(1).fillna(False)

            if cfg.regime_mode == "m1_compression":
                # Compression regime gate (D002): enable signals only when compression==True.
                # Fixed parameters:
                #   N = lookback_bars (existing fixed core config)
                #   K = floor(lookback_bars/2) (derived once; do not tune)
                n = int(core_cfg.lookback_bars)
                k = max(1, n // 2)
                median_prev = body.shift(1).rolling(n).median()
                is_small = body < median_prev
                compression = (is_small.rolling(k).sum() == k)
                compression_closed = compression.shift(1).fillna(False)
                burst_up_closed = burst_up_closed & compression_closed
                burst_dn_closed = burst_dn_closed & compression_closed

            sig_up_10s = burst_up_closed.reindex(df10.index, method="ffill").fillna(False).to_numpy(dtype=bool)
            sig_dn_10s = burst_dn_closed.reindex(df10.index, method="ffill").fillna(False).to_numpy(dtype=bool)
            return sig_up_10s, sig_dn_10s

        def _m1_momentum_burst_up(df10, core_cfg):  # noqa: ANN001
            up, _ = _m1_momentum_burst_signals(df10, core_cfg)
            return up

        def _m1_momentum_burst_down(df10, core_cfg):  # noqa: ANN001
            _, dn = _m1_momentum_burst_signals(df10, core_cfg)
            return dn

        patches.append(_temporary_attr(bc, "high_breakout_10s", _m1_momentum_burst_up))
        patches.append(_temporary_attr(bc, "low_breakout_10s", _m1_momentum_burst_down))

    if cfg.no_weekend_entry:
        # D005 filter: block *new entries only* near week transition.
        # Implemented by tightening core's time filter at evaluation timestamp `ts` (10s bar).
        # This does not touch exits/position management.
        from datetime import time as dtime

        _orig_is_trading_time = bc.is_trading_time

        def _is_trading_time_no_weekend(ts, core_cfg):  # noqa: ANN001
            if not _orig_is_trading_time(ts, core_cfg):
                return False
            # timestamps are expected in UTC (tz-aware). We intentionally evaluate on `ts`
            # (the 10s bar where signal is checked), not on entry_time (ts_next).
            dow = ts.dayofweek  # Mon=0 ... Sun=6
            t = ts.timetz().replace(tzinfo=None)
            if dow == 4 and t >= dtime(20, 0):  # Fri >= 20:00 UTC
                return False
            if dow == 0 and t < dtime(2, 0):  # Mon < 02:00 UTC
                return False
            return True

        patches.append(_temporary_attr(bc, "is_trading_time", _is_trading_time_no_weekend))

    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)

        print("=== backtest started (runner/inprocess) ===", flush=True)
        print(
            f"[cfg] hyp={cfg.hyp} range={core_cfg.from_month}..{core_cfg.to_month} "
            f"only_session={core_cfg.only_session} use_time_filter={core_cfg.use_time_filter} "
            f"use_h1_trend_filter={core_cfg.use_h1_trend_filter} entry_mode={cfg.entry_mode} bias_mode={cfg.bias_mode} "
            f"max_losing_streak={core_cfg.max_losing_streak}",
            flush=True,
        )
        print(f"[cfg] input parquet base: {bc.parquet_base(core_cfg)}", flush=True)

        files = bc.list_parquet_files(core_cfg)
        print(f"[debug] parquet files found (filtered): {len(files):,}", flush=True)
        if len(files) == 0:
            raise FileNotFoundError("No parquet files found for the given range.")

        run_dir = bc.make_run_dir(core_cfg)
        print(f"[run] output dir: {run_dir}", flush=True)

        (run_dir / "config.json").write_text(
            json.dumps({**asdict(core_cfg), "root": str(core_cfg.root), "results_root": str(core_cfg.results_root)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        runlog_path = run_dir / "runlog.jsonl"

        df_bid = bc.load_parquet_10s_bid(core_cfg)
        df10 = bc.add_synthetic_bidask(df_bid, core_cfg)

        if core_cfg.debug_day and core_cfg.dump_debug:
            pth = bc.dump_debug_signals(df10, core_cfg, run_dir)
            print(f"[done] debug_signals.csv written: {pth}", flush=True)

        trades, monthly, monthly_session = bc.backtest(df10, core_cfg, runlog_path, run_dir)

        # core.sanity_check reads CSVs back; when there are 0 trades, core may return an empty DataFrame
        # with no columns, which would serialize to an unreadable CSV. Keep headers for zero-trade runs.
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

        print(f"[done] trades: {len(trades):,}", flush=True)
        bc.sanity_check(run_dir)
        return run_dir


def _copy_artifacts(src_run_dir: Path, dst_dir: Path, *, keep_config_as: str = "config.json") -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ("monthly.csv", "monthly_by_session.csv", "config.json"):
        src = src_run_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing artifact: {src}")
        dst_name = keep_config_as if name == "config.json" else name
        shutil.copy2(src, dst_dir / dst_name)


def _write_trades_csv(src_trades_csv: Path, dst_trades_csv: Path) -> None:
    """
    Visualization-only trade log.
    Does NOT change trade generation/exit logic; just formats columns.
    """
    import pandas as pd

    if not src_trades_csv.exists():
        raise FileNotFoundError(str(src_trades_csv))

    df = pd.read_csv(src_trades_csv)
    if df.empty:
        out = pd.DataFrame(
            columns=[
                "trade_id",
                "side",
                "entry_time",
                "entry_price",
                "exit_time",
                "exit_price",
                "pnl_pips",
                "holding_secs",
                "reason_exit",
            ]
        )
        out.to_csv(dst_trades_csv, index=False)
        return

    df = df.copy()
    df.insert(0, "trade_id", range(1, len(df) + 1))
    if "exit_reason" in df.columns and "reason_exit" not in df.columns:
        df = df.rename(columns={"exit_reason": "reason_exit"})

    holding_secs = None
    if "entry_time" in df.columns and "exit_time" in df.columns:
        et = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
        xt = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
        holding_secs = (xt - et).dt.total_seconds()

    cols = [
        "trade_id",
        "side",
        "entry_time",
        "entry_price",
        "exit_time",
        "exit_price",
        "pnl_pips",
        "reason_exit",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None

    out = df[cols].copy()
    if holding_secs is not None:
        out.insert(out.columns.get_loc("pnl_pips") + 1, "holding_secs", holding_secs)
    else:
        out.insert(out.columns.get_loc("pnl_pips") + 1, "holding_secs", None)

    dst_trades_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst_trades_csv, index=False)


def _sum_pnl_pips(monthly_csv: Path) -> float:
    import csv

    with monthly_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        vals = [float(row["sum_pnl_pips"]) for row in r if row.get("sum_pnl_pips")]
    return float(sum(vals))


def _sum_trades(monthly_csv: Path) -> int:
    import csv

    with monthly_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        vals = [int(float(row["trades"])) for row in r if row.get("trades")]
    return int(sum(vals))


def _run_variant_inprocess(cfg: HypConfig, *, run_tag: str) -> dict:
    out_root = _results_root() / cfg.family / cfg.hyp
    out_root.mkdir(parents=True, exist_ok=True)

    # 2024 verification
    verify_run_dir = _run_core_inprocess(cfg, from_month=cfg.verify_from_month, to_month=cfg.verify_to_month, run_tag=run_tag)
    verify_out = out_root / "in_sample_2024"
    _copy_artifacts(verify_run_dir, verify_out, keep_config_as="core_config.json")
    if cfg.dump_trades:
        _write_trades_csv(verify_run_dir / "trades.csv", verify_out / "trades.csv")

    # 2025 forward
    forward_run_dir = _run_core_inprocess(cfg, from_month=cfg.forward_from_month, to_month=cfg.forward_to_month, run_tag=run_tag)
    forward_out = out_root / "forward_2025"
    _copy_artifacts(forward_run_dir, forward_out, keep_config_as="core_config.json")
    if cfg.dump_trades:
        _write_trades_csv(forward_run_dir / "trades.csv", forward_out / "trades.csv")

    # Required outputs at hyp root: forward artifacts + runner config.json
    _copy_artifacts(forward_run_dir, out_root, keep_config_as="core_config.json")
    shutil.copy2(forward_out / "monthly.csv", out_root / "monthly.csv")
    shutil.copy2(forward_out / "monthly_by_session.csv", out_root / "monthly_by_session.csv")
    if cfg.dump_trades:
        _write_trades_csv(forward_run_dir / "trades.csv", out_root / "trades.csv")

    meta = {
        "family": cfg.family,
        "hyp": cfg.hyp,
        "symbol": cfg.symbol,
        "root": cfg.root,
        "spread_pips": cfg.spread_pips,
        "verify": {
            "from_month": cfg.verify_from_month,
            "to_month": cfg.verify_to_month,
            "run_dir": str(verify_run_dir),
            "sum_pnl_pips": _sum_pnl_pips(verify_out / "monthly.csv"),
            "trades": _sum_trades(verify_out / "monthly.csv"),
        },
        "forward": {
            "from_month": cfg.forward_from_month,
            "to_month": cfg.forward_to_month,
            "run_dir": str(forward_run_dir),
            "sum_pnl_pips": _sum_pnl_pips(forward_out / "monthly.csv"),
            "trades": _sum_trades(forward_out / "monthly.csv"),
        },
        "diff": {
            "only_session": cfg.only_session,
            "use_h1_trend_filter": cfg.use_h1_trend_filter,
            "use_time_filter": cfg.use_time_filter,
            "disable_m1_bias_filter": cfg.disable_m1_bias_filter,
            "entry_mode": cfg.entry_mode,
            "regime_mode": cfg.regime_mode,
            "dump_trades": cfg.dump_trades,
            "momentum_mode": cfg.momentum_mode,
            "no_weekend_entry": cfg.no_weekend_entry,
            "allow_buy": cfg.allow_buy,
            "allow_sell": cfg.allow_sell,
            "exhaustion_filter_enabled": cfg.exhaustion_filter_enabled,
            "exhaustion_apply_side": cfg.exhaustion_apply_side,
            "exhaustion_feature": cfg.exhaustion_feature,
            "exhaustion_threshold": cfg.exhaustion_threshold,
            "exhaustion_nan_policy": cfg.exhaustion_nan_policy,
        },
    }
    (out_root / "config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def run_a002_w2_only(cfg: HypConfig) -> dict:
    core = Path(__file__).with_name("backtest_core.py")
    if not core.exists():
        raise FileNotFoundError(f"backtest_core.py not found: {core}")

    out_root = _results_root() / cfg.family / cfg.hyp
    out_root.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    common = [py, str(core), "--root", cfg.root, "--symbol", cfg.symbol]
    if cfg.only_session:
        common += ["--only_session", cfg.only_session]
    common += ["--spread_pips", str(cfg.spread_pips)]

    # 2024 verification
    verify_cmd = common + ["--from_month", cfg.verify_from_month, "--to_month", cfg.verify_to_month]
    verify_run_dir = _run_core(verify_cmd)
    verify_out = out_root / "in_sample_2024"
    _copy_artifacts(verify_run_dir, verify_out, keep_config_as="core_config.json")

    # 2025 forward
    forward_cmd = common + ["--from_month", cfg.forward_from_month, "--to_month", cfg.forward_to_month]
    forward_run_dir = _run_core(forward_cmd)
    forward_out = out_root / "forward_2025"
    _copy_artifacts(forward_run_dir, forward_out, keep_config_as="core_config.json")

    # Required outputs at hyp root: forward artifacts + runner config.json
    _copy_artifacts(forward_run_dir, out_root, keep_config_as="core_config.json")
    shutil.copy2(forward_out / "monthly.csv", out_root / "monthly.csv")
    shutil.copy2(forward_out / "monthly_by_session.csv", out_root / "monthly_by_session.csv")

    meta = {
        "family": cfg.family,
        "hyp": cfg.hyp,
        "symbol": cfg.symbol,
        "root": cfg.root,
        "spread_pips": cfg.spread_pips,
        "verify": {
            "from_month": cfg.verify_from_month,
            "to_month": cfg.verify_to_month,
            "run_dir": str(verify_run_dir),
            "pnl_sum_pips": _sum_pnl_pips(verify_out / "monthly.csv"),
        },
        "forward": {
            "from_month": cfg.forward_from_month,
            "to_month": cfg.forward_to_month,
            "run_dir": str(forward_run_dir),
            "pnl_sum_pips": _sum_pnl_pips(forward_out / "monthly.csv"),
        },
        "diff": {
            "only_session": cfg.only_session,
        },
    }
    (out_root / "config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def run_a001_baseline(cfg: HypConfig) -> dict:
    core = Path(__file__).with_name("backtest_core.py")
    if not core.exists():
        raise FileNotFoundError(f"backtest_core.py not found: {core}")

    out_root = _results_root() / cfg.family / cfg.hyp
    out_root.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    common = [py, str(core), "--root", cfg.root, "--symbol", cfg.symbol]
    if cfg.only_session:
        common += ["--only_session", cfg.only_session]
    common += ["--spread_pips", str(cfg.spread_pips)]

    verify_cmd = common + ["--from_month", cfg.verify_from_month, "--to_month", cfg.verify_to_month]
    verify_run_dir = _run_core(verify_cmd)
    verify_out = out_root / "in_sample_2024"
    _copy_artifacts(verify_run_dir, verify_out, keep_config_as="core_config.json")

    forward_cmd = common + ["--from_month", cfg.forward_from_month, "--to_month", cfg.forward_to_month]
    forward_run_dir = _run_core(forward_cmd)
    forward_out = out_root / "forward_2025"
    _copy_artifacts(forward_run_dir, forward_out, keep_config_as="core_config.json")

    _copy_artifacts(forward_run_dir, out_root, keep_config_as="core_config.json")
    shutil.copy2(forward_out / "monthly.csv", out_root / "monthly.csv")
    shutil.copy2(forward_out / "monthly_by_session.csv", out_root / "monthly_by_session.csv")

    meta = {
        "family": cfg.family,
        "hyp": cfg.hyp,
        "symbol": cfg.symbol,
        "root": cfg.root,
        "spread_pips": cfg.spread_pips,
        "verify": {
            "from_month": cfg.verify_from_month,
            "to_month": cfg.verify_to_month,
            "run_dir": str(verify_run_dir),
            "pnl_sum_pips": _sum_pnl_pips(verify_out / "monthly.csv"),
        },
        "forward": {
            "from_month": cfg.forward_from_month,
            "to_month": cfg.forward_to_month,
            "run_dir": str(forward_run_dir),
            "pnl_sum_pips": _sum_pnl_pips(forward_out / "monthly.csv"),
        },
        "diff": {
            "only_session": cfg.only_session,
        },
    }
    (out_root / "config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def run_family_a_variant(cfg: HypConfig) -> dict:
    out_root = _results_root() / cfg.family / cfg.hyp
    out_root.mkdir(parents=True, exist_ok=True)

    tag = f"hyp{cfg.hyp[1:]}"

    verify_run_dir = _run_core_inprocess(cfg, from_month=cfg.verify_from_month, to_month=cfg.verify_to_month, run_tag=tag)
    verify_out = out_root / "in_sample_2024"
    _copy_artifacts(verify_run_dir, verify_out, keep_config_as="core_config.json")

    forward_run_dir = _run_core_inprocess(cfg, from_month=cfg.forward_from_month, to_month=cfg.forward_to_month, run_tag=tag)
    forward_out = out_root / "forward_2025"
    _copy_artifacts(forward_run_dir, forward_out, keep_config_as="core_config.json")

    _copy_artifacts(forward_run_dir, out_root, keep_config_as="core_config.json")
    shutil.copy2(forward_out / "monthly.csv", out_root / "monthly.csv")
    shutil.copy2(forward_out / "monthly_by_session.csv", out_root / "monthly_by_session.csv")

    meta = {
        "family": cfg.family,
        "hyp": cfg.hyp,
        "symbol": cfg.symbol,
        "root": cfg.root,
        "spread_pips": cfg.spread_pips,
        "verify": {
            "from_month": cfg.verify_from_month,
            "to_month": cfg.verify_to_month,
            "run_dir": str(verify_run_dir),
            "pnl_sum_pips": _sum_pnl_pips(verify_out / "monthly.csv"),
        },
        "forward": {
            "from_month": cfg.forward_from_month,
            "to_month": cfg.forward_to_month,
            "run_dir": str(forward_run_dir),
            "pnl_sum_pips": _sum_pnl_pips(forward_out / "monthly.csv"),
        },
        "diff": {
            "only_session": cfg.only_session,
            "use_h1_trend_filter": cfg.use_h1_trend_filter,
            "use_time_filter": cfg.use_time_filter,
            "disable_m1_bias_filter": cfg.disable_m1_bias_filter,
        },
    }
    (out_root / "config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default=None, choices=[None, "family_BC", "family_BC_summary", "family_C_v2", "family_D_momentum"])
    p.add_argument(
        "--hyp",
        default="A002",
        choices=[
            "A001", "A002", "A003", "A004", "A005",
            "B001", "B002", "B003", "B004", "B005", "B006", "B007", "B008",
            "C001", "C002", "C003", "C004", "C005",
            "C101", "C102", "C103",
            "D001", "D002", "D003", "D004", "D005", "D006", "D009", "D011a", "D011b",
        ],
    )
    p.add_argument("--symbol", default="USDJPY")
    p.add_argument("--root", default=None, help="Data root (auto-detect if omitted)")
    p.add_argument("--spread_pips", type=float, default=1.0)
    p.add_argument("--verify_from_month", default="2024-01")
    p.add_argument("--verify_to_month", default="2024-12")
    p.add_argument("--forward_from_month", default="2025-01")
    p.add_argument("--forward_to_month", default="2025-12")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root
    if root is None:
        candidates = [
            Path("dukas_out_v2"),
            Path("FX/code/dukas_out_v2"),
            Path("../dukascopy_downloader/dukas_out_v2"),
            Path("../fx_hypo_analyzer/dukas_out_v2"),
        ]
        for c in candidates:
            if c.exists():
                root = str(c)
                break
    if root is None:
        raise FileNotFoundError("Could not auto-detect data root; pass --root /path/to/dukas_out_v2")

    def _judge_rows(rows: list[dict]) -> list[dict]:
        # judge (compare to baseline within each family); threshold is coarse and intentionally non-optimized
        threshold = 50.0

        by_family: dict[str, dict[str, dict]] = {}
        for r in rows:
            by_family.setdefault(r["family"], {})[r["hyp"]] = r

        def judge_baseline(fwd: float) -> str:
            return "conditional" if fwd >= -threshold else "dead"

        def judge_delta(fwd: float, baseline_forward: float) -> str:
            if fwd <= baseline_forward - threshold:
                return "dead"
            if fwd >= baseline_forward + threshold:
                return "conditional"
            return "conditional"

        for fam, m in by_family.items():
            baseline_key = "B001" if fam.startswith("family_B_") else "C001"
            baseline_forward = float(m[baseline_key]["forward_sum_pnl_pips"]) if baseline_key in m else None
            for hyp_key, row in m.items():
                fwd = float(row["forward_sum_pnl_pips"])
                if baseline_forward is None or hyp_key == baseline_key:
                    row["judge"] = judge_baseline(fwd)
                else:
                    row["judge"] = judge_delta(fwd, baseline_forward)
        return rows

    def _write_summary_csv(summary_path: Path, rows: list[dict]) -> None:
        import csv

        with summary_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "family",
                    "hyp",
                    "delta",
                    "verify_sum_pnl_pips",
                    "verify_trades",
                    "forward_sum_pnl_pips",
                    "forward_trades",
                    "judge",
                ],
            )
            w.writeheader()
            for row in sorted(rows, key=lambda x: (x["family"], x["hyp"])):
                w.writerow(row)

    def _build_rows_from_existing() -> list[dict]:
        import csv

        specs = [
            ("family_B_failedbreakout", "B001", "baseline"),
            ("family_B_failedbreakout", "B002", "only_session=W1"),
            ("family_B_failedbreakout", "B003", "use_h1_trend_filter=False"),
            ("family_B_failedbreakout", "B004", "max_losing_streak=999"),
            ("family_B_failedbreakout", "B005", "use_time_filter=False"),
            ("family_C_m1entry", "C001", "baseline"),
            ("family_C_m1entry", "C002", "only_session=W1"),
            ("family_C_m1entry", "C003", "use_h1_trend_filter=False"),
            ("family_C_m1entry", "C004", "max_losing_streak=999"),
            ("family_C_m1entry", "C005", "use_time_filter=False"),
        ]
        out: list[dict] = []
        for fam, hyp, delta in specs:
            base = _results_root() / fam / hyp
            v_csv = base / "in_sample_2024" / "monthly.csv"
            f_csv = base / "forward_2025" / "monthly.csv"
            if not v_csv.exists() or not f_csv.exists():
                raise FileNotFoundError(f"Missing monthly.csv for {fam}/{hyp}: {v_csv} {f_csv}")

            def read_sum_trades(p: Path) -> tuple[float, int]:
                with p.open("r", encoding="utf-8", newline="") as ff:
                    r = csv.DictReader(ff)
                    rows = list(r)
                s = sum(float(x["sum_pnl_pips"]) for x in rows)
                t = sum(int(float(x["trades"])) for x in rows)
                return float(s), int(t)

            v_sum, v_tr = read_sum_trades(v_csv)
            f_sum, f_tr = read_sum_trades(f_csv)
            out.append({
                "family": fam,
                "hyp": hyp,
                "delta": delta,
                "verify_sum_pnl_pips": v_sum,
                "verify_trades": v_tr,
                "forward_sum_pnl_pips": f_sum,
                "forward_trades": f_tr,
            })
        return out

    if args.suite == "family_BC_summary":
        summary_path = _results_root() / "summary_family_BC.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        rows = _build_rows_from_existing()
        rows = _judge_rows(rows)
        _write_summary_csv(summary_path, rows)
        print(f"[runner] wrote summary: {summary_path}", flush=True)
        return 0

    if args.suite == "family_D_momentum":
        import csv

        family = "family_D_momentum"
        summary_path = _results_root() / "summary_family_D_momentum.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        with summary_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "hyp",
                    "verify_sum_pnl_pips",
                    "verify_trades",
                    "forward_sum_pnl_pips",
                    "forward_trades",
                    "judge",
                ],
            )
            w.writeheader()
            for hyp, regime_mode, use_h1, no_weekend in [
                ("D001", None, None, False),
                ("D002", "m1_compression", None, False),
                ("D003", None, False, False),
                ("D004", None, None, False),
                ("D005", None, None, True),
                # D006: observation (24h) = no session/time filter, H1 ON
                ("D006", None, True, False),
            ]:
                cfg = HypConfig(
                    family=family,
                    hyp=hyp,
                    symbol=str(args.symbol).upper(),
                    root=str(root),
                    only_session=(None if hyp == "D006" else "W1"),
                    verify_from_month=str(args.verify_from_month),
                    verify_to_month=str(args.verify_to_month),
                    forward_from_month=str(args.forward_from_month),
                    forward_to_month=str(args.forward_to_month),
                    spread_pips=float(args.spread_pips),
                    use_h1_trend_filter=use_h1,
                    use_time_filter=(False if hyp == "D006" else None),
                    entry_mode="D_m1_momentum",
                    disable_m1_bias_filter=True,
                    regime_mode=regime_mode,
                    dump_trades=True,
                    momentum_mode=("D004_continuation" if hyp in ("D004", "D005", "D006") else None),
                    no_weekend_entry=no_weekend,
                )
                meta = _run_variant_inprocess(cfg, run_tag=hyp.lower())

                verify_sum = float(meta["verify"]["sum_pnl_pips"])
                verify_trades = int(meta["verify"]["trades"])
                forward_sum = float(meta["forward"]["sum_pnl_pips"])
                forward_trades = int(meta["forward"]["trades"])
                judge = "conditional" if (forward_sum >= 0) or (forward_sum > verify_sum) else "dead"

                w.writerow(
                    {
                        "hyp": hyp,
                        "verify_sum_pnl_pips": verify_sum,
                        "verify_trades": verify_trades,
                        "forward_sum_pnl_pips": forward_sum,
                        "forward_trades": forward_trades,
                        "judge": judge,
                    }
                )

        print(f"[runner] wrote summary: {summary_path}", flush=True)
        return 0

    if args.suite == "family_C_v2":
        import csv

        family = "family_C_m1entry_v2"
        summary_path = _results_root() / "summary_family_C_v2.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        specs: list[HypConfig] = [
            # C101 baseline: M1-close trigger, W1-only, H1 on, time filter on
            HypConfig(
                family=family,
                hyp="C101",
                symbol=str(args.symbol).upper(),
                root=str(root),
                only_session="W1",
                verify_from_month=str(args.verify_from_month),
                verify_to_month=str(args.verify_to_month),
                forward_from_month=str(args.forward_from_month),
                forward_to_month=str(args.forward_to_month),
                spread_pips=float(args.spread_pips),
                entry_mode="C_m1_close",
                bias_mode="m1_candle",
            ),
            # C102: disable H1 only
            HypConfig(
                family=family,
                hyp="C102",
                symbol=str(args.symbol).upper(),
                root=str(root),
                only_session="W1",
                verify_from_month=str(args.verify_from_month),
                verify_to_month=str(args.verify_to_month),
                forward_from_month=str(args.forward_from_month),
                forward_to_month=str(args.forward_to_month),
                spread_pips=float(args.spread_pips),
                use_h1_trend_filter=False,
                entry_mode="C_m1_close",
                bias_mode="m1_candle",
            ),
            # C103: disable time filter only
            HypConfig(
                family=family,
                hyp="C103",
                symbol=str(args.symbol).upper(),
                root=str(root),
                only_session="W1",
                verify_from_month=str(args.verify_from_month),
                verify_to_month=str(args.verify_to_month),
                forward_from_month=str(args.forward_from_month),
                forward_to_month=str(args.forward_to_month),
                spread_pips=float(args.spread_pips),
                use_time_filter=False,
                entry_mode="C_m1_close",
                bias_mode="m1_candle",
            ),
        ]

        metas: dict[str, dict] = {}
        for cfg in specs:
            metas[cfg.hyp] = _run_variant_inprocess(cfg, run_tag=cfg.hyp.lower())

        baseline_forward = float(metas["C101"]["forward"]["sum_pnl_pips"])

        def judge(hyp: str) -> str:
            fwd = float(metas[hyp]["forward"]["sum_pnl_pips"])
            if hyp == "C101":
                return "conditional" if fwd > 0 else "dead"
            return "conditional" if (fwd - baseline_forward) >= 50.0 else "dead"

        with summary_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["hyp", "verify_sum_pnl_pips", "forward_sum_pnl_pips", "trades", "judge"],
            )
            w.writeheader()
            for hyp in ["C101", "C102", "C103"]:
                w.writerow(
                    {
                        "hyp": hyp,
                        "verify_sum_pnl_pips": float(metas[hyp]["verify"]["sum_pnl_pips"]),
                        "forward_sum_pnl_pips": float(metas[hyp]["forward"]["sum_pnl_pips"]),
                        "trades": int(metas[hyp]["forward"]["trades"]),
                        "judge": judge(hyp),
                    }
                )

        print(f"[runner] wrote summary: {summary_path}", flush=True)
        return 0

    if args.suite == "family_BC":
        summary_path = _results_root() / "summary_family_BC.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        def preset(
            family: str,
            hyp: str,
            *,
            delta: str,
            only_session: str | None,
            use_h1: bool | None,
            entry_mode: str,
            bias_mode: str,
        ) -> tuple[HypConfig, str]:
            cfg = HypConfig(
                family=family,
                hyp=hyp,
                symbol=str(args.symbol).upper(),
                root=str(root),
                only_session=only_session,
                verify_from_month=str(args.verify_from_month),
                verify_to_month=str(args.verify_to_month),
                forward_from_month=str(args.forward_from_month),
                forward_to_month=str(args.forward_to_month),
                spread_pips=float(args.spread_pips),
                use_h1_trend_filter=use_h1,
                entry_mode=entry_mode,
                bias_mode=bias_mode,
            )
            return cfg, delta

        specs: list[tuple[HypConfig, str]] = [
            preset("family_B_failedbreakout", "B001", delta="baseline", only_session=None, use_h1=None, entry_mode="B_failed_breakout", bias_mode="default"),
            preset("family_B_failedbreakout", "B002", delta="only_session=W1", only_session="W1", use_h1=None, entry_mode="B_failed_breakout", bias_mode="default"),
            preset("family_B_failedbreakout", "B003", delta="use_h1_trend_filter=False", only_session=None, use_h1=False, entry_mode="B_failed_breakout", bias_mode="default"),
            # one-diff variants (004/005)
            (replace(preset("family_B_failedbreakout", "B004", delta="max_losing_streak=999", only_session=None, use_h1=None, entry_mode="B_failed_breakout", bias_mode="default")[0], max_losing_streak=999), "max_losing_streak=999"),
            (replace(preset("family_B_failedbreakout", "B005", delta="use_time_filter=False", only_session=None, use_h1=None, entry_mode="B_failed_breakout", bias_mode="default")[0], use_time_filter=False), "use_time_filter=False"),
            preset("family_C_m1entry", "C001", delta="baseline", only_session=None, use_h1=None, entry_mode="C_m1_close", bias_mode="m1_candle"),
            preset("family_C_m1entry", "C002", delta="only_session=W1", only_session="W1", use_h1=None, entry_mode="C_m1_close", bias_mode="m1_candle"),
            preset("family_C_m1entry", "C003", delta="use_h1_trend_filter=False", only_session=None, use_h1=False, entry_mode="C_m1_close", bias_mode="m1_candle"),
            (replace(preset("family_C_m1entry", "C004", delta="max_losing_streak=999", only_session=None, use_h1=None, entry_mode="C_m1_close", bias_mode="m1_candle")[0], max_losing_streak=999), "max_losing_streak=999"),
            (replace(preset("family_C_m1entry", "C005", delta="use_time_filter=False", only_session=None, use_h1=None, entry_mode="C_m1_close", bias_mode="m1_candle")[0], use_time_filter=False), "use_time_filter=False"),
        ]

        results: list[dict] = []
        for cfg, delta in specs:
            meta = _run_variant_inprocess(cfg, run_tag=cfg.hyp.lower())
            results.append({
                "family": cfg.family,
                "hyp": cfg.hyp,
                "delta": delta,
                "verify_sum_pnl_pips": meta["verify"]["sum_pnl_pips"],
                "verify_trades": meta["verify"]["trades"],
                "forward_sum_pnl_pips": meta["forward"]["sum_pnl_pips"],
                "forward_trades": meta["forward"]["trades"],
            })

        results = _judge_rows(results)
        _write_summary_csv(summary_path, results)

        print(f"[runner] wrote summary: {summary_path}", flush=True)
        return 0

    if args.hyp == "A001":
        cfg = HypConfig(
            family="family_A",
            hyp="A001",
            symbol=str(args.symbol).upper(),
            root=str(root),
            only_session=None,
            verify_from_month=str(args.verify_from_month),
            verify_to_month=str(args.verify_to_month),
            forward_from_month=str(args.forward_from_month),
            forward_to_month=str(args.forward_to_month),
            spread_pips=float(args.spread_pips),
        )
        meta = run_a001_baseline(cfg)
        print("[runner] done. summary:", json.dumps(meta, ensure_ascii=False), flush=True)
        return 0

    if args.hyp == "A002":
        cfg = HypConfig(
            family="family_A",
            hyp="A002",
            symbol=str(args.symbol).upper(),
            root=str(root),
            only_session="W2",
            verify_from_month=str(args.verify_from_month),
            verify_to_month=str(args.verify_to_month),
            forward_from_month=str(args.forward_from_month),
            forward_to_month=str(args.forward_to_month),
            spread_pips=float(args.spread_pips),
        )
        meta = run_a002_w2_only(cfg)
        print("[runner] done. summary:", json.dumps(meta, ensure_ascii=False), flush=True)
        return 0

    if args.hyp == "A003":
        cfg = HypConfig(
            family="family_A",
            hyp="A003",
            symbol=str(args.symbol).upper(),
            root=str(root),
            only_session=None,
            verify_from_month=str(args.verify_from_month),
            verify_to_month=str(args.verify_to_month),
            forward_from_month=str(args.forward_from_month),
            forward_to_month=str(args.forward_to_month),
            spread_pips=float(args.spread_pips),
            use_h1_trend_filter=False,
        )
        meta = run_family_a_variant(cfg)
        print("[runner] done. summary:", json.dumps(meta, ensure_ascii=False), flush=True)
        return 0

    if args.hyp == "A004":
        cfg = HypConfig(
            family="family_A",
            hyp="A004",
            symbol=str(args.symbol).upper(),
            root=str(root),
            only_session=None,
            verify_from_month=str(args.verify_from_month),
            verify_to_month=str(args.verify_to_month),
            forward_from_month=str(args.forward_from_month),
            forward_to_month=str(args.forward_to_month),
            spread_pips=float(args.spread_pips),
            disable_m1_bias_filter=True,
        )
        meta = run_family_a_variant(cfg)
        print("[runner] done. summary:", json.dumps(meta, ensure_ascii=False), flush=True)
        return 0

    if args.hyp == "A005":
        cfg = HypConfig(
            family="family_A",
            hyp="A005",
            symbol=str(args.symbol).upper(),
            root=str(root),
            only_session=None,
            verify_from_month=str(args.verify_from_month),
            verify_to_month=str(args.verify_to_month),
            forward_from_month=str(args.forward_from_month),
            forward_to_month=str(args.forward_to_month),
            spread_pips=float(args.spread_pips),
            use_time_filter=False,
        )
        meta = run_family_a_variant(cfg)
        print("[runner] done. summary:", json.dumps(meta, ensure_ascii=False), flush=True)
        return 0

    if args.hyp in ("C101", "C102", "C103"):
        family = "family_C_m1entry_v2"
        cfg = HypConfig(
            family=family,
            hyp=str(args.hyp),
            symbol=str(args.symbol).upper(),
            root=str(root),
            only_session="W1",
            verify_from_month=str(args.verify_from_month),
            verify_to_month=str(args.verify_to_month),
            forward_from_month=str(args.forward_from_month),
            forward_to_month=str(args.forward_to_month),
            spread_pips=float(args.spread_pips),
            entry_mode="C_m1_close",
            bias_mode="m1_candle",
            use_h1_trend_filter=(False if args.hyp == "C102" else None),
            use_time_filter=(False if args.hyp == "C103" else None),
        )
        meta = _run_variant_inprocess(cfg, run_tag=str(args.hyp).lower())
        print("[runner] done. summary:", json.dumps(meta, ensure_ascii=False), flush=True)
        return 0

    if args.hyp in ("D001", "D002", "D003", "D004", "D005", "D006"):
        cfg = HypConfig(
            family="family_D_momentum",
            hyp=str(args.hyp),
            symbol=str(args.symbol).upper(),
            root=str(root),
            only_session=(None if args.hyp == "D006" else "W1"),
            verify_from_month=str(args.verify_from_month),
            verify_to_month=str(args.verify_to_month),
            forward_from_month=str(args.forward_from_month),
            forward_to_month=str(args.forward_to_month),
            spread_pips=float(args.spread_pips),
            use_h1_trend_filter=(False if args.hyp == "D003" else (True if args.hyp == "D006" else None)),
            use_time_filter=(False if args.hyp == "D006" else None),
            entry_mode="D_m1_momentum",
            disable_m1_bias_filter=True,
            regime_mode=("m1_compression" if args.hyp == "D002" else None),
            dump_trades=True,
            momentum_mode=("D004_continuation" if args.hyp in ("D004", "D005", "D006") else None),
            no_weekend_entry=(args.hyp == "D005"),
        )
        meta = _run_variant_inprocess(cfg, run_tag=str(args.hyp).lower())
        print("[runner] done. summary:", json.dumps(meta, ensure_ascii=False), flush=True)
        return 0

    if args.hyp == "D009":
        # D009: sell-side observation.
        # - strategy: D004_continuation signals (unchanged trigger)
        # - allow_sell=True (and allow_buy left as True)
        # - enable bias (do NOT force bias=+1), otherwise sells cannot happen
        # - observe 24h by default (no session/time filter)
        cfg = HypConfig(
            family="family_D_momentum",
            hyp="D009",
            symbol=str(args.symbol).upper(),
            root=str(root),
            only_session=None,
            verify_from_month=str(args.verify_from_month),
            verify_to_month=str(args.verify_to_month),
            forward_from_month=str(args.forward_from_month),
            forward_to_month=str(args.forward_to_month),
            spread_pips=float(args.spread_pips),
            use_h1_trend_filter=True,
            use_time_filter=False,
            entry_mode="D_m1_momentum",
            disable_m1_bias_filter=False,
            dump_trades=True,
            momentum_mode="D004_continuation",
            allow_sell=True,
        )
        meta = _run_variant_inprocess(cfg, run_tag="d009")
        print("[runner] done. summary:", json.dumps(meta, ensure_ascii=False), flush=True)
        return 0

    def _read_trades_side_metrics(trades_csv: Path, *, side: str) -> dict:
        import pandas as pd

        df = pd.read_csv(trades_csv)
        if df.empty:
            return {"n_trades": 0, "n_early_loss": 0, "early_loss_rate": 0.0, "n_survivor": 0, "survivor_rate": 0.0, "sum_pnl_pips": 0.0, "avg_pnl_pips": 0.0}
        if "side" not in df.columns:
            raise ValueError(f"Missing side column: {trades_csv}")
        df["side"] = df["side"].astype(str).str.lower()
        df = df[df["side"] == side].copy()
        if df.empty:
            return {"n_trades": 0, "n_early_loss": 0, "early_loss_rate": 0.0, "n_survivor": 0, "survivor_rate": 0.0, "sum_pnl_pips": 0.0, "avg_pnl_pips": 0.0}

        df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
        df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
        df = df.dropna(subset=["entry_time", "exit_time"])
        holding_min = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 60.0
        df["holding_time_min"] = holding_min
        df = df[df["holding_time_min"].notna() & (df["holding_time_min"] >= 0)].copy()

        early = df["holding_time_min"] <= 3.0
        surv = df["holding_time_min"] >= 20.0
        n = int(len(df))
        n_early = int(early.sum())
        n_surv = int(surv.sum())
        sum_pnl = float(df["pnl_pips"].astype(float).sum()) if "pnl_pips" in df.columns else 0.0
        avg_pnl = float(df["pnl_pips"].astype(float).mean()) if ("pnl_pips" in df.columns and n > 0) else 0.0
        return {
            "n_trades": n,
            "n_early_loss": n_early,
            "early_loss_rate": float(n_early / n) if n > 0 else 0.0,
            "n_survivor": n_surv,
            "survivor_rate": float(n_surv / n) if n > 0 else 0.0,
            "sum_pnl_pips": sum_pnl,
            "avg_pnl_pips": avg_pnl,
        }

    def _compute_exhaustion_thresholds_from_d009_verify(*, side: str) -> dict:
        import numpy as np
        import pandas as pd
        import backtest_core as bc

        side = str(side).lower()
        if side not in ("buy", "sell"):
            raise ValueError("side must be buy/sell")

        d009 = _results_root() / "family_D_momentum" / "D009" / "in_sample_2024"
        trades_csv = d009 / "trades.csv"
        core_cfg_path = d009 / "core_config.json"
        if not trades_csv.exists() or not core_cfg_path.exists():
            raise FileNotFoundError(f"D009 verify artifacts not found: {trades_csv} {core_cfg_path}")

        core_cfg = json.loads(core_cfg_path.read_text(encoding="utf-8"))
        lookback = int(core_cfg.get("lookback_bars", 6))

        cfg_local = bc.Config(
            root=Path(core_cfg["root"]),
            symbol=str(core_cfg["symbol"]).upper(),
            from_month=str(core_cfg["from_month"]),
            to_month=str(core_cfg["to_month"]),
            run_tag="postprocess_thresholds",
            only_session=None,
            spread_pips=float(core_cfg.get("spread_pips", 1.0)),
            pip_size=float(core_cfg["pip_size"]),
        )

        df_bid = bc.load_parquet_10s_bid(cfg_local)
        df10 = bc.add_synthetic_bidask(df_bid, cfg_local)

        o = df10["open_bid"].resample("1min", label="right", closed="right").first()
        h = df10["high_bid"].resample("1min", label="right", closed="right").max()
        l = df10["low_bid"].resample("1min", label="right", closed="right").min()
        c = df10["close_bid"].resample("1min", label="right", closed="right").last()
        m1 = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna()

        body = (m1["close"] - m1["open"]).abs().astype(float)
        mean_prev_body = body.shift(1).rolling(int(lookback)).mean().astype(float)
        prev_high = m1["high"].shift(1).astype(float)
        prev_low = m1["low"].shift(1).astype(float)

        trig = pd.DataFrame(
            {
                "open": m1["open"].astype(float),
                "close": m1["close"].astype(float),
                "body": body,
                "mean_prev_body": mean_prev_body,
                "prev_high": prev_high,
                "prev_low": prev_low,
            },
            index=m1.index,
        )

        trades = pd.read_csv(trades_csv)
        trades["side"] = trades["side"].astype(str).str.lower()
        trades = trades[trades["side"] == side].copy()
        if trades.empty:
            raise ValueError(f"No {side} trades in D009 verify: {trades_csv}")

        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
        trades = trades.dropna(subset=["entry_time"]).copy()

        # Mapping: entry_time is ts_next; evaluate ts=entry_time-10s; burst_m1_end = floor(ts)-1min
        trades["signal_eval_ts"] = trades["entry_time"] - pd.Timedelta(seconds=10)
        trades["burst_m1_end"] = trades["signal_eval_ts"].dt.floor("min") - pd.Timedelta(minutes=1)

        trig_reset = trig.reset_index()
        trig_reset = trig_reset.rename(columns={str(trig_reset.columns[0]): "burst_m1_end"})
        merged = trades.merge(trig_reset, on="burst_m1_end", how="left")

        close = merged["close"].astype(float)
        open_ = merged["open"].astype(float)
        body_f = (close - open_).abs().astype(float)
        mean_prev = merged["mean_prev_body"].astype(float)
        prev_hi = merged["prev_high"].astype(float)
        prev_lo = merged["prev_low"].astype(float)

        if side == "buy":
            break_margin = (close - prev_hi).astype(float)
        else:
            break_margin = (prev_lo - close).astype(float)

        primary = np.where(mean_prev > 0, break_margin / mean_prev, np.nan).astype(float)
        fallback = np.where(body_f > 0, break_margin / body_f, np.nan).astype(float)

        primary_s = pd.Series(primary).dropna()
        fallback_s = pd.Series(fallback).dropna()

        primary_p80 = float(primary_s.quantile(0.8)) if len(primary_s) > 0 else float("nan")
        fallback_p80 = float(fallback_s.quantile(0.8)) if len(fallback_s) > 0 else float("nan")

        chosen_feature = "break_margin_over_mean_prev_body" if len(primary_s) > 0 else "break_margin_ratio"
        chosen_p80 = primary_p80 if chosen_feature == "break_margin_over_mean_prev_body" else fallback_p80
        if not np.isfinite(chosen_p80):
            raise ValueError("Could not compute p80 threshold (no non-NaN values)")

        return {
            "period_source": "D009/in_sample_2024",
            "side": side.upper(),
            "exclude_rate": 0.20,
            "primary_feature": "break_margin_over_mean_prev_body",
            "primary_p80": primary_p80,
            "primary_n": int(len(primary_s)),
            "fallback_feature": "break_margin_ratio",
            "fallback_p80": fallback_p80,
            "fallback_n": int(len(fallback_s)),
            "chosen_feature": chosen_feature,
            "chosen_p80": float(chosen_p80),
            "nan_policy": "pass",
        }

    def _upsert_family_d_summary_row(*, hyp: str, verify_sum: float, verify_trades: int, forward_sum: float, forward_trades: int) -> None:
        import csv

        summary_path = _results_root() / "summary_family_D_momentum.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        rows: list[dict] = []
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                rows = list(r)

        by_hyp = {row["hyp"]: row for row in rows if row.get("hyp")}
        judge = "conditional" if (forward_sum >= 0) or (forward_sum > verify_sum) else "dead"
        by_hyp[hyp] = {
            "hyp": hyp,
            "verify_sum_pnl_pips": float(verify_sum),
            "verify_trades": int(verify_trades),
            "forward_sum_pnl_pips": float(forward_sum),
            "forward_trades": int(forward_trades),
            "judge": judge,
        }

        out = [by_hyp[k] for k in sorted(by_hyp.keys())]
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "hyp",
                    "verify_sum_pnl_pips",
                    "verify_trades",
                    "forward_sum_pnl_pips",
                    "forward_trades",
                    "judge",
                ],
            )
            w.writeheader()
            for row in out:
                w.writerow(row)

    def _write_compare_vs_d009(*, out_csv: Path, model_dir: Path, model_hyp: str, side: str) -> None:
        import csv

        d009_dir = _results_root() / "family_D_momentum" / "D009"
        rows = []
        for period in ("in_sample_2024", "forward_2025"):
            d009_trades = d009_dir / period / "trades.csv"
            model_trades = model_dir / period / "trades.csv"
            if not d009_trades.exists() or not model_trades.exists():
                raise FileNotFoundError(f"Missing trades for compare: {d009_trades} {model_trades}")

            d009_m = _read_trades_side_metrics(d009_trades, side=side)
            model_m = _read_trades_side_metrics(model_trades, side=side)

            for model_name, m in [("D009", d009_m), (model_hyp, model_m)]:
                rows.append(
                    {
                        "period": ("verify" if period == "in_sample_2024" else "forward"),
                        "side": side.upper(),
                        "model": model_name,
                        **m,
                    }
                )

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "period",
                    "side",
                    "model",
                    "n_trades",
                    "n_early_loss",
                    "early_loss_rate",
                    "n_survivor",
                    "survivor_rate",
                    "sum_pnl_pips",
                    "avg_pnl_pips",
                ],
            )
            w.writeheader()
            for r in rows:
                w.writerow(r)

    def _append_results_to_md(md_path: Path, *, thresholds: dict, meta: dict, compare_csv: Path, base_compare_hyp: str) -> None:
        text = md_path.read_text(encoding="utf-8")
        if "## 結果メモ（後で記入）" not in text:
            raise ValueError(f"Missing results section in: {md_path}")
        if "feature=" in text and "compare CSV" in text:
            # already filled (avoid duplicates)
            return

        verify_sum = float(meta["verify"]["sum_pnl_pips"])
        verify_tr = int(meta["verify"]["trades"])
        forward_sum = float(meta["forward"]["sum_pnl_pips"])
        forward_tr = int(meta["forward"]["trades"])

        add = []
        add.append("")
        add.append(f"- p80閾値（verify基準）: feature={thresholds['chosen_feature']} p80={thresholds['chosen_p80']}")
        add.append(f"- 2024 verify: sum_pnl_pips={verify_sum}, trades={verify_tr}")
        add.append(f"- 2025 forward: sum_pnl_pips={forward_sum}, trades={forward_tr}")
        add.append(f"- thresholds.json: {(_results_root() / meta['family'] / meta['hyp'] / 'thresholds.json')}")
        add.append(f"- compare CSV（{base_compare_hyp}比）: {compare_csv}")

        md_path.write_text(text + "\n" + "\n".join(add) + "\n", encoding="utf-8")

    if args.hyp in ("D011a", "D011b"):
        # D011a/b: exclude top20% exhaustion signals.
        # Thresholds are computed from verify(2024) only (no forward leakage), using D009 verify trades.
        is_buy = (args.hyp == "D011a")
        side = "buy" if is_buy else "sell"

        thresholds = _compute_exhaustion_thresholds_from_d009_verify(side=side)

        cfg = HypConfig(
            family="family_D_momentum",
            hyp=str(args.hyp),
            symbol=str(args.symbol).upper(),
            root=str(root),
            only_session=None,
            verify_from_month=str(args.verify_from_month),
            verify_to_month=str(args.verify_to_month),
            forward_from_month=str(args.forward_from_month),
            forward_to_month=str(args.forward_to_month),
            spread_pips=float(args.spread_pips),
            use_h1_trend_filter=True,
            use_time_filter=False,
            entry_mode="D_m1_momentum",
            disable_m1_bias_filter=False,
            dump_trades=True,
            momentum_mode="D004_continuation",
            allow_buy=True if is_buy else False,
            allow_sell=False if is_buy else True,
            exhaustion_filter_enabled=True,
            exhaustion_apply_side=side,
            exhaustion_feature=str(thresholds["chosen_feature"]),
            exhaustion_threshold=float(thresholds["chosen_p80"]),
            exhaustion_nan_policy="pass",
        )
        meta = _run_variant_inprocess(cfg, run_tag=str(args.hyp).lower())

        out_root = _results_root() / "family_D_momentum" / str(args.hyp)
        (out_root / "thresholds.json").write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")

        compare_csv = out_root / "diagnostics" / ("early_loss_compare_vs_D009_buy.csv" if is_buy else "early_loss_compare_vs_D009_sell.csv")
        _write_compare_vs_d009(out_csv=compare_csv, model_dir=out_root, model_hyp=str(args.hyp), side=side)

        _upsert_family_d_summary_row(
            hyp=str(args.hyp),
            verify_sum=float(meta["verify"]["sum_pnl_pips"]),
            verify_trades=int(meta["verify"]["trades"]),
            forward_sum=float(meta["forward"]["sum_pnl_pips"]),
            forward_trades=int(meta["forward"]["trades"]),
        )

        md_path = Path("FX/30_hypotheses/family_D_momentum") / ("D011a_exhaustion_ratio_buy.md" if is_buy else "D011b_exhaustion_ratio_sell.md")
        _append_results_to_md(md_path, thresholds=thresholds, meta=meta, compare_csv=compare_csv, base_compare_hyp="D009")

        print("[runner] done. summary:", json.dumps(meta, ensure_ascii=False), flush=True)
        return 0

    if args.hyp in ("B001", "B002", "B003", "B004", "B005", "C001", "C002", "C003", "C004", "C005"):
        family = "family_B_failedbreakout" if args.hyp.startswith("B") else "family_C_m1entry"
        only_session = "W1" if args.hyp.endswith("002") else None
        use_h1 = False if args.hyp.endswith("003") else None
        use_time = False if args.hyp.endswith("005") else None
        max_ls = 999 if args.hyp.endswith("004") else None
        entry_mode = "B_failed_breakout" if args.hyp.startswith("B") else "C_m1_close"
        bias_mode = "default" if args.hyp.startswith("B") else "m1_candle"
        cfg = HypConfig(
            family=family,
            hyp=str(args.hyp),
            symbol=str(args.symbol).upper(),
            root=str(root),
            only_session=only_session,
            verify_from_month=str(args.verify_from_month),
            verify_to_month=str(args.verify_to_month),
            forward_from_month=str(args.forward_from_month),
            forward_to_month=str(args.forward_to_month),
            spread_pips=float(args.spread_pips),
            use_h1_trend_filter=use_h1,
            use_time_filter=use_time,
            entry_mode=entry_mode,
            bias_mode=bias_mode,
            max_losing_streak=max_ls,
        )
        meta = _run_variant_inprocess(cfg, run_tag=str(args.hyp).lower())
        print("[runner] done. summary:", json.dumps(meta, ensure_ascii=False), flush=True)
        return 0

    if args.hyp == "B006":
        import csv
        import backtest_core as bc
        import pandas as pd

        out_dir = _results_root() / "family_B_failedbreakout" / "B006_observation"
        out_dir.mkdir(parents=True, exist_ok=True)

        obs_cfg = bc.Config(
            root=Path(str(root)),
            symbol=str(args.symbol).upper(),
            from_month="2025-01",
            to_month="2025-12",
            run_tag="b006_obs",
        )

        df_bid = bc.load_parquet_10s_bid(obs_cfg)
        df10 = bc.add_synthetic_bidask(df_bid, obs_cfg)
        h1_up = bc.compute_h1_uptrend(df10, obs_cfg)

        months = df10.index.to_period("M").astype(str)
        ratio_df = pd.DataFrame({"month": months, "h1_up": h1_up.astype(int)})
        grp = ratio_df.groupby("month", as_index=False).agg(
            bars_total=("h1_up", "size"),
            h1_uptrend_true_bars=("h1_up", "sum"),
        )
        grp["h1_uptrend_ratio"] = grp["h1_uptrend_true_bars"] / grp["bars_total"]
        grp = grp.sort_values("month")

        p_ratio = out_dir / "h1_uptrend_monthly_ratio_2025.csv"
        grp.to_csv(p_ratio, index=False)

        # merge with B002 forward monthly
        b002_monthly = _results_root() / "family_B_failedbreakout" / "B002" / "monthly.csv"
        if not b002_monthly.exists():
            raise FileNotFoundError(f"B002 monthly.csv not found: {b002_monthly}")
        pnl = pd.read_csv(b002_monthly)
        merged = pnl.merge(grp[["month", "h1_uptrend_ratio"]], on="month", how="left")
        merged = merged[["month", "sum_pnl_pips", "trades", "h1_uptrend_ratio"]].sort_values("month")

        p_merged = out_dir / "merged_b002_pnl_vs_h1ratio_2025.csv"
        merged.to_csv(p_merged, index=False)

        # summary (win vs loss months)
        win = merged[merged["sum_pnl_pips"] > 0].copy()
        loss = merged[merged["sum_pnl_pips"] < 0].copy()

        win_mean = float(win["h1_uptrend_ratio"].mean()) if len(win) else float("nan")
        loss_mean = float(loss["h1_uptrend_ratio"].mean()) if len(loss) else float("nan")

        win_months = ",".join(win["month"].tolist())
        loss_months = ",".join(loss["month"].tolist())

        summary_lines = [
            "B006_observation (B002 / 2025 forward)",
            f"win_months_count={len(win)} win_mean_h1_uptrend_ratio={win_mean:.6f} months=[{win_months}]",
            f"loss_months_count={len(loss)} loss_mean_h1_uptrend_ratio={loss_mean:.6f} months=[{loss_months}]",
            f"ratio_csv={p_ratio}",
            f"merged_csv={p_merged}",
        ]
        p_summary = out_dir / "summary.txt"
        p_summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        cfg_out = {
            "family": "family_B_failedbreakout",
            "hyp": "B006_observation",
            "symbol": obs_cfg.symbol,
            "root": str(obs_cfg.root),
            "from_month": obs_cfg.from_month,
            "to_month": obs_cfg.to_month,
            "h1_ema_period": obs_cfg.h1_ema_period,
            "h1_ema_slope_bars": obs_cfg.h1_ema_slope_bars,
            "source_b002_monthly_csv": str(b002_monthly),
        }
        (out_dir / "config.json").write_text(json.dumps(cfg_out, ensure_ascii=False, indent=2), encoding="utf-8")

        print(p_summary.read_text(encoding="utf-8"), flush=True)
        return 0

    if args.hyp == "B007":
        import json as _json
        import numpy as np
        import pandas as pd

        import backtest_core as bc

        out_dir = _results_root() / "family_B_failedbreakout" / "B007_observation"
        out_dir.mkdir(parents=True, exist_ok=True)

        obs_cfg = bc.Config(
            root=Path(str(root)),
            symbol=str(args.symbol).upper(),
            from_month="2025-01",
            to_month="2025-12",
            run_tag="b007_obs",
        )

        df_bid = bc.load_parquet_10s_bid(obs_cfg)
        df10 = bc.add_synthetic_bidask(df_bid, obs_cfg)
        h1_up = bc.compute_h1_uptrend(df10, obs_cfg).astype(bool)
        ts = df10.index.to_numpy(dtype="datetime64[ns]")

        months = pd.PeriodIndex(df10.index, freq="M").astype(str)
        df = pd.DataFrame({"ts": df10.index, "month": months, "h1_up": h1_up})

        def run_stats_for_month(g: pd.DataFrame) -> dict:
            b = g["h1_up"].to_numpy(dtype=bool)
            t = g["ts"].to_numpy(dtype="datetime64[ns]")
            if len(b) == 0:
                return {
                    "max_true_run_hours": 0.0,
                    "mean_true_run_hours": 0.0,
                    "p90_true_run_hours": 0.0,
                    "true_runs_count": 0,
                    "total_true_hours": 0.0,
                    "total_hours": 0.0,
                    "true_ratio": 0.0,
                }

            # continuity requires: current True, previous True, and exact 10s spacing
            dt_ok = np.r_[False, (t[1:] - t[:-1]) == np.timedelta64(10, "s")]
            prev_true = np.r_[False, b[:-1]]
            cont = b & prev_true & dt_ok
            start = b & (~cont)
            run_id = np.cumsum(start).astype(np.int64)
            run_id = np.where(b, run_id, 0)
            lengths = np.bincount(run_id)[1:]  # bars per run

            hours = lengths.astype(float) / 360.0  # 10s bars => /360 = hours
            total_true_hours = float(b.sum()) / 360.0
            total_hours = float(len(b)) / 360.0
            true_ratio = total_true_hours / total_hours if total_hours > 0 else 0.0

            if len(hours) == 0:
                max_h = mean_h = p90_h = 0.0
            else:
                max_h = float(np.max(hours))
                mean_h = float(np.mean(hours))
                p90_h = float(np.quantile(hours, 0.90))

            return {
                "max_true_run_hours": max_h,
                "mean_true_run_hours": mean_h,
                "p90_true_run_hours": p90_h,
                "true_runs_count": int(len(hours)),
                "total_true_hours": total_true_hours,
                "total_hours": total_hours,
                "true_ratio": true_ratio,
            }

        stats_rows = []
        for month, g in df.groupby("month", sort=True):
            st = run_stats_for_month(g)
            stats_rows.append({"month": month, **st})

        stats = pd.DataFrame(stats_rows).sort_values("month")
        p_stats = out_dir / "h1_uptrend_run_stats_2025.csv"
        stats.to_csv(p_stats, index=False)

        # Merge with B006 merged CSV (month,sum_pnl_pips,trades,h1_uptrend_ratio)
        b006_merged = _results_root() / "family_B_failedbreakout" / "B006_observation" / "merged_b002_pnl_vs_h1ratio_2025.csv"
        if not b006_merged.exists():
            raise FileNotFoundError(f"B006 merged CSV not found: {b006_merged}")
        b006 = pd.read_csv(b006_merged)
        merged = b006.merge(stats, on="month", how="left")
        keep = [
            "month",
            "sum_pnl_pips",
            "trades",
            "true_ratio",
            "max_true_run_hours",
            "mean_true_run_hours",
            "p90_true_run_hours",
            "true_runs_count",
        ]
        merged_out = merged[keep].sort_values("month")
        p_merged = out_dir / "merged_b002_pnl_vs_h1run_2025.csv"
        merged_out.to_csv(p_merged, index=False)

        # Summary: win vs loss months (observation only)
        win = merged_out[merged_out["sum_pnl_pips"] > 0].copy()
        loss = merged_out[merged_out["sum_pnl_pips"] < 0].copy()

        def mean_or_nan(d: pd.DataFrame, col: str) -> float:
            return float(d[col].mean()) if len(d) else float("nan")

        metrics = [
            "true_ratio",
            "max_true_run_hours",
            "mean_true_run_hours",
            "p90_true_run_hours",
            "true_runs_count",
        ]

        lines = ["B007_observation (B002 / 2025 forward)"]
        lines.append(f"win_months_count={len(win)} months=[{','.join(win['month'].tolist())}]")
        lines.append(f"loss_months_count={len(loss)} months=[{','.join(loss['month'].tolist())}]")
        for m in metrics:
            lines.append(f"win_mean_{m}={mean_or_nan(win, m):.6f}")
            lines.append(f"loss_mean_{m}={mean_or_nan(loss, m):.6f}")
        lines.append(f"run_stats_csv={p_stats}")
        lines.append(f"merged_csv={p_merged}")

        p_summary = out_dir / "summary.txt"
        p_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

        cfg_out = {
            "family": "family_B_failedbreakout",
            "hyp": "B007_observation",
            "symbol": obs_cfg.symbol,
            "root": str(obs_cfg.root),
            "from_month": obs_cfg.from_month,
            "to_month": obs_cfg.to_month,
            "h1_ema_period": obs_cfg.h1_ema_period,
            "h1_ema_slope_bars": obs_cfg.h1_ema_slope_bars,
            "b006_merged_csv": str(b006_merged),
            "missing_bars_break_rule": "run continues only if previous bar exists at exactly +10s and both bars are true",
        }
        (out_dir / "config.json").write_text(_json.dumps(cfg_out, ensure_ascii=False, indent=2), encoding="utf-8")

        print(p_summary.read_text(encoding="utf-8"), flush=True)
        return 0

    if args.hyp == "B008":
        import json as _json
        import pandas as pd

        out_dir = _results_root() / "family_B_failedbreakout" / "B008_observation"
        out_dir.mkdir(parents=True, exist_ok=True)

        merged_b007 = _results_root() / "family_B_failedbreakout" / "B007_observation" / "merged_b002_pnl_vs_h1run_2025.csv"
        if not merged_b007.exists():
            raise FileNotFoundError(f"B007 merged CSV not found: {merged_b007}")

        df = pd.read_csv(merged_b007)
        required = {"month", "sum_pnl_pips", "trades", "true_ratio", "true_runs_count", "mean_true_run_hours", "p90_true_run_hours", "max_true_run_hours"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"B007 merged CSV missing columns: {sorted(missing)}")

        # Fixed rule: median split (no threshold search)
        med_runs = float(df["true_runs_count"].median())
        med_mean = float(df["mean_true_run_hours"].median())
        is_choppy = (df["true_runs_count"] > med_runs) & (df["mean_true_run_hours"] < med_mean)
        df["state"] = is_choppy.map({True: "choppy_trend", False: "persistent_trend"})

        labeled_cols = [
            "month",
            "sum_pnl_pips",
            "trades",
            "true_ratio",
            "true_runs_count",
            "mean_true_run_hours",
            "p90_true_run_hours",
            "max_true_run_hours",
            "state",
        ]
        labeled = df[labeled_cols].sort_values("month")
        p_labeled = out_dir / "state_labeled_2025.csv"
        labeled.to_csv(p_labeled, index=False)

        summary = labeled.groupby("state", as_index=False).agg(
            months_count=("month", "count"),
            sum_pnl_pips_sum=("sum_pnl_pips", "sum"),
            sum_pnl_pips_mean=("sum_pnl_pips", "mean"),
            trades_sum=("trades", "sum"),
        ).sort_values("state")
        p_state = out_dir / "state_summary_2025.csv"
        summary.to_csv(p_state, index=False)

        months_by_state = labeled.groupby("state")["month"].apply(list).to_dict()
        lines = [
            "B008_observation (B002 / 2025 forward)",
            "state rule: choppy_trend if (true_runs_count > median(true_runs_count)) AND (mean_true_run_hours < median(mean_true_run_hours)); else persistent_trend",
            f"median_true_runs_count={med_runs:.6f}",
            f"median_mean_true_run_hours={med_mean:.6f}",
            "",
            "state_summary_2025:",
        ]
        for _, r in summary.iterrows():
            lines.append(
                f"- {r['state']}: months={int(r['months_count'])} "
                f"sum_pnl_pips_sum={float(r['sum_pnl_pips_sum']):.1f} "
                f"sum_pnl_pips_mean={float(r['sum_pnl_pips_mean']):.3f} "
                f"trades_sum={int(r['trades_sum'])} "
                f"months={months_by_state.get(r['state'], [])}"
            )
        lines += [
            "",
            f"labeled_csv={p_labeled}",
            f"state_summary_csv={p_state}",
        ]
        p_summary = out_dir / "summary.txt"
        p_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

        cfg_out = {
            "family": "family_B_failedbreakout",
            "hyp": "B008_observation",
            "source_csv": str(merged_b007),
            "rule": "choppy_trend if (true_runs_count > median(true_runs_count)) AND (mean_true_run_hours < median(mean_true_run_hours)) else persistent_trend",
            "medians_2025": {
                "true_runs_count": med_runs,
                "mean_true_run_hours": med_mean,
            },
        }
        (out_dir / "config.json").write_text(_json.dumps(cfg_out, ensure_ascii=False, indent=2), encoding="utf-8")

        print(p_summary.read_text(encoding="utf-8"), flush=True)
        return 0
    raise AssertionError(f"Unsupported hyp: {args.hyp}")


if __name__ == "__main__":
    raise SystemExit(main())
