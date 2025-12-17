#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    # input
    root: Path
    symbol: str
    parquet_dirname: str = "bars10s_pq"

    # output
    results_root: Path = (Path("FX/results") if Path("FX/results").exists() else Path("results"))
    run_tag: str = "hyp"

    # month filter
    from_month: str | None = None
    to_month: str | None = None

    # debug
    debug_day: str | None = None
    debug_from: str | None = None
    debug_to: str | None = None
    dump_debug: bool = True

    # session filter (NEW)
    only_session: str | None = None   # "W1" or "W2" or None

    # --- Time filters (UTC) ---
    use_time_filter: bool = True
    w1_start: str = "08:00"
    w1_end: str = "11:00"
    w2_start: str = "13:30"
    w2_end: str = "16:00"
    rollover_from: str = "21:55"
    rollover_to: str = "22:10"
    fri_stop_after: str = "16:00"

    # --- Trend / bias ---
    use_h1_trend_filter: bool = True
    h1_ema_period: int = 50
    h1_ema_slope_bars: int = 6

    higher_tf: str = "1min"
    htf_swing_lookback: int = 5

    # --- LTF entry ---
    lookback_bars: int = 6
    allow_buy: bool = True
    allow_sell: bool = False

    # --- Exits ---
    stop_loss_pips: float = 10.0
    rr: float = 1.2
    pip_size: float = 0.01

    # --- daily risk control ---
    max_losing_streak: int = 2

    # --- execution model ---
    entry_on_next_open: bool = True
    sl_first_if_both: bool = True

    # --- spread model ---
    spread_pips: float = 1.0

    # --- progress logging ---
    log_every_files: int = 200
    log_every_sec: float = 5.0


# -------------------------
# Helpers
# -------------------------
def _parse_hhmm(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m), 0)


def _in_window(ts: pd.Timestamp, start: str, end: str) -> bool:
    t = ts.time()
    s = _parse_hhmm(start)
    e = _parse_hhmm(end)
    return (s <= t) and (t < e)


def is_trading_time(ts: pd.Timestamp, cfg: Config) -> bool:
    if ts.dayofweek in (5, 6):
        return False
    if ts.dayofweek == 4 and ts.time() >= _parse_hhmm(cfg.fri_stop_after):
        return False
    if _in_window(ts, cfg.rollover_from, cfg.rollover_to):
        return False
    in_w1 = _in_window(ts, cfg.w1_start, cfg.w1_end)
    in_w2 = _in_window(ts, cfg.w2_start, cfg.w2_end)
    return in_w1 or in_w2


def session_label(ts: pd.Timestamp, cfg: Config) -> str:
    if _in_window(ts, cfg.w1_start, cfg.w1_end):
        return "W1"
    if _in_window(ts, cfg.w2_start, cfg.w2_end):
        return "W2"
    return "OUT"


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def parquet_base(cfg: Config) -> Path:
    return cfg.root / cfg.symbol.upper() / cfg.parquet_dirname


def filter_files_by_month(files: list[Path], from_month: str | None, to_month: str | None) -> list[Path]:
    if not from_month and not to_month:
        return files
    fm = pd.Period(from_month, freq="M") if from_month else None
    tm = pd.Period(to_month, freq="M") if to_month else None

    out: list[Path] = []
    for f in files:
        parts = f.parts
        y = None
        m = None
        for i in range(len(parts) - 1, -1, -1):
            if m is None and parts[i].isdigit() and len(parts[i]) == 2:
                m = int(parts[i]); continue
            if y is None and parts[i].isdigit() and len(parts[i]) == 4:
                y = int(parts[i]); break
        if y is None or m is None:
            continue
        p = pd.Period(f"{y:04d}-{m:02d}", freq="M")
        if fm and p < fm:
            continue
        if tm and p > tm:
            continue
        out.append(f)
    return sorted(out)


def list_parquet_files(cfg: Config) -> list[Path]:
    base = parquet_base(cfg)
    files = sorted(base.rglob("*.parquet"))
    return filter_files_by_month(files, cfg.from_month, cfg.to_month)


def load_parquet_10s_bid(cfg: Config) -> pd.DataFrame:
    base = parquet_base(cfg)
    print(f"[debug] parquet base: {base}", flush=True)

    files = list_parquet_files(cfg)
    total = len(files)
    print(f"[load] parquet files (filtered): {total:,} range={cfg.from_month}..{cfg.to_month}", flush=True)
    if total == 0:
        raise FileNotFoundError(f"No parquet files found under: {base}")

    parts = []
    t0 = time.time()
    last = t0

    for i, fp in enumerate(files, start=1):
        df = pd.read_parquet(fp, engine="pyarrow")
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        parts.append(df)

        now = time.time()
        if (i % cfg.log_every_files == 0) or (now - last >= 2.0):
            elapsed = now - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            print(f"[load] {i:,}/{total:,} files read (rate={rate:.1f} files/s elapsed={elapsed:.1f}s)", flush=True)
            last = now

    all_df = pd.concat(parts, ignore_index=True).sort_values("ts").drop_duplicates("ts")
    all_df["ts"] = pd.to_datetime(all_df["ts"], utc=True)

    if cfg.debug_day:
        d = pd.to_datetime(cfg.debug_day).date()
        all_df = all_df[all_df["ts"].dt.date == d]
        print(f"[debug] filtered to debug_day={cfg.debug_day} rows={len(all_df):,}", flush=True)

    all_df = all_df.set_index("ts").sort_index()
    if len(all_df) == 0:
        raise ValueError("No rows after filtering (debug_day/range).")

    print(f"[load] bars: {len(all_df):,} from {all_df.index.min()} to {all_df.index.max()}", flush=True)
    return all_df


def add_synthetic_bidask(df_bid: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df_bid.copy()
    half = (cfg.spread_pips * cfg.pip_size) / 2.0

    df["open_bid"] = df["open"].astype(float)
    df["high_bid"] = df["high"].astype(float)
    df["low_bid"]  = df["low"].astype(float)
    df["close_bid"]= df["close"].astype(float)

    df["open_ask"] = df["open_bid"] + 2 * half
    df["high_ask"] = df["high_bid"] + 2 * half
    df["low_ask"]  = df["low_bid"]  + 2 * half
    df["close_ask"]= df["close_bid"]+ 2 * half
    return df


# -------------------------
# Signals
# -------------------------
def compute_h1_uptrend(df10: pd.DataFrame, cfg: Config) -> np.ndarray:
    close = df10["close_bid"]
    h1_close = close.resample("1H").last().dropna()
    h1_ema = ema(h1_close, cfg.h1_ema_period)
    ema_past = h1_ema.shift(cfg.h1_ema_slope_bars)
    ok = (h1_close > h1_ema) & (h1_ema > ema_past)
    ok_closed = ok.shift(1)
    return ok_closed.reindex(df10.index, method="ffill").fillna(False).to_numpy(dtype=bool)


def compute_bias_htf(df10: pd.DataFrame, cfg: Config) -> np.ndarray:
    o = df10["open_bid"].resample(cfg.higher_tf).first()
    h = df10["high_bid"].resample(cfg.higher_tf).max()
    l = df10["low_bid"].resample(cfg.higher_tf).min()
    c = df10["close_bid"].resample(cfg.higher_tf).last()
    htf = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna()

    look = cfg.htf_swing_lookback
    recent_high = htf["high"].shift(1).rolling(look).max()
    recent_low  = htf["low"].shift(1).rolling(look).min()

    bias = pd.Series(0, index=htf.index, dtype=np.int8)
    bias.loc[htf["close"] > recent_high] = 1
    bias.loc[htf["close"] < recent_low] = -1

    bias_closed = bias.shift(1)
    return bias_closed.reindex(df10.index, method="ffill").fillna(0).to_numpy(dtype=np.int8)


def high_breakout_10s(df10: pd.DataFrame, cfg: Config) -> np.ndarray:
    prev_high = df10["high_bid"].shift(1).rolling(cfg.lookback_bars).max()
    return (df10["close_bid"] > prev_high).fillna(False).to_numpy(dtype=bool)


def low_breakout_10s(df10: pd.DataFrame, cfg: Config) -> np.ndarray:
    prev_low = df10["low_bid"].shift(1).rolling(cfg.lookback_bars).min()
    return (df10["close_bid"] < prev_low).fillna(False).to_numpy(dtype=bool)


# -------------------------
# Debug signals dump
# -------------------------
def dump_debug_signals(df10: pd.DataFrame, cfg: Config, run_dir: Path) -> Path:
    t_from = _parse_hhmm(cfg.debug_from) if cfg.debug_from else dtime(0, 0, 0)
    t_to = _parse_hhmm(cfg.debug_to) if cfg.debug_to else dtime(23, 59, 59)

    idx = df10.index
    mask = np.array([(t_from <= ts.time() <= t_to) for ts in idx], dtype=bool)

    look = cfg.lookback_bars
    prev_high = df10["high_bid"].shift(1).rolling(look).max()
    breakout_hi = (df10["close_bid"] > prev_high).fillna(False)

    bias = compute_bias_htf(df10, cfg)
    h1_ok = compute_h1_uptrend(df10, cfg) if cfg.use_h1_trend_filter else np.ones(len(df10), dtype=bool)
    time_ok = np.array([(not cfg.use_time_filter) or is_trading_time(ts, cfg) for ts in idx], dtype=bool)
    sess = np.array([session_label(ts, cfg) for ts in idx])

    out = pd.DataFrame({
        "ts": idx.astype(str),
        "session": sess,
        "time_ok": time_ok.astype(int),
        "bias": bias.astype(int),
        "h1_ok": h1_ok.astype(int),
        "close_bid": df10["close_bid"].to_numpy(),
        f"prev_high_{cfg.lookback_bars}": prev_high.to_numpy(),
        "breakout_hi": breakout_hi.to_numpy().astype(int),
    }).loc[mask].copy()

    path = run_dir / "debug_signals.csv"
    out.to_csv(path, index=False)
    return path


# -------------------------
# Backtest engine (only_session support)
# -------------------------
def backtest(df10: pd.DataFrame, cfg: Config, runlog_path: Path, run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = df10.index
    n_bars = len(idx)

    print(f"[bt] preparing signals... bars={n_bars:,} only_session={cfg.only_session}", flush=True)

    time_ok = np.array([(not cfg.use_time_filter) or is_trading_time(ts, cfg) for ts in idx], dtype=bool)
    bias = compute_bias_htf(df10, cfg)
    h1_ok = compute_h1_uptrend(df10, cfg) if cfg.use_h1_trend_filter else np.ones(n_bars, dtype=bool)
    brk_hi = high_breakout_10s(df10, cfg)
    brk_lo = low_breakout_10s(df10, cfg)

    open_ask = df10["open_ask"].to_numpy()
    open_bid = df10["open_bid"].to_numpy()
    high_bid = df10["high_bid"].to_numpy()
    low_bid  = df10["low_bid"].to_numpy()
    high_ask = df10["high_ask"].to_numpy()
    low_ask  = df10["low_ask"].to_numpy()
    close_ask= df10["close_ask"].to_numpy()
    close_bid= df10["close_bid"].to_numpy()

    pos = None
    losing_streak = 0
    trading_stopped = False
    last_trade_date = idx[0].date()

    trades = []

    # session array for fast filtering
    sess_arr = np.array([session_label(ts, cfg) for ts in idx])

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

            # exit
            if pos is not None:
                if pos["side"] == "buy":
                    hit_sl = low_bid[i] <= pos["sl"]
                    hit_tp = high_bid[i] >= pos["tp"]
                    exit_price = None
                    reason = None
                    if hit_sl and hit_tp:
                        exit_price, reason = (pos["sl"], "sl_both") if cfg.sl_first_if_both else (pos["tp"], "tp_both")
                    elif hit_sl:
                        exit_price, reason = pos["sl"], "sl"
                    elif hit_tp:
                        exit_price, reason = pos["tp"], "tp"

                    if exit_price is not None:
                        pnl_pips = (exit_price - pos["entry_price"]) / cfg.pip_size
                        trades.append({
                            "side": "buy",
                            "entry_time": pos["entry_time"],
                            "exit_time": ts,
                            "entry_price": pos["entry_price"],
                            "exit_price": exit_price,
                            "pnl_pips": pnl_pips,
                            "exit_reason": reason,
                            "session": session_label(pos["entry_time"], cfg),
                        })
                        losing_streak = losing_streak + 1 if pnl_pips < 0 else 0
                        if losing_streak >= cfg.max_losing_streak:
                            trading_stopped = True
                            lg.write(json.dumps({"event": "daily_stop", "ts": str(ts), "losing_streak": losing_streak}) + "\n")
                        pos = None

                else:
                    hit_sl = high_ask[i] >= pos["sl"]
                    hit_tp = low_ask[i] <= pos["tp"]
                    exit_price = None
                    reason = None
                    if hit_sl and hit_tp:
                        exit_price, reason = (pos["sl"], "sl_both") if cfg.sl_first_if_both else (pos["tp"], "tp_both")
                    elif hit_sl:
                        exit_price, reason = pos["sl"], "sl"
                    elif hit_tp:
                        exit_price, reason = pos["tp"], "tp"

                    if exit_price is not None:
                        pnl_pips = (pos["entry_price"] - exit_price) / cfg.pip_size
                        trades.append({
                            "side": "sell",
                            "entry_time": pos["entry_time"],
                            "exit_time": ts,
                            "entry_price": pos["entry_price"],
                            "exit_price": exit_price,
                            "pnl_pips": pnl_pips,
                            "exit_reason": reason,
                            "session": session_label(pos["entry_time"], cfg),
                        })
                        losing_streak = losing_streak + 1 if pnl_pips < 0 else 0
                        if losing_streak >= cfg.max_losing_streak:
                            trading_stopped = True
                            lg.write(json.dumps({"event": "daily_stop", "ts": str(ts), "losing_streak": losing_streak}) + "\n")
                        pos = None

            # entry
            if pos is not None:
                continue
            if not time_ok[i]:
                continue

            # NEW: session restriction
            if cfg.only_session is not None:
                if sess_arr[i] != cfg.only_session:
                    continue

            b = int(bias[i])
            if b == 0:
                continue

            if b == 1 and cfg.use_h1_trend_filter and not bool(h1_ok[i]):
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


# -------------------------
# Sanity check (buy-only + entry conditions in debug mode)
# -------------------------
def sanity_check(run_dir: Path) -> None:
    trades_path = run_dir / "trades.csv"
    dbg_entries = run_dir / "debug_entries.csv"
    dbg_signals = run_dir / "debug_signals.csv"

    if not trades_path.exists():
        print("[check] trades.csv not found -> skip", flush=True)
        return

    trades = pd.read_csv(trades_path)
    if "side" in trades.columns:
        if (trades["side"] == "sell").any():
            raise AssertionError("SanityCheck failed: sell trades exist but should be buy-only.")
        print("[check] OK: buy-only (no sell trades)", flush=True)

    if dbg_entries.exists() and dbg_signals.exists():
        ent = pd.read_csv(dbg_entries)
        sig = pd.read_csv(dbg_signals).set_index("ts")

        bad = []
        for _, r in ent.iterrows():
            t = r["signal_bar_time"]
            if t not in sig.index:
                bad.append((t, "signal_bar_time_not_found"))
                continue
            row = sig.loc[t]
            cond = (row["time_ok"] == 1) and (row["bias"] == 1) and (row["h1_ok"] == 1) and (row["breakout_hi"] == 1)
            if not cond:
                bad.append((t, f"cond_failed time_ok={row['time_ok']} bias={row['bias']} h1_ok={row['h1_ok']} breakout={row['breakout_hi']}"))

        if bad:
            raise AssertionError(f"SanityCheck failed: entries violate conditions: {bad[:5]} ... total={len(bad)}")
        print("[check] OK: every entry satisfies (time_ok=1,bias=1,h1_ok=1,breakout_hi=1)", flush=True)


def make_run_dir(cfg: Config) -> Path:
    run_id = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    rng = f"{cfg.from_month or 'all'}_{cfg.to_month or 'all'}"
    sess = f"_sess_{cfg.only_session}" if cfg.only_session else ""
    dbg = f"_debug_{cfg.debug_day}" if cfg.debug_day else ""
    run_dir = cfg.results_root / f"{cfg.run_tag}{sess}_{rng}{dbg}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="dukas_out_v2")
    p.add_argument("--symbol", default="USDJPY")
    p.add_argument("--from_month", default=None)
    p.add_argument("--to_month", default=None)

    # NEW
    p.add_argument("--only_session", default=None, choices=[None, "W1", "W2"], help="Restrict entries to W1 or W2")

    # debug
    p.add_argument("--debug_day", default=None)
    p.add_argument("--debug_from", default=None)
    p.add_argument("--debug_to", default=None)
    p.add_argument("--no_dump_debug", action="store_true")

    p.add_argument("--spread_pips", type=float, default=1.0)
    args = p.parse_args()

    return Config(
        root=Path(args.root),
        symbol=str(args.symbol).upper(),
        from_month=args.from_month,
        to_month=args.to_month,
        only_session=args.only_session,
        debug_day=args.debug_day,
        debug_from=args.debug_from,
        debug_to=args.debug_to,
        spread_pips=float(args.spread_pips),
        dump_debug=(not args.no_dump_debug),
        run_tag="hyp002" if args.only_session == "W2" else "hyp001",
    )


def main():
    cfg = parse_args()
    print("=== backtest started ===", flush=True)
    print(f"[cfg] only_session={cfg.only_session} range={cfg.from_month}..{cfg.to_month} debug_day={cfg.debug_day}", flush=True)
    print(f"[cfg] input parquet base: {parquet_base(cfg)}", flush=True)

    files = list_parquet_files(cfg)
    print(f"[debug] parquet files found (filtered): {len(files):,}", flush=True)
    if len(files) == 0:
        raise FileNotFoundError("No parquet files found for the given range.")

    run_dir = make_run_dir(cfg)
    print(f"[run] output dir: {run_dir}", flush=True)

    (run_dir / "config.json").write_text(
        json.dumps({**asdict(cfg), "root": str(cfg.root), "results_root": str(cfg.results_root)}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    runlog_path = run_dir / "runlog.jsonl"

    df_bid = load_parquet_10s_bid(cfg)
    df10 = add_synthetic_bidask(df_bid, cfg)

    if cfg.debug_day and cfg.dump_debug:
        pth = dump_debug_signals(df10, cfg, run_dir)
        print(f"[done] debug_signals.csv written: {pth}", flush=True)

    trades, monthly, monthly_session = backtest(df10, cfg, runlog_path, run_dir)

    trades.to_csv(run_dir / "trades.csv", index=False)
    monthly.to_csv(run_dir / "monthly.csv", index=False)
    monthly_session.to_csv(run_dir / "monthly_by_session.csv", index=False)

    print(f"[done] trades: {len(trades):,}", flush=True)
    if not monthly.empty:
        print("[done] monthly (last 12):", flush=True)
        print(monthly.tail(12).to_string(index=False), flush=True)

    sanity_check(run_dir)


if __name__ == "__main__":
    main()
