#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dtime
from typing import Literal

import numpy as np
import pandas as pd


TrendSide = Literal["up", "down", "flat"]
BreakSide = Literal["up", "down"]


@dataclass(frozen=True)
class Params:
    # session windows (UTC)
    w1_start: str = "08:00"
    w1_end: str = "11:00"
    w2_start: str = "13:30"
    w2_end: str = "16:00"

    # 5m structure
    range_bars: int = 12  # 12*5min = 60min
    atr_period: int = 14
    ema_period: int = 50
    ema_slope_bars: int = 6

    # time alignment between 5m and 10s
    search_break_window_sec: int = 15 * 60

    # 10s event windows
    v_return_max_sec: int = 15 * 60
    rebreak_window_sec: int = 10 * 60

    # heuristic "ok" flags (not filters; just observation labels)
    range_width_atr_min: float = 0.5
    range_width_atr_max: float = 2.5
    fakeout_ratio_min: float = 0.05
    fakeout_ratio_max: float = 0.8


def pip_size_for(symbol: str) -> float:
    return 0.01 if symbol.upper().endswith("JPY") else 0.0001


def _parse_hhmm(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m), 0)


def _in_window(ts: pd.Timestamp, start: str, end: str) -> bool:
    t = ts.time()
    s = _parse_hhmm(start)
    e = _parse_hhmm(end)
    return (s <= t) and (t < e)


def session_label(ts: pd.Timestamp, params: Params) -> str:
    if _in_window(ts, params.w1_start, params.w1_end):
        return "W1"
    if _in_window(ts, params.w2_start, params.w2_end):
        return "W2"
    return "OUT"


def _require_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    candidates = [
        ("open", "high", "low", "close"),
        ("open_bid", "high_bid", "low_bid", "close_bid"),
    ]
    for o, h, l, c in candidates:
        if {o, h, l, c}.issubset(set(cols)):
            out = df[[o, h, l, c]].copy()
            out.columns = ["open", "high", "low", "close"]
            return out.astype(float)
    raise ValueError(f"Missing OHLC columns. Found: {sorted(df.columns)}")


def detect_trend_5m(df5: pd.DataFrame, params: Params) -> pd.DataFrame:
    """
    Trend proxy aligned with backtest_core spirit:
    - up: close > EMA and EMA rising vs ema_slope_bars ago
    - down: close < EMA and EMA falling vs ema_slope_bars ago
    - else flat
    """
    ohlc = _require_ohlc(df5)
    close = ohlc["close"]
    ema = close.ewm(span=params.ema_period, adjust=False).mean()
    ema_past = ema.shift(params.ema_slope_bars)

    up = (close > ema) & (ema > ema_past)
    down = (close < ema) & (ema < ema_past)
    side = np.where(up.to_numpy(), "up", np.where(down.to_numpy(), "down", "flat"))

    # Reference levels for later diagnostics: last swing high/low over the structure window.
    look = params.range_bars
    trend_break_high = ohlc["high"].shift(1).rolling(look).max()
    trend_break_low = ohlc["low"].shift(1).rolling(look).min()

    return pd.DataFrame(
        {
            "trend_side": side,
            "trend_break_high": trend_break_high.astype(float),
            "trend_break_low": trend_break_low.astype(float),
        },
        index=df5.index,
    )


def _atr(df5: pd.DataFrame, period: int) -> pd.Series:
    ohlc = _require_ohlc(df5)
    high = ohlc["high"]
    low = ohlc["low"]
    close = ohlc["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def detect_mid_range_5m(df5: pd.DataFrame, params: Params, *, pip_size: float) -> pd.DataFrame:
    ohlc = _require_ohlc(df5)
    rng = params.range_bars
    atr5 = _atr(ohlc, params.atr_period)

    range_high = ohlc["high"].shift(1).rolling(rng).max()
    range_low = ohlc["low"].shift(1).rolling(rng).min()
    width = (range_high - range_low).astype(float)
    width_pips = width / pip_size
    width_atr = width / atr5

    range_ok = (width_atr >= params.range_width_atr_min) & (width_atr <= params.range_width_atr_max)

    return pd.DataFrame(
        {
            "atr5": atr5.astype(float),
            "range_bars": rng,
            "range_high": range_high.astype(float),
            "range_low": range_low.astype(float),
            "range_width_pips": width_pips.astype(float),
            "range_width_atr": width_atr.astype(float),
            "range_ok": range_ok.fillna(False).astype(bool),
        },
        index=df5.index,
    )


def detect_fakeout_10s(
    df10: pd.DataFrame,
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    range_high: float,
    range_low: float,
    pip_size: float,
    params: Params,
) -> dict | None:
    ohlc = _require_ohlc(df10)
    w = ohlc.loc[(ohlc.index >= start_ts) & (ohlc.index <= end_ts)]
    if w.empty:
        return None

    close = w["close"]
    up_mask = close > range_high
    down_mask = close < range_low
    if not up_mask.any() and not down_mask.any():
        return None

    first_up = up_mask.idxmax() if up_mask.any() else None
    first_down = down_mask.idxmax() if down_mask.any() else None
    if first_up is not None and first_down is not None:
        break_side: BreakSide = "up" if first_up <= first_down else "down"
    elif first_up is not None:
        break_side = "up"
    else:
        break_side = "down"

    if break_side == "up":
        break_start_ts = first_up
        max_exc = float((w.loc[break_start_ts:, "high"].max() - range_high) / pip_size)
        width_pips = float((range_high - range_low) / pip_size) if range_high > range_low else 0.0
        fakeout_ratio = (max_exc / width_pips) if width_pips > 0 else np.nan
    else:
        break_start_ts = first_down
        max_exc = float((range_low - w.loc[break_start_ts:, "low"].min()) / pip_size)
        width_pips = float((range_high - range_low) / pip_size) if range_high > range_low else 0.0
        fakeout_ratio = (max_exc / width_pips) if width_pips > 0 else np.nan

    fakeout_ok = (
        np.isfinite(fakeout_ratio)
        and (params.fakeout_ratio_min <= fakeout_ratio <= params.fakeout_ratio_max)
    )

    return {
        "break_side": break_side,
        "break_start_ts": break_start_ts,
        "break_max_excursion_pips": max_exc,
        "fakeout_ratio": float(fakeout_ratio) if np.isfinite(fakeout_ratio) else np.nan,
        "fakeout_ok": bool(fakeout_ok),
    }


def detect_v_return_10s(
    df10: pd.DataFrame,
    *,
    break_side: BreakSide,
    break_start_ts: pd.Timestamp,
    range_high: float,
    range_low: float,
    params: Params,
) -> dict:
    ohlc = _require_ohlc(df10)
    end_ts = break_start_ts + pd.Timedelta(seconds=params.v_return_max_sec)
    w = ohlc.loc[(ohlc.index >= break_start_ts) & (ohlc.index <= end_ts)]
    if w.empty:
        return {"v_return_ts": pd.NaT, "v_return_sec": np.nan, "v_impulse_bars": 0, "v_return_ok": False}

    close = w["close"]
    if break_side == "up":
        ret = close <= range_high
    else:
        ret = close >= range_low

    if not ret.any():
        return {"v_return_ts": pd.NaT, "v_return_sec": np.nan, "v_impulse_bars": int(len(w)), "v_return_ok": False}

    v_return_ts = ret.idxmax()
    v_return_sec = float((v_return_ts - break_start_ts).total_seconds())
    v_impulse_bars = int((w.index <= v_return_ts).sum())
    v_return_ok = v_return_sec <= params.v_return_max_sec
    return {
        "v_return_ts": v_return_ts,
        "v_return_sec": v_return_sec,
        "v_impulse_bars": v_impulse_bars,
        "v_return_ok": bool(v_return_ok),
    }


def detect_rebreak_10s(
    df10: pd.DataFrame,
    *,
    break_side: BreakSide,
    v_return_ts: pd.Timestamp,
    range_high: float,
    range_low: float,
    params: Params,
) -> dict:
    if pd.isna(v_return_ts):
        return {"rebreak_window_sec": params.rebreak_window_sec, "rebreak_in_window": False}

    ohlc = _require_ohlc(df10)
    end_ts = v_return_ts + pd.Timedelta(seconds=params.rebreak_window_sec)
    w = ohlc.loc[(ohlc.index >= v_return_ts) & (ohlc.index <= end_ts)]
    if w.empty:
        return {"rebreak_window_sec": params.rebreak_window_sec, "rebreak_in_window": False}

    close = w["close"]
    if break_side == "up":
        rebreak = (close > range_high).any()
    else:
        rebreak = (close < range_low).any()

    return {"rebreak_window_sec": params.rebreak_window_sec, "rebreak_in_window": bool(rebreak)}


def extract_S001_setups(symbol: str, df5: pd.DataFrame, df10: pd.DataFrame, params: Params) -> pd.DataFrame:
    """
    Extract S001 setup candidates as *observations*.

    Inputs:
      - df5: 5-minute OHLC, indexed by pd.Timestamp (UTC recommended)
      - df10: 10-second OHLC, indexed by pd.Timestamp (UTC recommended)
    """
    symbol = str(symbol).upper()
    pip_size = pip_size_for(symbol)

    df5 = df5.copy()
    df10 = df10.copy()
    if "ts" in df5.columns:
        df5["ts"] = pd.to_datetime(df5["ts"], utc=True)
        df5 = df5.set_index("ts")
    if "ts" in df10.columns:
        df10["ts"] = pd.to_datetime(df10["ts"], utc=True)
        df10 = df10.set_index("ts")
    df5 = df5.sort_index()
    df10 = df10.sort_index()

    trend = detect_trend_5m(df5, params)
    rng = detect_mid_range_5m(df5, params, pip_size=pip_size)

    # Candidate anchor: each 5m bar defines a preceding range window.
    # Event search runs forward on 10s after the 5m bar start.
    out_rows: list[dict] = []
    for t5 in df5.index:
        r = rng.loc[t5]
        if pd.isna(r["range_high"]) or pd.isna(r["range_low"]) or pd.isna(r["atr5"]):
            continue

        t5_start_ts = pd.Timestamp(t5)
        t10_start = t5_start_ts
        t10_end = t10_start + pd.Timedelta(seconds=params.search_break_window_sec)

        fake = detect_fakeout_10s(
            df10,
            start_ts=t10_start,
            end_ts=t10_end,
            range_high=float(r["range_high"]),
            range_low=float(r["range_low"]),
            pip_size=pip_size,
            params=params,
        )
        if fake is None:
            continue

        vret = detect_v_return_10s(
            df10,
            break_side=fake["break_side"],
            break_start_ts=fake["break_start_ts"],
            range_high=float(r["range_high"]),
            range_low=float(r["range_low"]),
            params=params,
        )
        rebr = detect_rebreak_10s(
            df10,
            break_side=fake["break_side"],
            v_return_ts=vret["v_return_ts"],
            range_high=float(r["range_high"]),
            range_low=float(r["range_low"]),
            params=params,
        )

        t10_event_ts = fake["break_start_ts"]
        row = {
            "symbol": symbol,
            "date": pd.Timestamp(t10_event_ts).date().isoformat(),
            "session": session_label(pd.Timestamp(t10_event_ts), params),
            "t5_start_ts": t5_start_ts,
            "t10_event_ts": t10_event_ts,
            "trend_side": trend.loc[t5, "trend_side"],
            "trend_break_high": float(trend.loc[t5, "trend_break_high"]) if pd.notna(trend.loc[t5, "trend_break_high"]) else np.nan,
            "trend_break_low": float(trend.loc[t5, "trend_break_low"]) if pd.notna(trend.loc[t5, "trend_break_low"]) else np.nan,
            "atr5": float(r["atr5"]) if pd.notna(r["atr5"]) else np.nan,
            "range_bars": int(r["range_bars"]),
            "range_high": float(r["range_high"]),
            "range_low": float(r["range_low"]),
            "range_width_pips": float(r["range_width_pips"]) if pd.notna(r["range_width_pips"]) else np.nan,
            "range_width_atr": float(r["range_width_atr"]) if pd.notna(r["range_width_atr"]) else np.nan,
            "range_ok": bool(r["range_ok"]),
            "break_side": fake["break_side"],
            "break_start_ts": fake["break_start_ts"],
            "break_max_excursion_pips": float(fake["break_max_excursion_pips"]),
            "fakeout_ratio": float(fake["fakeout_ratio"]),
            "fakeout_ok": bool(fake["fakeout_ok"]),
            "v_return_ts": vret["v_return_ts"],
            "v_return_sec": float(vret["v_return_sec"]) if np.isfinite(vret["v_return_sec"]) else np.nan,
            "v_impulse_bars": int(vret["v_impulse_bars"]),
            "v_return_ok": bool(vret["v_return_ok"]),
            "rebreak_window_sec": int(rebr["rebreak_window_sec"]),
            "rebreak_in_window": bool(rebr["rebreak_in_window"]),
            # b3_filter: placeholder observation flag (not used as a trading filter here)
            "b3_filter": bool(r["range_ok"] and fake["fakeout_ok"] and vret["v_return_ok"]),
        }
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out

    # Normalize dtypes for CSV friendliness
    ts_cols = ["t5_start_ts", "t10_event_ts", "break_start_ts", "v_return_ts"]
    for c in ts_cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], utc=True, errors="coerce")
    return out

