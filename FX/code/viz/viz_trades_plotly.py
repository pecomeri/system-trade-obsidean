#!/usr/bin/env python3
"""
Visualize per-trade M1 candlesticks around entries/exits.

Usage (recommended with uv):
  uv run python FX/code/viz/viz_trades_plotly.py \
    --results_dir FX/results/family_D_momentum/D001 \
    --symbol USDJPY \
    --window_minutes 30 \
    --top_n 30 \
    --sort_by pnl_pips_asc

Inputs:
  {results_dir}/trades.csv

Outputs:
  {results_dir}/charts/trade_{trade_id}.html
  {results_dir}/charts/index.html

Notes:
  - Visualization only. Does not modify backtest logic or aggregates.
  - Prefers Plotly; falls back to Matplotlib -> single-file HTML (PNG embedded as base64).
  - Uses Dukascopy 10s bid parquet: {data_root}/{symbol}/bars10s_pq/YYYY/MM/*.parquet
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class PriceColumns:
    ts: str
    open: str
    high: str
    low: str
    close: str


def _pip_size(symbol: str) -> float:
    s = symbol.upper()
    return 0.01 if s.endswith("JPY") else 0.0001


def _iter_month_starts_utc(start: "datetime", end: "datetime") -> list[tuple[int, int]]:
    if start.tzinfo is None:
        raise ValueError("start must be timezone-aware")
    if end.tzinfo is None:
        raise ValueError("end must be timezone-aware")

    cur = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    last = datetime(end.year, end.month, 1, tzinfo=timezone.utc)
    out: list[tuple[int, int]] = []
    while cur <= last:
        out.append((cur.year, cur.month))
        if cur.month == 12:
            cur = datetime(cur.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            cur = datetime(cur.year, cur.month + 1, 1, tzinfo=timezone.utc)
    return out


def _resolve_10s_parquet_files(data_root: Path, symbol: str, start_utc: datetime, end_utc: datetime) -> list[Path]:
    base = data_root / symbol / "bars10s_pq"
    months = _iter_month_starts_utc(start_utc, end_utc)
    files: list[Path] = []
    for y, m in months:
        mm = f"{m:02d}"
        p = base / str(y) / mm
        if not p.exists():
            continue
        files.extend(sorted(p.glob("*.parquet")))
    return files


def _detect_price_columns(cols: list[str]) -> PriceColumns:
    # Preferred (most common)
    if all(c in cols for c in ("ts", "open", "high", "low", "close")):
        return PriceColumns(ts="ts", open="open", high="high", low="low", close="close")

    # Common alternative prefixes
    for prefix in ("bid_", "mid_", "price_"):
        want = [f"{prefix}{c}" for c in ("open", "high", "low", "close")]
        if "ts" in cols and all(c in cols for c in want):
            return PriceColumns(ts="ts", open=want[0], high=want[1], low=want[2], close=want[3])

    # If only close exists, approximate OHLC from close (fallback)
    for close_col in ("close", "bid_close", "mid_close", "price_close"):
        if "ts" in cols and close_col in cols:
            return PriceColumns(ts="ts", open=close_col, high=close_col, low=close_col, close=close_col)

    raise ValueError(f"Cannot detect OHLC columns. Available columns: {cols}")


def _read_parquet_robust(path: Path):
    import pandas as pd

    try:
        return pd.read_parquet(path)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Failed to read parquet: {path} ({e})") from e


def _detect_price_columns_from_data_root(data_root: Path, symbol: str) -> PriceColumns:
    base = data_root / symbol / "bars10s_pq"
    if not base.exists():
        raise FileNotFoundError(f"bars10s_pq not found: {base}")

    # Find any one parquet file (by year/month order).
    for fp in sorted(base.glob("*/*/*.parquet")):
        sample = _read_parquet_robust(fp)
        return _detect_price_columns(list(sample.columns))

    raise FileNotFoundError(f"No parquet files found under: {base}")


def load_m1_ohlc_from_10s_parquet_range(
    *,
    data_root: Path,
    symbol: str,
    start_utc: datetime,
    end_utc: datetime,
    price_columns: PriceColumns | None = None,
    freq: str = "1min",
) -> "object":
    """
    Loads only the time range needed and returns a pandas DataFrame:
      index: ts (UTC, minute)
      columns: open, high, low, close
    """
    import pandas as pd

    files = _resolve_10s_parquet_files(data_root, symbol, start_utc, end_utc)
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {data_root}/{symbol}/bars10s_pq for requested period")

    pc = price_columns or _detect_price_columns_from_data_root(data_root, symbol)

    need_cols = sorted(set([pc.ts, pc.open, pc.high, pc.low, pc.close]))
    parts = []
    for fp in files:
        df = _read_parquet_robust(fp)
        missing = [c for c in need_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Parquet missing expected columns {missing} in {fp.name}")
        df = df[need_cols].copy()
        df[pc.ts] = pd.to_datetime(df[pc.ts], utc=True, errors="coerce")
        df = df.dropna(subset=[pc.ts])
        df = df[(df[pc.ts] >= start_utc) & (df[pc.ts] <= end_utc)]
        if not df.empty:
            parts.append(df)

    if not parts:
        raise ValueError("No bars in selected time range (after filtering).")

    df10 = pd.concat(parts, ignore_index=True).sort_values(pc.ts).drop_duplicates(pc.ts)
    df10 = df10.set_index(pc.ts)

    if freq == "10s":
        out = df10.rename(columns={pc.open: "open", pc.high: "high", pc.low: "low", pc.close: "close"})
        out = out[["open", "high", "low", "close"]].astype(float).sort_index().dropna()
        return out

    if freq != "1min":
        raise ValueError(f"Unsupported freq: {freq}")

    o = df10[pc.open].astype(float).resample("1min").first()
    h = df10[pc.high].astype(float).resample("1min").max()
    l = df10[pc.low].astype(float).resample("1min").min()
    c = df10[pc.close].astype(float).resample("1min").last()

    m1 = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna()
    return m1


def _sort_trades(df, sort_by: str):
    import pandas as pd

    if sort_by == "pnl_pips_asc":
        return df.sort_values("pnl_pips", ascending=True)
    if sort_by == "pnl_pips_desc":
        return df.sort_values("pnl_pips", ascending=False)
    if sort_by == "abs_pnl_pips_desc":
        return df.assign(_abs=df["pnl_pips"].abs()).sort_values("_abs", ascending=False).drop(columns=["_abs"])
    if sort_by == "entry_time_asc":
        return df.sort_values("entry_time", ascending=True)
    if sort_by == "entry_time_desc":
        return df.sort_values("entry_time", ascending=False)
    raise ValueError(f"Unknown sort_by: {sort_by}")


def _try_plotly():
    try:
        import plotly.graph_objects as go  # type: ignore

        return go
    except Exception:  # noqa: BLE001
        return None


def _render_trade_plotly(*, m1, trade, out_html: Path) -> None:
    go = _try_plotly()
    if go is None:
        raise RuntimeError("plotly is not available")

    import pandas as pd

    t_id = int(trade["trade_id"])
    side = str(trade["side"])
    pnl = float(trade["pnl_pips"])

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=m1.index,
                open=m1["open"],
                high=m1["high"],
                low=m1["low"],
                close=m1["close"],
                name="M1",
            )
        ]
    )

    entry_ts = pd.to_datetime(trade["entry_time"], utc=True)
    exit_ts = pd.to_datetime(trade["exit_time"], utc=True)
    entry_px = float(trade["entry_price"])
    exit_px = float(trade["exit_price"])

    # Extra timestamps to avoid confusion between:
    #   - trigger (confirmed M1) vs
    #   - signal evaluation bar (10s) vs
    #   - execution (next 10s open; entry_time)
    #
    # We do NOT have raw 10s bars here. So we approximate:
    #   signal_eval_ts ~= entry_time - 10s
    # which matches core's entry_on_next_open model (ts_next).
    signal_eval_ts = entry_ts - pd.Timedelta(seconds=10)
    burst_m1_end = signal_eval_ts.ceil("min") - pd.Timedelta(minutes=1)
    burst_m1_start = burst_m1_end - pd.Timedelta(minutes=1)

    # Highlight trigger M1 candle window (start..end)
    fig.add_vrect(
        x0=burst_m1_start,
        x1=burst_m1_end,
        fillcolor="rgba(0,128,255,0.10)",
        line_width=0,
        layer="below",
    )

    fig.add_vline(x=burst_m1_end, line_color="rgba(0,128,255,0.9)", line_width=2)
    fig.add_annotation(
        x=burst_m1_end,
        y=1.0,
        yref="paper",
        text="burst_m1_end",
        showarrow=False,
        xanchor="left",
        font=dict(color="rgba(0,128,255,0.95)", size=11),
    )

    fig.add_vline(x=signal_eval_ts, line_color="rgba(255,140,0,0.9)", line_width=2, line_dash="dot")
    fig.add_annotation(
        x=signal_eval_ts,
        y=0.96,
        yref="paper",
        text="signal_eval_ts (~entry-10s)",
        showarrow=False,
        xanchor="left",
        font=dict(color="rgba(255,140,0,0.95)", size=11),
    )

    fig.add_vline(x=entry_ts, line_color="green", line_width=2)
    fig.add_vline(x=exit_ts, line_color="red", line_width=2)
    fig.add_trace(go.Scatter(x=[entry_ts], y=[entry_px], mode="markers", marker=dict(color="green", size=10), name="entry"))
    fig.add_trace(go.Scatter(x=[exit_ts], y=[exit_px], mode="markers", marker=dict(color="red", size=10), name="exit"))
    # make exit/entry level visible (not only vertical timing)
    fig.add_hline(y=entry_px, line_color="green", line_width=1, opacity=0.6)
    fig.add_hline(y=exit_px, line_color="red", line_width=1, opacity=0.6)

    # annotate markers (price + reason) so exit point is obvious
    reason = str(trade.get("reason_exit") or trade.get("exit_reason") or "")
    fig.add_annotation(
        x=entry_ts,
        y=entry_px,
        text=f"entry {entry_px:.3f}",
        showarrow=True,
        arrowhead=2,
        ax=20,
        ay=-30,
        font=dict(color="green"),
    )
    fig.add_annotation(
        x=exit_ts,
        y=exit_px,
        text=f"exit {exit_px:.3f} {reason}".strip(),
        showarrow=True,
        arrowhead=2,
        ax=20,
        ay=30,
        font=dict(color="red"),
    )

    fb = trade.get("__failed_breakout__")
    if isinstance(fb, dict) and fb.get("fail_ts") is not None:
        fail_ts = fb.get("fail_ts")
        fail_px = fb.get("fail_px")
        break_ts = fb.get("break_ts")
        break_px = fb.get("break_px")
        prev_low = fb.get("prev_low")

        if prev_low is not None:
            fig.add_hline(y=float(prev_low), line_color="#888", line_width=1, line_dash="dot", opacity=0.6)
        label = fb.get("label", "M1 proxy")
        if break_ts is not None and break_px is not None:
            fig.add_trace(
                go.Scatter(
                    x=[break_ts],
                    y=[break_px],
                    mode="markers+text",
                    text=[f"breakdown ({label})"],
                    textposition="top center",
                    marker=dict(symbol="triangle-down", color="#F58518", size=10),
                    name="breakdown",
                )
            )
        if fail_ts is not None and fail_px is not None:
            fig.add_trace(
                go.Scatter(
                    x=[fail_ts],
                    y=[fail_px],
                    mode="markers+text",
                    text=[f"fail ({label})"],
                    textposition="bottom center",
                    marker=dict(symbol="triangle-up", color="#4C78A8", size=10),
                    name="fail",
                )
            )

    # highlight the M1 candle that contains the exit_time (visual anchor)
    if len(m1.index) > 0:
        exit_m1 = exit_ts.floor("min")
        fig.add_vrect(
            x0=exit_m1,
            x1=exit_m1 + pd.Timedelta(minutes=1),
            fillcolor="rgba(255,0,0,0.08)",
            line_width=0,
        )

    fig.update_layout(
        title=f"trade_id={t_id} side={side} pnl_pips={pnl}",
        xaxis_title="time (UTC)",
        yaxis_title="price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=600,
        dragmode="pan",
        xaxis=dict(fixedrange=False),
        yaxis=dict(fixedrange=False),
    )
    if len(m1.index) > 0:
        fig.update_xaxes(range=[m1.index.min(), m1.index.max()])
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(out_html),
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "responsive": True,
        },
    )


def _render_trade_matplotlib_html(*, m1, trade, out_html: Path) -> None:
    import matplotlib.dates as mdates  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    import pandas as pd

    t_id = int(trade["trade_id"])
    side = str(trade["side"])
    pnl = float(trade["pnl_pips"])

    entry_ts = pd.to_datetime(trade["entry_time"], utc=True)
    exit_ts = pd.to_datetime(trade["exit_time"], utc=True)
    entry_px = float(trade["entry_price"])
    exit_px = float(trade["exit_price"])

    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)

    xs = mdates.date2num(m1.index.to_pydatetime())
    o = m1["open"].to_numpy()
    h = m1["high"].to_numpy()
    l = m1["low"].to_numpy()
    c = m1["close"].to_numpy()

    candle_width = 0.0006  # ~0.86min in matplotlib date units; visualization-only constant
    for x, op, hi, lo, cl in zip(xs, o, h, l, c):
        up = cl >= op
        color = "#2ca02c" if up else "#d62728"
        ax.vlines(x, lo, hi, color=color, linewidth=1)
        lower = min(op, cl)
        height = abs(cl - op)
        ax.add_patch(
            plt.Rectangle(
                (x - candle_width / 2, lower),
                candle_width,
                height if height != 0 else 1e-12,
                facecolor=color,
                edgecolor=color,
                alpha=0.6,
            )
        )

    ax.axvline(mdates.date2num(entry_ts.to_pydatetime()), color="green", linewidth=2)
    ax.axvline(mdates.date2num(exit_ts.to_pydatetime()), color="red", linewidth=2)
    ax.scatter([mdates.date2num(entry_ts.to_pydatetime())], [entry_px], color="green", s=30, zorder=10)
    ax.scatter([mdates.date2num(exit_ts.to_pydatetime())], [exit_px], color="red", s=30, zorder=10)

    fb = trade.get("__failed_breakout__")
    if isinstance(fb, dict) and fb.get("fail_ts") is not None:
        fail_ts = fb.get("fail_ts")
        fail_px = fb.get("fail_px")
        break_ts = fb.get("break_ts")
        break_px = fb.get("break_px")
        prev_low = fb.get("prev_low")
        if prev_low is not None:
            ax.axhline(prev_low, color="#888", linestyle="--", linewidth=1)
        if break_ts is not None and break_px is not None:
            ax.scatter([mdates.date2num(break_ts.to_pydatetime())], [break_px], color="#F58518", s=40, marker="v", zorder=10)
        if fail_ts is not None and fail_px is not None:
            ax.scatter([mdates.date2num(fail_ts.to_pydatetime())], [fail_px], color="#4C78A8", s=40, marker="^", zorder=10)

    ax.set_title(f"trade_id={t_id} side={side} pnl_pips={pnl}")
    ax.set_xlabel("time (UTC)")
    ax.set_ylabel("price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=timezone.utc))
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.25)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>trade_{t_id}</title>
  <style>
    body {{ font-family: sans-serif; margin: 16px; }}
    .meta {{ margin-bottom: 8px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
  </style>
</head>
<body>
  <div class="meta">trade_id={t_id} side={side} pnl_pips={pnl}</div>
  <img src="data:image/png;base64,{png_b64}" alt="candlestick" />
</body>
</html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")


def _render_trade_svg_html(*, m1, trade, out_html: Path) -> None:
    """
    Dependency-free fallback renderer (inline SVG candlestick).
    """
    import pandas as pd

    if m1 is None or len(m1) == 0:
        out_html.parent.mkdir(parents=True, exist_ok=True)
        out_html.write_text("No M1 data in window.\n", encoding="utf-8")
        return

    t_id = int(trade["trade_id"])
    side = str(trade["side"])
    pnl = float(trade["pnl_pips"])

    entry_ts = pd.to_datetime(trade["entry_time"], utc=True)
    exit_ts = pd.to_datetime(trade["exit_time"], utc=True)
    entry_px = float(trade["entry_price"])
    exit_px = float(trade["exit_price"])

    signal_eval_ts = entry_ts - pd.Timedelta(seconds=10)
    burst_m1_end = signal_eval_ts.ceil("min") - pd.Timedelta(minutes=1)
    burst_m1_start = burst_m1_end - pd.Timedelta(minutes=1)

    # Canvas
    width = 1100
    height = 520
    pad_l = 60
    pad_r = 20
    pad_t = 30
    pad_b = 40

    n = int(len(m1))
    if n < 2:
        out_html.parent.mkdir(parents=True, exist_ok=True)
        out_html.write_text("Not enough M1 bars to render.\n", encoding="utf-8")
        return

    prices_hi = float(m1["high"].max())
    prices_lo = float(m1["low"].min())
    if prices_hi <= prices_lo:
        prices_hi = prices_lo + 1e-9

    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    step = plot_w / max(1, n)
    body_w = max(1.0, step * 0.6)

    def x_at(i: int) -> float:
        return pad_l + (i + 0.5) * step

    def y_at(price: float) -> float:
        # Higher price => smaller y (SVG top-down)
        return pad_t + (prices_hi - price) / (prices_hi - prices_lo) * plot_h

    times = m1.index
    idx_entry = int(times.get_indexer([entry_ts], method="nearest")[0])
    idx_exit = int(times.get_indexer([exit_ts], method="nearest")[0])
    idx_signal = int(times.get_indexer([signal_eval_ts], method="nearest")[0])
    idx_burst_end = int(times.get_indexer([burst_m1_end], method="nearest")[0])
    idx_burst_start = int(times.get_indexer([burst_m1_start], method="nearest")[0])

    # Candles
    candle_svg = []
    for i, (op, hi, lo, cl) in enumerate(zip(m1["open"], m1["high"], m1["low"], m1["close"])):
        op = float(op)
        hi = float(hi)
        lo = float(lo)
        cl = float(cl)
        up = cl >= op
        color = "#2ca02c" if up else "#d62728"
        x = x_at(i)
        y_hi = y_at(hi)
        y_lo = y_at(lo)
        y_op = y_at(op)
        y_cl = y_at(cl)
        y_top = min(y_op, y_cl)
        y_bot = max(y_op, y_cl)
        candle_svg.append(f"<line x1='{x:.2f}' y1='{y_hi:.2f}' x2='{x:.2f}' y2='{y_lo:.2f}' stroke='{color}' stroke-width='1' />")
        candle_svg.append(
            f"<rect x='{(x - body_w / 2):.2f}' y='{y_top:.2f}' width='{body_w:.2f}' height='{max(1.0, y_bot - y_top):.2f}' fill='{color}' fill-opacity='0.6' stroke='{color}' />"
        )

    # Entry/Exit lines and markers (nearest M1 bar x)
    x_entry = x_at(max(0, min(n - 1, idx_entry)))
    x_exit = x_at(max(0, min(n - 1, idx_exit)))
    y_entry = y_at(entry_px)
    y_exit = y_at(exit_px)

    x_signal = x_at(max(0, min(n - 1, idx_signal)))
    x_burst_end = x_at(max(0, min(n - 1, idx_burst_end)))
    x_burst_start = x_at(max(0, min(n - 1, idx_burst_start)))

    overlays = [
        # burst window highlight (approx)
        f"<rect x='{x_burst_start:.2f}' y='{pad_t:.2f}' width='{max(1.0, x_burst_end - x_burst_start):.2f}' height='{plot_h:.2f}' fill='rgba(0,128,255,0.10)' />",
        f"<line x1='{x_burst_end:.2f}' y1='{pad_t:.2f}' x2='{x_burst_end:.2f}' y2='{(pad_t + plot_h):.2f}' stroke='rgba(0,128,255,0.9)' stroke-width='2' />",
        f"<line x1='{x_signal:.2f}' y1='{pad_t:.2f}' x2='{x_signal:.2f}' y2='{(pad_t + plot_h):.2f}' stroke='rgba(255,140,0,0.9)' stroke-width='2' stroke-dasharray='4,3' />",
        f"<line x1='{x_entry:.2f}' y1='{pad_t:.2f}' x2='{x_entry:.2f}' y2='{(pad_t + plot_h):.2f}' stroke='green' stroke-width='2' />",
        f"<line x1='{x_exit:.2f}' y1='{pad_t:.2f}' x2='{x_exit:.2f}' y2='{(pad_t + plot_h):.2f}' stroke='red' stroke-width='2' />",
        # horizontal price lines (so exit level is visible)
        f"<line x1='{pad_l:.2f}' y1='{y_entry:.2f}' x2='{(pad_l + plot_w):.2f}' y2='{y_entry:.2f}' stroke='green' stroke-width='1' stroke-opacity='0.5' />",
        f"<line x1='{pad_l:.2f}' y1='{y_exit:.2f}' x2='{(pad_l + plot_w):.2f}' y2='{y_exit:.2f}' stroke='red' stroke-width='1' stroke-opacity='0.5' />",
        f"<circle cx='{x_entry:.2f}' cy='{y_entry:.2f}' r='4' fill='green' />",
        f"<circle cx='{x_exit:.2f}' cy='{y_exit:.2f}' r='4' fill='red' />",
        f"<text x='{(x_entry + 6):.2f}' y='{(y_entry - 6):.2f}' font-size='11' font-family='sans-serif' fill='green'>entry {entry_px:.3f}</text>",
        f"<text x='{(x_exit + 6):.2f}' y='{(y_exit + 14):.2f}' font-size='11' font-family='sans-serif' fill='red'>exit {exit_px:.3f}</text>",
        f"<text x='{(x_signal + 6):.2f}' y='{(pad_t + 12):.2f}' font-size='11' font-family='sans-serif' fill='rgba(255,140,0,0.95)'>signal_eval_ts</text>",
        f"<text x='{(x_burst_end + 6):.2f}' y='{(pad_t + 26):.2f}' font-size='11' font-family='sans-serif' fill='rgba(0,128,255,0.95)'>burst_m1_end</text>",
    ]

    fb = trade.get("__failed_breakout__")
    if isinstance(fb, dict) and fb.get("fail_ts") is not None:
        fail_ts = fb.get("fail_ts")
        fail_px = fb.get("fail_px")
        break_ts = fb.get("break_ts")
        break_px = fb.get("break_px")
        prev_low = fb.get("prev_low")
        if prev_low is not None:
            y_prev = y_at(float(prev_low))
            overlays.append(
                f"<line x1='{pad_l:.2f}' y1='{y_prev:.2f}' x2='{(pad_l + plot_w):.2f}' y2='{y_prev:.2f}' stroke='#888' stroke-width='1' stroke-dasharray='4,3' />"
            )
        if break_ts is not None and break_px is not None:
            idx_break = int(times.get_indexer([break_ts], method='nearest')[0])
            x_break = x_at(max(0, min(n - 1, idx_break)))
            y_break = y_at(float(break_px))
            overlays.append(f"<polygon points='{x_break-4:.2f},{y_break-3:.2f} {x_break+4:.2f},{y_break-3:.2f} {x_break:.2f},{y_break+4:.2f}' fill='#F58518' />")
        if fail_ts is not None and fail_px is not None:
            idx_fail = int(times.get_indexer([fail_ts], method='nearest')[0])
            x_fail = x_at(max(0, min(n - 1, idx_fail)))
            y_fail = y_at(float(fail_px))
            overlays.append(f"<polygon points='{x_fail-4:.2f},{y_fail+3:.2f} {x_fail+4:.2f},{y_fail+3:.2f} {x_fail:.2f},{y_fail-4:.2f}' fill='#4C78A8' />")

    title = f"trade_id={t_id} side={side} pnl_pips={pnl}"
    x0 = times.min()
    x1 = times.max()
    caption = f"window={x0} .. {x1} (UTC)  entry={entry_ts} exit={exit_ts}"

    # Axes (minimal)
    axes = [
        f"<rect x='{pad_l}' y='{pad_t}' width='{plot_w}' height='{plot_h}' fill='white' stroke='#ddd' />",
        f"<text x='{pad_l}' y='{pad_t - 8}' font-size='12' font-family='sans-serif' fill='#111'>{title}</text>",
        f"<text x='{pad_l}' y='{height - 12}' font-size='11' font-family='sans-serif' fill='#444'>{caption}</text>",
        f"<text x='8' y='{pad_t + 12}' font-size='11' font-family='sans-serif' fill='#444'>high={prices_hi:.5f}</text>",
        f"<text x='8' y='{pad_t + plot_h - 4}' font-size='11' font-family='sans-serif' fill='#444'>low={prices_lo:.5f}</text>",
    ]

    svg = "\n".join(
        [
            f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>",
            *axes,
            *candle_svg,
            *overlays,
            "</svg>",
        ]
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>trade_{t_id}</title>
  <style>
    body {{ font-family: sans-serif; margin: 16px; }}
    .meta {{ margin-bottom: 8px; }}
  </style>
</head>
<body>
  <div class="meta">{title}</div>
  {svg}
</body>
</html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")


def _compute_failed_breakout_markers(*, m1, trade, lookback_bars: int):
    import pandas as pd

    entry_ts = pd.to_datetime(trade["entry_time"], utc=True)
    if m1 is None or len(m1) == 0:
        return None

    prev_low = m1["low"].shift(1).rolling(int(lookback_bars)).min()
    breakout = m1["close"] < prev_low
    fail = breakout.shift(1) & (m1["close"] > prev_low.shift(1))

    fail_idx = fail[fail].index
    fail_idx = fail_idx[fail_idx <= entry_ts]
    if len(fail_idx) == 0:
        return None
    fail_ts = fail_idx[-1]

    break_idx = breakout[breakout].index
    break_idx = break_idx[break_idx < fail_ts]
    break_ts = break_idx[-1] if len(break_idx) else None

    fail_px = float(m1.loc[fail_ts, "close"])
    break_px = float(m1.loc[break_ts, "close"]) if break_ts is not None else None
    prev_low_val = prev_low.loc[fail_ts]
    prev_low_val = float(prev_low_val) if pd.notna(prev_low_val) else None
    label = "10s" if str(trade.get("__viz_tf__", "")) == "10s" else "M1 proxy"

    return {
        "break_ts": break_ts,
        "break_px": break_px,
        "fail_ts": fail_ts,
        "fail_px": fail_px,
        "prev_low": prev_low_val,
        "label": label,
    }


def _render_trade_html(*, m1, trade, out_html: Path, annotate_failed_breakout: bool, lookback_bars: int) -> str:
    # Prefer Plotly, fallback to Matplotlib.
    if annotate_failed_breakout:
        fb = _compute_failed_breakout_markers(m1=m1, trade=trade, lookback_bars=lookback_bars)
        if fb is not None:
            trade = dict(trade)
            trade["__failed_breakout__"] = fb
    try:
        _render_trade_plotly(m1=m1, trade=trade, out_html=out_html)
        return "plotly"
    except Exception:  # noqa: BLE001
        try:
            _render_trade_matplotlib_html(m1=m1, trade=trade, out_html=out_html)
            return "matplotlib"
        except Exception as e:  # noqa: BLE001
            try:
                _render_trade_svg_html(m1=m1, trade=trade, out_html=out_html)
                return "svg"
            except Exception as e2:  # noqa: BLE001
                out_html.parent.mkdir(parents=True, exist_ok=True)
                out_html.write_text(f"Failed to render chart: {e} / {e2}\n", encoding="utf-8")
                return "error"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--data_root", default="FX/code/dukas_out_v2")
    p.add_argument("--window_minutes", type=int, default=30)
    p.add_argument("--tf", default="1min", choices=["1min", "10s"], help="Chart timeframe for OHLC.")
    p.add_argument(
        "--window_anchor",
        default="auto",
        choices=["auto", "entry", "exit", "span"],
        help=(
            "Which timestamp to anchor the window on. "
            "entry/exit=±window around that point. "
            "span=(entry-window .. exit+window). "
            "auto=entry unless exit is out of window, then exit."
        ),
    )
    p.add_argument("--top_n", type=int, default=30)
    p.add_argument(
        "--trade_id",
        type=int,
        action="append",
        help="Render only the specified trade_id (can be passed multiple times). Overrides --top_n/--sort_by.",
    )
    p.add_argument(
        "--sort_by",
        default="pnl_pips_asc",
        choices=["pnl_pips_asc", "pnl_pips_desc", "abs_pnl_pips_desc", "entry_time_asc", "entry_time_desc"],
    )
    p.add_argument("--annotate_failed_breakout", action="store_true", help="Annotate B failed-breakout proxy on charts (M1-based).")
    p.add_argument("--lookback_bars", type=int, default=None, help="Override lookback_bars for failed-breakout annotation.")
    args = p.parse_args()

    import pandas as pd

    results_dir = Path(args.results_dir)
    trades_csv = results_dir / "trades.csv"
    if not trades_csv.exists():
        print(f"[viz] trades.csv not found: {trades_csv}", file=sys.stderr)
        return 2

    trades = pd.read_csv(trades_csv)
    if trades.empty:
        print("[viz] trades.csv is empty; nothing to plot.", flush=True)
        return 0

    if "trade_id" not in trades.columns:
        trades = trades.copy()
        trades.insert(0, "trade_id", range(1, len(trades) + 1))

    required = ["trade_id", "side", "entry_time", "entry_price", "exit_time", "exit_price", "pnl_pips"]
    missing = [c for c in required if c not in trades.columns]
    if missing:
        print(f"[viz] trades.csv missing columns: {missing}", file=sys.stderr)
        return 2

    trades = trades.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce")
    trades = trades.dropna(subset=["entry_time", "exit_time"])

    if args.trade_id:
        want = set(int(x) for x in args.trade_id)
        trades = trades[trades["trade_id"].astype(int).isin(want)].copy()
        trades = trades.sort_values("trade_id")
    else:
        trades = _sort_trades(trades, args.sort_by).head(int(args.top_n))
    if trades.empty:
        print("[viz] no valid trades after filtering.", flush=True)
        return 0

    window = pd.Timedelta(minutes=int(args.window_minutes))
    print(
        f"[viz] symbol={args.symbol} pip_size={_pip_size(args.symbol)} window_minutes={args.window_minutes} top_n={args.top_n} sort_by={args.sort_by}",
        flush=True,
    )
    data_root = Path(args.data_root)
    pc = _detect_price_columns_from_data_root(data_root, args.symbol)
    run_cfg = results_dir / "config.json"
    lookback_bars = int(args.lookback_bars) if args.lookback_bars is not None else None
    if lookback_bars is None and run_cfg.exists():
        try:
            cfg = json.loads(run_cfg.read_text(encoding="utf-8"))
            lookback_bars = int(cfg.get("lookback_bars", 6))
        except Exception:  # noqa: BLE001
            lookback_bars = 6
    if lookback_bars is None:
        lookback_bars = 6

    charts_dir = results_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    engines: dict[str, int] = {}
    m1_cache: dict[tuple[int, int], "object"] = {}
    for _, t in trades.iterrows():
        t_id = int(t["trade_id"])
        entry_ts = pd.to_datetime(t["entry_time"], utc=True)
        exit_ts = pd.to_datetime(t["exit_time"], utc=True)

        def _window_for(anchor: str):
            if anchor == "span":
                return entry_ts - window, exit_ts + window, "span"
            if anchor == "exit":
                return exit_ts - window, exit_ts + window, "exit"
            return entry_ts - window, entry_ts + window, "entry"

        if args.window_anchor == "auto":
            s, e, anchored = _window_for("entry")
            if exit_ts < s or exit_ts > e:
                s, e, anchored = _window_for("exit")
        else:
            s, e, anchored = _window_for(args.window_anchor)

        # Cache by month to avoid re-loading the same parquet multiple times.
        months = _iter_month_starts_utc(s.to_pydatetime(), e.to_pydatetime())
        parts = []
        for y, m in months:
            key = (y, m)
            if key not in m1_cache:
                ms = datetime(y, m, 1, tzinfo=timezone.utc)
                me = datetime(y + 1, 1, 1, tzinfo=timezone.utc) if m == 12 else datetime(y, m + 1, 1, tzinfo=timezone.utc)
                m1_cache[key] = load_m1_ohlc_from_10s_parquet_range(
                    data_root=data_root,
                    symbol=args.symbol,
                    start_utc=ms,
                    end_utc=me,
                    price_columns=pc,
                    freq=args.tf,
                )
            parts.append(m1_cache[key])

        m1 = pd.concat(parts).sort_index().drop_duplicates()
        sub = m1[(m1.index >= s) & (m1.index <= e)]

        out_html = charts_dir / f"trade_{t_id}.html"
        trade_dict = t.to_dict()
        trade_dict["__viz_window_anchor__"] = anchored
        trade_dict["__viz_window_start__"] = str(s)
        trade_dict["__viz_window_end__"] = str(e)
        trade_dict["__viz_tf__"] = args.tf
        engine = _render_trade_html(
            m1=sub,
            trade=trade_dict,
            out_html=out_html,
            annotate_failed_breakout=bool(args.annotate_failed_breakout),
            lookback_bars=lookback_bars,
        )
        engines[engine] = engines.get(engine, 0) + 1

        index_rows.append(
            {
                "trade_id": t_id,
                "side": str(t["side"]),
                "pnl_pips": float(t["pnl_pips"]),
                "entry_time": str(t["entry_time"]),
                "exit_time": str(t["exit_time"]),
                "file": out_html.name,
                "engine": engine,
                "window_anchor": anchored,
            }
        )

    # index.html
    index_rows_sorted = sorted(index_rows, key=lambda r: r["trade_id"])
    rows_html = "\n".join(
        [
            f"<tr><td>{r['trade_id']}</td><td>{r['side']}</td><td>{r['pnl_pips']:.2f}</td>"
            f"<td>{r['entry_time']}</td><td>{r['exit_time']}</td>"
            f"<td>{r['engine']}</td><td>{r['window_anchor']}</td><td><a href=\"{r['file']}\">{r['file']}</a></td></tr>"
            for r in index_rows_sorted
        ]
    )
    index_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Trade Charts</title>
  <style>
    body {{ font-family: sans-serif; margin: 16px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; }}
    th {{ background: #f6f6f6; text-align: left; }}
    .meta {{ margin-bottom: 12px; color: #333; }}
  </style>
</head>
<body>
  <div class="meta">
    results_dir={results_dir}<br/>
    symbol={args.symbol} window_minutes={args.window_minutes} top_n={args.top_n} sort_by={args.sort_by}<br/>
    engines={engines}
  </div>
  <table>
    <thead>
      <tr>
        <th>trade_id</th><th>side</th><th>pnl_pips</th><th>entry_time</th><th>exit_time</th><th>engine</th><th>window_anchor</th><th>file</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</body>
</html>
"""
    (charts_dir / "index.html").write_text(index_html, encoding="utf-8")

    print(f"[viz] wrote charts: {charts_dir}", flush=True)
    print(f"[viz] wrote index:  {charts_dir / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
