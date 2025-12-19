#!/usr/bin/env python3
"""
Overview chart (TradingView-like) for a date range with trade markers.

Design goals:
  - Works without Plotly (SVG+JS pan/zoom fallback).
  - Supports a 1-week window for M1/M5 without becoming unreadable.
  - Visualization-only; does not alter backtest logic or aggregates.

Usage (recommended with uv):
  uv run python FX/code/viz/viz_overview_tf.py \
    --results_dir FX/results/family_D_momentum/D004 \
    --symbol USDJPY \
    --tf 1min \
    --from 2025-06-16 \
    --to 2025-06-22

If uv fails in this environment, fallback:
  ./.venv/bin/python FX/code/viz/viz_overview_tf.py ...

Inputs:
  {results_dir}/trades.csv
  Dukascopy 10s bid parquet under: {data_root}/{symbol}/bars10s_pq/YYYY/MM/*.parquet

Outputs:
  {results_dir}/charts/overview_{tf}_{from}_{to}.html
"""

from __future__ import annotations

import argparse
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


def _parse_date_utc(s: str) -> datetime:
    dt = datetime.strptime(s, "%Y-%m-%d")
    return dt.replace(tzinfo=timezone.utc)


def _iter_month_starts_utc(start: datetime, end: datetime) -> list[tuple[int, int]]:
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
    files: list[Path] = []
    for y, m in _iter_month_starts_utc(start_utc, end_utc):
        mm = f"{m:02d}"
        p = base / str(y) / mm
        if not p.exists():
            continue
        files.extend(sorted(p.glob("*.parquet")))
    return files


def _detect_price_columns(cols: list[str]) -> PriceColumns:
    if all(c in cols for c in ("ts", "open", "high", "low", "close")):
        return PriceColumns(ts="ts", open="open", high="high", low="low", close="close")
    for prefix in ("bid_", "mid_", "price_"):
        want = [f"{prefix}{c}" for c in ("open", "high", "low", "close")]
        if "ts" in cols and all(c in cols for c in want):
            return PriceColumns(ts="ts", open=want[0], high=want[1], low=want[2], close=want[3])
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
    for fp in sorted(base.glob("*/*/*.parquet")):
        sample = _read_parquet_robust(fp)
        return _detect_price_columns(list(sample.columns))
    raise FileNotFoundError(f"No parquet files found under: {base}")


def load_ohlc_from_10s_parquet(
    *,
    data_root: Path,
    symbol: str,
    start_utc: datetime,
    end_utc: datetime,
    freq: str,
    price_columns: PriceColumns | None = None,
):
    import pandas as pd

    files = _resolve_10s_parquet_files(data_root, symbol, start_utc, end_utc)
    if not files:
        raise FileNotFoundError(f"No parquet files found for {symbol} under {data_root}")

    pc = price_columns or _detect_price_columns_from_data_root(data_root, symbol)
    need_cols = sorted(set([pc.ts, pc.open, pc.high, pc.low, pc.close]))

    parts = []
    for fp in files:
        df = _read_parquet_robust(fp)
        df = df[[c for c in need_cols if c in df.columns]].copy()
        df[pc.ts] = pd.to_datetime(df[pc.ts], utc=True, errors="coerce")
        df = df.dropna(subset=[pc.ts])
        df = df[(df[pc.ts] >= start_utc) & (df[pc.ts] <= end_utc)]
        if not df.empty:
            parts.append(df)
    if not parts:
        raise ValueError("No bars in selected time range (after filtering).")

    df10 = pd.concat(parts, ignore_index=True).sort_values(pc.ts).drop_duplicates(pc.ts).set_index(pc.ts)
    o = df10[pc.open].astype(float).resample(freq).first()
    h = df10[pc.high].astype(float).resample(freq).max()
    l = df10[pc.low].astype(float).resample(freq).min()
    c = df10[pc.close].astype(float).resample(freq).last()
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna()


def _try_plotly():
    try:
        import plotly.graph_objects as go  # type: ignore

        return go
    except Exception:  # noqa: BLE001
        return None


def _render_svg_overview(*, out_html: Path, tf: str, ohlc, trades_df, start_utc, end_utc) -> None:
    """
    Dependency-free fallback: inline SVG close line with entry/exit markers + JS pan/zoom.
    """
    import pandas as pd

    if len(ohlc) == 0:
        out_html.parent.mkdir(parents=True, exist_ok=True)
        out_html.write_text("No OHLC data.\n", encoding="utf-8")
        return

    # Fixed downsample cap to keep file size reasonable (not tuned; visualization-only).
    MAX_POINTS = 12000
    m = ohlc.copy()
    step = max(1, int(len(m) // MAX_POINTS) + 1)
    m = m.iloc[::step]

    x = m.index
    y = m["close"].astype(float)
    y_hi = float(y.max())
    y_lo = float(y.min())
    if y_hi <= y_lo:
        y_hi = y_lo + 1e-9

    width = 1400
    height = 700
    pad_l = 70
    pad_r = 20
    pad_t = 30
    pad_b = 70
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def x_at(i: int) -> float:
        return pad_l + (i / max(1, len(x) - 1)) * plot_w

    def y_at(v: float) -> float:
        return pad_t + (y_hi - v) / (y_hi - y_lo) * plot_h

    pts = " ".join([f"{x_at(i):.2f},{y_at(float(v)):.2f}" for i, v in enumerate(y)])

    t = trades_df.copy()
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True, errors="coerce")
    t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True, errors="coerce")
    t = t.dropna(subset=["entry_time", "exit_time"])
    # IMPORTANT: filter trades to the visible window; otherwise markers outside
    # expand x-axis and make the plot look like it spans the full year.
    in_window = ((t["entry_time"] >= start_utc) & (t["entry_time"] < end_utc)) | ((t["exit_time"] >= start_utc) & (t["exit_time"] < end_utc))
    t = t[in_window]
    t = t.sort_values("entry_time")

    times = x

    def nearest_idx(ts: pd.Timestamp) -> int:
        idx = times.get_indexer([ts], method="nearest")[0]
        if idx < 0:
            return 0
        return int(idx)

    markers = []
    for _, r in t.iterrows():
        tid = int(r.get("trade_id", 0))
        side = str(r.get("side", ""))
        pnl = r.get("pnl_pips", "")
        reason = str(r.get("reason_exit", r.get("exit_reason", "")) or "")
        title = f"trade_id={tid} side={side} pnl={pnl} reason={reason} entry={r['entry_time']} exit={r['exit_time']}"

        ei = nearest_idx(r["entry_time"])
        xe = x_at(ei)
        ye = y_at(float(r["entry_price"]))
        markers.append(f"<circle cx='{xe:.2f}' cy='{ye:.2f}' r='5' fill='green' stroke='white' stroke-width='1'><title>{title}</title></circle>")
        markers.append(f"<text x='{(xe + 7):.2f}' y='{(ye - 7):.2f}' font-size='11' font-family='sans-serif' fill='green'>{tid}</text>")

        xi = nearest_idx(r["exit_time"])
        xx = x_at(xi)
        yx = y_at(float(r["exit_price"]))
        markers.append(f"<rect x='{(xx-4):.2f}' y='{(yx-4):.2f}' width='8' height='8' fill='red' stroke='white' stroke-width='1'><title>{title}</title></rect>")
        markers.append(f"<text x='{(xx + 7):.2f}' y='{(yx + 14):.2f}' font-size='11' font-family='sans-serif' fill='red'>{tid}</text>")

    x0 = str(x.min())
    x1 = str(x.max())
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Overview {tf}</title>
  <style>
    body {{ font-family: sans-serif; margin: 16px; }}
    .meta {{ margin-bottom: 10px; color: #333; }}
    .hint {{ color: #666; font-size: 12px; }}
    #wrap {{ border: 1px solid #ddd; overflow: hidden; width: {width}px; }}
    svg {{ display: block; }}
  </style>
</head>
<body>
  <div class="meta">
    Overview (SVG fallback): close line ({tf}, downsample step={step}) with entry/exit markers<br/>
    Green circle=entry, Red square=exit. Hover markers for trade_id. Range: {x0} .. {x1} (UTC)
    <div class="hint">Wheel=zoom, Drag=pan (client-side). Install plotly for candlesticks.</div>
  </div>
  <div id="wrap">
    <svg id="svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
      <g id="viewport">
        <rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="white" stroke="#ddd" />
        <polyline points="{pts}" fill="none" stroke="#1f77b4" stroke-width="1" />
        {''.join(markers)}
        <text x="10" y="{pad_t+12}" font-size="12" font-family="sans-serif" fill="#444">high={y_hi:.5f}</text>
        <text x="10" y="{pad_t+plot_h-4}" font-size="12" font-family="sans-serif" fill="#444">low={y_lo:.5f}</text>
      </g>
    </svg>
  </div>
  <script>
    (function() {{
      const svg = document.getElementById('svg');
      const vp = document.getElementById('viewport');
      let scale = 1.0;
      let tx = 0.0, ty = 0.0;
      let dragging = false;
      let lastX = 0, lastY = 0;

      function apply() {{
        vp.setAttribute('transform', `translate(${tx},${ty}) scale(${scale})`);
      }}

      svg.addEventListener('wheel', (e) => {{
        e.preventDefault();
        const delta = Math.sign(e.deltaY);
        const factor = (delta < 0) ? 1.12 : 0.89;
        scale = Math.max(0.3, Math.min(12.0, scale * factor));
        apply();
      }}, {{ passive: false }});

      svg.addEventListener('mousedown', (e) => {{
        dragging = true;
        lastX = e.clientX;
        lastY = e.clientY;
      }});
      window.addEventListener('mouseup', () => {{ dragging = false; }});
      window.addEventListener('mousemove', (e) => {{
        if (!dragging) return;
        tx += (e.clientX - lastX);
        ty += (e.clientY - lastY);
        lastX = e.clientX;
        lastY = e.clientY;
        apply();
      }});

      apply();
    }})();
  </script>
</body>
</html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")


def render_overview(*, out_html: Path, tf: str, ohlc, trades_df, start_utc, end_utc) -> str:
    import pandas as pd

    go = _try_plotly()
    if go is None:
        _render_svg_overview(out_html=out_html, tf=tf, ohlc=ohlc, trades_df=trades_df, start_utc=start_utc, end_utc=end_utc)
        return "svg"

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=ohlc.index,
                open=ohlc["open"],
                high=ohlc["high"],
                low=ohlc["low"],
                close=ohlc["close"],
                name=tf,
            )
        ]
    )

    t = trades_df.copy()
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True, errors="coerce")
    t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True, errors="coerce")
    t = t.dropna(subset=["entry_time", "exit_time"])
    # IMPORTANT: filter trades to the visible window; otherwise markers outside
    # expand x-axis and make the plot look like it spans the full year.
    in_window = ((t["entry_time"] >= start_utc) & (t["entry_time"] < end_utc)) | ((t["exit_time"] >= start_utc) & (t["exit_time"] < end_utc))
    t = t[in_window]

    hover = (
        "trade_id=%{customdata[0]}<br>"
        "side=%{customdata[1]}<br>"
        "pnl_pips=%{customdata[2]}<br>"
        "reason=%{customdata[3]}"
    )
    custom = t[["trade_id", "side", "pnl_pips"]].copy()
    custom["reason"] = t.get("reason_exit", t.get("exit_reason", "")).fillna("")

    show_labels = bool(getattr(trades_df, "_viz_show_labels", True))
    marker_size = int(getattr(trades_df, "_viz_marker_size", 12))

    entry_kwargs = {}
    exit_kwargs = {}
    if show_labels:
        entry_kwargs = {
            "mode": "markers+text",
            "text": t["trade_id"].astype(str),
            "textposition": "top center",
            "textfont": dict(color="green", size=10),
        }
        exit_kwargs = {
            "mode": "markers+text",
            "text": t["trade_id"].astype(str),
            "textposition": "bottom center",
            "textfont": dict(color="red", size=10),
        }
    else:
        entry_kwargs = {"mode": "markers"}
        exit_kwargs = {"mode": "markers"}

    fig.add_trace(
        go.Scatter(
            x=t["entry_time"],
            y=t["entry_price"],
            marker=dict(color="green", size=marker_size, symbol="triangle-up", line=dict(width=1, color="white")),
            name="entry",
            customdata=custom.to_numpy(),
            hovertemplate=hover,
            **entry_kwargs,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t["exit_time"],
            y=t["exit_price"],
            marker=dict(color="red", size=marker_size, symbol="x", line=dict(width=1, color="white")),
            name="exit",
            customdata=custom.to_numpy(),
            hovertemplate=hover,
            **exit_kwargs,
        )
    )

    fig.update_layout(
        title=f"Overview {tf} + trade markers",
        xaxis_title="time (UTC)",
        yaxis_title="price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=720,
        dragmode="pan",
        xaxis=dict(fixedrange=False),
        yaxis=dict(fixedrange=False),
        xaxis_range=[start_utc, end_utc],
    )
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
    return "plotly"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--data_root", default="FX/code/dukas_out_v2")
    p.add_argument("--tf", default="5min", choices=["1min", "5min", "15min"])
    p.add_argument("--from", dest="from_date", required=True, help="YYYY-MM-DD (UTC)")
    p.add_argument("--to", dest="to_date", required=True, help="YYYY-MM-DD (UTC)")
    p.add_argument("--max_days", type=int, default=8, help="Safety limit for date span (inclusive); set --force to override")
    p.add_argument("--force", action="store_true", help="Allow longer ranges than --max_days")
    p.add_argument("--marker_size", type=int, default=12, help="Marker size for entry/exit points (visual only)")
    p.add_argument("--no_labels", action="store_true", help="Hide trade_id text labels (visual only)")
    args = p.parse_args()

    import pandas as pd

    results_dir = Path(args.results_dir)
    trades_csv = results_dir / "trades.csv"
    if not trades_csv.exists():
        print(f"[overview] trades.csv not found: {trades_csv}", file=sys.stderr)
        return 2

    trades = pd.read_csv(trades_csv)
    if trades.empty:
        print("[overview] trades.csv empty", flush=True)
        return 0

    # Pass visualization preferences without expanding the function signature too much.
    trades._viz_show_labels = (not args.no_labels)  # type: ignore[attr-defined]
    trades._viz_marker_size = int(args.marker_size)  # type: ignore[attr-defined]

    start = _parse_date_utc(args.from_date)
    end_inclusive = _parse_date_utc(args.to_date)
    span_days = int((end_inclusive.date() - start.date()).days) + 1
    if (not args.force) and (span_days > int(args.max_days)):
        print(f"[overview] range too long: {span_days} days > max_days={args.max_days}. Use --force to override.", file=sys.stderr)
        return 2

    # inclusive end date
    end = end_inclusive + pd.Timedelta(days=1)

    freq = args.tf
    print(f"[overview] symbol={args.symbol} pip_size={_pip_size(args.symbol)} tf={freq} range={start.date()}..{end_inclusive.date()} ({span_days}d)", flush=True)

    data_root = Path(args.data_root)
    pc = _detect_price_columns_from_data_root(data_root, args.symbol)
    ohlc = load_ohlc_from_10s_parquet(data_root=data_root, symbol=args.symbol, start_utc=start, end_utc=end, freq=freq, price_columns=pc)

    charts_dir = results_dir / "charts"
    out_html = charts_dir / f"overview_{args.tf}_{args.from_date}_{args.to_date}.html"
    engine = render_overview(out_html=out_html, tf=args.tf, ohlc=ohlc, trades_df=trades, start_utc=start, end_utc=end)
    print(f"[overview] wrote: {out_html} engine={engine}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
