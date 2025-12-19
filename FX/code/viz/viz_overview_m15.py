#!/usr/bin/env python3
"""
Overview chart (TradingView-like) for a whole period with trade markers.

Purpose:
  - Provide a bird's-eye view (M15) and clickable trade_id list.
  - Visualization-only; does not alter backtest logic or aggregates.

Usage (recommended with uv):
  uv run python FX/code/viz/viz_overview_m15.py \
    --results_dir FX/results/family_D_momentum/D004 \
    --symbol USDJPY \
    --from 2025-01-01 \
    --to 2025-12-31

If uv fails in this environment, fallback:
  ./.venv/bin/python FX/code/viz/viz_overview_m15.py ...

Inputs:
  {results_dir}/trades.csv
  Dukascopy 10s bid parquet under: {data_root}/{symbol}/bars10s_pq/YYYY/MM/*.parquet

Outputs:
  {results_dir}/charts/overview_m15.html
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
    # Accept YYYY-MM-DD
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


def load_m15_ohlc_from_10s_parquet(
    *,
    data_root: Path,
    symbol: str,
    start_utc: datetime,
    end_utc: datetime,
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
    o = df10[pc.open].astype(float).resample("15min").first()
    h = df10[pc.high].astype(float).resample("15min").max()
    l = df10[pc.low].astype(float).resample("15min").min()
    c = df10[pc.close].astype(float).resample("15min").last()
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna()


def _try_plotly():
    try:
        import plotly.graph_objects as go  # type: ignore

        return go
    except Exception:  # noqa: BLE001
        return None


def render_overview(*, out_html: Path, m15, trades_df) -> str:
    import pandas as pd

    go = _try_plotly()
    if go is None:
        # Dependency-free fallback: inline SVG line chart (close) with entry/exit markers.
        # Fixed downsample to keep file size reasonable (not tuned; visualization-only).
        MAX_POINTS = 4000

        if len(m15) == 0:
            out_html.parent.mkdir(parents=True, exist_ok=True)
            out_html.write_text("No M15 data.\n", encoding="utf-8")
            return "svg_empty"

        m = m15.copy()
        step = max(1, int(len(m) // MAX_POINTS) + 1)
        m = m.iloc[::step]

        x = m.index
        y = m["close"].astype(float)
        y_hi = float(y.max())
        y_lo = float(y.min())
        if y_hi <= y_lo:
            y_hi = y_lo + 1e-9

        width = 1400
        height = 650
        pad_l = 70
        pad_r = 20
        pad_t = 30
        pad_b = 60
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
        t = t.sort_values("entry_time")

        # map timestamps to nearest x index in downsampled series (simple nearest by time)
        times = x
        def nearest_idx(ts: pd.Timestamp) -> int:
            try:
                return int(times.get_indexer([ts], method="nearest")[0])
            except Exception:
                return int(max(0, min(len(times) - 1, 0)))

        markers = []
        for _, r in t.iterrows():
            tid = int(r.get("trade_id", 0))
            side = str(r.get("side", ""))
            pnl = r.get("pnl_pips", "")
            reason = str(r.get("reason_exit", r.get("exit_reason", "")) or "")

            ei = nearest_idx(r["entry_time"])
            xi = x_at(ei)
            ye = y_at(float(r["entry_price"]))
            title = f"trade_id={tid} side={side} pnl={pnl} reason={reason} entry={r['entry_time']} exit={r['exit_time']}"
            markers.append(f"<circle cx='{xi:.2f}' cy='{ye:.2f}' r='3.5' fill='green'><title>{title}</title></circle>")

            xo_i = nearest_idx(r["exit_time"])
            xo = x_at(xo_i)
            yx = y_at(float(r["exit_price"]))
            markers.append(f"<rect x='{(xo-3):.2f}' y='{(yx-3):.2f}' width='6' height='6' fill='red'><title>{title}</title></rect>")

        x0 = str(x.min())
        x1 = str(x.max())
        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Overview M15</title>
  <style>
    body {{ font-family: sans-serif; margin: 16px; }}
    .meta {{ margin-bottom: 10px; color: #333; }}
    .hint {{ color: #666; font-size: 12px; }}
  </style>
</head>
<body>
  <div class="meta">
    Overview (fallback SVG): close line (M15, downsampled step={step}) with entry/exit markers<br/>
    Green circle=entry, Red square=exit. Hover markers for trade_id. Range: {x0} .. {x1} (UTC)
    <div class="hint">Install plotly to get candlestick + pan/zoom UI.</div>
  </div>
  <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
    <rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="white" stroke="#ddd" />
    <polyline points="{pts}" fill="none" stroke="#1f77b4" stroke-width="1" />
    {''.join(markers)}
    <text x="10" y="{pad_t+12}" font-size="12" font-family="sans-serif" fill="#444">high={y_hi:.5f}</text>
    <text x="10" y="{pad_t+plot_h-4}" font-size="12" font-family="sans-serif" fill="#444">low={y_lo:.5f}</text>
  </svg>
</body>
</html>
"""
        out_html.parent.mkdir(parents=True, exist_ok=True)
        out_html.write_text(html, encoding="utf-8")
        return "svg"

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=m15.index,
                open=m15["open"],
                high=m15["high"],
                low=m15["low"],
                close=m15["close"],
                name="M15",
            )
        ]
    )

    # Markers: entry and exit
    t = trades_df.copy()
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True, errors="coerce")
    t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True, errors="coerce")
    t = t.dropna(subset=["entry_time", "exit_time"])

    hover = (
        "trade_id=%{customdata[0]}<br>"
        "side=%{customdata[1]}<br>"
        "pnl_pips=%{customdata[2]}<br>"
        "reason=%{customdata[3]}"
    )
    custom = t[["trade_id", "side", "pnl_pips"]].copy()
    custom["reason"] = t.get("reason_exit", t.get("exit_reason", "")).fillna("")

    fig.add_trace(
        go.Scatter(
            x=t["entry_time"],
            y=t["entry_price"],
            mode="markers",
            marker=dict(color="green", size=6, symbol="triangle-up"),
            name="entry",
            customdata=custom.to_numpy(),
            hovertemplate=hover,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t["exit_time"],
            y=t["exit_price"],
            mode="markers",
            marker=dict(color="red", size=6, symbol="x"),
            name="exit",
            customdata=custom.to_numpy(),
            hovertemplate=hover,
        )
    )

    fig.update_layout(
        title="Overview M15 + trade markers",
        xaxis_title="time (UTC)",
        yaxis_title="price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=700,
        dragmode="pan",
        xaxis=dict(fixedrange=False),
        yaxis=dict(fixedrange=False),
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
    p.add_argument("--from", dest="from_date", required=True, help="YYYY-MM-DD (UTC)")
    p.add_argument("--to", dest="to_date", required=True, help="YYYY-MM-DD (UTC)")
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

    start = _parse_date_utc(args.from_date)
    end = _parse_date_utc(args.to_date) + pd.Timedelta(days=1)  # inclusive end date
    print(f"[overview] symbol={args.symbol} pip_size={_pip_size(args.symbol)} range={start.date()}..{(end - pd.Timedelta(days=1)).date()}", flush=True)

    data_root = Path(args.data_root)
    pc = _detect_price_columns_from_data_root(data_root, args.symbol)
    m15 = load_m15_ohlc_from_10s_parquet(data_root=data_root, symbol=args.symbol, start_utc=start, end_utc=end, price_columns=pc)

    charts_dir = results_dir / "charts"
    out_html = charts_dir / "overview_m15.html"
    engine = render_overview(out_html=out_html, m15=m15, trades_df=trades)
    print(f"[overview] wrote: {out_html} engine={engine}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
