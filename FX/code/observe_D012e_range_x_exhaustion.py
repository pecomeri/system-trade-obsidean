#!/usr/bin/env python3
"""
Observe D012e: pre_range_length x exhaustion interaction on D009 trades.

Definition summary:
  - pre_range_length: close-only range persistence count (lookback N=20).
  - pre_range_group: fixed by D012a thresholds (short/mid/long).
  - exhaustion_ratio: primary=break_margin_over_mean_prev_body, fallback=break_margin_ratio.
  - exhaustion_group: verify p80 (per side) -> weak/strong (no filtering).
  - early_loss: holding_time_min in [0, 3].

Outputs (overwritten):
  - results/family_D_momentum/D012e/summary_verify.csv
  - results/family_D_momentum/D012e/summary_forward.csv
  - results/family_D_momentum/D012e/thresholds.json
  - results/family_D_momentum/D012e/plots/*.png
  - results/family_D_momentum/D012e/README.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


# Ensure `import backtest_core` works when executed from repo root.
_FX_CODE_DIR = Path(__file__).resolve().parent
if str(_FX_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_FX_CODE_DIR))


@dataclass(frozen=True)
class PeriodInputs:
    period: str
    trades_csv: Path
    core_config_json: Path


RANGE_ORDER = ["short", "mid", "long"]
EXHAUSTION_ORDER = ["weak", "strong"]
EARLY_LOSS_MAX_MIN = 3.0

# D012a fixed thresholds
PRE_RANGE_LOOKBACK = 20
PRE_RANGE_Q33 = 0.0
PRE_RANGE_Q66 = 3.0


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
    if not out:
        raise FileNotFoundError(f"No in_sample_2024/forward_2025 trades under: {d009_dir}")
    return out


def _read_core_cfg(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_m1_ohlc(*, core_cfg: dict, data_root: Path | None, symbol_override: str | None):
    import pandas as pd
    import backtest_core as bc

    root = Path(data_root) if data_root is not None else Path(core_cfg["root"])
    symbol = str(symbol_override or core_cfg["symbol"]).upper()

    cfg = bc.Config(
        root=root,
        symbol=symbol,
        from_month=str(core_cfg["from_month"]),
        to_month=str(core_cfg["to_month"]),
        run_tag="d012e_m1",
        only_session=None,
        spread_pips=float(core_cfg.get("spread_pips", 1.0)),
        pip_size=float(core_cfg.get("pip_size", 0.01)),
    )
    df_bid = bc.load_parquet_10s_bid(cfg)
    df10 = bc.add_synthetic_bidask(df_bid, cfg)

    o = df10["open_bid"].resample("1min", label="right", closed="right").first()
    h = df10["high_bid"].resample("1min", label="right", closed="right").max()
    l = df10["low_bid"].resample("1min", label="right", closed="right").min()
    c = df10["close_bid"].resample("1min", label="right", closed="right").last()
    m1 = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna()
    return m1


def _compute_pre_range_length(close, *, lookback: int):
    """
    Close-based range length (same as D012a):
      in_range[t] = (close[t] <= max(close[t-1..t-lookback]))
                    and (close[t] >= min(close[t-1..t-lookback]))
    pre_range_length[t] = consecutive count of in_range ending at t.
    """
    import numpy as np

    prev_high = close.shift(1).rolling(int(lookback)).max()
    prev_low = close.shift(1).rolling(int(lookback)).min()
    in_range = ((close <= prev_high) & (close >= prev_low)).fillna(False).to_numpy(dtype=bool)

    counts = np.zeros(len(in_range), dtype=int)
    for i, flag in enumerate(in_range):
        if flag:
            counts[i] = counts[i - 1] + 1 if i > 0 else 1
        else:
            counts[i] = 0
    return close.index, counts


def _map_pre_range_length(entry_times, *, m1_index, pre_range_counts):
    import numpy as np

    pos = m1_index.searchsorted(entry_times.to_numpy(), side="left") - 1
    out = np.full(len(pos), np.nan, dtype=float)
    valid = (pos >= 0) & (pos < len(pre_range_counts))
    if valid.any():
        out[valid] = pre_range_counts[pos[valid]]
    return out


def _classify_pre_range(values):
    import numpy as np

    out = np.full(len(values), None, dtype=object)
    mask = ~np.isnan(values)
    if mask.any():
        v = values[mask]
        out[mask] = np.where(
            v <= PRE_RANGE_Q33,
            "short",
            np.where(v <= PRE_RANGE_Q66, "mid", "long"),
        )
    return out


def _compute_exhaustion_table(*, m1, lookback_bars: int):
    import pandas as pd

    body = (m1["close"] - m1["open"]).abs().astype(float)
    mean_prev_body = body.shift(1).rolling(int(lookback_bars)).mean().astype(float)
    prev_high = m1["high"].shift(1).astype(float)
    prev_low = m1["low"].shift(1).astype(float)

    out = pd.DataFrame(
        {
            "open": m1["open"].astype(float),
            "high": m1["high"].astype(float),
            "low": m1["low"].astype(float),
            "close": m1["close"].astype(float),
            "body": body,
            "mean_prev_body": mean_prev_body,
            "prev_high": prev_high,
            "prev_low": prev_low,
        },
        index=m1.index,
    )
    return out


def _attach_exhaustion(trades, *, trig):
    import pandas as pd
    import numpy as np

    trades = trades.copy()
    trades["signal_eval_ts"] = trades["entry_time"] - pd.Timedelta(seconds=10)
    trades["burst_m1_end"] = trades["signal_eval_ts"].dt.floor("min") - pd.Timedelta(minutes=1)

    trig_reset = trig.reset_index()
    trig_reset = trig_reset.rename(columns={str(trig_reset.columns[0]): "burst_m1_end"})
    merged = trades.merge(trig_reset, on="burst_m1_end", how="left")

    side = merged["side"].astype(str)
    close = merged["close"].astype(float)
    open_ = merged["open"].astype(float)
    prev_high = merged["prev_high"].astype(float)
    prev_low = merged["prev_low"].astype(float)
    body = (close - open_).abs().astype(float)
    mean_prev_body = merged["mean_prev_body"].astype(float)

    break_margin = np.where(side == "buy", close - prev_high, prev_low - close).astype(float)
    merged["break_margin_ratio"] = np.where(body > 0, break_margin / body, np.nan)
    merged["break_margin_over_mean_prev_body"] = np.where(mean_prev_body > 0, break_margin / mean_prev_body, np.nan)

    ratio = merged["break_margin_over_mean_prev_body"].astype(float)
    fallback = merged["break_margin_ratio"].astype(float)
    merged["exhaustion_ratio"] = ratio.where(ratio.notna(), fallback)
    return merged


def _compute_p80_by_side(verify_df):
    import numpy as np

    p80 = {}
    for side in ("buy", "sell"):
        s = verify_df.loc[verify_df["side"] == side, "exhaustion_ratio"].dropna().astype(float)
        if len(s) == 0:
            raise ValueError(f"No exhaustion_ratio values for verify side={side}")
        p80[side] = float(np.nanquantile(s, 0.80))
    all_vals = verify_df["exhaustion_ratio"].dropna().astype(float)
    p80["all"] = float(np.nanquantile(all_vals, 0.80)) if len(all_vals) else float("nan")
    return p80


def _classify_exhaustion(values, *, side_values, p80_by_side: dict):
    import numpy as np

    out = np.full(len(values), None, dtype=object)
    for side in ("buy", "sell"):
        mask = (side_values == side) & ~np.isnan(values)
        if not mask.any():
            continue
        threshold = p80_by_side[side]
        out[mask] = np.where(values[mask] < threshold, "weak", "strong")
    return out


def _summarize(df, *, period_label: str):
    import numpy as np
    import pandas as pd

    columns = [
        "period",
        "side",
        "range_group",
        "exhaustion_group",
        "trades",
        "early_loss_trades",
        "early_loss_rate",
        "holding_mean_min",
        "holding_median_min",
        "holding_p25_min",
        "holding_p75_min",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    base = df.copy()
    base["side"] = base["side"].astype(str).str.lower()
    all_side = base.copy()
    all_side["side"] = "all"
    merged = pd.concat([base, all_side], ignore_index=True)

    def _p25(x):
        return float(np.nanquantile(x, 0.25)) if len(x) else float("nan")

    def _p75(x):
        return float(np.nanquantile(x, 0.75)) if len(x) else float("nan")

    out = (
        merged.groupby(["side", "range_group", "exhaustion_group"], as_index=False)
        .agg(
            trades=("holding_time_min", "count"),
            early_loss_trades=("early_loss", "sum"),
            early_loss_rate=("early_loss", "mean"),
            holding_mean_min=("holding_time_min", "mean"),
            holding_median_min=("holding_time_min", "median"),
            holding_p25_min=("holding_time_min", _p25),
            holding_p75_min=("holding_time_min", _p75),
        )
        .assign(period=period_label)
    )

    out["trades"] = out["trades"].fillna(0).astype(int)
    out["early_loss_trades"] = out["early_loss_trades"].fillna(0).astype(int)
    out["early_loss_rate"] = out["early_loss_rate"].astype(float)

    out["range_group"] = pd.Categorical(out["range_group"], categories=RANGE_ORDER, ordered=True)
    out["exhaustion_group"] = pd.Categorical(out["exhaustion_group"], categories=EXHAUSTION_ORDER, ordered=True)
    out["side"] = pd.Categorical(out["side"], categories=["all", "buy", "sell"], ordered=True)
    out = out.sort_values(["side", "range_group", "exhaustion_group"]).reset_index(drop=True)
    return out.loc[:, columns]


def _plot_early_loss_rate_2x2(summary_df, *, period_label: str, side: str, out_dir: Path):
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = summary_df[(summary_df["period"] == period_label) & (summary_df["side"] == side)].copy()
    if d.empty:
        return

    d = d[d["range_group"].isin(["short", "long"])]
    if d.empty:
        return

    order = [
        ("short", "weak"),
        ("short", "strong"),
        ("long", "weak"),
        ("long", "strong"),
    ]
    labels = [f"{r}/{e}" for r, e in order]
    rates = []
    for r, e in order:
        row = d[(d["range_group"] == r) & (d["exhaustion_group"] == e)]
        rates.append(float(row["early_loss_rate"].iloc[0]) if not row.empty else float("nan"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, rates, color=["#4C78A8", "#72B7B2", "#F58518", "#E45756"])
    ymax = float(np.nanmax(rates)) if np.isfinite(rates).any() else 1.0
    ax.set_ylim(0.0, min(1.0, ymax * 1.2 if ymax > 0 else 1.0))
    ax.set_title(f"Early loss rate 2x2 ({period_label}, {side})")
    ax.set_xlabel("cell")
    ax.set_ylabel("early_loss_rate")
    fig.tight_layout()
    out_path = out_dir / f"early_loss_rate_{period_label}_{side}_2x2.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_holding_hist_2x2(df, *, period_label: str, side: str, out_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.copy()
    if side != "all":
        d = d[d["side"] == side]
    d = d[d["range_group"].isin(["short", "long"])]
    d = d[d["exhaustion_group"].isin(EXHAUSTION_ORDER)]
    if d.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
    grid = [
        ("short", "weak", (0, 0)),
        ("short", "strong", (0, 1)),
        ("long", "weak", (1, 0)),
        ("long", "strong", (1, 1)),
    ]

    for rg, eg, (i, j) in grid:
        ax = axes[i, j]
        vals = d[(d["range_group"] == rg) & (d["exhaustion_group"] == eg)]["holding_time_min"].dropna().to_numpy()
        ax.hist(vals, bins=40, alpha=0.7, color="#4C78A8")
        ax.set_title(f"{rg}/{eg}")
        ax.set_xlabel("holding_time_min")
        ax.set_ylabel("trades")

    fig.suptitle(f"Holding time hist 2x2 ({period_label}, {side})")
    fig.tight_layout()
    out_path = out_dir / f"holding_time_hist_{period_label}_{side}_2x2.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_thresholds(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "thresholds.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_readme(out_dir: Path, *, meta: dict, thresholds: dict, d009_dir: Path, data_root: Path | None) -> None:
    lines: list[str] = []
    lines.append("# D012e pre_range_length x exhaustion (observe)")
    lines.append("")
    lines.append("## Purpose")
    lines.append("- Observe interaction between pre_range_length (D012a fixed) and exhaustion_ratio (D010).")
    lines.append("- No filtering; attach features and aggregate only.")
    lines.append("")
    lines.append("## Leak control")
    lines.append("- pre_range thresholds fixed (q33/q66 from D012a).")
    lines.append("- exhaustion p80 computed on verify only; forward uses fixed values.")
    lines.append("")
    lines.append("## Definitions")
    lines.append("- pre_range_length: close-only range persistence (lookback N=20).")
    lines.append("- pre_range_group: short<=0 / mid=1..3 / long>=4 (D012a fixed).")
    lines.append("- exhaustion_ratio: break_margin_over_mean_prev_body (fallback=break_margin_ratio).")
    lines.append("- exhaustion_group: weak < p80 / strong >= p80 (per side, verify fixed).")
    lines.append(f"- early_loss: holding_time_min in [0, {EARLY_LOSS_MAX_MIN}]")
    lines.append("")
    lines.append("## Thresholds")
    lines.append(f"- pre_range q33={PRE_RANGE_Q33} / q66={PRE_RANGE_Q66}")
    lines.append(f"- exhaustion p80 (verify): buy={thresholds['exhaustion_p80']['buy']} sell={thresholds['exhaustion_p80']['sell']}")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- d009_dir: `{d009_dir}`")
    if data_root is not None:
        lines.append(f"- data_root (override): `{data_root}`")
    for period, info in meta.items():
        lines.append(f"- {period}: trades_csv=`{info['trades_csv']}` core_config=`{info['core_config_json']}`")
        lines.append(
            f"  raw={info['n_trades_raw']} time_valid={info['n_time_valid']} side_valid={info['n_side_valid']} "
            f"pre_range_valid={info['n_pre_range_valid']} exhaustion_valid={info['n_exhaustion_valid']} "
            f"exhaustion_missing={info['n_exhaustion_missing']}"
        )
    lines.append("")
    lines.append("## Outputs")
    lines.append("- summary_verify.csv / summary_forward.csv")
    lines.append("- thresholds.json")
    lines.append("- plots/*.png (2x2: short/long x weak/strong)")
    lines.append("- README.md")
    lines.append("")
    lines.append("## Usage")
    lines.append("```bash")
    lines.append("python FX/code/observe_D012e_range_x_exhaustion.py \\")
    lines.append("  --d009_dir FX/results/family_D_momentum/D009 \\")
    lines.append("  --out_dir FX/results/family_D_momentum/D012e")
    lines.append("```")
    lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d009_dir", type=Path, default=Path("FX/results/family_D_momentum/D009"))
    parser.add_argument("--out_dir", type=Path, default=Path("FX/results/family_D_momentum/D012e"))
    parser.add_argument("--data_root", type=Path, default=None)
    args = parser.parse_args()

    d009_dir = args.d009_dir
    out_dir = args.out_dir
    data_root = args.data_root

    periods = _read_period_inputs(d009_dir)
    period_map = {"in_sample_2024": "verify", "forward_2025": "forward"}

    period_frames: dict[str, "object"] = {}
    period_meta: dict[str, dict] = {}

    for p in periods:
        print(f"[D012e] loading trades: {p.trades_csv}", flush=True)
        core_cfg = _read_core_cfg(p.core_config_json)
        lookback_bars = int(core_cfg["lookback_bars"])

        import pandas as pd
        import numpy as np

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
                "n_pre_range_valid": 0,
                "n_exhaustion_valid": 0,
                "n_exhaustion_missing": 0,
                "lookback_bars": lookback_bars,
            }
            continue

        required = {"entry_time", "exit_time", "side"}
        missing = required - set(trades_raw.columns)
        if missing:
            raise ValueError(f"{p.trades_csv} missing columns: {sorted(missing)}")

        trades = trades_raw.copy()
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce")

        holding_sec = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds()
        if "holding_secs" in trades.columns:
            holding_sec = holding_sec.fillna(pd.to_numeric(trades["holding_secs"], errors="coerce"))
        trades["holding_time_min"] = holding_sec / 60.0

        time_valid = trades["holding_time_min"].notna() & (trades["holding_time_min"] >= 0)
        trades = trades[time_valid].copy()
        n_time_valid = int(len(trades))

        trades["side"] = trades["side"].astype(str).str.lower()
        side_valid = trades["side"].isin(["buy", "sell"])
        trades = trades[side_valid].copy()
        n_side_valid = int(len(trades))

        symbol_col = trades["symbol"] if "symbol" in trades.columns else None
        symbols = (
            sorted(symbol_col.dropna().astype(str).str.upper().unique().tolist()) if symbol_col is not None else [str(core_cfg["symbol"]).upper()]
        )
        print(f"[D012e] load M1 OHLC: period={p.period} symbols={symbols}", flush=True)

        mapped_parts = []
        for sym in symbols:
            sub = trades if symbol_col is None else trades[trades["symbol"].astype(str).str.upper() == sym].copy()
            if sub.empty:
                continue
            m1 = _load_m1_ohlc(core_cfg=core_cfg, data_root=data_root, symbol_override=sym)
            m1_index, pre_range_counts = _compute_pre_range_length(m1["close"], lookback=PRE_RANGE_LOOKBACK)
            sub["pre_range_length"] = _map_pre_range_length(
                sub["entry_time"],
                m1_index=m1_index,
                pre_range_counts=pre_range_counts,
            )
            trig = _compute_exhaustion_table(m1=m1, lookback_bars=lookback_bars)
            sub = _attach_exhaustion(sub, trig=trig)
            mapped_parts.append(sub)

        trades = pd.concat(mapped_parts, ignore_index=True) if mapped_parts else trades.iloc[0:0].copy()
        n_pre_range_valid = int(trades["pre_range_length"].notna().sum()) if not trades.empty else 0
        n_exhaustion_valid = int(trades["exhaustion_ratio"].notna().sum()) if not trades.empty else 0
        n_exhaustion_missing = int(len(trades) - n_exhaustion_valid)

        trades["range_group"] = _classify_pre_range(trades["pre_range_length"].to_numpy(dtype=float))
        trades["early_loss"] = (trades["holding_time_min"] >= 0.0) & (trades["holding_time_min"] <= EARLY_LOSS_MAX_MIN)

        period_frames[p.period] = trades
        period_meta[p.period] = {
            "trades_csv": str(p.trades_csv),
            "core_config_json": str(p.core_config_json),
            "n_trades_raw": n_raw,
            "n_time_valid": n_time_valid,
            "n_side_valid": n_side_valid,
            "n_pre_range_valid": n_pre_range_valid,
            "n_exhaustion_valid": n_exhaustion_valid,
            "n_exhaustion_missing": n_exhaustion_missing,
            "lookback_bars": lookback_bars,
        }

    verify_key = next((k for k in period_frames if k == "in_sample_2024"), None)
    if verify_key is None:
        raise ValueError("Verify period (in_sample_2024) not found.")

    verify_df = period_frames[verify_key]
    if verify_df.empty or verify_df["exhaustion_ratio"].dropna().empty:
        raise ValueError("Verify trades missing exhaustion_ratio; cannot compute thresholds.")

    p80_by_side = _compute_p80_by_side(verify_df)

    for key, df in period_frames.items():
        if df.empty:
            continue
        values = df["exhaustion_ratio"].to_numpy(dtype=float)
        sides = df["side"].astype(str).to_numpy()
        df["exhaustion_group"] = _classify_exhaustion(values, side_values=sides, p80_by_side=p80_by_side)
        period_frames[key] = df

    summaries = {}
    for key, df in period_frames.items():
        period_label = period_map.get(key, key)
        valid = df[
            df["range_group"].isin(RANGE_ORDER) & df["exhaustion_group"].isin(EXHAUSTION_ORDER)
        ].copy() if not df.empty else df
        summaries[key] = _summarize(valid, period_label=period_label)

    out_dir.mkdir(parents=True, exist_ok=True)
    verify_summary = summaries.get("in_sample_2024")
    forward_summary = summaries.get("forward_2025")

    if verify_summary is not None:
        verify_summary.to_csv(out_dir / "summary_verify.csv", index=False)
    if forward_summary is not None:
        forward_summary.to_csv(out_dir / "summary_forward.csv", index=False)

    thresholds_payload = {
        "pre_range_thresholds": {
            "lookback": PRE_RANGE_LOOKBACK,
            "q33": PRE_RANGE_Q33,
            "q66": PRE_RANGE_Q66,
            "short": "pre_range_length <= 0",
            "mid": "1 <= pre_range_length <= 3",
            "long": "pre_range_length >= 4",
        },
        "exhaustion_ratio_definition": {
            "primary": "break_margin_over_mean_prev_body",
            "fallback": "break_margin_ratio",
        },
        "exhaustion_p80": p80_by_side,
        "early_loss_definition": f"holding_time_min in [0, {EARLY_LOSS_MAX_MIN}] (D007)",
        "period_map": period_map,
        "meta": period_meta,
    }
    _write_thresholds(out_dir, thresholds_payload)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for key, df in period_frames.items():
        if df.empty:
            continue
        period_label = period_map.get(key, key)
        for side in ("all", "buy", "sell"):
            _plot_early_loss_rate_2x2(
                summaries[key], period_label=period_label, side=side, out_dir=plots_dir
            )
            _plot_holding_hist_2x2(df, period_label=period_label, side=side, out_dir=plots_dir)

    _write_readme(
        out_dir,
        meta=period_meta,
        thresholds={"exhaustion_p80": p80_by_side},
        d009_dir=d009_dir,
        data_root=data_root,
    )

    print(f"[D012e] done: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
