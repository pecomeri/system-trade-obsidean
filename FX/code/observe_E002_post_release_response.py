#!/usr/bin/env python3
"""
Observe E002: post-release response distribution on D009 trades.

Definitions (fixed; no optimization):
  - Window: next K M1 bars after entry (K fixed).
  - MFE_K: max favorable excursion (pips) within K bars.
  - MAE_K: max adverse excursion (pips) within K bars.
  - Labels: pre_range_length / range_tightness / probe_count / exhaustion (from E001 thresholds).

Outputs (overwritten):
  - results/family_E_range_release/E002/summary_verify.csv
  - results/family_E_range_release/E002/summary_forward.csv
  - results/family_E_range_release/E002/thresholds.json
  - results/family_E_range_release/E002/plots/*.png
  - results/family_E_range_release/E002/README.md
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


EARLY_LOSS_MAX_MIN = 3.0
DEFAULT_K_BARS = 3

RANGE_CATS = ["short", "mid", "long"]
TIGHTNESS_CATS = ["tight", "normal", "wide"]
PROBE_CATS = ["clean", "messy"]
EXHAUSTION_CATS = ["weak", "strong"]


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
    if out:
        return out

    trades = d009_dir / "trades.csv"
    core_cfg = d009_dir / "core_config.json"
    if trades.exists() and core_cfg.exists():
        return [PeriodInputs(period="all", trades_csv=trades, core_config_json=core_cfg)]
    raise FileNotFoundError(f"No trades.csv/core_config.json found under: {d009_dir}")


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
        run_tag="e002_m1",
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


def _compute_range_tightness(m1, *, lookback: int):
    high = m1["high"].astype(float)
    low = m1["low"].astype(float)
    close = m1["close"].astype(float)
    open_ = m1["open"].astype(float)

    range_width = high.shift(1).rolling(int(lookback)).max() - low.shift(1).rolling(int(lookback)).min()
    body = (close - open_).abs()
    mean_body = body.shift(1).rolling(int(lookback)).mean()
    eps = 1e-9
    tightness = range_width / (mean_body + eps)
    return tightness


def _compute_probe_count(m1, *, lookback: int):
    high = m1["high"].astype(float)
    low = m1["low"].astype(float)
    close = m1["close"].astype(float)

    prev_high = high.shift(1).rolling(int(lookback)).max()
    prev_low = low.shift(1).rolling(int(lookback)).min()

    wick_only = ((high > prev_high) | (low < prev_low)) & (close <= prev_high) & (close >= prev_low)
    probe_count = wick_only.shift(1).rolling(int(lookback)).sum()
    return probe_count


def _map_series(entry_times, *, m1_index, series):
    import numpy as np

    pos = m1_index.searchsorted(entry_times.to_numpy(), side="left") - 1
    out = np.full(len(pos), np.nan, dtype=float)
    valid = (pos >= 0) & (pos < len(series))
    if valid.any():
        out[valid] = series.iloc[pos[valid]].to_numpy(dtype=float)
    return out


def _classify_pre_range(values, *, q33: float, q66: float):
    import numpy as np

    out = np.full(len(values), None, dtype=object)
    mask = ~np.isnan(values)
    if mask.any():
        v = values[mask]
        out[mask] = np.where(v <= q33, "short", np.where(v <= q66, "mid", "long"))
    return out


def _classify_tightness(values, *, q33: float, q66: float):
    import numpy as np

    out = np.full(len(values), None, dtype=object)
    mask = ~np.isnan(values)
    if mask.any():
        v = values[mask]
        out[mask] = np.where(v <= q33, "tight", np.where(v <= q66, "normal", "wide"))
    return out


def _classify_probe(values, *, p50: float):
    import numpy as np

    out = np.full(len(values), None, dtype=object)
    mask = ~np.isnan(values)
    if mask.any():
        v = values[mask]
        out[mask] = np.where(v <= p50, "clean", "messy")
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
    import numpy as np
    import pandas as pd

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


def _compute_mfe_mae(trades, *, m1, k_bars: int, pip_size: float):
    import numpy as np
    import pandas as pd

    idx = m1.index
    highs = m1["high"].to_numpy(dtype=float)
    lows = m1["low"].to_numpy(dtype=float)

    entry_ts = trades["entry_time"]
    entry_end = entry_ts.dt.floor("min") + pd.Timedelta(minutes=1)
    pos = idx.searchsorted(entry_end.to_numpy(), side="left")

    entry_price = trades["entry_price"].to_numpy(dtype=float)
    side = trades["side"].astype(str).to_numpy()

    mfe = np.full(len(trades), np.nan, dtype=float)
    mae = np.full(len(trades), np.nan, dtype=float)

    for i, p in enumerate(pos):
        if p < 0 or (p + k_bars) > len(idx):
            continue
        hi = float(np.max(highs[p : p + k_bars]))
        lo = float(np.min(lows[p : p + k_bars]))
        if side[i] == "buy":
            mfe[i] = (hi - entry_price[i]) / pip_size
            mae[i] = (entry_price[i] - lo) / pip_size
        else:
            mfe[i] = (entry_price[i] - lo) / pip_size
            mae[i] = (hi - entry_price[i]) / pip_size
    return mfe, mae


def _summarize_label(df, *, period_label: str, label: str, categories: list[str]):
    import numpy as np
    import pandas as pd

    rows = []
    for side in ("all", "buy", "sell"):
        if side == "all":
            d = df.copy()
        else:
            d = df[df["side"] == side].copy()
        for cat in categories:
            g = d[d[label] == cat].copy()
            g = g[g["mfe_k"].notna() & g["mae_k"].notna()].copy()
            n = int(len(g))
            if n == 0:
                rows.append(
                    {
                        "period": period_label,
                        "side": side,
                        "label": label,
                        "category": cat,
                        "trades": 0,
                        "mfe_median": float("nan"),
                        "mfe_p75": float("nan"),
                        "mfe_p90": float("nan"),
                        "mae_median": float("nan"),
                        "mae_p75": float("nan"),
                        "mae_p90": float("nan"),
                        "diff_median": float("nan"),
                        "diff_p75": float("nan"),
                        "diff_p90": float("nan"),
                    }
                )
                continue
            mfe = g["mfe_k"].to_numpy(dtype=float)
            mae = g["mae_k"].to_numpy(dtype=float)
            diff = mfe - mae
            rows.append(
                {
                    "period": period_label,
                    "side": side,
                    "label": label,
                    "category": cat,
                    "trades": n,
                    "mfe_median": float(np.nanquantile(mfe, 0.50)),
                    "mfe_p75": float(np.nanquantile(mfe, 0.75)),
                    "mfe_p90": float(np.nanquantile(mfe, 0.90)),
                    "mae_median": float(np.nanquantile(mae, 0.50)),
                    "mae_p75": float(np.nanquantile(mae, 0.75)),
                    "mae_p90": float(np.nanquantile(mae, 0.90)),
                    "diff_median": float(np.nanquantile(diff, 0.50)),
                    "diff_p75": float(np.nanquantile(diff, 0.75)),
                    "diff_p90": float(np.nanquantile(diff, 0.90)),
                }
            )
    return pd.DataFrame(rows)


def _plot_box_by_label(summary_df, *, df_raw, period_label: str, label: str, categories: list[str], metric: str, out_dir: Path):
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df_raw.copy()
    d = d[d["side"].isin(["buy", "sell"])].copy()
    d = d[d[metric].notna()]
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    positions = []
    data = []
    colors = []

    for i, cat in enumerate(categories, start=1):
        for side, offset, color in (("buy", -0.18, "#4C78A8"), ("sell", 0.18, "#E45756")):
            vals = d[(d["side"] == side) & (d[label] == cat)][metric].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            positions.append(i + offset)
            data.append(vals)
            colors.append(color)

    if not data:
        plt.close(fig)
        return

    bp = ax.boxplot(data, positions=positions, widths=0.32, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticks(range(1, len(categories) + 1))
    ax.set_xticklabels(categories)
    ax.set_title(f"{metric} ({label}, {period_label})")
    ax.set_ylabel("pips")
    ax.legend(handles=[
        plt.Line2D([0], [0], color="#4C78A8", lw=6, label="buy"),
        plt.Line2D([0], [0], color="#E45756", lw=6, label="sell"),
    ])
    fig.tight_layout()

    out_path = out_dir / f"{metric}_{label}_{period_label}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_thresholds(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "thresholds.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_readme(out_dir: Path, *, meta: dict, thresholds: dict, d009_dir: Path, e001_dir: Path, data_root: Path | None) -> None:
    lines: list[str] = []
    lines.append("# E002 post-release response distribution (observe)")
    lines.append("")
    lines.append("## Purpose")
    lines.append("- Observe MFE/MAE distribution after release using E001 labels.")
    lines.append("- No filtering; aggregation only.")
    lines.append("")
    lines.append("## Definitions")
    lines.append(f"- K bars (M1): {thresholds['k_bars']}")
    lines.append("- MFE_K: max favorable excursion in next K M1 bars (pips).")
    lines.append("- MAE_K: max adverse excursion in next K M1 bars (pips).")
    lines.append("- Labels: pre_range_length / range_tightness / probe_count / exhaustion (from E001).")
    lines.append("")
    lines.append("## Thresholds (verify fixed)")
    lines.append(f"- pre_range q33={thresholds['pre_range_thresholds']['q33']} / q66={thresholds['pre_range_thresholds']['q66']}")
    lines.append(f"- tightness q33={thresholds['tightness_q33']} / q66={thresholds['tightness_q66']}")
    lines.append(f"- probe p50={thresholds['probe_p50']}")
    lines.append(f"- exhaustion p80 (buy={thresholds['exhaustion_p80']['buy']} / sell={thresholds['exhaustion_p80']['sell']})")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- d009_dir: `{d009_dir}`")
    lines.append(f"- e001_dir: `{e001_dir}` (thresholds)")
    if data_root is not None:
        lines.append(f"- data_root (override): `{data_root}`")
    for period, info in meta.items():
        lines.append(f"- {period}: trades_csv=`{info['trades_csv']}` core_config=`{info['core_config_json']}`")
        lines.append(
            "  raw={raw} time_valid={time_valid} side_valid={side_valid} "
            "mfe_valid={mfe_valid} mae_valid={mae_valid} "
            "pre_range_valid={pre_range_valid} tightness_valid={tightness_valid} "
            "probe_valid={probe_valid} exhaustion_valid={exhaustion_valid}".format(**info)
        )
    lines.append("")
    lines.append("## Outputs")
    lines.append("- summary_verify.csv / summary_forward.csv")
    lines.append("- thresholds.json")
    lines.append("- plots/*.png (boxplots)")
    lines.append("- README.md")
    lines.append("")
    lines.append("## Usage")
    lines.append("```bash")
    lines.append("python FX/code/observe_E002_post_release_response.py \\")
    lines.append("  --d009_dir FX/results/family_D_momentum/D009 \\")
    lines.append("  --e001_dir FX/results/family_E_range_release/E001 \\")
    lines.append("  --out_dir FX/results/family_E_range_release/E002")
    lines.append("```")
    lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d009_dir", type=Path, default=Path("FX/results/family_D_momentum/D009"))
    parser.add_argument("--e001_dir", type=Path, default=Path("FX/results/family_E_range_release/E001"))
    parser.add_argument("--out_dir", type=Path, default=Path("FX/results/family_E_range_release/E002"))
    parser.add_argument("--k_bars", type=int, default=DEFAULT_K_BARS)
    parser.add_argument("--data_root", type=Path, default=None)
    args = parser.parse_args()

    d009_dir = args.d009_dir
    e001_dir = args.e001_dir
    out_dir = args.out_dir
    k_bars = int(args.k_bars)
    data_root = args.data_root

    thresholds_path = e001_dir / "thresholds.json"
    if not thresholds_path.exists():
        raise FileNotFoundError(f"E001 thresholds.json not found: {thresholds_path}")
    e001_thr = json.loads(thresholds_path.read_text(encoding="utf-8"))

    pre_thr = e001_thr["pre_range_thresholds"]
    tight_q33 = float(e001_thr["tightness_q33"])
    tight_q66 = float(e001_thr["tightness_q66"])
    probe_p50 = float(e001_thr["probe_p50"])
    p80_by_side = e001_thr["exhaustion_p80"]

    periods = _read_period_inputs(d009_dir)
    period_map = {"in_sample_2024": "verify", "forward_2025": "forward", "all": "all"}

    period_frames = {}
    period_meta = {}

    for p in periods:
        import pandas as pd
        import numpy as np

        trades_raw = pd.read_csv(p.trades_csv)
        n_raw = int(len(trades_raw))

        if trades_raw.empty:
            period_frames[p.period] = trades_raw
            period_meta[p.period] = {
                "trades_csv": str(p.trades_csv),
                "core_config_json": str(p.core_config_json),
                "raw": n_raw,
                "time_valid": 0,
                "side_valid": 0,
                "mfe_valid": 0,
                "mae_valid": 0,
                "pre_range_valid": 0,
                "tightness_valid": 0,
                "probe_valid": 0,
                "exhaustion_valid": 0,
            }
            continue

        required = {"entry_time", "exit_time", "side", "entry_price"}
        missing = required - set(trades_raw.columns)
        if missing:
            raise ValueError(f"{p.trades_csv} missing columns: {sorted(missing)}")

        trades = trades_raw.copy()
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce")
        trades["holding_time_min"] = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 60.0
        time_valid = trades["holding_time_min"].notna() & (trades["holding_time_min"] >= 0)
        trades = trades[time_valid].copy()
        n_time_valid = int(len(trades))

        trades["side"] = trades["side"].astype(str).str.lower()
        side_valid = trades["side"].isin(["buy", "sell"])
        trades = trades[side_valid].copy()
        n_side_valid = int(len(trades))

        core_cfg = _read_core_cfg(p.core_config_json)
        pip_size = float(core_cfg.get("pip_size", 0.01))
        lookback_bars = int(core_cfg["lookback_bars"])

        symbol_col = trades["symbol"] if "symbol" in trades.columns else None
        symbols = (
            sorted(symbol_col.dropna().astype(str).str.upper().unique().tolist()) if symbol_col is not None else [str(core_cfg["symbol"]).upper()]
        )
        print(f"[E002] load M1 OHLC: period={p.period} symbols={symbols}", flush=True)

        mapped_parts = []
        for sym in symbols:
            sub = trades if symbol_col is None else trades[trades["symbol"].astype(str).str.upper() == sym].copy()
            if sub.empty:
                continue
            m1 = _load_m1_ohlc(core_cfg=core_cfg, data_root=data_root, symbol_override=sym)

            # Range features
            m1_index, pre_range_counts = _compute_pre_range_length(m1["close"], lookback=int(pre_thr["lookback"]))
            tightness = _compute_range_tightness(m1, lookback=int(pre_thr["lookback"]))
            probe_count = _compute_probe_count(m1, lookback=int(pre_thr["lookback"]))

            sub["pre_range_length"] = _map_series(sub["entry_time"], m1_index=m1_index, series=pd.Series(pre_range_counts, index=m1_index))
            sub["range_tightness"] = _map_series(sub["entry_time"], m1_index=m1_index, series=tightness)
            sub["probe_count"] = _map_series(sub["entry_time"], m1_index=m1_index, series=probe_count)

            # Exhaustion (D010)
            trig = _compute_exhaustion_table(m1=m1, lookback_bars=lookback_bars)
            sub = _attach_exhaustion(sub, trig=trig)

            # MFE/MAE
            mfe, mae = _compute_mfe_mae(sub, m1=m1, k_bars=k_bars, pip_size=pip_size)
            sub["mfe_k"] = mfe
            sub["mae_k"] = mae
            sub["diff_k"] = sub["mfe_k"] - sub["mae_k"]

            mapped_parts.append(sub)

        trades = pd.concat(mapped_parts, ignore_index=True) if mapped_parts else trades.iloc[0:0].copy()

        trades["pre_range_group"] = _classify_pre_range(
            trades["pre_range_length"].to_numpy(dtype=float),
            q33=float(pre_thr["q33"]),
            q66=float(pre_thr["q66"]),
        )
        trades["tightness_group"] = _classify_tightness(trades["range_tightness"].to_numpy(dtype=float), q33=tight_q33, q66=tight_q66)
        trades["probe_group"] = _classify_probe(trades["probe_count"].to_numpy(dtype=float), p50=probe_p50)
        trades["exhaustion_group"] = _classify_exhaustion(
            trades["exhaustion_ratio"].to_numpy(dtype=float),
            side_values=trades["side"].astype(str).to_numpy(),
            p80_by_side=p80_by_side,
        )
        trades["early_loss"] = (trades["holding_time_min"] >= 0.0) & (trades["holding_time_min"] <= EARLY_LOSS_MAX_MIN)

        period_frames[p.period] = trades
        period_meta[p.period] = {
            "trades_csv": str(p.trades_csv),
            "core_config_json": str(p.core_config_json),
            "raw": n_raw,
            "time_valid": n_time_valid,
            "side_valid": n_side_valid,
            "mfe_valid": int(trades["mfe_k"].notna().sum()),
            "mae_valid": int(trades["mae_k"].notna().sum()),
            "pre_range_valid": int(trades["pre_range_length"].notna().sum()),
            "tightness_valid": int(trades["range_tightness"].notna().sum()),
            "probe_valid": int(trades["probe_count"].notna().sum()),
            "exhaustion_valid": int(trades["exhaustion_ratio"].notna().sum()),
        }

    summaries = {}
    for key, df in period_frames.items():
        period_label = period_map.get(key, key)
        parts = []
        parts.append(_summarize_label(df, period_label=period_label, label="pre_range_group", categories=RANGE_CATS))
        parts.append(_summarize_label(df, period_label=period_label, label="tightness_group", categories=TIGHTNESS_CATS))
        parts.append(_summarize_label(df, period_label=period_label, label="probe_group", categories=PROBE_CATS))
        parts.append(_summarize_label(df, period_label=period_label, label="exhaustion_group", categories=EXHAUSTION_CATS))
        summaries[key] = pd.concat(parts, ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    if "in_sample_2024" in summaries:
        summaries["in_sample_2024"].to_csv(out_dir / "summary_verify.csv", index=False)
    if "forward_2025" in summaries:
        summaries["forward_2025"].to_csv(out_dir / "summary_forward.csv", index=False)
    if "all" in summaries:
        summaries["all"].to_csv(out_dir / "summary_all.csv", index=False)

    thresholds_payload = {
        "k_bars": k_bars,
        "pip_size": float(_read_core_cfg(periods[0].core_config_json).get("pip_size", 0.01)),
        "pre_range_thresholds": pre_thr,
        "tightness_q33": tight_q33,
        "tightness_q66": tight_q66,
        "probe_p50": probe_p50,
        "exhaustion_p80": p80_by_side,
        "exhaustion_ratio_definition": e001_thr["exhaustion_ratio_definition"],
        "early_loss_definition": f"holding_time_min in [0, {EARLY_LOSS_MAX_MIN}] (D007)",
        "response_definition": {
            "mfe_k": "max favorable excursion (pips) within next K M1 bars",
            "mae_k": "max adverse excursion (pips) within next K M1 bars",
        },
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
        for label, cats in (
            ("pre_range_group", RANGE_CATS),
            ("tightness_group", TIGHTNESS_CATS),
            ("probe_group", PROBE_CATS),
            ("exhaustion_group", EXHAUSTION_CATS),
        ):
            _plot_box_by_label(
                summaries[key],
                df_raw=df,
                period_label=period_label,
                label=label,
                categories=cats,
                metric="mfe_k",
                out_dir=plots_dir,
            )
            _plot_box_by_label(
                summaries[key],
                df_raw=df,
                period_label=period_label,
                label=label,
                categories=cats,
                metric="mae_k",
                out_dir=plots_dir,
            )

    _write_readme(out_dir, meta=period_meta, thresholds=thresholds_payload, d009_dir=d009_dir, e001_dir=e001_dir, data_root=data_root)
    print(f"[E002] done: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
