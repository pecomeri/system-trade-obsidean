#!/usr/bin/env python3
"""
Observe E001: range-release signature on D009 trades.

Definitions (fixed; no optimization):
  - pre_range_length: close-based range persistence (N=20), D012a thresholds.
  - range_tightness: range_width / (mean_body_prevN + eps), N=20.
  - probe_count: wick-only probes beyond prior range, count over N=20 bars.
  - exhaustion_ratio: break_margin_over_mean_prev_body (fallback=break_margin_ratio).
  - strong/weak exhaustion: verify p80 (per side), forward fixed.
  - next_bar_confirm: next M1 close moves in entry direction.
  - quick_reject: within next 2 M1 closes moves opposite to entry direction.
  - early_loss: holding_time_min in [0, 3] (reference only).

Outputs (overwritten):
  - results/family_E_range_release/E001/summary_verify.csv
  - results/family_E_range_release/E001/summary_forward.csv
  - results/family_E_range_release/E001/thresholds.json
  - results/family_E_range_release/E001/plots/*.png
  - results/family_E_range_release/E001/README.md
"""

from __future__ import annotations

import argparse
import json
import math
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
RANGE_LOOKBACK = 20
PRE_RANGE_Q33 = 0.0
PRE_RANGE_Q66 = 3.0

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
        run_tag="e001_m1",
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
    import numpy as np

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


def _compute_entry_response(trades, *, m1_close):
    import numpy as np
    import pandas as pd

    entry_time = trades["entry_time"]
    entry_price = trades["entry_price"].astype(float)
    side = trades["side"].astype(str)

    entry_bar_end = entry_time.dt.floor("min") + pd.Timedelta(minutes=1)
    close1 = m1_close.reindex(entry_bar_end + pd.Timedelta(minutes=1))
    close2 = m1_close.reindex(entry_bar_end + pd.Timedelta(minutes=2))

    close1 = close1.to_numpy(dtype=float)
    close2 = close2.to_numpy(dtype=float)
    entry_price_np = entry_price.to_numpy(dtype=float)
    side_np = side.to_numpy(dtype=str)

    confirm = np.full(len(trades), np.nan, dtype=float)
    reject = np.full(len(trades), np.nan, dtype=float)

    valid1 = np.isfinite(close1)
    if valid1.any():
        buy = (side_np == "buy") & valid1
        sell = (side_np == "sell") & valid1
        confirm[buy] = (close1[buy] > entry_price_np[buy]).astype(float)
        confirm[sell] = (close1[sell] < entry_price_np[sell]).astype(float)

    valid_any = np.isfinite(close1) | np.isfinite(close2)
    if valid_any.any():
        buy = (side_np == "buy") & valid_any
        sell = (side_np == "sell") & valid_any

        opp_buy = np.zeros(len(trades), dtype=bool)
        opp_sell = np.zeros(len(trades), dtype=bool)

        if np.isfinite(close1).any():
            opp_buy = opp_buy | (np.isfinite(close1) & (close1 < entry_price_np))
            opp_sell = opp_sell | (np.isfinite(close1) & (close1 > entry_price_np))
        if np.isfinite(close2).any():
            opp_buy = opp_buy | (np.isfinite(close2) & (close2 < entry_price_np))
            opp_sell = opp_sell | (np.isfinite(close2) & (close2 > entry_price_np))

        reject[buy] = opp_buy[buy].astype(float)
        reject[sell] = opp_sell[sell].astype(float)

    trades["next_bar_confirm"] = confirm
    trades["quick_reject"] = reject
    return trades


def _summarize_feature(df, *, period_label: str, feature: str, categories: list[str]):
    import numpy as np
    import pandas as pd

    rows = []
    for side in ("all", "buy", "sell"):
        if side == "all":
            d = df.copy()
        else:
            d = df[df["side"] == side].copy()
        for cat in categories:
            g = d[d[feature] == cat].copy()
            n = int(len(g))
            if n == 0:
                rows.append(
                    {
                        "period": period_label,
                        "side": side,
                        "feature": feature,
                        "category": cat,
                        "trades": 0,
                        "next_bar_confirm_rate": float("nan"),
                        "quick_reject_rate": float("nan"),
                        "early_loss_rate": float("nan"),
                    }
                )
                continue
            rows.append(
                {
                    "period": period_label,
                    "side": side,
                    "feature": feature,
                    "category": cat,
                    "trades": n,
                    "next_bar_confirm_rate": float(np.nanmean(g["next_bar_confirm"].to_numpy(dtype=float))),
                    "quick_reject_rate": float(np.nanmean(g["quick_reject"].to_numpy(dtype=float))),
                    "early_loss_rate": float(np.nanmean(g["early_loss"].to_numpy(dtype=float))),
                }
            )
    return pd.DataFrame(rows)


def _summarize_cross(df, *, period_label: str):
    import numpy as np
    import pandas as pd

    rows = []
    for side in ("all", "buy", "sell"):
        if side == "all":
            d = df.copy()
        else:
            d = df[df["side"] == side].copy()

        long_strong = d[(d["pre_range_group"] == "long") & (d["exhaustion_group"] == "strong")]
        tight_strong = d[(d["tightness_group"] == "tight") & (d["exhaustion_group"] == "strong")]

        for name, g in (("long_strong", long_strong), ("tight_strong", tight_strong)):
            n = int(len(g))
            rows.append(
                {
                    "period": period_label,
                    "side": side,
                    "feature": "cross",
                    "category": name,
                    "trades": n,
                    "next_bar_confirm_rate": float(np.nanmean(g["next_bar_confirm"].to_numpy(dtype=float))) if n else float("nan"),
                    "quick_reject_rate": float(np.nanmean(g["quick_reject"].to_numpy(dtype=float))) if n else float("nan"),
                    "early_loss_rate": float(np.nanmean(g["early_loss"].to_numpy(dtype=float))) if n else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _plot_feature(summary_df, *, period_label: str, side: str, feature: str, out_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = summary_df[(summary_df["period"] == period_label) & (summary_df["side"] == side) & (summary_df["feature"] == feature)]
    if d.empty:
        return

    categories = d["category"].tolist()
    x = range(len(categories))
    confirm = d["next_bar_confirm_rate"].to_numpy(dtype=float)
    reject = d["quick_reject_rate"].to_numpy(dtype=float)
    early = d["early_loss_rate"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.25
    ax.bar([i - width for i in x], confirm, width=width, label="confirm", color="#4C78A8")
    ax.bar(x, reject, width=width, label="reject", color="#E45756")
    ax.bar([i + width for i in x], early, width=width, label="early_loss", color="#F58518")
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories)
    ax.set_ylim(0.0, min(1.0, max(0.1, float(max([v for v in list(confirm) + list(reject) + list(early) if math.isfinite(v)], default=0.1)) * 1.2)))
    ax.set_title(f"{feature} rates ({period_label}, {side})")
    ax.set_ylabel("rate")
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / f"{feature}_rates_{period_label}_{side}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_thresholds(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "thresholds.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_readme(out_dir: Path, *, meta: dict, thresholds: dict, d009_dir: Path, data_root: Path | None) -> None:
    lines: list[str] = []
    lines.append("# E001 range-release signature (observe)")
    lines.append("")
    lines.append("## Purpose")
    lines.append("- Observe range-release signature on D009 trades (buy/sell, 24h).")
    lines.append("- No filtering; classify and aggregate only.")
    lines.append("")
    lines.append("## Definitions")
    lines.append(f"- pre_range_length: close-based range persistence (N={RANGE_LOOKBACK}, D012a fixed).")
    lines.append("- range_tightness: range_width / (mean_body_prevN + eps).")
    lines.append("- probe_count: wick-only probes beyond prior range (count over N).")
    lines.append("- exhaustion_ratio: break_margin_over_mean_prev_body (fallback=break_margin_ratio).")
    lines.append("- next_bar_confirm: next M1 close moves in entry direction.")
    lines.append("- quick_reject: within next 2 M1 closes moves opposite to entry direction.")
    lines.append(f"- early_loss: holding_time_min in [0, {EARLY_LOSS_MAX_MIN}]")
    lines.append("")
    lines.append("## Thresholds (verify fixed)")
    lines.append(f"- pre_range q33={PRE_RANGE_Q33} / q66={PRE_RANGE_Q66}")
    lines.append(f"- tightness q33={thresholds['tightness_q33']} / q66={thresholds['tightness_q66']}")
    lines.append(f"- probe_count p50={thresholds['probe_p50']}")
    lines.append(f"- exhaustion p80 (buy={thresholds['exhaustion_p80']['buy']} / sell={thresholds['exhaustion_p80']['sell']})")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- d009_dir: `{d009_dir}`")
    if data_root is not None:
        lines.append(f"- data_root (override): `{data_root}`")
    for period, info in meta.items():
        lines.append(f"- {period}: trades_csv=`{info['trades_csv']}` core_config=`{info['core_config_json']}`")
        lines.append(
            "  raw={raw} time_valid={time_valid} side_valid={side_valid} "
            "pre_range_valid={pre_range_valid} tightness_valid={tightness_valid} "
            "probe_valid={probe_valid} exhaustion_valid={exhaustion_valid} "
            "confirm_valid={confirm_valid} reject_valid={reject_valid}".format(**info)
        )
    lines.append("")
    lines.append("## Outputs")
    lines.append("- summary_verify.csv / summary_forward.csv")
    lines.append("- thresholds.json")
    lines.append("- plots/*.png")
    lines.append("- README.md")
    lines.append("")
    lines.append("## Usage")
    lines.append("```bash")
    lines.append("python FX/code/observe_E001_range_release_signature.py \\")
    lines.append("  --d009_dir FX/results/family_D_momentum/D009 \\")
    lines.append("  --out_dir FX/results/family_E_range_release/E001")
    lines.append("```")
    lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d009_dir", type=Path, default=Path("FX/results/family_D_momentum/D009"))
    parser.add_argument("--out_dir", type=Path, default=Path("FX/results/family_E_range_release/E001"))
    parser.add_argument("--data_root", type=Path, default=None)
    args = parser.parse_args()

    d009_dir = args.d009_dir
    out_dir = args.out_dir
    data_root = args.data_root

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
                "pre_range_valid": 0,
                "tightness_valid": 0,
                "probe_valid": 0,
                "exhaustion_valid": 0,
                "confirm_valid": 0,
                "reject_valid": 0,
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

        symbol_col = trades["symbol"] if "symbol" in trades.columns else None
        symbols = (
            sorted(symbol_col.dropna().astype(str).str.upper().unique().tolist()) if symbol_col is not None else [str(_read_core_cfg(p.core_config_json)["symbol"]).upper()]
        )
        print(f"[E001] load M1 OHLC: period={p.period} symbols={symbols}", flush=True)

        mapped_parts = []
        for sym in symbols:
            sub = trades if symbol_col is None else trades[trades["symbol"].astype(str).str.upper() == sym].copy()
            if sub.empty:
                continue
            core_cfg = _read_core_cfg(p.core_config_json)
            m1 = _load_m1_ohlc(core_cfg=core_cfg, data_root=data_root, symbol_override=sym)

            # Range features
            m1_index, pre_range_counts = _compute_pre_range_length(m1["close"], lookback=RANGE_LOOKBACK)
            tightness = _compute_range_tightness(m1, lookback=RANGE_LOOKBACK)
            probe_count = _compute_probe_count(m1, lookback=RANGE_LOOKBACK)

            sub["pre_range_length"] = _map_series(sub["entry_time"], m1_index=m1_index, series=pd.Series(pre_range_counts, index=m1_index))
            sub["range_tightness"] = _map_series(sub["entry_time"], m1_index=m1_index, series=tightness)
            sub["probe_count"] = _map_series(sub["entry_time"], m1_index=m1_index, series=probe_count)

            # Exhaustion (D010)
            trig = _compute_exhaustion_table(m1=m1, lookback_bars=int(core_cfg["lookback_bars"]))
            sub = _attach_exhaustion(sub, trig=trig)

            # Response
            sub = _compute_entry_response(sub, m1_close=m1["close"])
            mapped_parts.append(sub)

        trades = pd.concat(mapped_parts, ignore_index=True) if mapped_parts else trades.iloc[0:0].copy()

        trades["pre_range_group"] = _classify_pre_range(trades["pre_range_length"].to_numpy(dtype=float))
        trades["early_loss"] = (trades["holding_time_min"] >= 0.0) & (trades["holding_time_min"] <= EARLY_LOSS_MAX_MIN)

        period_frames[p.period] = trades
        period_meta[p.period] = {
            "trades_csv": str(p.trades_csv),
            "core_config_json": str(p.core_config_json),
            "raw": n_raw,
            "time_valid": n_time_valid,
            "side_valid": n_side_valid,
            "pre_range_valid": int(trades["pre_range_length"].notna().sum()),
            "tightness_valid": int(trades["range_tightness"].notna().sum()),
            "probe_valid": int(trades["probe_count"].notna().sum()),
            "exhaustion_valid": int(trades["exhaustion_ratio"].notna().sum()),
            "confirm_valid": int(trades["next_bar_confirm"].notna().sum()),
            "reject_valid": int(trades["quick_reject"].notna().sum()),
        }

    verify_key = "in_sample_2024" if "in_sample_2024" in period_frames else "all"
    verify_df = period_frames[verify_key]
    if verify_df.empty:
        raise ValueError("Verify period data is empty; cannot compute thresholds.")

    tight_q33 = float(verify_df["range_tightness"].dropna().quantile(0.33))
    tight_q66 = float(verify_df["range_tightness"].dropna().quantile(0.66))
    probe_p50 = float(verify_df["probe_count"].dropna().quantile(0.50))
    p80_by_side = _compute_p80_by_side(verify_df)

    for key, df in period_frames.items():
        if df.empty:
            continue
        df["tightness_group"] = _classify_tightness(df["range_tightness"].to_numpy(dtype=float), q33=tight_q33, q66=tight_q66)
        df["probe_group"] = _classify_probe(df["probe_count"].to_numpy(dtype=float), p50=probe_p50)
        df["exhaustion_group"] = _classify_exhaustion(df["exhaustion_ratio"].to_numpy(dtype=float), side_values=df["side"].astype(str).to_numpy(), p80_by_side=p80_by_side)
        period_frames[key] = df

    summaries = {}
    for key, df in period_frames.items():
        period_label = period_map.get(key, key)
        parts = []
        parts.append(_summarize_feature(df, period_label=period_label, feature="pre_range_group", categories=RANGE_CATS))
        parts.append(_summarize_feature(df, period_label=period_label, feature="tightness_group", categories=TIGHTNESS_CATS))
        parts.append(_summarize_feature(df, period_label=period_label, feature="probe_group", categories=PROBE_CATS))
        parts.append(_summarize_feature(df, period_label=period_label, feature="exhaustion_group", categories=EXHAUSTION_CATS))
        parts.append(_summarize_cross(df, period_label=period_label))
        summaries[key] = pd.concat(parts, ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    if "in_sample_2024" in summaries:
        summaries["in_sample_2024"].to_csv(out_dir / "summary_verify.csv", index=False)
    if "forward_2025" in summaries:
        summaries["forward_2025"].to_csv(out_dir / "summary_forward.csv", index=False)
    if "all" in summaries:
        summaries["all"].to_csv(out_dir / "summary_all.csv", index=False)

    thresholds_payload = {
        "pre_range_thresholds": {
            "lookback": RANGE_LOOKBACK,
            "q33": PRE_RANGE_Q33,
            "q66": PRE_RANGE_Q66,
            "short": "pre_range_length <= 0",
            "mid": "1 <= pre_range_length <= 3",
            "long": "pre_range_length >= 4",
        },
        "tightness_q33": tight_q33,
        "tightness_q66": tight_q66,
        "probe_p50": probe_p50,
        "probe_definition": "wick-only probes beyond prior range (count over N bars)",
        "exhaustion_ratio_definition": {
            "primary": "break_margin_over_mean_prev_body",
            "fallback": "break_margin_ratio",
        },
        "exhaustion_p80": p80_by_side,
        "response_definition": {
            "next_bar_confirm": "next M1 close moves in entry direction",
            "quick_reject": "within next 2 M1 closes moves opposite to entry direction",
        },
        "early_loss_definition": f"holding_time_min in [0, {EARLY_LOSS_MAX_MIN}] (D007)",
        "period_map": period_map,
        "meta": period_meta,
    }
    _write_thresholds(out_dir, thresholds_payload)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for key, summary in summaries.items():
        if summary.empty:
            continue
        period_label = period_map.get(key, key)
        for side in ("all", "buy", "sell"):
            for feature in ("pre_range_group", "tightness_group", "probe_group", "exhaustion_group", "cross"):
                _plot_feature(summary, period_label=period_label, side=side, feature=feature, out_dir=plots_dir)

    _write_readme(out_dir, meta=period_meta, thresholds=thresholds_payload, d009_dir=d009_dir, data_root=data_root)
    print(f"[E001] done: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
