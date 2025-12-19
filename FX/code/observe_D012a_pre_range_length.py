#!/usr/bin/env python3
"""
Observe D012a: pre_range_length on top of D009 trades (buy/sell, 24h).

Definition (close-based, no optimization):
  pre_range_length = consecutive count of M1 closes that stay within the
  rolling close range of the previous N bars (no new close-high/low).
  Computed on the last completed M1 bar strictly before entry_time.

Leak control:
  - Quantiles (q33, q66) computed on verify only.
  - Forward uses the same thresholds (no re-fit).

Output (overwritten):
  - results/family_D_momentum/D012a/summary_verify.csv
  - results/family_D_momentum/D012a/summary_forward.csv
  - results/family_D_momentum/D012a/thresholds.json
  - results/family_D_momentum/D012a/plots/*.png
  - results/family_D_momentum/D012a/README.md
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
EARLY_LOSS_MAX_MIN = 3.0
DEFAULT_LOOKBACK = 20


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


def _load_m1_close(*, core_cfg: dict, data_root: Path | None, symbol_override: str | None):
    import pandas as pd
    import backtest_core as bc

    root = Path(data_root) if data_root is not None else Path(core_cfg["root"])
    symbol = str(symbol_override or core_cfg["symbol"]).upper()

    cfg = bc.Config(
        root=root,
        symbol=symbol,
        from_month=str(core_cfg["from_month"]),
        to_month=str(core_cfg["to_month"]),
        run_tag="d012a_m1",
        only_session=None,
        spread_pips=float(core_cfg.get("spread_pips", 1.0)),
        pip_size=float(core_cfg.get("pip_size", 0.01)),
    )
    df_bid = bc.load_parquet_10s_bid(cfg)
    df10 = bc.add_synthetic_bidask(df_bid, cfg)

    close = df10["close_bid"].resample("1min", label="right", closed="right").last().dropna()
    if close.empty:
        raise ValueError("M1 close series is empty after resampling.")
    return close


def _compute_pre_range_length(close, *, lookback: int):
    """
    Close-based range length.

    For each M1 bar t:
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


def _classify_ranges(values, *, q33: float, q66: float):
    import numpy as np

    out = np.full(len(values), None, dtype=object)
    mask = ~np.isnan(values)
    if mask.any():
        v = values[mask]
        out[mask] = np.where(v <= q33, "short", np.where(v <= q66, "mid", "long"))
    return out


def _summarize(df, *, period_label: str):
    import numpy as np
    import pandas as pd

    columns = [
        "period",
        "side",
        "range_group",
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
        merged.groupby(["side", "range_group"], as_index=False)
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
    out["side"] = pd.Categorical(out["side"], categories=["all", "buy", "sell"], ordered=True)
    out = out.sort_values(["side", "range_group"]).reset_index(drop=True)
    return out.loc[:, columns]


def _plot_holding_hist(df, *, period_label: str, side: str, out_dir: Path):
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df.copy()
    if side != "all":
        d = d[d["side"] == side]
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for grp in RANGE_ORDER:
        vals = d.loc[d["range_group"] == grp, "holding_time_min"].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=50, alpha=0.55, label=grp)

    ax.set_title(f"Holding time hist ({period_label}, {side})")
    ax.set_xlabel("holding_time_min")
    ax.set_ylabel("trades")
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / f"holding_time_hist_{period_label}_{side}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_early_loss_rate(summary_df, *, period_label: str, side: str, out_dir: Path):
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = summary_df[(summary_df["period"] == period_label) & (summary_df["side"] == side)].copy()
    if d.empty:
        return

    d = d.set_index("range_group").reindex(RANGE_ORDER)
    rates = d["early_loss_rate"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(RANGE_ORDER, rates, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylim(0.0, min(1.0, float(np.nanmax(rates)) * 1.2 if np.isfinite(rates).any() else 1.0))
    ax.set_title(f"Early loss rate ({period_label}, {side})")
    ax.set_xlabel("range_group")
    ax.set_ylabel("early_loss_rate")
    fig.tight_layout()
    out_path = out_dir / f"early_loss_rate_{period_label}_{side}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_thresholds(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "thresholds.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_readme(out_dir: Path, *, meta: dict, thresholds: dict, lookback: int, d009_dir: Path, data_root: Path | None) -> None:
    lines: list[str] = []
    lines.append("# D012a pre_range_length (observe)")
    lines.append("")
    lines.append("## Purpose")
    lines.append("- Observe pre_range_length on D009 trades (buy/sell, 24h).")
    lines.append("- No filtering; attach feature and aggregate only.")
    lines.append("")
    lines.append("## Leak control")
    lines.append("- q33/q66 computed on verify only.")
    lines.append("- Forward uses fixed thresholds (no re-fit).")
    lines.append("")
    lines.append("## Definitions")
    lines.append(f"- lookback N = {lookback}")
    lines.append("- pre_range_length: consecutive M1 closes that stay within the")
    lines.append("  rolling close range of the previous N bars (close-only).")
    lines.append("- uses the last completed M1 bar strictly before entry_time.")
    lines.append(f"- early_loss: holding_time_min in [0, {EARLY_LOSS_MAX_MIN}]")
    lines.append("")
    lines.append("## Thresholds (verify)")
    lines.append(f"- q33: {thresholds['q33']}")
    lines.append(f"- q66: {thresholds['q66']}")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- d009_dir: `{d009_dir}`")
    if data_root is not None:
        lines.append(f"- data_root (override): `{data_root}`")
    for period, info in meta.items():
        lines.append(f"- {period}: trades_csv=`{info['trades_csv']}` core_config=`{info['core_config_json']}`")
        lines.append(
            f"  raw={info['n_trades_raw']} time_valid={info['n_time_valid']} side_valid={info['n_side_valid']} "
            f"pre_range_valid={info['n_pre_range_valid']} pre_range_missing={info['n_pre_range_missing']}"
        )
    lines.append("")
    lines.append("## Outputs")
    lines.append("- summary_verify.csv / summary_forward.csv")
    lines.append("- thresholds.json")
    lines.append("- plots/*.png (holding_time hist, early_loss rate)")
    lines.append("- README.md")
    lines.append("")
    lines.append("## Usage")
    lines.append("```bash")
    lines.append("python FX/code/observe_D012a_pre_range_length.py \\")
    lines.append("  --d009_dir FX/results/family_D_momentum/D009 \\")
    lines.append("  --out_dir FX/results/family_D_momentum/D012a")
    lines.append("```")
    lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d009_dir", type=Path, default=Path("FX/results/family_D_momentum/D009"))
    parser.add_argument("--out_dir", type=Path, default=Path("FX/results/family_D_momentum/D012a"))
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--data_root", type=Path, default=None)
    args = parser.parse_args()

    d009_dir = args.d009_dir
    out_dir = args.out_dir
    lookback = int(args.lookback)
    data_root = args.data_root

    periods = _read_period_inputs(d009_dir)
    period_map = {"in_sample_2024": "verify", "forward_2025": "forward"}

    period_frames: dict[str, "object"] = {}
    period_meta: dict[str, dict] = {}

    for p in periods:
        print(f"[D012a] loading trades: {p.trades_csv}", flush=True)
        core_cfg = _read_core_cfg(p.core_config_json)

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
                "n_pre_range_missing": 0,
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
        print(f"[D012a] load M1 close: period={p.period} symbols={symbols}", flush=True)

        mapped_parts = []
        for sym in symbols:
            sub = trades if symbol_col is None else trades[trades["symbol"].astype(str).str.upper() == sym].copy()
            if sub.empty:
                continue
            close = _load_m1_close(core_cfg=core_cfg, data_root=data_root, symbol_override=sym)
            m1_index, pre_range_counts = _compute_pre_range_length(close, lookback=lookback)
            sub["pre_range_length"] = _map_pre_range_length(
                sub["entry_time"],
                m1_index=m1_index,
                pre_range_counts=pre_range_counts,
            )
            mapped_parts.append(sub)

        trades = pd.concat(mapped_parts, ignore_index=True) if mapped_parts else trades.iloc[0:0].copy()
        n_pre_range_valid = int(trades["pre_range_length"].notna().sum()) if not trades.empty else 0
        n_pre_range_missing = int(len(trades) - n_pre_range_valid)

        trades["early_loss"] = (trades["holding_time_min"] >= 0.0) & (trades["holding_time_min"] <= EARLY_LOSS_MAX_MIN)

        period_frames[p.period] = trades
        period_meta[p.period] = {
            "trades_csv": str(p.trades_csv),
            "core_config_json": str(p.core_config_json),
            "n_trades_raw": n_raw,
            "n_time_valid": n_time_valid,
            "n_side_valid": n_side_valid,
            "n_pre_range_valid": n_pre_range_valid,
            "n_pre_range_missing": n_pre_range_missing,
        }

    verify_key = next((k for k in period_frames if k == "in_sample_2024"), None)
    if verify_key is None:
        raise ValueError("Verify period (in_sample_2024) not found.")

    verify_df = period_frames[verify_key]
    if verify_df.empty or verify_df["pre_range_length"].dropna().empty:
        raise ValueError("Verify trades missing pre_range_length; cannot compute thresholds.")

    q33 = float(verify_df["pre_range_length"].dropna().quantile(0.33))
    q66 = float(verify_df["pre_range_length"].dropna().quantile(0.66))

    for key, df in period_frames.items():
        if df.empty:
            continue
        values = df["pre_range_length"].to_numpy(dtype=float)
        df["range_group"] = _classify_ranges(values, q33=q33, q66=q66)
        period_frames[key] = df

    summaries = {}
    for key, df in period_frames.items():
        period_label = period_map.get(key, key)
        valid = df[df["range_group"].isin(RANGE_ORDER)].copy() if not df.empty else df
        summaries[key] = _summarize(valid, period_label=period_label)

    out_dir.mkdir(parents=True, exist_ok=True)
    verify_summary = summaries.get("in_sample_2024")
    forward_summary = summaries.get("forward_2025")

    if verify_summary is not None:
        verify_summary.to_csv(out_dir / "summary_verify.csv", index=False)
    if forward_summary is not None:
        forward_summary.to_csv(out_dir / "summary_forward.csv", index=False)

    thresholds_payload = {
        "lookback": lookback,
        "early_loss_definition": f"holding_time_min in [0, {EARLY_LOSS_MAX_MIN}] (D007)",
        "pre_range_length_definition": "close stays within previous N close range; consecutive count",
        "q33": q33,
        "q66": q66,
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
            _plot_holding_hist(df, period_label=period_label, side=side, out_dir=plots_dir)

    if verify_summary is not None:
        for side in ("all", "buy", "sell"):
            _plot_early_loss_rate(verify_summary, period_label="verify", side=side, out_dir=plots_dir)
    if forward_summary is not None:
        for side in ("all", "buy", "sell"):
            _plot_early_loss_rate(forward_summary, period_label="forward", side=side, out_dir=plots_dir)

    _write_readme(
        out_dir,
        meta=period_meta,
        thresholds={"q33": q33, "q66": q66},
        lookback=lookback,
        d009_dir=d009_dir,
        data_root=data_root,
    )

    print(f"[D012a] done: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
