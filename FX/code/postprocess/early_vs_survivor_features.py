#!/usr/bin/env python3
"""
Postprocess: compare trigger features between early-loss (0–3 min) vs survivor (>=20 min).

Scope:
  - Aggregation only (no strategy logic changes).
  - Uses D004 trades.csv when diag_trades.csv is not available.
  - Minimal feature recomputation from M1 OHLC built from Dukascopy 10s bid parquet.

Group definitions (fixed):
  - early_loss: holding_time_min in [0, 3]
  - survivor: holding_time_min >= 20

Outputs (overwritten):
  - {hyp_dir}/diagnostics/early_vs_survivor_features.csv
  - {hyp_dir}/diagnostics/early_vs_survivor_summary.md

Example:
  uv run python FX/code/postprocess/early_vs_survivor_features.py \
    --hyp_dir FX/results/family_D_momentum/D004
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PeriodInputs:
    period: str
    period_dir: Path
    trades_csv: Path
    core_config_json: Path


FEATURES_NUMERIC = [
    "body",
    "mean_prev_body",
    "burst_strength",
    "body_ratio",
    "break_margin_pips",
    "next_m1_body_ratio",
]

FEATURES_BOOLEAN = [
    "next_m1_is_opposite_body",
]

# Ensure we can import `FX/code/backtest_core.py` as `backtest_core` when running from repo root.
_FX_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_FX_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_FX_CODE_DIR))


def _read_core_cfg(core_cfg_path: Path) -> dict:
    import json

    return json.loads(core_cfg_path.read_text(encoding="utf-8"))


def _load_m1_ohlc_from_10s_bid(*, core_cfg: dict):
    import pandas as pd
    import backtest_core as bc

    cfg = bc.Config(
        root=Path(core_cfg["root"]),
        symbol=str(core_cfg["symbol"]).upper(),
        from_month=str(core_cfg["from_month"]),
        to_month=str(core_cfg["to_month"]),
        run_tag="postprocess_m1",
        only_session=None,
        spread_pips=float(core_cfg.get("spread_pips", 1.0)),
        pip_size=float(core_cfg["pip_size"]),
    )

    df_bid = bc.load_parquet_10s_bid(cfg)
    df10 = bc.add_synthetic_bidask(df_bid, cfg)

    o = df10["open_bid"].resample("1min", label="right", closed="right").first()
    h = df10["high_bid"].resample("1min", label="right", closed="right").max()
    l = df10["low_bid"].resample("1min", label="right", closed="right").min()
    c = df10["close_bid"].resample("1min", label="right", closed="right").last()
    m1 = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna()
    return m1


def _compute_trigger_series(*, m1, lookback_bars: int):
    import numpy as np
    import pandas as pd

    body = (m1["close"] - m1["open"]).abs()
    rng = (m1["high"] - m1["low"]).astype(float)
    body_ratio = (body / rng.replace(0.0, np.nan)).astype(float)

    mean_prev_body = body.shift(1).rolling(int(lookback_bars)).mean()
    burst_strength = (body / mean_prev_body.replace(0.0, np.nan)).astype(float)

    prev_high = m1["high"].shift(1).astype(float)
    prev_low = m1["low"].shift(1).astype(float)

    out = pd.DataFrame(
        {
            "m1_open": m1["open"].astype(float),
            "m1_high": m1["high"].astype(float),
            "m1_low": m1["low"].astype(float),
            "m1_close": m1["close"].astype(float),
            "body": body.astype(float),
            "range": rng.astype(float),
            "body_ratio": body_ratio.astype(float),
            "mean_prev_body": mean_prev_body.astype(float),
            "burst_strength": burst_strength.astype(float),
            "prev_high": prev_high,
            "prev_low": prev_low,
        },
        index=m1.index,
    )
    return out


def _read_period_inputs(hyp_dir: Path) -> list[PeriodInputs]:
    periods: list[PeriodInputs] = []
    for name in ("in_sample_2024", "forward_2025"):
        d = hyp_dir / name
        if not d.exists():
            continue
        trades_csv = d / "trades.csv"
        core_cfg = d / "core_config.json"
        if not trades_csv.exists() or not core_cfg.exists():
            continue
        periods.append(PeriodInputs(period=name, period_dir=d, trades_csv=trades_csv, core_config_json=core_cfg))
    if not periods:
        raise FileNotFoundError(f"Expected in_sample_2024/forward_2025 under {hyp_dir}")
    return periods


def _label_group(holding_min: float) -> str:
    if 0.0 <= holding_min <= 3.0:
        return "early_loss"
    if holding_min >= 20.0:
        return "survivor"
    return "other"


def _build_trade_feature_rows(*, period: str, trades_csv: Path, core_cfg_path: Path):
    import numpy as np
    import pandas as pd

    core_cfg = _read_core_cfg(core_cfg_path)
    pip_size = float(core_cfg["pip_size"])
    lookback = int(core_cfg["lookback_bars"])

    trades = pd.read_csv(trades_csv)
    required = {"trade_id", "side", "entry_time", "exit_time", "pnl_pips"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"{trades_csv} missing columns: {sorted(missing)}")

    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce")
    trades = trades.dropna(subset=["entry_time", "exit_time"])
    trades["pnl_pips"] = pd.to_numeric(trades["pnl_pips"], errors="coerce")
    trades = trades.dropna(subset=["pnl_pips"])

    holding_sec = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds()
    trades["holding_time_min"] = holding_sec / 60.0
    trades = trades[(trades["holding_time_min"] >= 0) & trades["holding_time_min"].notna()].copy()

    trades["group"] = trades["holding_time_min"].astype(float).apply(_label_group)
    trades = trades[trades["group"].isin(["early_loss", "survivor"])].copy()
    if trades.empty:
        return pd.DataFrame(), {"core_config_json": str(core_cfg_path), "trades_csv": str(trades_csv)}

    # Load M1 OHLC for the period and compute trigger-derived series (body, mean_prev_body, prev_high/low, etc.)
    m1 = _load_m1_ohlc_from_10s_bid(core_cfg=core_cfg)
    trig = _compute_trigger_series(m1=m1, lookback_bars=lookback)

    # Map each trade to the trigger M1 candle end timestamp.
    # Rationale (core behavior):
    # - entry_time is ts_next when entry_on_next_open=True (10s next open).
    # - signal is evaluated on ts (10s bar) and comes from M1 signals shifted(1) and ffilled to 10s.
    # - Therefore, the trigger M1 close timestamp is approximately:
    #     burst_m1_end = floor_minute(signal_eval_ts) - 1 minute
    #   where signal_eval_ts ~= entry_time - 10 seconds.
    trades["signal_eval_ts"] = trades["entry_time"] - pd.Timedelta(seconds=10)
    trades["burst_m1_end"] = trades["signal_eval_ts"].dt.floor("min") - pd.Timedelta(minutes=1)
    trades["next_m1_end"] = trades["burst_m1_end"] + pd.Timedelta(minutes=1)

    # Join trigger candle features
    trig_reset = trig.reset_index()
    idx_col = str(trig_reset.columns[0])
    trig_reset = trig_reset.rename(columns={idx_col: "burst_m1_end"})
    merged = trades.merge(
        trig_reset,
        on="burst_m1_end",
        how="left",
    )
    # Join next candle OHLC for "immediate negation" features
    next_ohlc = trig[["m1_open", "m1_close", "body_ratio"]].reset_index()
    nidx_col = str(next_ohlc.columns[0])
    next_ohlc = next_ohlc.rename(
        columns={
            nidx_col: "next_m1_end",
            "m1_open": "next_m1_open",
            "m1_close": "next_m1_close",
            "body_ratio": "next_m1_body_ratio",
        }
    )
    merged = merged.merge(next_ohlc, on="next_m1_end", how="left")

    side = merged["side"].astype(str).str.lower()
    close = merged["m1_close"].astype(float)
    prev_high = merged["prev_high"].astype(float)
    prev_low = merged["prev_low"].astype(float)

    break_margin = np.where(side == "buy", close - prev_high, prev_low - close)
    merged["break_margin_pips"] = break_margin / pip_size

    # Next candle opposite-body (only if next candle exists)
    next_open = merged["next_m1_open"].astype(float)
    next_close = merged["next_m1_close"].astype(float)
    merged["next_m1_is_opposite_body"] = np.where(
        side == "buy",
        (next_close < next_open),
        (next_close > next_open),
    )

    merged["period"] = period

    keep_cols = (
        [
            "period",
            "trade_id",
            "side",
            "entry_time",
            "exit_time",
            "pnl_pips",
            "holding_time_min",
            "group",
            "signal_eval_ts",
            "burst_m1_end",
        ]
        + FEATURES_NUMERIC
        + FEATURES_BOOLEAN
    )
    out = merged.loc[:, [c for c in keep_cols if c in merged.columns]].copy()
    return out, {"core_config_json": str(core_cfg_path), "trades_csv": str(trades_csv), "pip_size": pip_size, "lookback_bars": lookback}


def _describe_feature(series):
    import numpy as np

    s = series.dropna().astype(float)
    if len(s) == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "p25": np.nan, "p75": np.nan}
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
    }


def _build_feature_stats(features_df):
    import pandas as pd

    rows = []
    for period, grp_period in features_df.groupby("period"):
        for feature in FEATURES_NUMERIC + FEATURES_BOOLEAN:
            for group in ("early_loss", "survivor"):
                sub = grp_period[grp_period["group"] == group]
                if feature not in sub.columns:
                    stats = {"n": 0, "mean": float("nan"), "median": float("nan"), "p25": float("nan"), "p75": float("nan")}
                else:
                    # bool -> float rate
                    s = sub[feature]
                    if feature in FEATURES_BOOLEAN:
                        s = s.astype("float")
                    stats = _describe_feature(s)
                rows.append({"period": period, "feature": feature, "group": group, **stats})

    long = pd.DataFrame(rows)

    # wide comparison per (period, feature)
    wide = long.pivot_table(
        index=["period", "feature"],
        columns="group",
        values=["n", "mean", "median", "p25", "p75"],
        aggfunc="first",
    ).reset_index()

    # flatten columns safely
    flat_cols = []
    for c in wide.columns:
        if isinstance(c, tuple):
            a, b = c
            if b == "":
                flat_cols.append(str(a))
            else:
                flat_cols.append(f"{a}_{b}")
        else:
            flat_cols.append(str(c))
    wide.columns = flat_cols

    # ensure expected columns exist
    for g in ("early_loss", "survivor"):
        for k in ("n", "mean", "median", "p25", "p75"):
            col = f"{k}_{g}"
            if col not in wide.columns:
                wide[col] = float("nan")

    wide["delta_mean_survivor_minus_early"] = wide["mean_survivor"] - wide["mean_early_loss"]
    wide["delta_median_survivor_minus_early"] = wide["median_survivor"] - wide["median_early_loss"]
    return long, wide


def _write_summary_md(out_path: Path, *, meta: dict, features_df, wide_df) -> None:
    import math

    lines: list[str] = []
    lines.append("# early_loss vs survivor（特徴量比較）")
    lines.append("")
    lines.append("## 前提")
    lines.append("- グループ定義（固定）:")
    lines.append("  - early_loss: holding_time_min ∈ [0, 3]")
    lines.append("  - survivor: holding_time_min ≥ 20")
    lines.append("- holding_time_min = (exit_time - entry_time).total_seconds()/60.0")
    lines.append("")
    lines.append("## 入力")
    for period, info in meta["periods"].items():
        lines.append(f"- {period}: trades_csv=`{info['trades_csv']}` core_config=`{info['core_config_json']}` (lookback_bars={info['lookback_bars']} pip_size={info['pip_size']})")
    lines.append("")
    lines.append("## トリガー時刻のマッピング（diag_tradesが無い場合の近似）")
    lines.append("- signal_eval_ts ~= entry_time - 10秒（coreの entry_on_next_open=ts_next に基づく）")
    lines.append("- burst_m1_end = floor_minute(signal_eval_ts) - 1分（M1シグナルの shift(1) + 10sへのffill を前提）")
    lines.append("")

    if features_df.empty:
        lines.append("## 集計結果")
        lines.append("- （early_loss/survivor に該当するトレードが無い）")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append("## 件数")
    for period, g in features_df.groupby("period"):
        c = g["group"].value_counts().to_dict()
        lines.append(f"- {period}: early_loss={int(c.get('early_loss', 0))} survivor={int(c.get('survivor', 0))}")
    lines.append("")

    lines.append("## 差が大きい特徴量（|delta_median| 上位、事実のみ）")
    for period, g in wide_df.groupby("period"):
        g2 = g.copy()
        g2["abs_delta_median"] = (g2["delta_median_survivor_minus_early"]).abs()
        top = g2.sort_values("abs_delta_median", ascending=False).head(6)
        lines.append(f"### {period}")

        def fmt(x: float) -> str:
            try:
                if x is None or (isinstance(x, float) and math.isnan(x)):
                    return "nan"
            except Exception:  # noqa: BLE001
                return "nan"
            return f"{float(x):.4f}"

        for _, r in top.iterrows():
            feat = r["feature"]
            lines.append(
                f"- {feat}: "
                f"median_early={fmt(r.get('median_early_loss'))} median_survivor={fmt(r.get('median_survivor'))} "
                f"delta_median(survivor-early)={fmt(r.get('delta_median_survivor_minus_early'))}"
            )
    lines.append("")

    lines.append("## 出せなかった特徴量")
    missing = []
    for f in FEATURES_NUMERIC + FEATURES_BOOLEAN:
        if f not in features_df.columns:
            missing.append(f)
    if not missing:
        lines.append("- （なし）")
    else:
        lines.append(f"- missing_cols_in_output={missing}")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hyp_dir", required=True, help="e.g. FX/results/family_D_momentum/D004")
    args = p.parse_args()

    hyp_dir = Path(args.hyp_dir)
    periods = _read_period_inputs(hyp_dir)

    all_rows = []
    meta = {"periods": {}}
    for pi in periods:
        df, info = _build_trade_feature_rows(period=pi.period, trades_csv=pi.trades_csv, core_cfg_path=pi.core_config_json)
        meta["periods"][pi.period] = info
        if df is not None and len(df) > 0:
            all_rows.append(df)

    import pandas as pd

    features_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    _, wide = _build_feature_stats(features_df) if not features_df.empty else (pd.DataFrame(), pd.DataFrame())

    out_dir = hyp_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "early_vs_survivor_features.csv"
    out_md = out_dir / "early_vs_survivor_summary.md"
    wide.to_csv(out_csv, index=False)
    _write_summary_md(out_md, meta=meta, features_df=features_df, wide_df=wide)

    print(f"[early_vs_survivor_features] wrote: {out_csv}", flush=True)
    print(f"[early_vs_survivor_features] wrote: {out_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
