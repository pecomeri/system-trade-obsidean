#!/usr/bin/env python3
"""
Postprocess (observation-only):
Split trades into early_loss (0–3min) vs survivor (>=20min) and compare trigger features,
additionally split by side (buy/sell).

This is a side-split extension of D008-style feature extraction, designed for D009.
It does not modify strategy logic, SL/TP, or trade generation.

Outputs (overwritten):
  - {hyp_dir}/diagnostics/early_vs_survivor_features_by_side.csv
  - {hyp_dir}/diagnostics/early_vs_survivor_summary_by_side.md

Example:
  uv run python FX/code/postprocess/early_vs_survivor_features_by_side.py \
    --hyp_dir FX/results/family_D_momentum/D009
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path


# Ensure `import backtest_core` works when executed from repo root.
_FX_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_FX_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_FX_CODE_DIR))


@dataclass(frozen=True)
class PeriodInputs:
    period: str
    period_dir: Path
    trades_csv: Path
    core_config_json: Path


GROUP_EARLY = "early_loss"
GROUP_SURV = "survivor"

FEATURES_NUMERIC = [
    "body",
    "mean_prev_body",
    "burst_strength",
    "body_ratio",
    "break_margin_pips",
    "next_m1_body_ratio",
]
FEATURES_BOOLEAN = [
    "next_m1_is_opposite_body",  # rate (0..1)
]


def _read_core_cfg(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _read_period_inputs(hyp_dir: Path) -> list[PeriodInputs]:
    periods: list[PeriodInputs] = []
    for name in ("in_sample_2024", "forward_2025"):
        d = hyp_dir / name
        if not d.exists():
            continue
        trades = d / "trades.csv"
        core_cfg = d / "core_config.json"
        if trades.exists() and core_cfg.exists():
            periods.append(PeriodInputs(period=name, period_dir=d, trades_csv=trades, core_config_json=core_cfg))
    if not periods:
        # Fall back: single-period usage (less preferred)
        trades = hyp_dir / "trades.csv"
        core_cfg = hyp_dir / "core_config.json"
        if trades.exists() and core_cfg.exists():
            periods.append(PeriodInputs(period="all", period_dir=hyp_dir, trades_csv=trades, core_config_json=core_cfg))
    if not periods:
        raise FileNotFoundError(f"No period trades.csv/core_config.json found under: {hyp_dir}")
    return periods


def _label_group(holding_min: float) -> str:
    if 0.0 <= holding_min <= 3.0:
        return GROUP_EARLY
    if holding_min >= 20.0:
        return GROUP_SURV
    return "other"


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
    trades = trades[trades["group"].isin([GROUP_EARLY, GROUP_SURV])].copy()
    if trades.empty:
        return pd.DataFrame(), {"core_config_json": str(core_cfg_path), "trades_csv": str(trades_csv), "pip_size": pip_size, "lookback_bars": lookback}

    trades["side"] = trades["side"].astype(str).str.lower()

    m1 = _load_m1_ohlc_from_10s_bid(core_cfg=core_cfg)
    trig = _compute_trigger_series(m1=m1, lookback_bars=lookback)

    # Approximate mapping (same as D008):
    trades["signal_eval_ts"] = trades["entry_time"] - pd.Timedelta(seconds=10)
    trades["burst_m1_end"] = trades["signal_eval_ts"].dt.floor("min") - pd.Timedelta(minutes=1)
    trades["next_m1_end"] = trades["burst_m1_end"] + pd.Timedelta(minutes=1)

    trig_reset = trig.reset_index()
    trig_reset = trig_reset.rename(columns={str(trig_reset.columns[0]): "burst_m1_end"})
    merged = trades.merge(trig_reset, on="burst_m1_end", how="left")

    next_ohlc = trig[["m1_open", "m1_close", "body_ratio"]].reset_index()
    next_ohlc = next_ohlc.rename(
        columns={
            str(next_ohlc.columns[0]): "next_m1_end",
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

    next_open = merged["next_m1_open"].astype(float)
    next_close = merged["next_m1_close"].astype(float)
    merged["next_m1_is_opposite_body"] = np.where(
        side == "buy",
        (next_close < next_open),
        (next_close > next_open),
    )

    merged["period"] = period
    return merged, {"core_config_json": str(core_cfg_path), "trades_csv": str(trades_csv), "pip_size": pip_size, "lookback_bars": lookback}


def _describe(series):
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


def _build_long_stats(df):
    import pandas as pd

    rows = []
    for (period, side, group), g in df.groupby(["period", "side", "group"]):
        for feat in FEATURES_NUMERIC:
            rows.append({"period": period, "side": side, "group": group, "feature": feat, **_describe(g[feat] if feat in g.columns else pd.Series(dtype=float))})
        for feat in FEATURES_BOOLEAN:
            # bool -> rate
            s = g[feat].astype("float") if feat in g.columns else pd.Series(dtype=float)
            rows.append({"period": period, "side": side, "group": group, "feature": feat, **_describe(s)})

    out = pd.DataFrame(rows)
    return out


def _delta_table(long_df):
    import pandas as pd

    if long_df.empty:
        return pd.DataFrame()
    med = long_df.pivot_table(index=["period", "side", "feature"], columns="group", values=["n", "median"], aggfunc="first").reset_index()

    # flatten columns
    flat_cols = []
    for c in med.columns:
        if isinstance(c, tuple):
            a, b = c
            if b == "":
                flat_cols.append(str(a))
            else:
                flat_cols.append(f"{a}_{b}")
        else:
            flat_cols.append(str(c))
    med.columns = flat_cols

    # expected
    for g in (GROUP_EARLY, GROUP_SURV):
        for k in ("n", "median"):
            col = f"{k}_{g}"
            if col not in med.columns:
                med[col] = float("nan")

    med["delta_median"] = med[f"median_{GROUP_SURV}"] - med[f"median_{GROUP_EARLY}"]
    return med


def _write_md(path: Path, *, hyp_dir: Path, meta: dict, long_df, delta_df):
    lines: list[str] = []
    lines.append("# early_loss vs survivor（by side）")
    lines.append("")
    lines.append("## 前提")
    lines.append("- holding_time_min = (exit_time - entry_time).total_seconds()/60")
    lines.append(f"- early_loss: 0 <= holding_time_min <= 3 / survivor: holding_time_min >= 20")
    lines.append("- side: trades.csv の side 列（buy/sell）を使用")
    lines.append("")
    lines.append("## 入力")
    for period, info in meta["periods"].items():
        lines.append(f"- {period}: trades_csv=`{info['trades_csv']}` core_config=`{info['core_config_json']}` (lookback_bars={info['lookback_bars']} pip_size={info['pip_size']})")
    lines.append("")
    lines.append("## トリガー時刻のマッピング（diag_trades無しの近似）")
    lines.append("- signal_eval_ts ~= entry_time - 10秒（entry_on_next_open=ts_next に基づく）")
    lines.append("- burst_m1_end = floor_minute(signal_eval_ts) - 1分（M1シグナルの shift(1) + 10sへのffill を前提）")
    lines.append("")

    if long_df.empty or delta_df.empty:
        lines.append("## 集計結果")
        lines.append("- （early_loss/survivor に該当するデータが無い）")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    # group counts by period/side
    lines.append("## 件数（period×side×group）")
    counts = long_df[long_df["feature"] == FEATURES_NUMERIC[0]][["period", "side", "group", "n"]].copy()
    for (period, side), g in counts.groupby(["period", "side"]):
        m = {r["group"]: int(r["n"]) for _, r in g.iterrows()}
        lines.append(f"- {period} {side}: early_loss={m.get(GROUP_EARLY, 0)} survivor={m.get(GROUP_SURV, 0)}")
    lines.append("")

    lines.append("## 差が大きい特徴量（side別 top3 / delta_median = median_survivor - median_early）")
    # Choose ranking based on forward_2025 if present, else in_sample_2024.
    rank_period = "forward_2025" if (delta_df["period"] == "forward_2025").any() else delta_df["period"].iloc[0]
    other_period = "in_sample_2024" if rank_period == "forward_2025" and (delta_df["period"] == "in_sample_2024").any() else None

    def fmt(x):
        try:
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return "nan"
        except Exception:  # noqa: BLE001
            return "nan"
        return f"{float(x):.4f}"

    for side, gside in delta_df[delta_df["period"] == rank_period].groupby("side"):
        g2 = gside.copy()
        g2["abs_delta"] = g2["delta_median"].abs()
        # require both groups to have >=1
        g2 = g2[(g2[f"n_{GROUP_EARLY}"] > 0) & (g2[f"n_{GROUP_SURV}"] > 0)]
        top = g2.sort_values("abs_delta", ascending=False).head(3)
        lines.append(f"### {side} (rank_period={rank_period})")
        for _, r in top.iterrows():
            feat = r["feature"]
            d_fwd = float(r["delta_median"])
            sign_match = "n/a"
            d_other = None
            if other_period is not None:
                r2 = delta_df[(delta_df["period"] == other_period) & (delta_df["side"] == side) & (delta_df["feature"] == feat)]
                if len(r2) == 1:
                    d_other = float(r2.iloc[0]["delta_median"])
                    sign_match = "match" if (d_other == 0 and d_fwd == 0) or ((d_other > 0) == (d_fwd > 0)) else "mismatch"
            lines.append(
                f"- {feat}: delta_median({rank_period})={fmt(d_fwd)}"
                + (f" delta_median({other_period})={fmt(d_other)} sign={sign_match}" if other_period is not None else "")
            )
    lines.append("")

    lines.append("## 備考")
    lines.append("- このMDは集計結果の事実のみ（解釈・提案は含めない）")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hyp_dir", required=True)
    args = p.parse_args()

    hyp_dir = Path(args.hyp_dir)
    periods = _read_period_inputs(hyp_dir)

    import pandas as pd

    meta = {"periods": {}}
    all_trades = []
    for pi in periods:
        df, info = _build_trade_feature_rows(period=pi.period, trades_csv=pi.trades_csv, core_cfg_path=pi.core_config_json)
        meta["periods"][pi.period] = info
        if df is not None and not df.empty:
            all_trades.append(df)

    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if trades_df.empty:
        long_df = pd.DataFrame()
        delta_df = pd.DataFrame()
    else:
        long_df = _build_long_stats(trades_df)
        delta_df = _delta_table(long_df)

    out_dir = hyp_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "early_vs_survivor_features_by_side.csv"
    out_md = out_dir / "early_vs_survivor_summary_by_side.md"

    long_df.to_csv(out_csv, index=False)
    _write_md(out_md, hyp_dir=hyp_dir, meta=meta, long_df=long_df, delta_df=delta_df)

    print(f"[early_vs_survivor_by_side] wrote: {out_csv}", flush=True)
    print(f"[early_vs_survivor_by_side] wrote: {out_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

