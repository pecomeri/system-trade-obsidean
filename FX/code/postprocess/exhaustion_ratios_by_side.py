#!/usr/bin/env python3
"""
Postprocess (observation-only):
Compute exhaustion ratios for D009 (buy/sell) and compare early_loss vs survivor.

Definitions (fixed; no optimization):
  holding_time_min = (exit_time - entry_time).total_seconds()/60
  early_loss: 0 <= holding_time_min <= 3
  survivor: holding_time_min >= 20

Ratios:
  body = abs(close - open)
  break_margin (price):
    - buy:  close - prev_high
    - sell: prev_low - close
  break_margin_ratio = break_margin / body           (body==0 -> NaN)
  burst_strength     = body / mean_prev_body         (mean_prev_body==0 -> NaN)
  break_margin_over_mean_prev_body = break_margin / mean_prev_body  (mean_prev_body==0 -> NaN)

Output (overwritten):
  - {hyp_dir}/diagnostics/exhaustion_ratios_by_side.csv
  - {hyp_dir}/diagnostics/exhaustion_ratios_summary.md

Example:
  uv run python FX/code/postprocess/exhaustion_ratios_by_side.py \
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
    trades_csv: Path
    core_config_json: Path


GROUP_EARLY = "early_loss"
GROUP_SURV = "survivor"


def _label_group(holding_min: float) -> str:
    if 0.0 <= holding_min <= 3.0:
        return GROUP_EARLY
    if holding_min >= 20.0:
        return GROUP_SURV
    return "other"


def _read_period_inputs(hyp_dir: Path) -> list[PeriodInputs]:
    out: list[PeriodInputs] = []
    for name in ("in_sample_2024", "forward_2025"):
        d = hyp_dir / name
        if not d.exists():
            continue
        trades = d / "trades.csv"
        core_cfg = d / "core_config.json"
        if trades.exists() and core_cfg.exists():
            out.append(PeriodInputs(period=name, trades_csv=trades, core_config_json=core_cfg))
    if not out:
        raise FileNotFoundError(f"No in_sample_2024/forward_2025 trades under: {hyp_dir}")
    return out


def _read_core_cfg(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


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
    # ratios (computed later per-trade due to side)
    return out


def _build_trade_rows(*, period: str, trades_csv: Path, core_cfg_path: Path):
    import numpy as np
    import pandas as pd

    core_cfg = _read_core_cfg(core_cfg_path)
    lookback = int(core_cfg["lookback_bars"])

    trades = pd.read_csv(trades_csv)
    required = {"trade_id", "side", "entry_time", "exit_time", "pnl_pips"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"{trades_csv} missing columns: {sorted(missing)}")

    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce")
    trades = trades.dropna(subset=["entry_time", "exit_time"])
    holding_min = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 60.0
    trades["holding_time_min"] = holding_min
    trades = trades[(trades["holding_time_min"] >= 0) & trades["holding_time_min"].notna()].copy()

    trades["group"] = trades["holding_time_min"].astype(float).apply(_label_group)
    trades = trades[trades["group"].isin([GROUP_EARLY, GROUP_SURV])].copy()
    if trades.empty:
        return pd.DataFrame(), {"trades_csv": str(trades_csv), "core_config_json": str(core_cfg_path), "lookback_bars": lookback}

    trades["side"] = trades["side"].astype(str).str.lower()
    trades["period"] = period

    # Load M1 and compute per-bar prerequisites
    m1 = _load_m1_ohlc_from_10s_bid(core_cfg=core_cfg)
    trig = _compute_trigger_series(m1=m1, lookback_bars=lookback)

    # Approximate mapping (same as D008/D009 feature scripts):
    trades["signal_eval_ts"] = trades["entry_time"] - pd.Timedelta(seconds=10)
    trades["burst_m1_end"] = trades["signal_eval_ts"].dt.floor("min") - pd.Timedelta(minutes=1)

    trig_reset = trig.reset_index()
    trig_reset = trig_reset.rename(columns={str(trig_reset.columns[0]): "burst_m1_end"})
    merged = trades.merge(trig_reset, on="burst_m1_end", how="left")

    # Compute ratios (price-based; consistent units)
    side = merged["side"].astype(str)
    close = merged["close"].astype(float)
    open_ = merged["open"].astype(float)
    prev_high = merged["prev_high"].astype(float)
    prev_low = merged["prev_low"].astype(float)
    body = (close - open_).abs().astype(float)
    mean_prev_body = merged["mean_prev_body"].astype(float)

    break_margin = np.where(side == "buy", close - prev_high, prev_low - close).astype(float)

    merged["body"] = body
    merged["break_margin_price"] = break_margin

    merged["break_margin_ratio"] = np.where(body > 0, break_margin / body, np.nan)
    merged["burst_strength"] = np.where(mean_prev_body > 0, body / mean_prev_body, np.nan)
    merged["break_margin_over_mean_prev_body"] = np.where(mean_prev_body > 0, break_margin / mean_prev_body, np.nan)

    keep = [
        "period",
        "side",
        "group",
        "trade_id",
        "entry_time",
        "exit_time",
        "holding_time_min",
        "burst_m1_end",
        "break_margin_ratio",
        "burst_strength",
        "break_margin_over_mean_prev_body",
    ]
    return merged.loc[:, [c for c in keep if c in merged.columns]].copy(), {
        "trades_csv": str(trades_csv),
        "core_config_json": str(core_cfg_path),
        "lookback_bars": lookback,
    }


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


def _build_long_stats(trades_df):
    import pandas as pd

    features = ["break_margin_ratio", "burst_strength", "break_margin_over_mean_prev_body"]
    rows = []
    for (period, side, group), g in trades_df.groupby(["period", "side", "group"]):
        for feat in features:
            if feat not in g.columns:
                stats = _describe(pd.Series(dtype=float))
            else:
                stats = _describe(g[feat])
            rows.append({"period": period, "side": side, "group": group, "feature": feat, **stats})
    return pd.DataFrame(rows)


def _delta_table(long_df):
    import pandas as pd

    piv = long_df.pivot_table(index=["period", "side", "feature"], columns="group", values=["n", "median"], aggfunc="first").reset_index()
    # flatten columns
    flat = []
    for c in piv.columns:
        if isinstance(c, tuple):
            a, b = c
            flat.append(str(a) if b == "" else f"{a}_{b}")
        else:
            flat.append(str(c))
    piv.columns = flat
    for g in (GROUP_EARLY, GROUP_SURV):
        for k in ("n", "median"):
            col = f"{k}_{g}"
            if col not in piv.columns:
                piv[col] = float("nan")
    piv["delta_median"] = piv[f"median_{GROUP_SURV}"] - piv[f"median_{GROUP_EARLY}"]
    return piv


def _sign_match(delta_verify: float, delta_forward: float) -> str:
    if delta_verify is None or delta_forward is None:
        return "判定不可"
    if math.isnan(delta_verify) or math.isnan(delta_forward):
        return "判定不可"
    if delta_verify == 0 and delta_forward == 0:
        return "一致"
    return "一致" if ((delta_verify > 0) == (delta_forward > 0)) else "不一致"


def _write_md(path: Path, *, hyp_dir: Path, meta: dict, long_df, delta_df) -> None:
    lines: list[str] = []
    lines.append("# exhaustion ratios（by side）")
    lines.append("")
    lines.append("## 入力")
    for period, info in meta["periods"].items():
        lines.append(f"- {period}: trades_csv=`{info['trades_csv']}` core_config=`{info['core_config_json']}` (lookback_bars={info['lookback_bars']})")
    lines.append("")
    lines.append("## ratio定義")
    lines.append("- body = abs(close - open)")
    lines.append("- break_margin:")
    lines.append("  - BUY: close - prev_high")
    lines.append("  - SELL: prev_low - close")
    lines.append("- break_margin_ratio = break_margin / body（body==0 -> NaN）")
    lines.append("- burst_strength = body / mean_prev_body（mean_prev_body==0 -> NaN）")
    lines.append("- break_margin_over_mean_prev_body = break_margin / mean_prev_body（mean_prev_body==0 -> NaN）")
    lines.append("")
    lines.append("## group定義（固定）")
    lines.append("- holding_time_min = (exit_time - entry_time).total_seconds()/60")
    lines.append("- early_loss: 0 <= holding_time_min <= 3")
    lines.append("- survivor: holding_time_min >= 20")
    lines.append("")

    if delta_df.empty:
        lines.append("## 集計結果")
        lines.append("- （データなし）")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    # side別 top3 by |delta_median| on forward (fallback verify)
    rank_period = "forward_2025" if (delta_df["period"] == "forward_2025").any() else "in_sample_2024"
    other_period = "in_sample_2024" if rank_period == "forward_2025" and (delta_df["period"] == "in_sample_2024").any() else None

    lines.append(f"## side別 top3（|delta_median|が大きい順, rank_period={rank_period}）")
    for side, g in delta_df[delta_df["period"] == rank_period].groupby("side"):
        g2 = g.copy()
        g2 = g2[(g2[f"n_{GROUP_EARLY}"] > 0) & (g2[f"n_{GROUP_SURV}"] > 0)]
        g2["abs_delta"] = g2["delta_median"].abs()
        top = g2.sort_values("abs_delta", ascending=False).head(3)
        lines.append(f"### {side}")
        for _, r in top.iterrows():
            feat = r["feature"]
            dv = float(r["delta_median"])
            dfw = dv
            dver = math.nan
            match = "判定不可"
            if other_period is not None:
                rr = delta_df[(delta_df["period"] == other_period) & (delta_df["side"] == side) & (delta_df["feature"] == feat)]
                if len(rr) == 1:
                    dver = float(rr.iloc[0]["delta_median"])
                    match = _sign_match(dver, dfw)
            lines.append(f"- {feat}: delta_median({rank_period})={dfw:.4f}" + (f" delta_median({other_period})={dver:.4f} sign_match={match}" if other_period else ""))
    lines.append("")

    # sign_match lists
    if other_period is not None:
        lines.append("## verify/forwardで符号一致するfeature一覧（side別）")
        for side in sorted(delta_df["side"].unique()):
            items = []
            for feat in sorted(delta_df["feature"].unique()):
                dv_row = delta_df[(delta_df["period"] == other_period) & (delta_df["side"] == side) & (delta_df["feature"] == feat)]
                df_row = delta_df[(delta_df["period"] == rank_period) & (delta_df["side"] == side) & (delta_df["feature"] == feat)]
                if len(dv_row) != 1 or len(df_row) != 1:
                    continue
                dv = float(dv_row.iloc[0]["delta_median"])
                dfw = float(df_row.iloc[0]["delta_median"])
                sm = _sign_match(dv, dfw)
                if sm == "一致":
                    items.append(f"{feat}({dfw:+.4f}/{dv:+.4f})")
            lines.append(f"- {side}: {', '.join(items) if items else '（なし）'}")
        lines.append("")

    # missing ratios
    lines.append("## 出せなかったratio")
    present = set(long_df["feature"].unique()) if not long_df.empty else set()
    expected = {"break_margin_ratio", "burst_strength", "break_margin_over_mean_prev_body"}
    missing = sorted(expected - present)
    lines.append(f"- missing_features={missing if missing else '（なし）'}")
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
    all_rows = []
    for pi in periods:
        df, info = _build_trade_rows(period=pi.period, trades_csv=pi.trades_csv, core_cfg_path=pi.core_config_json)
        meta["periods"][pi.period] = info
        if df is not None and not df.empty:
            all_rows.append(df)

    trades_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    long_df = _build_long_stats(trades_df) if not trades_df.empty else pd.DataFrame()
    delta_df = _delta_table(long_df) if not long_df.empty else pd.DataFrame()

    out_dir = hyp_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "exhaustion_ratios_by_side.csv"
    out_md = out_dir / "exhaustion_ratios_summary.md"

    long_df.to_csv(out_csv, index=False)
    _write_md(out_md, hyp_dir=hyp_dir, meta=meta, long_df=long_df, delta_df=delta_df)

    print(f"[exhaustion_ratios] wrote: {out_csv}", flush=True)
    print(f"[exhaustion_ratios] wrote: {out_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

