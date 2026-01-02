#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_core as core


EARLY_LOSS_MIN = 3.0
LABEL_WINDOW_MIN = 3
IMMEDIATE_DIR_WINDOW_MIN = 3


@dataclass(frozen=True)
class RunSpec:
    label: str
    from_month: str
    to_month: str
    run_dir: Path


def _load_d009_config() -> dict:
    path = Path("FX/results/family_D_momentum/D009/config.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_core_config() -> dict:
    path = Path("FX/results/family_D_momentum/D009/core_config.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _atr_m1(df10: pd.DataFrame, period: int) -> pd.Series:
    o = df10["open_bid"].resample("1min").first()
    h = df10["high_bid"].resample("1min").max()
    l = df10["low_bid"].resample("1min").min()
    c = df10["close_bid"].resample("1min").last()
    m1 = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna()

    high = m1["high"]
    low = m1["low"]
    close = m1["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _early_loss_flag(holding_secs: float) -> bool:
    return (holding_secs / 60.0) <= EARLY_LOSS_MIN


def _immediate_directional_alignment(
    df_win: pd.DataFrame,
    *,
    side: str,
    entry_price: float,
    sl_first_if_both: bool,
) -> str:
    if df_win.empty:
        return "flat"

    for _, row in df_win.iterrows():
        if side == "buy":
            favorable = row["high_bid"] > entry_price
            unfavorable = row["low_bid"] < entry_price
        else:
            favorable = row["low_ask"] < entry_price
            unfavorable = row["high_ask"] > entry_price

        if favorable and unfavorable:
            return "counter" if sl_first_if_both else "align"
        if favorable:
            return "align"
        if unfavorable:
            return "counter"

    return "flat"


def _early_retrace_presence(
    df_win: pd.DataFrame,
    *,
    side: str,
    entry_price: float,
) -> str:
    if df_win.empty:
        return "no_retrace"
    if side == "buy":
        favorable = (df_win["high_bid"] > entry_price).any()
    else:
        favorable = (df_win["low_ask"] < entry_price).any()
    return "with_retrace" if favorable else "no_retrace"


def _initial_mae_stage(
    df_win: pd.DataFrame,
    *,
    side: str,
    entry_price: float,
    pip_size: float,
    stop_loss_pips: float,
) -> str:
    if df_win.empty:
        return "small"
    if side == "buy":
        adverse_pips = (entry_price - df_win["low_bid"].min()) / pip_size
    else:
        adverse_pips = (df_win["high_ask"].max() - entry_price) / pip_size

    t1 = stop_loss_pips / 3.0
    t2 = (2.0 * stop_loss_pips) / 3.0
    if adverse_pips <= t1:
        return "small"
    if adverse_pips <= t2:
        return "medium"
    return "large"


def _immediate_volatility_expansion(
    atr_m1: pd.Series,
    *,
    entry_time: pd.Timestamp,
) -> tuple[str, bool]:
    entry_min = entry_time.floor("min")
    pre_idx = [entry_min - pd.Timedelta(minutes=3), entry_min - pd.Timedelta(minutes=2), entry_min - pd.Timedelta(minutes=1)]
    post_idx = [entry_min, entry_min + pd.Timedelta(minutes=1), entry_min + pd.Timedelta(minutes=2)]

    pre = atr_m1.reindex(pre_idx)
    post = atr_m1.reindex(post_idx)
    if pre.isna().any() or post.isna().any():
        return "not_expanded", True

    return ("expanded", False) if float(post.mean()) > float(pre.mean()) else ("not_expanded", False)


def _time_to_failure_stage(holding_secs: float) -> str:
    mins = holding_secs / 60.0
    if mins <= 1.0:
        return "very_fast"
    if mins <= 2.0:
        return "fast"
    return "slow"


def _label_trades(
    trades: pd.DataFrame,
    df10: pd.DataFrame,
    atr_m1: pd.Series,
    *,
    pip_size: float,
    stop_loss_pips: float,
    sl_first_if_both: bool,
) -> tuple[pd.DataFrame, int]:
    labels = []
    missing_vol_count = 0
    window_delta = pd.Timedelta(minutes=LABEL_WINDOW_MIN)
    immediate_delta = pd.Timedelta(minutes=IMMEDIATE_DIR_WINDOW_MIN)

    for row in trades.itertuples(index=False):
        entry_time = pd.Timestamp(row.entry_time)
        if entry_time.tzinfo is None:
            entry_time = entry_time.tz_localize("UTC")
        side = str(row.side).lower()
        entry_price = float(row.entry_price)

        holding_secs = float(row.holding_secs) if hasattr(row, "holding_secs") else (pd.Timestamp(row.exit_time) - entry_time).total_seconds()
        early_loss = _early_loss_flag(holding_secs)

        win_start = entry_time
        win_end = entry_time + window_delta
        df_win = df10.loc[win_start:win_end]

        align = _immediate_directional_alignment(
            df10.loc[win_start: entry_time + immediate_delta],
            side=side,
            entry_price=entry_price,
            sl_first_if_both=sl_first_if_both,
        )
        mae_stage = _initial_mae_stage(
            df_win,
            side=side,
            entry_price=entry_price,
            pip_size=pip_size,
            stop_loss_pips=stop_loss_pips,
        )
        retrace = _early_retrace_presence(df_win, side=side, entry_price=entry_price)
        vol_expand, vol_missing = _immediate_volatility_expansion(atr_m1, entry_time=entry_time)
        if vol_missing:
            missing_vol_count += 1
        ttf_stage = _time_to_failure_stage(holding_secs)

        labels.append(
            {
                "trade_id": row.trade_id if hasattr(row, "trade_id") else None,
                "early_loss": early_loss,
                "immediate_directional_alignment": align,
                "initial_mae_stage": mae_stage,
                "early_retrace_presence": retrace,
                "immediate_volatility_expansion": vol_expand,
                "time_to_failure_stage": ttf_stage,
            }
        )

    return pd.DataFrame(labels), missing_vol_count


def _summary_by_label(df: pd.DataFrame, label_cols: list[str]) -> pd.DataFrame:
    rows = []
    for label in label_cols:
        for value, grp in df.groupby(label, dropna=False):
            total = len(grp)
            early = int(grp["early_loss"].sum())
            ratio = early / total if total else 0.0
            rows.append(
                {
                    "label": label,
                    "value": str(value),
                    "total_trades": total,
                    "early_loss_trades": early,
                    "early_loss_ratio": ratio,
                }
            )
    return pd.DataFrame(rows)


def _write_readme(
    out_dir: Path,
    *,
    pip_size: float,
    stop_loss_pips: float,
    atr_period: int,
    sl_first_if_both: bool,
    data_root: str,
    symbol: str,
    missing_vol_counts: dict[str, int],
) -> None:
    lines = [
        "# D014: Early Loss Classifiability (Price Action Labels)",
        "",
        "## Data source",
        f"- trades: results/family_D_momentum/D009/",
        f"- symbol: {symbol}",
        f"- data_root: {data_root}",
        "",
        "## Fixed definitions",
        f"- early_loss: holding_time_min <= {EARLY_LOSS_MIN}",
        f"- label window: entry_time to entry_time + {LABEL_WINDOW_MIN} minutes (10s bars)",
        f"- immediate_directional_alignment window: entry_time to entry_time + {IMMEDIATE_DIR_WINDOW_MIN} minutes",
        f"- ATR period (M1): {atr_period}",
        f"- stop_loss_pips (scale): {stop_loss_pips}",
        f"- pip_size: {pip_size}",
        f"- sl_first_if_both: {sl_first_if_both}",
        "",
        "## Label definitions",
        "- immediate_directional_alignment:",
        "  - align: favorable price update occurs first",
        "  - counter: unfavorable price update occurs first",
        "  - flat: no price update in window",
        "- initial_mae_stage:",
        "  - small: MAE <= 1/3 SL",
        "  - medium: 1/3 SL < MAE <= 2/3 SL",
        "  - large: MAE > 2/3 SL",
        "- early_retrace_presence:",
        "  - with_retrace: favorable price update occurs at least once in window",
        "  - no_retrace: otherwise",
        "- immediate_volatility_expansion:",
        "  - expanded: mean ATR(post 3 mins) > mean ATR(pre 3 mins)",
        "  - not_expanded: otherwise",
        "- time_to_failure_stage (entry->exit time):",
        "  - very_fast: <= 1 min",
        "  - fast: <= 2 min",
        "  - slow: > 2 min",
        "",
        "## Notes",
        "- No entry/exit/SL/TP/time/direction changes.",
        "- No filtering or optimization.",
        "- Summary files contain early_loss ratios only.",
        f"- immediate_volatility_expansion missing ATR windows: {missing_vol_counts}",
        "",
        "## Columns used",
        "- trades.csv: trade_id, side, entry_time, entry_price, exit_time, holding_secs",
        "- 10s bars: open_bid, high_bid, low_bid, close_bid, open_ask, high_ask, low_ask, close_ask",
    ]
    out_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cfg = _load_d009_config()
    core_cfg = _load_core_config()

    data_root = cfg["root"]
    symbol = cfg["symbol"]
    pip_size = float(core_cfg["pip_size"])
    stop_loss_pips = float(core_cfg["stop_loss_pips"])
    sl_first_if_both = bool(core_cfg["sl_first_if_both"])
    atr_period = 14

    specs = [
        RunSpec(
            label="verify",
            from_month=cfg["verify"]["from_month"],
            to_month=cfg["verify"]["to_month"],
            run_dir=Path(cfg["verify"]["run_dir"]),
        ),
        RunSpec(
            label="forward",
            from_month=cfg["forward"]["from_month"],
            to_month=cfg["forward"]["to_month"],
            run_dir=Path(cfg["forward"]["run_dir"]),
        ),
    ]

    out_dir = Path("results/family_D_momentum/D014")
    out_dir.mkdir(parents=True, exist_ok=True)
    missing_vol_counts: dict[str, int] = {}

    for spec in specs:
        cfg_bt = core.Config(
            root=Path(data_root),
            symbol=symbol,
            from_month=spec.from_month,
            to_month=spec.to_month,
        )
        df10 = core.load_parquet_10s_bid(cfg_bt)
        df10 = core.add_synthetic_bidask(df10, cfg_bt)
        atr_m1 = _atr_m1(df10, atr_period)

        trades = pd.read_csv(spec.run_dir / "trades.csv")
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)

        labeled, missing_vol_count = _label_trades(
            trades,
            df10,
            atr_m1,
            pip_size=pip_size,
            stop_loss_pips=stop_loss_pips,
            sl_first_if_both=sl_first_if_both,
        )

        summary = _summary_by_label(
            labeled,
            [
                "immediate_directional_alignment",
                "initial_mae_stage",
                "early_retrace_presence",
                "immediate_volatility_expansion",
                "time_to_failure_stage",
            ],
        )

        summary_path = out_dir / f"summary_{spec.label}.csv"
        summary.to_csv(summary_path, index=False)
        missing_vol_counts = missing_vol_counts | {spec.label: missing_vol_count}

    _write_readme(
        out_dir,
        pip_size=pip_size,
        stop_loss_pips=stop_loss_pips,
        atr_period=atr_period,
        sl_first_if_both=sl_first_if_both,
        data_root=data_root,
        symbol=symbol,
        missing_vol_counts=missing_vol_counts,
    )

    print(f"[ok] outputs: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
