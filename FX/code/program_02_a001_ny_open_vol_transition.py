#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_core as core


K_PRE_BARS = 30
K_POST_BARS = 30
ATR_PERIOD = 14


@dataclass(frozen=True)
class Period:
    label: str
    from_month: str
    to_month: str


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _resample_m1(df10: pd.DataFrame) -> pd.DataFrame:
    o = df10["open_bid"].resample("1min").first()
    h = df10["high_bid"].resample("1min").max()
    l = df10["low_bid"].resample("1min").min()
    c = df10["close_bid"].resample("1min").last()
    m1 = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna()
    return m1


def _classify_state(value: float, q_low: float, q_high: float) -> str:
    if value <= q_low:
        return "low"
    if value <= q_high:
        return "mid"
    return "high"


def _transition_type(pre_state: str, post_state: str) -> str:
    if pre_state == post_state:
        return "stay"
    order = {"low": 0, "mid": 1, "high": 2}
    return "up" if order[post_state] > order[pre_state] else "down"


def _has_continuous_minutes(idx: pd.Index) -> bool:
    if len(idx) < 2:
        return False
    diffs = idx.to_series().diff().dropna()
    return bool((diffs == pd.Timedelta(minutes=1)).all())


def _extract_events(
    atr: pd.Series,
    *,
    t0_time: pd.Timedelta,
    q_low: float,
    q_high: float,
) -> pd.DataFrame:
    one_min = pd.Timedelta(minutes=1)
    events: list[dict[str, str]] = []

    for day in atr.index.normalize().unique():
        t0 = day + t0_time
        if t0 not in atr.index:
            continue
        pre = atr.loc[: t0 - one_min].tail(K_PRE_BARS)
        post = atr.loc[t0:].head(K_POST_BARS)
        if len(pre) != K_PRE_BARS or len(post) != K_POST_BARS:
            continue
        if not _has_continuous_minutes(pre.index) or not _has_continuous_minutes(post.index):
            continue
        if pre.isna().any() or post.isna().any():
            continue

        pre_mean = float(pre.mean())
        post_mean = float(post.mean())
        pre_state = _classify_state(pre_mean, q_low, q_high)
        post_state = _classify_state(post_mean, q_low, q_high)
        events.append(
            {
                "pre_state": pre_state,
                "post_state": post_state,
                "transition_type": _transition_type(pre_state, post_state),
            }
        )

    return pd.DataFrame(events, columns=["pre_state", "post_state", "transition_type"])


def _transition_summary(events: pd.DataFrame) -> dict[str, object]:
    total = int(len(events))
    type_counts = events["transition_type"].value_counts().reindex(["stay", "up", "down"], fill_value=0)
    matrix = (
        events.groupby(["pre_state", "post_state"])
        .size()
        .reindex(
            pd.MultiIndex.from_product(
                [["low", "mid", "high"], ["low", "mid", "high"]],
                names=["pre_state", "post_state"],
            ),
            fill_value=0,
        )
        .unstack("post_state")
    )
    return {
        "total": total,
        "type_counts": type_counts,
        "matrix": matrix,
    }


def _sign_from_counts(counts: pd.Series) -> str:
    up = int(counts.get("up", 0))
    down = int(counts.get("down", 0))
    if up > down:
        return "+"
    if up < down:
        return "-"
    return "0"


def _sign_by_pre_state(events: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for state in ["low", "mid", "high"]:
        subset = events[events["pre_state"] == state]
        if subset.empty:
            out[state] = "NA"
            continue
        counts = subset["transition_type"].value_counts()
        out[state] = _sign_from_counts(counts)
    return out


def _write_summary(
    path: Path,
    *,
    verify: dict[str, object],
    forward: dict[str, object],
    verify_signs: dict[str, str],
    forward_signs: dict[str, str],
    overall_verify: str,
    overall_forward: str,
    thresholds: tuple[float, float],
    ny_open_time: str,
) -> None:
    q_low, q_high = thresholds
    lines = [
        "# program_02 / A001 NY open volatility state transitions",
        "",
        "## Fixed parameters",
        f"- NY open (UTC): {ny_open_time}",
        f"- K_pre (bars): {K_PRE_BARS}",
        f"- K_post (bars): {K_POST_BARS}",
        f"- ATR period: {ATR_PERIOD}",
        f"- State thresholds (verify ATR tertiles): q33={q_low:.6f}, q66={q_high:.6f}",
        "",
        "## Verify summary",
        f"- total_events: {verify['total']}",
        f"- transition_type_counts: {verify['type_counts'].to_dict()}",
        "",
        "### pre_state -> post_state matrix (verify)",
        verify["matrix"].to_string(),
        "",
        "## Forward summary",
        f"- total_events: {forward['total']}",
        f"- transition_type_counts: {forward['type_counts'].to_dict()}",
        "",
        "### pre_state -> post_state matrix (forward)",
        forward["matrix"].to_string(),
        "",
        "## Sign consistency (up vs down)",
        f"- overall: verify={overall_verify} / forward={overall_forward}",
        f"- by pre_state: verify={verify_signs} / forward={forward_signs}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_m1_atr(cfg: core.Config) -> pd.Series:
    df10 = core.load_parquet_10s_bid(cfg)
    df10 = core.add_synthetic_bidask(df10, cfg)
    m1 = _resample_m1(df10)
    atr = _atr(m1, ATR_PERIOD)
    return atr


def main() -> None:
    parser = argparse.ArgumentParser(description="program_02 A001 NY open volatility state transition observer")
    parser.add_argument("--symbol", default="USDJPY")
    parser.add_argument("--root", default="FX/code/dukas_out_v2")
    parser.add_argument("--verify-from", default="2024-01")
    parser.add_argument("--verify-to", default="2024-12")
    parser.add_argument("--forward-from", default="2025-01")
    parser.add_argument("--forward-to", default="2025-12")
    parser.add_argument("--output-dir", default="results/program_02_open_volatilyty_state/family_A")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = core.Config(root=Path(args.root), symbol=args.symbol)
    ny_open_time = core._parse_hhmm(base_cfg.w2_start)
    ny_open_delta = pd.Timedelta(hours=ny_open_time.hour, minutes=ny_open_time.minute)

    verify_cfg = core.Config(
        root=Path(args.root),
        symbol=args.symbol,
        from_month=args.verify_from,
        to_month=args.verify_to,
    )
    forward_cfg = core.Config(
        root=Path(args.root),
        symbol=args.symbol,
        from_month=args.forward_from,
        to_month=args.forward_to,
    )

    atr_verify = _load_m1_atr(verify_cfg).dropna()
    if atr_verify.empty:
        raise ValueError("No ATR values in verify period.")

    q_low, q_high = atr_verify.quantile([1 / 3, 2 / 3]).to_list()
    verify_events = _extract_events(
        atr_verify,
        t0_time=ny_open_delta,
        q_low=q_low,
        q_high=q_high,
    )

    atr_forward = _load_m1_atr(forward_cfg).dropna()
    if atr_forward.empty:
        raise ValueError("No ATR values in forward period.")
    forward_events = _extract_events(
        atr_forward,
        t0_time=ny_open_delta,
        q_low=q_low,
        q_high=q_high,
    )

    verify_events.to_csv(out_dir / "events_verify.csv", index=False)
    forward_events.to_csv(out_dir / "events_forward.csv", index=False)

    verify_summary = _transition_summary(verify_events)
    forward_summary = _transition_summary(forward_events)
    verify_signs = _sign_by_pre_state(verify_events)
    forward_signs = _sign_by_pre_state(forward_events)
    overall_verify = _sign_from_counts(verify_summary["type_counts"])
    overall_forward = _sign_from_counts(forward_summary["type_counts"])

    _write_summary(
        out_dir / "summary.md",
        verify=verify_summary,
        forward=forward_summary,
        verify_signs=verify_signs,
        forward_signs=forward_signs,
        overall_verify=overall_verify,
        overall_forward=overall_forward,
        thresholds=(float(q_low), float(q_high)),
        ny_open_time=base_cfg.w2_start,
    )

    print(f"[ok] outputs: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
