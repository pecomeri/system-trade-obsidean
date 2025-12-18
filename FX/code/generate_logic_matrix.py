#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class HypRow:
    family: str
    hyp: str
    delta: str
    entry_mode: str
    bias_mode: str
    use_h1_trend_filter: bool | None
    use_time_filter: bool | None
    only_session: str | None
    disable_m1_bias_filter: bool
    max_losing_streak: int | None


def _effective(flag: bool | None, default: bool) -> bool:
    return default if flag is None else bool(flag)


def _entry_impl(entry_mode: str) -> str:
    if entry_mode == "A_10s_breakout":
        return "core.high_breakout_10s: close_bid > rolling(prev_high, lookback_bars)"
    if entry_mode == "B_failed_breakout":
        return "runner.patch high_breakout_10s: down-break (close<prev_low) then next-bar negate -> long"
    if entry_mode == "C_m1_close":
        return "runner.patch high_breakout_10s: M1 close > rolling(prev_M1_high, lookback_bars)"
    return f"UNKNOWN({entry_mode})"


def _bias_impl(bias_mode: str, disable_m1_bias_filter: bool) -> str:
    if disable_m1_bias_filter:
        return "runner.patch compute_bias_htf: bias=1 for all bars (filter disabled)"
    if bias_mode == "default":
        return "core.compute_bias_htf: HTF close vs rolling recent_high/recent_low -> +1/-1/0 (shifted)"
    if bias_mode == "m1_candle":
        return "runner.patch compute_bias_htf: last closed M1 close>open -> 1 else 0 (shifted)"
    return f"UNKNOWN({bias_mode})"


def _core_gates_note() -> str:
    # Keep this stable & short; detail lives in core.
    return "core entry gate ≈ time_ok & (bias==1) & h1_ok & breakout_hi; long-only by default"


def _build_rows() -> list[HypRow]:
    # A-family variants (CLI mapping in backtest_runner.py)
    rows: list[HypRow] = [
        HypRow(
            family="family_A",
            hyp="A001",
            delta="baseline",
            entry_mode="A_10s_breakout",
            bias_mode="default",
            use_h1_trend_filter=None,
            use_time_filter=None,
            only_session=None,
            disable_m1_bias_filter=False,
            max_losing_streak=None,
        ),
        HypRow(
            family="family_A",
            hyp="A002",
            delta="only_session=W2",
            entry_mode="A_10s_breakout",
            bias_mode="default",
            use_h1_trend_filter=None,
            use_time_filter=None,
            only_session="W2",
            disable_m1_bias_filter=False,
            max_losing_streak=None,
        ),
        HypRow(
            family="family_A",
            hyp="A003",
            delta="use_h1_trend_filter=False",
            entry_mode="A_10s_breakout",
            bias_mode="default",
            use_h1_trend_filter=False,
            use_time_filter=None,
            only_session=None,
            disable_m1_bias_filter=False,
            max_losing_streak=None,
        ),
        HypRow(
            family="family_A",
            hyp="A004",
            delta="disable_m1_bias_filter=True",
            entry_mode="A_10s_breakout",
            bias_mode="default",
            use_h1_trend_filter=None,
            use_time_filter=None,
            only_session=None,
            disable_m1_bias_filter=True,
            max_losing_streak=None,
        ),
        HypRow(
            family="family_A",
            hyp="A005",
            delta="use_time_filter=False",
            entry_mode="A_10s_breakout",
            bias_mode="default",
            use_h1_trend_filter=None,
            use_time_filter=False,
            only_session=None,
            disable_m1_bias_filter=False,
            max_losing_streak=None,
        ),
    ]

    # family_B_failedbreakout (runner presets)
    for hyp, delta, only_sess, use_h1, use_time, max_ls in [
        ("B001", "baseline", None, None, None, None),
        ("B002", "only_session=W1", "W1", None, None, None),
        ("B003", "use_h1_trend_filter=False", None, False, None, None),
        ("B004", "max_losing_streak=999", None, None, None, 999),
        ("B005", "use_time_filter=False", None, None, False, None),
    ]:
        rows.append(
            HypRow(
                family="family_B_failedbreakout",
                hyp=hyp,
                delta=delta,
                entry_mode="B_failed_breakout",
                bias_mode="default",
                use_h1_trend_filter=use_h1,
                use_time_filter=use_time,
                only_session=only_sess,
                disable_m1_bias_filter=False,
                max_losing_streak=max_ls,
            )
        )

    # family_C_m1entry (runner presets)
    for hyp, delta, only_sess, use_h1, use_time, max_ls in [
        ("C001", "baseline", None, None, None, None),
        ("C002", "only_session=W1", "W1", None, None, None),
        ("C003", "use_h1_trend_filter=False", None, False, None, None),
        ("C004", "max_losing_streak=999", None, None, None, 999),
        ("C005", "use_time_filter=False", None, None, False, None),
    ]:
        rows.append(
            HypRow(
                family="family_C_m1entry",
                hyp=hyp,
                delta=delta,
                entry_mode="C_m1_close",
                bias_mode="m1_candle",
                use_h1_trend_filter=use_h1,
                use_time_filter=use_time,
                only_session=only_sess,
                disable_m1_bias_filter=False,
                max_losing_streak=max_ls,
            )
        )

    # family_C_m1entry_v2 (runner suite family_C_v2)
    for hyp, delta, use_h1, use_time in [
        ("C101", "baseline (W1 only)", None, None),
        ("C102", "use_h1_trend_filter=False", False, None),
        ("C103", "use_time_filter=False", None, False),
    ]:
        rows.append(
            HypRow(
                family="family_C_m1entry_v2",
                hyp=hyp,
                delta=delta,
                entry_mode="C_m1_close",
                bias_mode="m1_candle",
                use_h1_trend_filter=use_h1,
                use_time_filter=use_time,
                only_session="W1",
                disable_m1_bias_filter=False,
                max_losing_streak=None,
            )
        )

    return rows


def _to_markdown_table(df: pd.DataFrame) -> str:
    """
    Minimal markdown table writer (avoids optional dependency `tabulate`).
    """
    cols = list(df.columns)
    rows = df.astype(str).values.tolist()

    def esc(s: str) -> str:
        return s.replace("\n", " ").replace("|", "\\|")

    widths = [len(c) for c in cols]
    for r in rows:
        for i, v in enumerate(r):
            widths[i] = max(widths[i], len(esc(v)))

    header = "| " + " | ".join(esc(c).ljust(widths[i]) for i, c in enumerate(cols)) + " |"
    sep = "| " + " | ".join(("-" * widths[i]) for i in range(len(cols))) + " |"
    body_lines = [
        "| " + " | ".join(esc(v).ljust(widths[i]) for i, v in enumerate(r)) + " |" for r in rows
    ]
    return "\n".join([header, sep, *body_lines])


def main() -> int:
    out_dir = Path("FX/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _build_rows()

    # core defaults (documented; not re-importing core to avoid side effects / path issues)
    default_use_h1 = True
    default_use_time = True
    default_max_losing_streak = 2

    table_rows: list[dict] = []
    for r in rows:
        eff_h1 = _effective(r.use_h1_trend_filter, default_use_h1)
        eff_time = _effective(r.use_time_filter, default_use_time)
        eff_max_ls = default_max_losing_streak if r.max_losing_streak is None else int(r.max_losing_streak)

        notes: list[str] = [_core_gates_note()]
        if r.only_session is not None:
            notes.append("note: only_session gate still applies even if use_time_filter=False (session_label based)")
        if r.family == "family_C_m1entry_v2" and r.hyp == "C101":
            notes.append("note: C101 is config-identical to C002 (legacy) in current runner")

        table_rows.append(
            {
                "family": r.family,
                "hyp": r.hyp,
                "delta": r.delta,
                "entry_mode": r.entry_mode,
                "entry_impl": _entry_impl(r.entry_mode),
                "bias_mode": r.bias_mode,
                "bias_impl": _bias_impl(r.bias_mode, r.disable_m1_bias_filter),
                "use_h1_trend_filter": eff_h1,
                "use_time_filter": eff_time,
                "only_session": r.only_session,
                "max_losing_streak": eff_max_ls,
                "notes": " | ".join(notes),
            }
        )

    df = pd.DataFrame(table_rows).sort_values(["family", "hyp"]).reset_index(drop=True)

    csv_path = out_dir / "logic_matrix_ABC.csv"
    df.to_csv(csv_path, index=False)

    # A compact markdown table (keep it readable in Obsidian)
    md_cols = [
        "family",
        "hyp",
        "delta",
        "entry_mode",
        "bias_mode",
        "only_session",
        "use_h1_trend_filter",
        "use_time_filter",
        "max_losing_streak",
    ]
    md_df = df[md_cols].copy()
    md_path = out_dir / "logic_matrix_ABC.md"
    md_path.write_text(_to_markdown_table(md_df) + "\n", encoding="utf-8")

    meta_path = out_dir / "logic_matrix_ABC.meta.json"
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "backtest_core": "FX/code/backtest_core.py",
            "backtest_runner": "FX/code/backtest_runner.py",
        },
        "outputs": {
            "csv": str(csv_path),
            "md": str(md_path),
        },
        "rows": len(df),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[ok] wrote: {csv_path}")
    print(f"[ok] wrote: {md_path}")
    print(f"[ok] wrote: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
