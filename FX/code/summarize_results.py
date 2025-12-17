#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Row:
    run_id: str
    experiment_id: str
    period: str
    trade_count: int | None
    session_pnl_range: float | None
    session_pnl_std: float | None
    symbol: str | None
    only_session: str | None
    use_time_filter: bool | None
    use_h1_trend_filter: bool | None
    run_tag: str | None
    config_path: str
    group_key: str


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        rows = list(r)
    if not rows:
        return 0
    return max(0, len(rows) - 1)


def _session_stats(path: Path) -> tuple[float | None, float | None]:
    """
    Computes per-session sum_pnl_pips across rows and returns (range, stddev).
    """
    if not path.exists():
        return None, None
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        session_to_sum: dict[str, float] = {}
        for row in r:
            s = str(row.get("session", "")).strip() or "NA"
            try:
                v = float(row.get("sum_pnl_pips", "") or 0.0)
            except Exception:
                v = 0.0
            session_to_sum[s] = session_to_sum.get(s, 0.0) + v

    vals = list(session_to_sum.values())
    if not vals:
        return 0.0, 0.0
    vmin = min(vals)
    vmax = max(vals)
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    std = math.sqrt(var)
    return float(vmax - vmin), float(std)


def _infer_experiment_id(run_id: str) -> str:
    # convention: prefix before first "_" is the experiment tag (e.g., b002, hyp001)
    return run_id.split("_", 1)[0] if "_" in run_id else run_id


def _infer_period(cfg: dict[str, Any] | None) -> str:
    if not cfg:
        return ""
    fm = cfg.get("from_month")
    tm = cfg.get("to_month")
    if fm or tm:
        return f"{fm or ''}..{tm or ''}".strip(".")
    dbg = cfg.get("debug_day")
    return f"debug:{dbg}" if dbg else ""


def _group_key(experiment_id: str, period: str, only_session: str | None) -> str:
    sess = only_session or ""
    return f"{experiment_id}|{period}|{sess}"


def scan_results(results_root: Path) -> list[Row]:
    rows: list[Row] = []

    # Walk directories that contain config.json (run folders)
    for cfg_path in sorted(results_root.rglob("config.json")):
        # Ignore family-level config.json (family/hyp outputs) by requiring sibling trades.csv
        run_dir = cfg_path.parent
        trades_path = run_dir / "trades.csv"
        if not trades_path.exists():
            continue

        run_id = str(run_dir.relative_to(results_root)).replace(os.sep, "/")
        cfg = _safe_read_json(cfg_path) or {}
        experiment_id = _infer_experiment_id(run_dir.name)
        period = _infer_period(cfg)

        trade_count = _count_csv_rows(trades_path)
        rng, std = _session_stats(run_dir / "monthly_by_session.csv")

        symbol = cfg.get("symbol")
        only_session = cfg.get("only_session")
        use_time_filter = cfg.get("use_time_filter")
        use_h1_trend_filter = cfg.get("use_h1_trend_filter")
        run_tag = cfg.get("run_tag")

        rows.append(
            Row(
                run_id=run_id,
                experiment_id=str(experiment_id),
                period=str(period),
                trade_count=int(trade_count) if trade_count is not None else None,
                session_pnl_range=float(rng) if rng is not None else None,
                session_pnl_std=float(std) if std is not None else None,
                symbol=str(symbol) if symbol is not None else None,
                only_session=str(only_session) if only_session is not None else None,
                use_time_filter=bool(use_time_filter) if isinstance(use_time_filter, bool) else None,
                use_h1_trend_filter=bool(use_h1_trend_filter) if isinstance(use_h1_trend_filter, bool) else None,
                run_tag=str(run_tag) if run_tag is not None else None,
                config_path=str(cfg_path),
                group_key=_group_key(str(experiment_id), str(period), str(only_session) if only_session is not None else None),
            )
        )

    return rows


def write_summary(rows: list[Row], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run_id",
                "experiment_id",
                "period",
                "trade_count",
                "session_pnl_range",
                "session_pnl_std",
                "symbol",
                "only_session",
                "use_time_filter",
                "use_h1_trend_filter",
                "run_tag",
                "group_key",
                "config_path",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.run_id,
                    r.experiment_id,
                    r.period,
                    "" if r.trade_count is None else r.trade_count,
                    "" if r.session_pnl_range is None else f"{r.session_pnl_range:.6f}",
                    "" if r.session_pnl_std is None else f"{r.session_pnl_std:.6f}",
                    r.symbol or "",
                    r.only_session or "",
                    "" if r.use_time_filter is None else str(int(r.use_time_filter)),
                    "" if r.use_h1_trend_filter is None else str(int(r.use_h1_trend_filter)),
                    r.run_tag or "",
                    r.group_key,
                    r.config_path,
                ]
            )


def main() -> int:
    # Prefer FX/results
    results_root = Path("FX/results") if Path("FX/results").exists() else Path("results")
    if not results_root.exists():
        raise SystemExit(f"results root not found: {results_root}")

    out_path = results_root / "_summary.csv"
    rows = scan_results(results_root)
    write_summary(rows, out_path)
    print(f"[summarize_results] wrote: {out_path} rows={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

