#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class HypConfig:
    family: str
    hyp: str
    symbol: str
    root: str
    only_session: str | None
    verify_from_month: str
    verify_to_month: str
    forward_from_month: str
    forward_to_month: str
    spread_pips: float = 1.0
    use_h1_trend_filter: bool | None = None
    use_time_filter: bool | None = None
    disable_m1_bias_filter: bool = False
    entry_mode: str = "A_10s_breakout"  # "A_10s_breakout" | "B_failed_breakout" | "C_m1_close"
    bias_mode: str = "default"          # "default" | "m1_candle"
    max_losing_streak: int | None = None


def _run_core(cmd: list[str]) -> Path:
    print("[runner] exec:", " ".join(cmd), flush=True)

    run_dir: Path | None = None
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        if "[run] output dir:" in line:
            _, rhs = line.split("[run] output dir:", 1)
            run_dir = Path(rhs.strip())
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"backtest_core failed (code={proc.returncode})")
    if run_dir is None:
        raise RuntimeError("Could not detect run_dir from backtest_core output.")
    return run_dir


@contextlib.contextmanager
def _temporary_attr(obj, name: str, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def _run_core_inprocess(cfg: HypConfig, *, from_month: str, to_month: str, run_tag: str) -> Path:
    import numpy as np

    import backtest_core as bc

    core_cfg = bc.Config(
        root=Path(cfg.root),
        symbol=str(cfg.symbol).upper(),
        from_month=str(from_month),
        to_month=str(to_month),
        only_session=cfg.only_session,
        spread_pips=float(cfg.spread_pips),
        run_tag=str(run_tag),
    )
    if cfg.use_h1_trend_filter is not None:
        core_cfg = replace(core_cfg, use_h1_trend_filter=bool(cfg.use_h1_trend_filter))
    if cfg.use_time_filter is not None:
        core_cfg = replace(core_cfg, use_time_filter=bool(cfg.use_time_filter))
    if cfg.max_losing_streak is not None:
        core_cfg = replace(core_cfg, max_losing_streak=int(cfg.max_losing_streak))

    patches: list[contextlib.AbstractContextManager] = []
    if cfg.disable_m1_bias_filter:
        def _bias_all_one(df10, _cfg):  # noqa: ANN001
            return np.ones(len(df10), dtype=np.int8)
        patches.append(_temporary_attr(bc, "compute_bias_htf", _bias_all_one))

    if cfg.bias_mode == "m1_candle":
        def _bias_m1_candle(df10, _cfg):  # noqa: ANN001
            o = df10["open_bid"].resample("1min", label="right", closed="right").first()
            c = df10["close_bid"].resample("1min", label="right", closed="right").last()
            bias_m1 = (c > o).astype(np.int8)
            # use last closed M1
            bias_closed = bias_m1.shift(1).fillna(0).astype(np.int8)
            return bias_closed.reindex(df10.index, method="ffill").fillna(0).to_numpy(dtype=np.int8)
        patches.append(_temporary_attr(bc, "compute_bias_htf", _bias_m1_candle))

    if cfg.entry_mode == "B_failed_breakout":
        def _failed_breakout_long(df10, core_cfg):  # noqa: ANN001
            prev_low = df10["low_bid"].shift(1).rolling(core_cfg.lookback_bars).min()
            breakout_down = (df10["close_bid"] < prev_low)
            # failure = next bar closes back above the breakout reference (2-step structure)
            fail = breakout_down.shift(1) & (df10["close_bid"] > prev_low.shift(1))
            return fail.fillna(False).to_numpy(dtype=bool)
        patches.append(_temporary_attr(bc, "high_breakout_10s", _failed_breakout_long))

    if cfg.entry_mode == "C_m1_close":
        def _m1_close_breakout(df10, core_cfg):  # noqa: ANN001
            hi = df10["high_bid"].resample("1min", label="right", closed="right").max()
            cl = df10["close_bid"].resample("1min", label="right", closed="right").last()
            prev_hi = hi.shift(1).rolling(core_cfg.lookback_bars).max()
            trig = (cl > prev_hi).fillna(False)
            # trigger only after the M1 candle is closed (no lookahead)
            trig_closed = trig.astype(bool)
            return trig_closed.reindex(df10.index, method="ffill").fillna(False).to_numpy(dtype=bool)
        patches.append(_temporary_attr(bc, "high_breakout_10s", _m1_close_breakout))

    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)

        print("=== backtest started (runner/inprocess) ===", flush=True)
        print(
            f"[cfg] hyp={cfg.hyp} range={core_cfg.from_month}..{core_cfg.to_month} "
            f"only_session={core_cfg.only_session} use_time_filter={core_cfg.use_time_filter} "
            f"use_h1_trend_filter={core_cfg.use_h1_trend_filter} entry_mode={cfg.entry_mode} bias_mode={cfg.bias_mode} "
            f"max_losing_streak={core_cfg.max_losing_streak}",
            flush=True,
        )
        print(f"[cfg] input parquet base: {bc.parquet_base(core_cfg)}", flush=True)

        files = bc.list_parquet_files(core_cfg)
        print(f"[debug] parquet files found (filtered): {len(files):,}", flush=True)
        if len(files) == 0:
            raise FileNotFoundError("No parquet files found for the given range.")

        run_dir = bc.make_run_dir(core_cfg)
        print(f"[run] output dir: {run_dir}", flush=True)

        (run_dir / "config.json").write_text(
            json.dumps({**asdict(core_cfg), "root": str(core_cfg.root), "results_root": str(core_cfg.results_root)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        runlog_path = run_dir / "runlog.jsonl"

        df_bid = bc.load_parquet_10s_bid(core_cfg)
        df10 = bc.add_synthetic_bidask(df_bid, core_cfg)

        if core_cfg.debug_day and core_cfg.dump_debug:
            pth = bc.dump_debug_signals(df10, core_cfg, run_dir)
            print(f"[done] debug_signals.csv written: {pth}", flush=True)

        trades, monthly, monthly_session = bc.backtest(df10, core_cfg, runlog_path, run_dir)
        trades.to_csv(run_dir / "trades.csv", index=False)
        monthly.to_csv(run_dir / "monthly.csv", index=False)
        monthly_session.to_csv(run_dir / "monthly_by_session.csv", index=False)

        print(f"[done] trades: {len(trades):,}", flush=True)
        bc.sanity_check(run_dir)
        return run_dir


def _copy_artifacts(src_run_dir: Path, dst_dir: Path, *, keep_config_as: str = "config.json") -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ("monthly.csv", "monthly_by_session.csv", "config.json"):
        src = src_run_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing artifact: {src}")
        dst_name = keep_config_as if name == "config.json" else name
        shutil.copy2(src, dst_dir / dst_name)


def _sum_pnl_pips(monthly_csv: Path) -> float:
    import csv

    with monthly_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        vals = [float(row["sum_pnl_pips"]) for row in r if row.get("sum_pnl_pips")]
    return float(sum(vals))


def _sum_trades(monthly_csv: Path) -> int:
    import csv

    with monthly_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        vals = [int(float(row["trades"])) for row in r if row.get("trades")]
    return int(sum(vals))


def _run_variant_inprocess(cfg: HypConfig, *, run_tag: str) -> dict:
    out_root = Path("results") / cfg.family / cfg.hyp
    out_root.mkdir(parents=True, exist_ok=True)

    # 2024 verification
    verify_run_dir = _run_core_inprocess(cfg, from_month=cfg.verify_from_month, to_month=cfg.verify_to_month, run_tag=run_tag)
    verify_out = out_root / "in_sample_2024"
    _copy_artifacts(verify_run_dir, verify_out, keep_config_as="core_config.json")

    # 2025 forward
    forward_run_dir = _run_core_inprocess(cfg, from_month=cfg.forward_from_month, to_month=cfg.forward_to_month, run_tag=run_tag)
    forward_out = out_root / "forward_2025"
    _copy_artifacts(forward_run_dir, forward_out, keep_config_as="core_config.json")

    # Required outputs at hyp root: forward artifacts + runner config.json
    _copy_artifacts(forward_run_dir, out_root, keep_config_as="core_config.json")
    shutil.copy2(forward_out / "monthly.csv", out_root / "monthly.csv")
    shutil.copy2(forward_out / "monthly_by_session.csv", out_root / "monthly_by_session.csv")

    meta = {
        "family": cfg.family,
        "hyp": cfg.hyp,
        "symbol": cfg.symbol,
        "root": cfg.root,
        "spread_pips": cfg.spread_pips,
        "verify": {
            "from_month": cfg.verify_from_month,
            "to_month": cfg.verify_to_month,
            "run_dir": str(verify_run_dir),
            "sum_pnl_pips": _sum_pnl_pips(verify_out / "monthly.csv"),
            "trades": _sum_trades(verify_out / "monthly.csv"),
        },
        "forward": {
            "from_month": cfg.forward_from_month,
            "to_month": cfg.forward_to_month,
            "run_dir": str(forward_run_dir),
            "sum_pnl_pips": _sum_pnl_pips(forward_out / "monthly.csv"),
            "trades": _sum_trades(forward_out / "monthly.csv"),
        },
        "diff": {
            "only_session": cfg.only_session,
            "use_h1_trend_filter": cfg.use_h1_trend_filter,
            "use_time_filter": cfg.use_time_filter,
            "disable_m1_bias_filter": cfg.disable_m1_bias_filter,
        },
    }
    (out_root / "config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def run_a002_w2_only(cfg: HypConfig) -> dict:
    core = Path(__file__).with_name("backtest_core.py")
    if not core.exists():
        raise FileNotFoundError(f"backtest_core.py not found: {core}")

    out_root = Path("results") / cfg.family / cfg.hyp
    out_root.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    common = [py, str(core), "--root", cfg.root, "--symbol", cfg.symbol]
    if cfg.only_session:
        common += ["--only_session", cfg.only_session]
    common += ["--spread_pips", str(cfg.spread_pips)]

    # 2024 verification
    verify_cmd = common + ["--from_month", cfg.verify_from_month, "--to_month", cfg.verify_to_month]
    verify_run_dir = _run_core(verify_cmd)
    verify_out = out_root / "in_sample_2024"
    _copy_artifacts(verify_run_dir, verify_out, keep_config_as="core_config.json")

    # 2025 forward
    forward_cmd = common + ["--from_month", cfg.forward_from_month, "--to_month", cfg.forward_to_month]
    forward_run_dir = _run_core(forward_cmd)
    forward_out = out_root / "forward_2025"
    _copy_artifacts(forward_run_dir, forward_out, keep_config_as="core_config.json")

    # Required outputs at hyp root: forward artifacts + runner config.json
    _copy_artifacts(forward_run_dir, out_root, keep_config_as="core_config.json")
    shutil.copy2(forward_out / "monthly.csv", out_root / "monthly.csv")
    shutil.copy2(forward_out / "monthly_by_session.csv", out_root / "monthly_by_session.csv")

    meta = {
        "family": cfg.family,
        "hyp": cfg.hyp,
        "symbol": cfg.symbol,
        "root": cfg.root,
        "spread_pips": cfg.spread_pips,
        "verify": {
            "from_month": cfg.verify_from_month,
            "to_month": cfg.verify_to_month,
            "run_dir": str(verify_run_dir),
            "pnl_sum_pips": _sum_pnl_pips(verify_out / "monthly.csv"),
        },
        "forward": {
            "from_month": cfg.forward_from_month,
            "to_month": cfg.forward_to_month,
            "run_dir": str(forward_run_dir),
            "pnl_sum_pips": _sum_pnl_pips(forward_out / "monthly.csv"),
        },
        "diff": {
            "only_session": cfg.only_session,
        },
    }
    (out_root / "config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def run_a001_baseline(cfg: HypConfig) -> dict:
    core = Path(__file__).with_name("backtest_core.py")
    if not core.exists():
        raise FileNotFoundError(f"backtest_core.py not found: {core}")

    out_root = Path("results") / cfg.family / cfg.hyp
    out_root.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    common = [py, str(core), "--root", cfg.root, "--symbol", cfg.symbol]
    if cfg.only_session:
        common += ["--only_session", cfg.only_session]
    common += ["--spread_pips", str(cfg.spread_pips)]

    verify_cmd = common + ["--from_month", cfg.verify_from_month, "--to_month", cfg.verify_to_month]
    verify_run_dir = _run_core(verify_cmd)
    verify_out = out_root / "in_sample_2024"
    _copy_artifacts(verify_run_dir, verify_out, keep_config_as="core_config.json")

    forward_cmd = common + ["--from_month", cfg.forward_from_month, "--to_month", cfg.forward_to_month]
    forward_run_dir = _run_core(forward_cmd)
    forward_out = out_root / "forward_2025"
    _copy_artifacts(forward_run_dir, forward_out, keep_config_as="core_config.json")

    _copy_artifacts(forward_run_dir, out_root, keep_config_as="core_config.json")
    shutil.copy2(forward_out / "monthly.csv", out_root / "monthly.csv")
    shutil.copy2(forward_out / "monthly_by_session.csv", out_root / "monthly_by_session.csv")

    meta = {
        "family": cfg.family,
        "hyp": cfg.hyp,
        "symbol": cfg.symbol,
        "root": cfg.root,
        "spread_pips": cfg.spread_pips,
        "verify": {
            "from_month": cfg.verify_from_month,
            "to_month": cfg.verify_to_month,
            "run_dir": str(verify_run_dir),
            "pnl_sum_pips": _sum_pnl_pips(verify_out / "monthly.csv"),
        },
        "forward": {
            "from_month": cfg.forward_from_month,
            "to_month": cfg.forward_to_month,
            "run_dir": str(forward_run_dir),
            "pnl_sum_pips": _sum_pnl_pips(forward_out / "monthly.csv"),
        },
        "diff": {
            "only_session": cfg.only_session,
        },
    }
    (out_root / "config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def run_family_a_variant(cfg: HypConfig) -> dict:
    out_root = Path("results") / cfg.family / cfg.hyp
    out_root.mkdir(parents=True, exist_ok=True)

    tag = f"hyp{cfg.hyp[1:]}"

    verify_run_dir = _run_core_inprocess(cfg, from_month=cfg.verify_from_month, to_month=cfg.verify_to_month, run_tag=tag)
    verify_out = out_root / "in_sample_2024"
    _copy_artifacts(verify_run_dir, verify_out, keep_config_as="core_config.json")

    forward_run_dir = _run_core_inprocess(cfg, from_month=cfg.forward_from_month, to_month=cfg.forward_to_month, run_tag=tag)
    forward_out = out_root / "forward_2025"
    _copy_artifacts(forward_run_dir, forward_out, keep_config_as="core_config.json")

    _copy_artifacts(forward_run_dir, out_root, keep_config_as="core_config.json")
    shutil.copy2(forward_out / "monthly.csv", out_root / "monthly.csv")
    shutil.copy2(forward_out / "monthly_by_session.csv", out_root / "monthly_by_session.csv")

    meta = {
        "family": cfg.family,
        "hyp": cfg.hyp,
        "symbol": cfg.symbol,
        "root": cfg.root,
        "spread_pips": cfg.spread_pips,
        "verify": {
            "from_month": cfg.verify_from_month,
            "to_month": cfg.verify_to_month,
            "run_dir": str(verify_run_dir),
            "pnl_sum_pips": _sum_pnl_pips(verify_out / "monthly.csv"),
        },
        "forward": {
            "from_month": cfg.forward_from_month,
            "to_month": cfg.forward_to_month,
            "run_dir": str(forward_run_dir),
            "pnl_sum_pips": _sum_pnl_pips(forward_out / "monthly.csv"),
        },
        "diff": {
            "only_session": cfg.only_session,
            "use_h1_trend_filter": cfg.use_h1_trend_filter,
            "use_time_filter": cfg.use_time_filter,
            "disable_m1_bias_filter": cfg.disable_m1_bias_filter,
        },
    }
    (out_root / "config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default=None, choices=[None, "family_BC", "family_BC_summary"])
    p.add_argument(
        "--hyp",
        default="A002",
        choices=[
            "A001", "A002", "A003", "A004", "A005",
            "B001", "B002", "B003", "B004", "B005", "B006", "B007", "B008",
            "C001", "C002", "C003", "C004", "C005",
        ],
    )
    p.add_argument("--symbol", default="USDJPY")
    p.add_argument("--root", default=None, help="Data root (auto-detect if omitted)")
    p.add_argument("--spread_pips", type=float, default=1.0)
    p.add_argument("--verify_from_month", default="2024-01")
    p.add_argument("--verify_to_month", default="2024-12")
    p.add_argument("--forward_from_month", default="2025-01")
    p.add_argument("--forward_to_month", default="2025-12")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root
    if root is None:
        candidates = [
            Path("dukas_out_v2"),
            Path("FX/code/dukas_out_v2"),
            Path("../dukascopy_downloader/dukas_out_v2"),
            Path("../fx_hypo_analyzer/dukas_out_v2"),
        ]
        for c in candidates:
            if c.exists():
                root = str(c)
                break
    if root is None:
        raise FileNotFoundError("Could not auto-detect data root; pass --root /path/to/dukas_out_v2")

    def _judge_rows(rows: list[dict]) -> list[dict]:
        # judge (compare to baseline within each family); threshold is coarse and intentionally non-optimized
        threshold = 50.0

        by_family: dict[str, dict[str, dict]] = {}
        for r in rows:
            by_family.setdefault(r["family"], {})[r["hyp"]] = r

        def judge_baseline(fwd: float) -> str:
            return "conditional" if fwd >= -threshold else "dead"

        def judge_delta(fwd: float, baseline_forward: float) -> str:
            if fwd <= baseline_forward - threshold:
                return "dead"
            if fwd >= baseline_forward + threshold:
                return "conditional"
            return "conditional"

        for fam, m in by_family.items():
            baseline_key = "B001" if fam.startswith("family_B_") else "C001"
            baseline_forward = float(m[baseline_key]["forward_sum_pnl_pips"]) if baseline_key in m else None
            for hyp_key, row in m.items():
                fwd = float(row["forward_sum_pnl_pips"])
                if baseline_forward is None or hyp_key == baseline_key:
                    row["judge"] = judge_baseline(fwd)
                else:
                    row["judge"] = judge_delta(fwd, baseline_forward)
        return rows

    def _write_summary_csv(summary_path: Path, rows: list[dict]) -> None:
        import csv

        with summary_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "family",
                    "hyp",
                    "delta",
                    "verify_sum_pnl_pips",
                    "verify_trades",
                    "forward_sum_pnl_pips",
                    "forward_trades",
                    "judge",
                ],
            )
            w.writeheader()
            for row in sorted(rows, key=lambda x: (x["family"], x["hyp"])):
                w.writerow(row)

    def _build_rows_from_existing() -> list[dict]:
        import csv

        specs = [
            ("family_B_failedbreakout", "B001", "baseline"),
            ("family_B_failedbreakout", "B002", "only_session=W1"),
            ("family_B_failedbreakout", "B003", "use_h1_trend_filter=False"),
            ("family_B_failedbreakout", "B004", "max_losing_streak=999"),
            ("family_B_failedbreakout", "B005", "use_time_filter=False"),
            ("family_C_m1entry", "C001", "baseline"),
            ("family_C_m1entry", "C002", "only_session=W1"),
            ("family_C_m1entry", "C003", "use_h1_trend_filter=False"),
            ("family_C_m1entry", "C004", "max_losing_streak=999"),
            ("family_C_m1entry", "C005", "use_time_filter=False"),
        ]
        out: list[dict] = []
        for fam, hyp, delta in specs:
            base = Path("results") / fam / hyp
            v_csv = base / "in_sample_2024" / "monthly.csv"
            f_csv = base / "forward_2025" / "monthly.csv"
            if not v_csv.exists() or not f_csv.exists():
                raise FileNotFoundError(f"Missing monthly.csv for {fam}/{hyp}: {v_csv} {f_csv}")

            def read_sum_trades(p: Path) -> tuple[float, int]:
                with p.open("r", encoding="utf-8", newline="") as ff:
                    r = csv.DictReader(ff)
                    rows = list(r)
                s = sum(float(x["sum_pnl_pips"]) for x in rows)
                t = sum(int(float(x["trades"])) for x in rows)
                return float(s), int(t)

            v_sum, v_tr = read_sum_trades(v_csv)
            f_sum, f_tr = read_sum_trades(f_csv)
            out.append({
                "family": fam,
                "hyp": hyp,
                "delta": delta,
                "verify_sum_pnl_pips": v_sum,
                "verify_trades": v_tr,
                "forward_sum_pnl_pips": f_sum,
                "forward_trades": f_tr,
            })
        return out

    if args.suite == "family_BC_summary":
        summary_path = Path("results") / "summary_family_BC.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        rows = _build_rows_from_existing()
        rows = _judge_rows(rows)
        _write_summary_csv(summary_path, rows)
        print(f"[runner] wrote summary: {summary_path}", flush=True)
        return 0

    if args.suite == "family_BC":
        summary_path = Path("results") / "summary_family_BC.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        def preset(
            family: str,
            hyp: str,
            *,
            delta: str,
            only_session: str | None,
            use_h1: bool | None,
            entry_mode: str,
            bias_mode: str,
        ) -> tuple[HypConfig, str]:
            cfg = HypConfig(
                family=family,
                hyp=hyp,
                symbol=str(args.symbol).upper(),
                root=str(root),
                only_session=only_session,
                verify_from_month=str(args.verify_from_month),
                verify_to_month=str(args.verify_to_month),
                forward_from_month=str(args.forward_from_month),
                forward_to_month=str(args.forward_to_month),
                spread_pips=float(args.spread_pips),
                use_h1_trend_filter=use_h1,
                entry_mode=entry_mode,
                bias_mode=bias_mode,
            )
            return cfg, delta

        specs: list[tuple[HypConfig, str]] = [
            preset("family_B_failedbreakout", "B001", delta="baseline", only_session=None, use_h1=None, entry_mode="B_failed_breakout", bias_mode="default"),
            preset("family_B_failedbreakout", "B002", delta="only_session=W1", only_session="W1", use_h1=None, entry_mode="B_failed_breakout", bias_mode="default"),
            preset("family_B_failedbreakout", "B003", delta="use_h1_trend_filter=False", only_session=None, use_h1=False, entry_mode="B_failed_breakout", bias_mode="default"),
            # one-diff variants (004/005)
            (replace(preset("family_B_failedbreakout", "B004", delta="max_losing_streak=999", only_session=None, use_h1=None, entry_mode="B_failed_breakout", bias_mode="default")[0], max_losing_streak=999), "max_losing_streak=999"),
            (replace(preset("family_B_failedbreakout", "B005", delta="use_time_filter=False", only_session=None, use_h1=None, entry_mode="B_failed_breakout", bias_mode="default")[0], use_time_filter=False), "use_time_filter=False"),
            preset("family_C_m1entry", "C001", delta="baseline", only_session=None, use_h1=None, entry_mode="C_m1_close", bias_mode="m1_candle"),
            preset("family_C_m1entry", "C002", delta="only_session=W1", only_session="W1", use_h1=None, entry_mode="C_m1_close", bias_mode="m1_candle"),
            preset("family_C_m1entry", "C003", delta="use_h1_trend_filter=False", only_session=None, use_h1=False, entry_mode="C_m1_close", bias_mode="m1_candle"),
            (replace(preset("family_C_m1entry", "C004", delta="max_losing_streak=999", only_session=None, use_h1=None, entry_mode="C_m1_close", bias_mode="m1_candle")[0], max_losing_streak=999), "max_losing_streak=999"),
            (replace(preset("family_C_m1entry", "C005", delta="use_time_filter=False", only_session=None, use_h1=None, entry_mode="C_m1_close", bias_mode="m1_candle")[0], use_time_filter=False), "use_time_filter=False"),
        ]

        results: list[dict] = []
        for cfg, delta in specs:
            meta = _run_variant_inprocess(cfg, run_tag=cfg.hyp.lower())
            results.append({
                "family": cfg.family,
                "hyp": cfg.hyp,
                "delta": delta,
                "verify_sum_pnl_pips": meta["verify"]["sum_pnl_pips"],
                "verify_trades": meta["verify"]["trades"],
                "forward_sum_pnl_pips": meta["forward"]["sum_pnl_pips"],
                "forward_trades": meta["forward"]["trades"],
            })

        results = _judge_rows(results)
        _write_summary_csv(summary_path, results)

        print(f"[runner] wrote summary: {summary_path}", flush=True)
        return 0

    if args.hyp == "A001":
        cfg = HypConfig(
            family="family_A",
            hyp="A001",
            symbol=str(args.symbol).upper(),
            root=str(root),
            only_session=None,
            verify_from_month=str(args.verify_from_month),
            verify_to_month=str(args.verify_to_month),
            forward_from_month=str(args.forward_from_month),
            forward_to_month=str(args.forward_to_month),
            spread_pips=float(args.spread_pips),
        )
        meta = run_a001_baseline(cfg)
        print("[runner] done. summary:", json.dumps(meta, ensure_ascii=False), flush=True)
        return 0

    if args.hyp == "A002":
        cfg = HypConfig(
            family="family_A",
            hyp="A002",
            symbol=str(args.symbol).upper(),
            root=str(root),
            only_session="W2",
            verify_from_month=str(args.verify_from_month),
            verify_to_month=str(args.verify_to_month),
            forward_from_month=str(args.forward_from_month),
            forward_to_month=str(args.forward_to_month),
            spread_pips=float(args.spread_pips),
        )
        meta = run_a002_w2_only(cfg)
        print("[runner] done. summary:", json.dumps(meta, ensure_ascii=False), flush=True)
        return 0

    if args.hyp == "A003":
        cfg = HypConfig(
            family="family_A",
            hyp="A003",
            symbol=str(args.symbol).upper(),
            root=str(root),
            only_session=None,
            verify_from_month=str(args.verify_from_month),
            verify_to_month=str(args.verify_to_month),
            forward_from_month=str(args.forward_from_month),
            forward_to_month=str(args.forward_to_month),
            spread_pips=float(args.spread_pips),
            use_h1_trend_filter=False,
        )
        meta = run_family_a_variant(cfg)
        print("[runner] done. summary:", json.dumps(meta, ensure_ascii=False), flush=True)
        return 0

    if args.hyp == "A004":
        cfg = HypConfig(
            family="family_A",
            hyp="A004",
            symbol=str(args.symbol).upper(),
            root=str(root),
            only_session=None,
            verify_from_month=str(args.verify_from_month),
            verify_to_month=str(args.verify_to_month),
            forward_from_month=str(args.forward_from_month),
            forward_to_month=str(args.forward_to_month),
            spread_pips=float(args.spread_pips),
            disable_m1_bias_filter=True,
        )
        meta = run_family_a_variant(cfg)
        print("[runner] done. summary:", json.dumps(meta, ensure_ascii=False), flush=True)
        return 0

    if args.hyp == "A005":
        cfg = HypConfig(
            family="family_A",
            hyp="A005",
            symbol=str(args.symbol).upper(),
            root=str(root),
            only_session=None,
            verify_from_month=str(args.verify_from_month),
            verify_to_month=str(args.verify_to_month),
            forward_from_month=str(args.forward_from_month),
            forward_to_month=str(args.forward_to_month),
            spread_pips=float(args.spread_pips),
            use_time_filter=False,
        )
        meta = run_family_a_variant(cfg)
        print("[runner] done. summary:", json.dumps(meta, ensure_ascii=False), flush=True)
        return 0

    if args.hyp in ("B001", "B002", "B003", "B004", "B005", "C001", "C002", "C003", "C004", "C005"):
        family = "family_B_failedbreakout" if args.hyp.startswith("B") else "family_C_m1entry"
        only_session = "W1" if args.hyp.endswith("002") else None
        use_h1 = False if args.hyp.endswith("003") else None
        use_time = False if args.hyp.endswith("005") else None
        max_ls = 999 if args.hyp.endswith("004") else None
        entry_mode = "B_failed_breakout" if args.hyp.startswith("B") else "C_m1_close"
        bias_mode = "default" if args.hyp.startswith("B") else "m1_candle"
        cfg = HypConfig(
            family=family,
            hyp=str(args.hyp),
            symbol=str(args.symbol).upper(),
            root=str(root),
            only_session=only_session,
            verify_from_month=str(args.verify_from_month),
            verify_to_month=str(args.verify_to_month),
            forward_from_month=str(args.forward_from_month),
            forward_to_month=str(args.forward_to_month),
            spread_pips=float(args.spread_pips),
            use_h1_trend_filter=use_h1,
            use_time_filter=use_time,
            entry_mode=entry_mode,
            bias_mode=bias_mode,
            max_losing_streak=max_ls,
        )
        meta = _run_variant_inprocess(cfg, run_tag=str(args.hyp).lower())
        print("[runner] done. summary:", json.dumps(meta, ensure_ascii=False), flush=True)
        return 0

    if args.hyp == "B006":
        import csv
        import backtest_core as bc
        import pandas as pd

        out_dir = Path("results") / "family_B_failedbreakout" / "B006_observation"
        out_dir.mkdir(parents=True, exist_ok=True)

        obs_cfg = bc.Config(
            root=Path(str(root)),
            symbol=str(args.symbol).upper(),
            from_month="2025-01",
            to_month="2025-12",
            run_tag="b006_obs",
        )

        df_bid = bc.load_parquet_10s_bid(obs_cfg)
        df10 = bc.add_synthetic_bidask(df_bid, obs_cfg)
        h1_up = bc.compute_h1_uptrend(df10, obs_cfg)

        months = df10.index.to_period("M").astype(str)
        ratio_df = pd.DataFrame({"month": months, "h1_up": h1_up.astype(int)})
        grp = ratio_df.groupby("month", as_index=False).agg(
            bars_total=("h1_up", "size"),
            h1_uptrend_true_bars=("h1_up", "sum"),
        )
        grp["h1_uptrend_ratio"] = grp["h1_uptrend_true_bars"] / grp["bars_total"]
        grp = grp.sort_values("month")

        p_ratio = out_dir / "h1_uptrend_monthly_ratio_2025.csv"
        grp.to_csv(p_ratio, index=False)

        # merge with B002 forward monthly
        b002_monthly = Path("results") / "family_B_failedbreakout" / "B002" / "monthly.csv"
        if not b002_monthly.exists():
            raise FileNotFoundError(f"B002 monthly.csv not found: {b002_monthly}")
        pnl = pd.read_csv(b002_monthly)
        merged = pnl.merge(grp[["month", "h1_uptrend_ratio"]], on="month", how="left")
        merged = merged[["month", "sum_pnl_pips", "trades", "h1_uptrend_ratio"]].sort_values("month")

        p_merged = out_dir / "merged_b002_pnl_vs_h1ratio_2025.csv"
        merged.to_csv(p_merged, index=False)

        # summary (win vs loss months)
        win = merged[merged["sum_pnl_pips"] > 0].copy()
        loss = merged[merged["sum_pnl_pips"] < 0].copy()

        win_mean = float(win["h1_uptrend_ratio"].mean()) if len(win) else float("nan")
        loss_mean = float(loss["h1_uptrend_ratio"].mean()) if len(loss) else float("nan")

        win_months = ",".join(win["month"].tolist())
        loss_months = ",".join(loss["month"].tolist())

        summary_lines = [
            "B006_observation (B002 / 2025 forward)",
            f"win_months_count={len(win)} win_mean_h1_uptrend_ratio={win_mean:.6f} months=[{win_months}]",
            f"loss_months_count={len(loss)} loss_mean_h1_uptrend_ratio={loss_mean:.6f} months=[{loss_months}]",
            f"ratio_csv={p_ratio}",
            f"merged_csv={p_merged}",
        ]
        p_summary = out_dir / "summary.txt"
        p_summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

        cfg_out = {
            "family": "family_B_failedbreakout",
            "hyp": "B006_observation",
            "symbol": obs_cfg.symbol,
            "root": str(obs_cfg.root),
            "from_month": obs_cfg.from_month,
            "to_month": obs_cfg.to_month,
            "h1_ema_period": obs_cfg.h1_ema_period,
            "h1_ema_slope_bars": obs_cfg.h1_ema_slope_bars,
            "source_b002_monthly_csv": str(b002_monthly),
        }
        (out_dir / "config.json").write_text(json.dumps(cfg_out, ensure_ascii=False, indent=2), encoding="utf-8")

        print(p_summary.read_text(encoding="utf-8"), flush=True)
        return 0

    if args.hyp == "B007":
        import json as _json
        import numpy as np
        import pandas as pd

        import backtest_core as bc

        out_dir = Path("results") / "family_B_failedbreakout" / "B007_observation"
        out_dir.mkdir(parents=True, exist_ok=True)

        obs_cfg = bc.Config(
            root=Path(str(root)),
            symbol=str(args.symbol).upper(),
            from_month="2025-01",
            to_month="2025-12",
            run_tag="b007_obs",
        )

        df_bid = bc.load_parquet_10s_bid(obs_cfg)
        df10 = bc.add_synthetic_bidask(df_bid, obs_cfg)
        h1_up = bc.compute_h1_uptrend(df10, obs_cfg).astype(bool)
        ts = df10.index.to_numpy(dtype="datetime64[ns]")

        months = pd.PeriodIndex(df10.index, freq="M").astype(str)
        df = pd.DataFrame({"ts": df10.index, "month": months, "h1_up": h1_up})

        def run_stats_for_month(g: pd.DataFrame) -> dict:
            b = g["h1_up"].to_numpy(dtype=bool)
            t = g["ts"].to_numpy(dtype="datetime64[ns]")
            if len(b) == 0:
                return {
                    "max_true_run_hours": 0.0,
                    "mean_true_run_hours": 0.0,
                    "p90_true_run_hours": 0.0,
                    "true_runs_count": 0,
                    "total_true_hours": 0.0,
                    "total_hours": 0.0,
                    "true_ratio": 0.0,
                }

            # continuity requires: current True, previous True, and exact 10s spacing
            dt_ok = np.r_[False, (t[1:] - t[:-1]) == np.timedelta64(10, "s")]
            prev_true = np.r_[False, b[:-1]]
            cont = b & prev_true & dt_ok
            start = b & (~cont)
            run_id = np.cumsum(start).astype(np.int64)
            run_id = np.where(b, run_id, 0)
            lengths = np.bincount(run_id)[1:]  # bars per run

            hours = lengths.astype(float) / 360.0  # 10s bars => /360 = hours
            total_true_hours = float(b.sum()) / 360.0
            total_hours = float(len(b)) / 360.0
            true_ratio = total_true_hours / total_hours if total_hours > 0 else 0.0

            if len(hours) == 0:
                max_h = mean_h = p90_h = 0.0
            else:
                max_h = float(np.max(hours))
                mean_h = float(np.mean(hours))
                p90_h = float(np.quantile(hours, 0.90))

            return {
                "max_true_run_hours": max_h,
                "mean_true_run_hours": mean_h,
                "p90_true_run_hours": p90_h,
                "true_runs_count": int(len(hours)),
                "total_true_hours": total_true_hours,
                "total_hours": total_hours,
                "true_ratio": true_ratio,
            }

        stats_rows = []
        for month, g in df.groupby("month", sort=True):
            st = run_stats_for_month(g)
            stats_rows.append({"month": month, **st})

        stats = pd.DataFrame(stats_rows).sort_values("month")
        p_stats = out_dir / "h1_uptrend_run_stats_2025.csv"
        stats.to_csv(p_stats, index=False)

        # Merge with B006 merged CSV (month,sum_pnl_pips,trades,h1_uptrend_ratio)
        b006_merged = Path("results") / "family_B_failedbreakout" / "B006_observation" / "merged_b002_pnl_vs_h1ratio_2025.csv"
        if not b006_merged.exists():
            raise FileNotFoundError(f"B006 merged CSV not found: {b006_merged}")
        b006 = pd.read_csv(b006_merged)
        merged = b006.merge(stats, on="month", how="left")
        keep = [
            "month",
            "sum_pnl_pips",
            "trades",
            "true_ratio",
            "max_true_run_hours",
            "mean_true_run_hours",
            "p90_true_run_hours",
            "true_runs_count",
        ]
        merged_out = merged[keep].sort_values("month")
        p_merged = out_dir / "merged_b002_pnl_vs_h1run_2025.csv"
        merged_out.to_csv(p_merged, index=False)

        # Summary: win vs loss months (observation only)
        win = merged_out[merged_out["sum_pnl_pips"] > 0].copy()
        loss = merged_out[merged_out["sum_pnl_pips"] < 0].copy()

        def mean_or_nan(d: pd.DataFrame, col: str) -> float:
            return float(d[col].mean()) if len(d) else float("nan")

        metrics = [
            "true_ratio",
            "max_true_run_hours",
            "mean_true_run_hours",
            "p90_true_run_hours",
            "true_runs_count",
        ]

        lines = ["B007_observation (B002 / 2025 forward)"]
        lines.append(f"win_months_count={len(win)} months=[{','.join(win['month'].tolist())}]")
        lines.append(f"loss_months_count={len(loss)} months=[{','.join(loss['month'].tolist())}]")
        for m in metrics:
            lines.append(f"win_mean_{m}={mean_or_nan(win, m):.6f}")
            lines.append(f"loss_mean_{m}={mean_or_nan(loss, m):.6f}")
        lines.append(f"run_stats_csv={p_stats}")
        lines.append(f"merged_csv={p_merged}")

        p_summary = out_dir / "summary.txt"
        p_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

        cfg_out = {
            "family": "family_B_failedbreakout",
            "hyp": "B007_observation",
            "symbol": obs_cfg.symbol,
            "root": str(obs_cfg.root),
            "from_month": obs_cfg.from_month,
            "to_month": obs_cfg.to_month,
            "h1_ema_period": obs_cfg.h1_ema_period,
            "h1_ema_slope_bars": obs_cfg.h1_ema_slope_bars,
            "b006_merged_csv": str(b006_merged),
            "missing_bars_break_rule": "run continues only if previous bar exists at exactly +10s and both bars are true",
        }
        (out_dir / "config.json").write_text(_json.dumps(cfg_out, ensure_ascii=False, indent=2), encoding="utf-8")

        print(p_summary.read_text(encoding="utf-8"), flush=True)
        return 0

    if args.hyp == "B008":
        import json as _json
        import pandas as pd

        out_dir = Path("results") / "family_B_failedbreakout" / "B008_observation"
        out_dir.mkdir(parents=True, exist_ok=True)

        merged_b007 = Path("results") / "family_B_failedbreakout" / "B007_observation" / "merged_b002_pnl_vs_h1run_2025.csv"
        if not merged_b007.exists():
            raise FileNotFoundError(f"B007 merged CSV not found: {merged_b007}")

        df = pd.read_csv(merged_b007)
        required = {"month", "sum_pnl_pips", "trades", "true_ratio", "true_runs_count", "mean_true_run_hours", "p90_true_run_hours", "max_true_run_hours"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"B007 merged CSV missing columns: {sorted(missing)}")

        # Fixed rule: median split (no threshold search)
        med_runs = float(df["true_runs_count"].median())
        med_mean = float(df["mean_true_run_hours"].median())
        is_choppy = (df["true_runs_count"] > med_runs) & (df["mean_true_run_hours"] < med_mean)
        df["state"] = is_choppy.map({True: "choppy_trend", False: "persistent_trend"})

        labeled_cols = [
            "month",
            "sum_pnl_pips",
            "trades",
            "true_ratio",
            "true_runs_count",
            "mean_true_run_hours",
            "p90_true_run_hours",
            "max_true_run_hours",
            "state",
        ]
        labeled = df[labeled_cols].sort_values("month")
        p_labeled = out_dir / "state_labeled_2025.csv"
        labeled.to_csv(p_labeled, index=False)

        summary = labeled.groupby("state", as_index=False).agg(
            months_count=("month", "count"),
            sum_pnl_pips_sum=("sum_pnl_pips", "sum"),
            sum_pnl_pips_mean=("sum_pnl_pips", "mean"),
            trades_sum=("trades", "sum"),
        ).sort_values("state")
        p_state = out_dir / "state_summary_2025.csv"
        summary.to_csv(p_state, index=False)

        months_by_state = labeled.groupby("state")["month"].apply(list).to_dict()
        lines = [
            "B008_observation (B002 / 2025 forward)",
            "state rule: choppy_trend if (true_runs_count > median(true_runs_count)) AND (mean_true_run_hours < median(mean_true_run_hours)); else persistent_trend",
            f"median_true_runs_count={med_runs:.6f}",
            f"median_mean_true_run_hours={med_mean:.6f}",
            "",
            "state_summary_2025:",
        ]
        for _, r in summary.iterrows():
            lines.append(
                f"- {r['state']}: months={int(r['months_count'])} "
                f"sum_pnl_pips_sum={float(r['sum_pnl_pips_sum']):.1f} "
                f"sum_pnl_pips_mean={float(r['sum_pnl_pips_mean']):.3f} "
                f"trades_sum={int(r['trades_sum'])} "
                f"months={months_by_state.get(r['state'], [])}"
            )
        lines += [
            "",
            f"labeled_csv={p_labeled}",
            f"state_summary_csv={p_state}",
        ]
        p_summary = out_dir / "summary.txt"
        p_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

        cfg_out = {
            "family": "family_B_failedbreakout",
            "hyp": "B008_observation",
            "source_csv": str(merged_b007),
            "rule": "choppy_trend if (true_runs_count > median(true_runs_count)) AND (mean_true_run_hours < median(mean_true_run_hours)) else persistent_trend",
            "medians_2025": {
                "true_runs_count": med_runs,
                "mean_true_run_hours": med_mean,
            },
        }
        (out_dir / "config.json").write_text(_json.dumps(cfg_out, ensure_ascii=False, indent=2), encoding="utf-8")

        print(p_summary.read_text(encoding="utf-8"), flush=True)
        return 0
    raise AssertionError(f"Unsupported hyp: {args.hyp}")


if __name__ == "__main__":
    raise SystemExit(main())
