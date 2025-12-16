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

    patches: list[contextlib.AbstractContextManager] = []
    if cfg.disable_m1_bias_filter:
        def _bias_all_one(df10, _cfg):  # noqa: ANN001
            return np.ones(len(df10), dtype=np.int8)
        patches.append(_temporary_attr(bc, "compute_bias_htf", _bias_all_one))

    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)

        print("=== backtest started (runner/inprocess) ===", flush=True)
        print(
            f"[cfg] hyp={cfg.hyp} range={core_cfg.from_month}..{core_cfg.to_month} "
            f"only_session={core_cfg.only_session} use_time_filter={core_cfg.use_time_filter} "
            f"use_h1_trend_filter={core_cfg.use_h1_trend_filter} disable_m1_bias_filter={cfg.disable_m1_bias_filter}",
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
    p.add_argument("--suite", default=None, choices=[None, "family_BC"])
    p.add_argument(
        "--hyp",
        default="A002",
        choices=[
            "A001", "A002", "A003", "A004", "A005",
            "B001", "B002", "B003",
            "C001", "C002", "C003",
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

    if args.suite == "family_BC":
        summary_path = Path("results") / "summary_family_BC.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        def preset(family: str, hyp: str, *, delta: str, only_session: str | None, use_h1: bool | None) -> tuple[HypConfig, str]:
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
            )
            return cfg, delta

        specs: list[tuple[HypConfig, str]] = [
            preset("family_B_failedbreakout", "B001", delta="baseline", only_session=None, use_h1=None),
            preset("family_B_failedbreakout", "B002", delta="only_session=W1", only_session="W1", use_h1=None),
            preset("family_B_failedbreakout", "B003", delta="use_h1_trend_filter=False", only_session=None, use_h1=False),
            preset("family_C_m1entry", "C001", delta="baseline", only_session=None, use_h1=None),
            preset("family_C_m1entry", "C002", delta="only_session=W1", only_session="W1", use_h1=None),
            preset("family_C_m1entry", "C003", delta="use_h1_trend_filter=False", only_session=None, use_h1=False),
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

        # judge (compare to baseline within each family)
        by_family: dict[str, dict[str, dict]] = {}
        for r in results:
            by_family.setdefault(r["family"], {})[r["hyp"]] = r

        def judge_row(row: dict, baseline_forward: float | None) -> str:
            fwd = float(row["forward_sum_pnl_pips"])
            if baseline_forward is None:
                return "conditional" if fwd >= 0 else "dead"
            if fwd <= baseline_forward - 50.0:
                return "dead"
            if fwd >= baseline_forward + 50.0:
                return "conditional"
            return "conditional"

        for fam, m in by_family.items():
            baseline_key = "B001" if fam.startswith("family_B_") else "C001"
            baseline_forward = float(m[baseline_key]["forward_sum_pnl_pips"]) if baseline_key in m else None
            for hyp_key, row in m.items():
                row["judge"] = judge_row(row, baseline_forward if hyp_key != baseline_key else None)

        # write CSV
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
            for row in sorted(results, key=lambda x: (x["family"], x["hyp"])):
                w.writerow(row)

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

    if args.hyp in ("B001", "B002", "B003", "C001", "C002", "C003"):
        family = "family_B_failedbreakout" if args.hyp.startswith("B") else "family_C_m1entry"
        only_session = "W1" if args.hyp.endswith("002") else None
        use_h1 = False if args.hyp.endswith("003") else None
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
        )
        meta = _run_variant_inprocess(cfg, run_tag=str(args.hyp).lower())
        print("[runner] done. summary:", json.dumps(meta, ensure_ascii=False), flush=True)
        return 0
    raise AssertionError(f"Unsupported hyp: {args.hyp}")


if __name__ == "__main__":
    raise SystemExit(main())
