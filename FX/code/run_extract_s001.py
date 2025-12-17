#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from core_setup_s001 import Params, extract_S001_setups

# -------------------------
# Configure input CSV paths
# -------------------------
SYMBOL = "USDJPY"

# Expected columns (either set is OK):
# - ts, open, high, low, close
# - ts, open_bid, high_bid, low_bid, close_bid
CSV_5M_PATH = Path("FX/data_local/usdjpy_5m.csv")
CSV_10S_PATH = Path("FX/data_local/usdjpy_10s.csv")

OUT_PATH = Path("FX/results/s001_setups.csv")


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if "ts" not in df.columns:
        raise ValueError(f"CSV must have 'ts' column: {path}")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def main() -> int:
    params = Params()

    df5 = _load_csv(CSV_5M_PATH)
    df10 = _load_csv(CSV_10S_PATH)

    df5i = df5.set_index("ts").sort_index()
    df10i = df10.set_index("ts").sort_index()

    print("[s001] extract start", flush=True)
    print(f"[s001] symbol={SYMBOL}", flush=True)
    print(f"[s001] df5  rows={len(df5i):,} range={df5i.index.min()}..{df5i.index.max()}", flush=True)
    print(f"[s001] df10 rows={len(df10i):,} range={df10i.index.min()}..{df10i.index.max()}", flush=True)
    print(f"[s001] params={json.dumps(asdict(params), ensure_ascii=False)}", flush=True)

    out = extract_S001_setups(SYMBOL, df5i, df10i, params)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    if out.empty:
        print("[s001] extracted 0 rows", flush=True)
    else:
        print(f"[s001] extracted rows={len(out):,}", flush=True)
        print(f"[s001] date range={out['date'].min()}..{out['date'].max()}", flush=True)
        print(f"[s001] sessions={sorted(out['session'].unique().tolist())}", flush=True)

    print(f"[s001] wrote: {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

