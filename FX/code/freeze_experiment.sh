#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "" ]; then
  echo "usage: $0 <experiment_id>" >&2
  exit 2
fi

EXPERIMENT_ID="$1"
ROOT="FX"
REGISTRY="$ROOT/_review/2025-12-protocol/experiment_registry.csv"
FREEZE_DIR="$ROOT/40_freeze"

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
DAY="$(date -u +%Y-%m-%d)"
FREEZE_NOTE="$FREEZE_DIR/freeze_${DAY}.md"

if [ ! -f "$REGISTRY" ]; then
  echo "experiment_registry.csv not found: $REGISTRY" >&2
  exit 1
fi

python3 - <<'PY'
import csv
import os
import sys
from datetime import datetime, timezone

exp_id = os.environ["EXPERIMENT_ID"]
registry = os.environ["REGISTRY"]
ts = os.environ["TS"]

with open(registry, "r", encoding="utf-8", newline="") as f:
    r = csv.DictReader(f)
    rows = list(r)
    fieldnames = r.fieldnames or []

if "experiment_id" not in fieldnames:
    raise SystemExit("experiment_registry.csv missing experiment_id column")

found = False
for row in rows:
    if row.get("experiment_id") == exp_id:
        row["status"] = "frozen"
        row["frozen_at"] = ts
        found = True

if not found:
    # add minimal row if absent
    new = {k: "" for k in fieldnames}
    new["experiment_id"] = exp_id
    new["status"] = "frozen"
    new["frozen_at"] = ts
    rows.append(new)

tmp = registry + ".tmp"
with open(tmp, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for row in rows:
        w.writerow(row)
os.replace(tmp, registry)
print(f"[freeze] registry updated: {registry}", flush=True)
PY

mkdir -p "$FREEZE_DIR"
if [ ! -f "$FREEZE_NOTE" ]; then
  cat > "$FREEZE_NOTE" <<EOF
# freeze ${DAY} (UTC)

## Frozen experiments
EOF
fi

if ! grep -q "^- ${EXPERIMENT_ID}\\b" "$FREEZE_NOTE" 2>/dev/null; then
  echo "- ${EXPERIMENT_ID} frozen_at=${TS}" >> "$FREEZE_NOTE"
fi

echo "[freeze] wrote/updated: $FREEZE_NOTE"

