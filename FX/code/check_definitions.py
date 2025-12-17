#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckItem:
    path: Path
    missing: list[str]


def _is_empty_after_colon(line: str) -> bool:
    if "：" not in line:
        return False
    head, tail = line.split("：", 1)
    return tail.strip() == ""


def _scan_file(path: Path, required_prefixes: list[str]) -> CheckItem | None:
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    missing: list[str] = []
    for prefix in required_prefixes:
        hits = [ln for ln in text if ln.strip().startswith(prefix)]
        if not hits:
            missing.append(f"NOT_FOUND: {prefix}")
            continue
        # If any matching line is non-empty after colon, accept
        ok = any(not _is_empty_after_colon(h) for h in hits)
        if not ok:
            missing.append(f"EMPTY: {prefix}")
    if not missing:
        return None
    return CheckItem(path=path, missing=missing)


def main() -> int:
    root = Path("FX")
    if not root.exists():
        print("FX/ not found", flush=True)
        return 0

    checks: list[CheckItem] = []

    # Setups
    for p in sorted((root / "15_setups").glob("S-001_*.md")):
        item = _scan_file(
            p,
            required_prefixes=[
                "  - トレンドの定義：",
                "  - ミッドレンジの定義：",
                "  - フェイクアウトの定義：",
            ],
        )
        if item:
            checks.append(item)

    # Entries
    for p in sorted((root / "25_entries").glob("E-*.md")):
        item = _scan_file(
            p,
            required_prefixes=[
                "- 必須条件（Yes/No）：",
            ],
        )
        if item:
            checks.append(item)

    # Filters
    for p in sorted((root / "27_filters").glob("F-*.md")):
        item = _scan_file(
            p,
            required_prefixes=[
                "- Yes（通す）：",
                "- No（弾く）：",
            ],
        )
        if item:
            checks.append(item)

    out_dir = root / "_review" / "2025-12-protocol"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "definition_check.md"

    lines: list[str] = []
    lines.append("# 定義チェック（自動）")
    lines.append("")
    lines.append("このレポートは自動生成。ノートの自動修正はしない。")
    lines.append("")
    if not checks:
        lines.append("- OK: 必須フィールドの空欄は検出されなかった")
    else:
        lines.append(f"- WARN: 空欄/未記入の可能性あり（files={len(checks)}）")
        lines.append("")
        for c in checks:
            lines.append(f"## {c.path}")
            for m in c.missing:
                lines.append(f"- TODO: {m}")
            lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"[check_definitions] wrote: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

