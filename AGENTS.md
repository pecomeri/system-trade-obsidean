# Repository Guidelines

## Project Structure & Module Organization
- Root folders: `00_Index/` (meta-notes), `Crypto/` (exchange/coin notes), `FX/` (main research and code).
- `FX/code/`: Python backtest engine (`backtest_core.py`) and a thin wrapper (`backtest_runner.py`).
- Research docs: `FX/10_regimes/`, `FX/20_strategies/`, `FX/30_hypotheses/`, `FX/99_logs/` plus higher-level notes in `FX/00_Index/`.
- Data is expected outside the repo by default: `dukas_out_v2/<SYMBOL>/bars10s_pq/*.parquet`.
- Backtest outputs are written under `results/` as timestamped run folders containing CSVs and `config.json`.

## Build, Test, and Development Commands
- Run a full backtest (USDJPY example, filtered months):  
  `python FX/code/backtest_core.py --symbol USDJPY --from_month 2024-01 --to_month 2024-06`
- Restrict to a session: add `--only_session W2` (or `W1`).
- Quick debug slice: `python FX/code/backtest_core.py --symbol USDJPY --debug_day 2024-01-15 --debug_from 09:00 --debug_to 10:00`
- Point to another data root if needed: `--root /path/to/data_root`.
- Runs emit progress to stdout, CSVs to the run directory, and a sanity check runs automatically at the end.

## Coding Style & Naming Conventions
- Python 3; follow PEP 8 with 4-space indents and `snake_case` names. Prefer type hints, dataclasses for configs, and clear defaults.
- Keep computations vectorized in pandas/numpy; avoid per-row loops unless justified.
- Log via `print(..., flush=True)` as used in the engine; prefer structured logs (JSON lines) when adding new events.
- When adding params, thread them through `Config`, CLI args, and saved `config.json` so runs remain reproducible.

## Testing Guidelines
- No formal test suite yet; validate changes with a focused debug run before longer batches.
- Use `--debug_day/--debug_from/--debug_to` to limit scope and inspect generated `debug_signals.csv` and `debug_entries.csv`.
- Ensure the built-in `sanity_check` passes; if you add new entry conditions, update the check accordingly.
- Prefer small, deterministic data slices for regression checks and keep result artifacts out of version control unless summarizing findings.

## Data Management & Security
- Do not commit parquet data or large result folders; keep them local or in external storage.
- Avoid embedding credentials or API keys in code or configs; use environment variables or local `.env` files excluded from git.
- Document any new data expectations (paths, schema) in the relevant strategy/regime memo.

## Commit & Pull Request Guidelines
- Commits: short, imperative subjects (e.g., `Add session filter gating`) with focused scope.
- PRs: include a short description, the exact command(s) run, key metrics (trades, winrate, PnL), and any new debug artifacts examined.
- Link related strategy/regime/hypothesis docs when changes affect them; attach small CSV excerpts or screenshots only when clarifying behavior.
