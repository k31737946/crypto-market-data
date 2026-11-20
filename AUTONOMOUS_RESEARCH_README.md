# Autonomous Market Research Engine (v1)

A local Python research prototype for 5-minute OHLCV data. It does **not** connect to an exchange or place orders.

## Files
- `autonomous_research.py`: feature generation, hypothesis search, independent-trade backtest, loss diagnostics, SQLite memory and CSV leaderboard.
- `requirements.txt`: Python dependencies.

## Setup
Python 3.10+ recommended.

```bash
pip install -r requirements.txt
python autonomous_research.py --data . --pattern "*-5m-*.csv"
```

The repository's `BTCUSDT-5m-YYYY-MM.csv` files are discovered automatically when the script is run from the cloned repository root.

## Research rules
- Signal is known only after candle close; entry is next candle open.
- Each valid signal is evaluated as an independent trade. Signals are not suppressed by another open trade.
- Default stop distance is 1 x ATR(14), target is 2 x stop distance, maximum holding period is 48 bars.
- If stop and target are both touched in one candle, stop is assumed first (conservative).
- First 70% of time-ordered bars are used for candidate ranking; final 30% is reported as out-of-sample.
- Feature thresholds are derived from the training period only.
- Gross results by default: set `--fee-bps` to a realistic round-trip fee estimate for net estimates. Slippage is not yet modeled.
- Results are appended to `research_memory.sqlite`; leaderboard is exported to `research_leaderboard.csv`.

## Important limitations
This is v1, not a production-grade evolutionary agent. The initial generator searches single-factor thresholds, then cross-combines leading single factors. Loss diagnostics are descriptive only and do not automatically promote filters. Repeated experimentation against the same 30% holdout will eventually contaminate it; reserve a final untouched period and use walk-forward validation before trusting any result. The default 0 fee is for exploratory research only.

## Suggested next upgrades
1. Purged walk-forward validation and embargo around label horizons.
2. Slippage, funding, exchange fee tiers and contract tick/lot constraints.
3. Loss-driven mutation proposals evaluated on fresh time windows.
4. Parallel search, experiment deduplication and resource budgets.
5. A separate paper-trading monitor; never let research code place live orders.
