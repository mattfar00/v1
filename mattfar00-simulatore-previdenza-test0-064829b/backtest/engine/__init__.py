"""backtest.engine — esecuzione deterministica del backtest storico."""
from .engine import (
    backtest_lump_sum,
    backtest_con_contributi,
    backtest_via_motore,
    equity_to_returns,
)

__all__ = [
    "backtest_lump_sum", "backtest_con_contributi",
    "backtest_via_motore", "equity_to_returns",
]
