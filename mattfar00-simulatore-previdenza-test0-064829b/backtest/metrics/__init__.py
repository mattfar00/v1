"""
backtest.metrics — metriche di backtest ispirate a Portfolio Visualizer.
Lavorano tutte su una serie di RENDIMENTI MENSILI (np.array 1D).
"""
from .metrics import (
    cagr,
    volatilita_annua,
    downside_deviation_annua,
    max_drawdown,
    sharpe,
    sortino,
    calmar,
    var_storico,
    cvar_storico,
    best_worst_year,
    rolling_returns,
    upside_downside_capture,
    scheda_completa,
)

__all__ = [
    "cagr", "volatilita_annua", "downside_deviation_annua", "max_drawdown",
    "sharpe", "sortino", "calmar", "var_storico", "cvar_storico",
    "best_worst_year", "rolling_returns", "upside_downside_capture",
    "scheda_completa",
]
