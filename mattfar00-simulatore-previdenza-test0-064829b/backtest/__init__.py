"""
backtest — package di backtesting storico per il simulatore previdenziale.

Sottomoduli:
    backtest.metrics : metriche stile Portfolio Visualizer (CAGR, Sharpe,
                       Sortino, max drawdown, VaR/CVaR, rolling returns...)
    backtest.data    : estrazione delle serie storiche reali dal main
    backtest.engine  : esecuzione deterministica (lump sum / DCA / via motore)
    backtest.ui      : tab Streamlit (import di streamlit/plotly solo qui)

I moduli metrics/data/engine NON importano streamlit: sono usabili anche in
uno script di test. La UI riceve tutto dal main via dependency injection.
"""
__version__ = "0.1.0"
