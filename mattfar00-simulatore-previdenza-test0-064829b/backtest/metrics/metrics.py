"""
backtest_metrics.py — metriche di backtest ispirate a Portfolio Visualizer.

Tutte le funzioni lavorano su una serie di RENDIMENTI MENSILI (np.array 1D,
es. 0.012 = +1,2% nel mese). Coerente col resto del tuo simulatore, che è
già mensile. Nessuna dipendenza oltre numpy/pandas.

Convenzione: risk-free passato come tasso ANNUO (default 0). PV usa il
T-Bill USA a 3 mesi; nel tuo contesto EUR puoi mettere l'Euribor/BOT medio
del periodo, oppure 0 per semplicità.
"""
import numpy as np
import pandas as pd


def _ann_factor(freq_per_anno=12):
    return freq_per_anno


def cagr(rend_mensili):
    r = np.asarray(rend_mensili, float)
    n = r.size
    if n == 0:
        return np.nan
    tot = np.prod(1 + r)
    return tot ** (12 / n) - 1


def volatilita_annua(rend_mensili):
    r = np.asarray(rend_mensili, float)
    return r.std(ddof=1) * np.sqrt(12) if r.size > 1 else np.nan


def downside_deviation_annua(rend_mensili, mar_mensile=0.0):
    r = np.asarray(rend_mensili, float)
    downside = np.minimum(r - mar_mensile, 0.0)
    return np.sqrt(np.mean(downside ** 2)) * np.sqrt(12)


def max_drawdown(rend_mensili):
    """Ritorna (mdd, idx_picco, idx_valle). mdd è negativo (es. -0.235)."""
    r = np.asarray(rend_mensili, float)
    equity = np.cumprod(1 + r)
    picco = np.maximum.accumulate(equity)
    dd = equity / picco - 1
    idx_valle = int(np.argmin(dd))
    idx_picco = int(np.argmax(equity[:idx_valle + 1])) if idx_valle > 0 else 0
    return float(dd.min()), idx_picco, idx_valle


def sharpe(rend_mensili, rf_annuo=0.0):
    r = np.asarray(rend_mensili, float)
    rf_m = (1 + rf_annuo) ** (1 / 12) - 1
    ex = r - rf_m
    sd = ex.std(ddof=1)
    return (ex.mean() / sd) * np.sqrt(12) if sd > 0 else np.nan


def sortino(rend_mensili, rf_annuo=0.0):
    r = np.asarray(rend_mensili, float)
    rf_m = (1 + rf_annuo) ** (1 / 12) - 1
    ex = r - rf_m
    dd = np.sqrt(np.mean(np.minimum(ex, 0.0) ** 2))
    return (ex.mean() / dd) * np.sqrt(12) if dd > 0 else np.nan


def calmar(rend_mensili):
    mdd, _, _ = max_drawdown(rend_mensili)
    c = cagr(rend_mensili)
    return c / abs(mdd) if mdd < 0 else np.nan


def var_storico(rend_mensili, alpha=0.05):
    """VaR mensile storico (valore positivo = perdita)."""
    r = np.asarray(rend_mensili, float)
    return -np.percentile(r, alpha * 100)


def cvar_storico(rend_mensili, alpha=0.05):
    r = np.asarray(rend_mensili, float)
    soglia = np.percentile(r, alpha * 100)
    coda = r[r <= soglia]
    return -coda.mean() if coda.size else np.nan


def best_worst_year(rend_mensili, anni_label=None):
    """Rendimenti per anno solare. Se anni_label è dato (stessa lunghezza),
    raggruppa per anno reale; altrimenti a blocchi di 12 mesi."""
    r = np.asarray(rend_mensili, float)
    if anni_label is not None:
        df = pd.DataFrame({"y": anni_label, "r": r})
        annui = df.groupby("y")["r"].apply(lambda s: np.prod(1 + s) - 1)
        return float(annui.max()), float(annui.min()), annui
    n_anni = r.size // 12
    blocchi = r[:n_anni * 12].reshape(n_anni, 12)
    annui = np.prod(1 + blocchi, axis=1) - 1
    return float(annui.max()), float(annui.min()), annui


def rolling_returns(rend_mensili, anni=(1, 3, 5, 10)):
    """CAGR rolling su finestre mobili. Ritorna dict {anni: (media, high, low)}."""
    r = np.asarray(rend_mensili, float)
    out = {}
    for a in anni:
        w = a * 12
        if r.size < w:
            continue
        vals = [np.prod(1 + r[i:i + w]) ** (12 / w) - 1 for i in range(r.size - w + 1)]
        vals = np.array(vals)
        out[a] = (float(vals.mean()), float(vals.max()), float(vals.min()))
    return out


def upside_downside_capture(rend_port, rend_bench):
    """Capture ratio vs benchmark (entrambi mensili, stessa lunghezza)."""
    p = np.asarray(rend_port, float)
    b = np.asarray(rend_bench, float)
    up = b > 0
    dn = b < 0
    def _ann(x):
        return np.prod(1 + x) ** (12 / x.size) - 1 if x.size else np.nan
    up_c = _ann(p[up]) / _ann(b[up]) if up.any() else np.nan
    dn_c = _ann(p[dn]) / _ann(b[dn]) if dn.any() else np.nan
    return up_c, dn_c


def scheda_completa(rend_mensili, rf_annuo=0.0, anni_label=None):
    """Riepilogo stile 'Performance Summary' di Portfolio Visualizer."""
    r = np.asarray(rend_mensili, float)
    mdd, ip, iv = max_drawdown(r)
    by, wy, _ = best_worst_year(r, anni_label)
    pos = (r > 0).sum()
    return {
        "CAGR": cagr(r),
        "Volatilita_annua": volatilita_annua(r),
        "Downside_dev_annua": downside_deviation_annua(r),
        "Best_year": by,
        "Worst_year": wy,
        "Max_drawdown": mdd,
        "Sharpe": sharpe(r, rf_annuo),
        "Sortino": sortino(r, rf_annuo),
        "Calmar": calmar(r),
        "VaR_5pct_mensile": var_storico(r),
        "CVaR_5pct_mensile": cvar_storico(r),
        "Mesi_positivi_pct": pos / r.size if r.size else np.nan,
        "N_mesi": int(r.size),
    }
