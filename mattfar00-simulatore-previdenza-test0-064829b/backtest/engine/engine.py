"""
backtest.engine — esegue il backtest DETERMINISTICO su una singola traiettoria
storica reale (non stocastica). Tre modalità:

1) lump_sum          : capitale iniziale unico, nessun versamento (stile PV
                       "growth of $10k"). Ideale per confrontare allocazioni
                       "pulite".
2) con_contributi    : capitale iniziale + versamento mensile costante (DCA).
                       Il drawdown risulta attenuato (media dei prezzi d'entrata).
3) via_motore        : riusa il TUO simula_capitale, iniettato come argomento,
                       alimentandolo con i rendimenti storici reali invece che
                       con le traiettorie bootstrap. Mantiene tutta la tua
                       logica fiscale/contributiva (TFR, IRPEF, aliquote uscita).

NB: nessun import dal main. `simula_capitale` e gli altri oggetti vengono
passati come parametri.
"""
import numpy as np
import pandas as pd


def backtest_lump_sum(rend_mensili, capitale_iniziale: float = 10_000.0):
    """
    Ritorna (equity_curve, montante_finale). equity_curve è l'array del
    montante mese per mese partendo da capitale_iniziale, senza versamenti.
    """
    r = np.asarray(rend_mensili, dtype=float)
    equity = capitale_iniziale * np.cumprod(1 + r)
    return equity, float(equity[-1]) if equity.size else capitale_iniziale


def backtest_con_contributi(rend_mensili, contributo_mensile: float,
                            capitale_iniziale: float = 0.0):
    """
    DCA: versamento a inizio mese, poi rendimento del mese. Ritorna
    (equity_curve, montante_finale, totale_versato).
    """
    r = np.asarray(rend_mensili, dtype=float)
    cap = float(capitale_iniziale)
    versato = float(capitale_iniziale)
    equity = np.empty(r.size)
    for i, ri in enumerate(r):
        cap += contributo_mensile
        versato += contributo_mensile
        cap *= (1 + ri)
        equity[i] = cap
    finale = float(equity[-1]) if r.size else cap
    return equity, finale, versato


def backtest_via_motore(simula_capitale, fattori, rend_fondo_mensili,
                        rend_pac_mensili, sched, scal, vol_extra_serie,
                        vp_serie) -> pd.DataFrame:
    """
    Chiama UNA volta il tuo simula_capitale con la traiettoria STORICA REALE.
    `simula_capitale` è la funzione del main, passata come argomento.
    Le serie di rendimento devono essere lunghe len(sched)*12 (mensili).
    Ritorna il DataFrame anno-per-anno prodotto da simula_capitale.
    """
    n_mesi = len(sched) * 12
    rf = np.asarray(rend_fondo_mensili, dtype=float)
    rp = np.asarray(rend_pac_mensili, dtype=float)
    if rf.size < n_mesi or rp.size < n_mesi:
        raise ValueError(
            f"Storico insufficiente per il backtest via motore: servono "
            f"{n_mesi} mesi, disponibili fondo={rf.size}, pac={rp.size}. "
            f"Riduci la durata o accorcia l'orizzonte."
        )
    return simula_capitale(fattori, rf[:n_mesi], rp[:n_mesi], sched, scal,
                           vol_extra_serie, vp_serie)


def equity_to_returns(equity_curve, capitale_iniziale):
    """Da una equity curve (montante) ai rendimenti mensili impliciti.
    Solo per lump-sum (senza versamenti), altrimenti i versamenti falsano i
    rendimenti — per il DCA le metriche vanno calcolate sui rendimenti
    dell'asset, non sul montante."""
    eq = np.asarray(equity_curve, dtype=float)
    prev = np.concatenate([[capitale_iniziale], eq[:-1]])
    return eq / prev - 1
