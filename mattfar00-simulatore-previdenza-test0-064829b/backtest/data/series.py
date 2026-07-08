"""
backtest.data.series — costruisce le serie di RENDIMENTI MENSILI REALI da dare
in pasto al backtest, a partire dalle strutture che il main già possiede.

Nessun import dal main: tutto viene passato come argomento (dependency
injection). Così questo package resta autonomo e testabile.
"""
import numpy as np


def serie_fondo_reale(storico_mensile: dict, fondo: str, comparto: str) -> np.ndarray:
    """
    Rendimenti mensili reali della quota del comparto, in ordine cronologico.
    `storico_mensile` è il tuo STORICO_MENSILE[fondo][comparto].
    """
    serie = storico_mensile.get(fondo, {}).get(comparto, [])
    return np.asarray(serie, dtype=float)


def serie_pac_reale(portafoglio_info: dict) -> np.ndarray:
    """
    Rendimenti mensili reali del portafoglio PAC pesato, dal dizionario
    restituito da stima_parametri_portafoglio (chiave 'rend_mensili_pesato').
    Ritorna array vuoto se non disponibile (PAC in modalità 'Semplice').
    """
    if not portafoglio_info:
        return np.array([], dtype=float)
    return np.asarray(portafoglio_info.get("rend_mensili_pesato", []), dtype=float)


def serie_singolo_asset(portafoglio_info: dict, ticker: str) -> np.ndarray:
    """
    Rendimenti mensili reali di un singolo ticker (es. per usarlo come
    benchmark, tipo SWDA/VWCE). Ricavati dal prezzi_df conservato nel
    portafoglio_info. Ritorna array vuoto se il ticker non c'è.
    """
    if not portafoglio_info or "prezzi_df" not in portafoglio_info:
        return np.array([], dtype=float)
    df = portafoglio_info["prezzi_df"]
    if ticker not in df.columns:
        return np.array([], dtype=float)
    return df[ticker].pct_change().dropna().to_numpy(dtype=float)


def allinea_ultimi_mesi(serie: np.ndarray, mesi_richiesti: int):
    """
    Prende gli ULTIMI `mesi_richiesti` rendimenti (i più recenti). Se la serie
    è più corta, la ritorna intera segnalando quanti mesi effettivi ci sono.
    Ritorna (serie_tagliata, n_effettivi, troncata_bool).
    """
    s = np.asarray(serie, dtype=float)
    if s.size == 0:
        return s, 0, False
    if s.size < mesi_richiesti:
        return s, int(s.size), False
    return s[-mesi_richiesti:], int(mesi_richiesti), True


def allinea_coppia(serie_a: np.ndarray, serie_b: np.ndarray):
    """
    Allinea due serie sull'ultimo tratto COMUNE (stessa lunghezza), utile per
    confronti portafoglio vs benchmark che hanno storici di lunghezza diversa.
    Assunzione: entrambe terminano allo stesso mese (ultimo dato disponibile).
    """
    a = np.asarray(serie_a, dtype=float)
    b = np.asarray(serie_b, dtype=float)
    n = min(a.size, b.size)
    if n == 0:
        return a[:0], b[:0], 0
    return a[-n:], b[-n:], int(n)
