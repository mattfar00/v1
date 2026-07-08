# ---------------------------------------------------------------------------
# PAC ENGINE — logica PAC estratta dal main
# ---------------------------------------------------------------------------
# Contiene tutto cio' che riguarda il PAC "base" e il portafoglio a ticker:
#   - catalogo ETF curato + whitelist accumulazione UCITS + flag distribuzione
#   - classifica_ticker (controllo acc/UCITS)
#   - quota titoli di Stato per ticker (tassazione 12,5%)
#   - GBM parametrico mensile (modalita' "Semplice")
#   - download prezzi Yahoo + stima parametri (Markowitz) + GBM Cholesky
#   - utilita' generiche: mensili->annui, netto TER, selezione percentile
# Il main importa da qui; il tab "PAC avanzato" (pac_avanzato.py) e' separato.
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# CATALOGO ETF PREDEFINITI — ticker Yahoo Finance (SOLO accumulazione UCITS)
# ---------------------------------------------------------------------------
CATALOGO_ETF = {
    "Azionario Globale": {
        "iShares Core MSCI World Acc (SWDA.MI)": "SWDA.MI",
        "Vanguard FTSE All-World Acc (VWCE.DE)": "VWCE.DE",
        "Xtrackers MSCI World Acc (XDWD.MI)": "XDWD.MI",
        "iShares MSCI ACWI Acc (SSAC.MI)": "SSAC.MI",
    },
    "Azionario USA": {
        "iShares Core S&P 500 Acc (CSSPX.MI)": "CSSPX.MI",
        "Xtrackers S&P 500 Acc (XSPX.MI)": "XSPX.MI",
        "Invesco Nasdaq-100 Acc (EQAC.MI)": "EQAC.MI",
    },
    "Azionario Europa": {
        "Xtrackers Euro Stoxx 50 Acc (XESC.MI)": "XESC.MI",
        "iShares Core MSCI EMU Acc (CEBL.MI)": "CEBL.MI",
    },
    "Azionario Mercati Emergenti": {
        "iShares Core MSCI EM IMI Acc (EIMI.MI)": "EIMI.MI",
        "Xtrackers MSCI Emerging Markets Acc (XMME.MI)": "XMME.MI",
    },
    "Obbligazionario": {
        "iShares Core Global Aggregate Bond EUR-H Acc (AGGH.MI)": "AGGH.MI",
        "Xtrackers Global Government Bond EUR-H Acc (XG7S.MI)": "XG7S.MI",
    },
    "Oro e Materie Prime": {
        "iShares Physical Gold ETC (SGLN.MI)": "SGLN.MI",
        "Invesco Physical Gold ETC (SGLD.MI)": "SGLD.MI",
        "WisdomTree Broad Commodities Acc (WCOA.MI)": "WCOA.MI",
    },
    "Immobiliare (REIT)": {
        "Xtrackers FTSE EPRA/NAREIT Global Acc (XREA.MI)": "XREA.MI",
    },
}
TICKER_TO_NOME = {t: nome for cat in CATALOGO_ETF.values() for nome, t in cat.items()}
WHITELIST_ACC_UCITS = set(TICKER_TO_NOME.keys())

# Classi di asset per il modello multi-bucket del tab PAC unificato.
# "Azioni singole" e' per stock individuali da input manuale (es. AAPL,
# ENI.MI): bucket separato, cosi' il rischio idiosincratico non inquina
# la stima della classe azionaria diversificata.
CLASSI_ASSET = ["Azionario", "Obbligazionario", "Oro/Materie prime",
                "Immobiliare", "Azioni singole"]
_CATEGORIA_TO_CLASSE = {
    "Azionario Globale": "Azionario",
    "Azionario USA": "Azionario",
    "Azionario Europa": "Azionario",
    "Azionario Mercati Emergenti": "Azionario",
    "Obbligazionario": "Obbligazionario",
    "Oro e Materie Prime": "Oro/Materie prime",
    "Immobiliare (REIT)": "Immobiliare",
}
TICKER_TO_CLASSE = {t: _CATEGORIA_TO_CLASSE.get(cat_nome, "Azionario")
                    for cat_nome, etfs in CATALOGO_ETF.items()
                    for t in etfs.values()}

# Quota "titoli di Stato/white list" per ticker (aliquota ridotta 12,5%).
# Solo i ticker con composizione inequivocabile; i misti vanno impostati a mano.
QUOTA_TITOLI_STATO_TICKER = {
    "XG7S.MI": 1.00,   # Xtrackers Global Government Bond: solo titoli di Stato
}

# Ticker noti come NON ad accumulazione (a distribuzione) o da verificare.
ETF_FLAG = {
    "EQQQ.MI": "a DISTRIBUZIONE — la versione ad accumulo è EQAC.MI (o SB.. classi acc)",
    "EXSA.MI": "a DISTRIBUZIONE (iShares STOXX Europe 600, dist)",
    "IWDP.MI": "a DISTRIBUZIONE (Property *Yield*)",
    "IBGX.MI": "a DISTRIBUZIONE (iShares Euro Gov Bond 3-5y, dist)",
    "IEBC.MI": "a DISTRIBUZIONE (iShares Euro Corporate Bond, dist)",
    "EMU.MI":  "DA VERIFICARE (esistono classi dist e acc con ticker vicini)",
    "IWRD.MI": "a DISTRIBUZIONE (versione acc: SWDA.MI)",
    "VWRL.MI": "a DISTRIBUZIONE (versione acc: VWCE.DE)",
}


def classifica_ticker(ticker: str):
    """
    Ritorna (stato, nota) sullo stato di accumulazione/UCITS di un ticker.
    stato ∈ {"ok", "warn", "sconosciuto"}. Yahoo non espone in modo affidabile
    il flag dist/acc: la verifica si basa sulla whitelist curata.
    """
    t = ticker.strip().upper()
    if t in WHITELIST_ACC_UCITS:
        return "ok", "accumulazione UCITS (da catalogo curato)"
    if t in ETF_FLAG:
        return "warn", ETF_FLAG[t]
    return "sconosciuto", ("non in whitelist: può essere un ETF non catalogato "
                           "o un'AZIONE SINGOLA. Le azioni singole sono ammesse "
                           "ma hanno volatilità molto più alta di un ETF "
                           "diversificato e possono compromettere l'analisi "
                           "(la stima storica di rendimento/rischio su un solo "
                           "titolo è poco affidabile). Se è un ETF, verifica "
                           "sul KID che sia ad accumulazione e UCITS.")


# ---------------------------------------------------------------------------
# UTILITÀ GENERICHE (usate anche dal main per il fondo)
# ---------------------------------------------------------------------------
def mensili_ad_annui(mat_mensile: np.ndarray) -> np.ndarray:
    """Compone una matrice di rendimenti mensili (n x anni*12) in annui (n x anni)."""
    n, mesi = mat_mensile.shape
    anni = mesi // 12
    return np.prod(1 + mat_mensile[:, :anni * 12].reshape(n, anni, 12), axis=2) - 1


def rendimento_netto_pac(r, ter_p):
    """
    Rendimento netto annuo del PAC al netto del solo TER: il PAC in Italia NON
    tassa il rendimento anno per anno — la plusvalenza è tassata solo in
    uscita/vendita. Netto di SOLI COSTI ricorrenti, non di imposta.
    """
    r = np.asarray(r, dtype=float)
    return (1 + r) * (1 - ter_p) - 1


def seleziona_traiettoria_per_percentile(rendimenti: np.ndarray, percentile: int):
    montanti = np.prod(1 + rendimenti, axis=1)
    ordine = np.argsort(montanti)
    idx = int(round((percentile / 100) * (len(ordine) - 1)))
    return rendimenti[ordine[idx]]


# ---------------------------------------------------------------------------
# SHRINKAGE DEL DRIFT VERSO UN'ANCORA DI LUNGO PERIODO
# ---------------------------------------------------------------------------
# La media stimata su un campione corto e' molto rumorosa (con 10 anni di dati
# e vol 15% l'errore standard sulla media annua e' ~4,7 punti). La correzione
# standard (stile Bayes/James-Stein) e' combinare la stima campionaria con
# un'ancora di lungo periodo:  drift = w*storico + (1-w)*ancora,
# con w = mesi_campione / MESI_PIENA_FIDUCIA (cap a 1).
# L'ancora e' un input dell'utente: puo' essere la media secolare (~6,5%
# nominale per l'azionario mondiale) o una Capital Market Assumption
# aggiornata (JPMorgan LTCMA, Vanguard, BlackRock: pubbliche e gratuite).
MESI_PIENA_FIDUCIA = 240  # 20 anni: oltre, lo storico pesa il 100%


def cagr_da_mensili(serie_mensile) -> float:
    """CAGR annuo (geometrico) da una serie di rendimenti mensili semplici."""
    serie = np.asarray(serie_mensile, dtype=float)
    if serie.size == 0:
        return 0.0
    log_tot = np.sum(np.log1p(serie))
    return float(np.exp(log_tot * 12.0 / serie.size) - 1.0)


def shrink_verso_ancora(cagr_campione: float, n_mesi: int, ancora: float,
                        mesi_pieni: int = MESI_PIENA_FIDUCIA):
    """
    Ritorna (cagr_corretto, peso_campione). peso = min(1, n_mesi/mesi_pieni).
    """
    w = min(1.0, max(0.0, n_mesi / float(mesi_pieni)))
    return w * cagr_campione + (1.0 - w) * ancora, w


def ricentra_mensili(serie_mensile, cagr_target: float):
    """
    Ricentra una serie di rendimenti mensili in modo MOLTIPLICATIVO cosicche'
    il suo CAGR diventi esattamente `cagr_target`, preservando volatilita',
    autocorrelazione e forma della distribuzione (a meno di una traslazione
    dei log-rendimenti). Usata per correggere le serie prima del bootstrap.
    """
    serie = np.asarray(serie_mensile, dtype=float)
    if serie.size == 0:
        return serie
    cagr_camp = cagr_da_mensili(serie)
    fattore = ((1.0 + cagr_target) / (1.0 + cagr_camp)) ** (1.0 / 12.0)
    return (1.0 + serie) * fattore - 1.0


def _shock_t_student(rng, size, nu, condividi_righe=False):
    """
    Shock a code grasse: T di Student con nu gradi di liberta', riscalata a
    varianza unitaria (nu>2). Con nu=None/0 ritorna normali standard.
    Con condividi_righe=True e size 2D (periodi x asset) il fattore
    chi-quadro e' unico per riga: T multivariata, le code arrivano
    INSIEME su tutti gli asset dello stesso mese (crash congiunti).
    """
    if not nu or nu <= 2:
        return rng.standard_normal(size)
    z = rng.standard_normal(size)
    if condividi_righe and len(size) == 2:
        g = rng.chisquare(nu, size=(size[0], 1)) / nu   # condiviso sugli asset
    else:
        g = rng.chisquare(nu, size=size) / nu
    return z / np.sqrt(g) * np.sqrt((nu - 2.0) / nu)


# ---------------------------------------------------------------------------
# GBM PARAMETRICO MENSILE (modalità PAC "Semplice" + fallback)
# ---------------------------------------------------------------------------
@st.cache_data
def genera_rendimenti_gbm(rend_cagr: float, vol: float, durata: int,
                          n: int = 200, seed: int = 7, nu: float = 0.0):
    """
    GBM lognormale MENSILE. `rend_cagr` e' il rendimento composto annuo
    (CAGR) atteso: la traiettoria MEDIANA compone esattamente a quel tasso.
    (La media aritmetica implicita e' piu' alta di ~vol^2/2: la correzione
    di Ito' e' incorporata nella parametrizzazione del log-drift.)
    `vol` e' la volatilita' annua. `nu`>2 attiva code grasse (T di Student).
    Ritorna (n x durata*12).
    """
    rng = np.random.default_rng(seed)
    rend_m = (1 + rend_cagr) ** (1 / 12) - 1
    vol_m = vol / np.sqrt(12)
    sigma = np.sqrt(np.log(1 + (vol_m**2) / ((1 + rend_m)**2)))
    mu = np.log(1 + rend_m)   # mediana mensile = (1+CAGR)^(1/12)
    z = _shock_t_student(rng, (n, durata * 12), nu)
    return np.exp(mu + sigma * z) - 1.0


# ---------------------------------------------------------------------------
# PORTAFOGLIO A TICKER: download storico, stima parametri, Cholesky
# ---------------------------------------------------------------------------
def parse_ticker_pesi(tickers_str: str, pesi_str: str):
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    pesi_raw = [p.strip() for p in pesi_str.split(",") if p.strip()]
    if len(tickers) == 0:
        raise ValueError("Inserisci almeno un ticker.")
    if len(pesi_raw) != len(tickers):
        raise ValueError(f"Hai {len(tickers)} ticker ma {len(pesi_raw)} pesi.")
    pesi = np.array([float(p) for p in pesi_raw])
    if pesi.sum() <= 0:
        raise ValueError("La somma dei pesi deve essere positiva.")
    pesi = pesi / pesi.sum()
    return tickers, pesi


@st.cache_data(show_spinner=False)
def scarica_prezzi_mensili(tickers: tuple, anni: int):
    """
    Prezzi mensili AGGIUSTATI (auto_adjust=True: dividendi/split incorporati).
    Fondamentale per strumenti a distribuzione o azioni singole: senza
    aggiustamento ogni stacco appare come un calo fittizio.
    """
    import yfinance as yf
    from datetime import date
    from dateutil.relativedelta import relativedelta

    end = date.today()
    start = end - relativedelta(years=anni)

    serie = {}
    for t in tickers:
        data = yf.download(t, start=start.isoformat(), end=end.isoformat(),
                           progress=False, auto_adjust=True, actions=False)
        if data is None or data.empty:
            raise ValueError(f"Nessun dato scaricato per '{t}'. Verifica il ticker su Yahoo.")
        if "Close" in data.columns:
            col_data = data["Close"]
        else:
            col_data = data.iloc[:, 0]
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]
        serie[t] = col_data.resample("ME").last()

    df = pd.DataFrame(serie).dropna()
    if len(df) < 24:
        raise ValueError(
            f"Storico comune troppo corto ({len(df)} mesi): riduci gli anni o "
            f"verifica i ticker."
        )
    return df


def stima_parametri_portafoglio(prezzi_df: pd.DataFrame, pesi: np.ndarray):
    rend_mensili = prezzi_df.pct_change().dropna()
    media_mensile = rend_mensili.mean().values
    cov_mensile = rend_mensili.cov().values
    corr = rend_mensili.corr().values
    rend_annuo_asset = (1 + media_mensile) ** 12 - 1
    vol_annua_asset = rend_mensili.std().values * np.sqrt(12)
    rend_portafoglio = float(np.dot(pesi, rend_annuo_asset))
    vol_portafoglio = float(np.sqrt(pesi @ (cov_mensile * 12) @ pesi))
    cov_reg = cov_mensile + np.eye(len(pesi)) * 1e-10
    L = np.linalg.cholesky(cov_reg)
    rend_mensili_pesato = (rend_mensili.values @ pesi)
    # CAGR (geometrico) del portafoglio ribilanciato mensilmente: e' il tasso
    # a cui il campione ha COMPOSTO davvero — sempre < media aritmetica
    # annualizzata (vol drag ~ sigma^2/2). E' la base per lo shrinkage.
    cagr_portafoglio = cagr_da_mensili(rend_mensili_pesato)
    cagr_asset = np.array([cagr_da_mensili(rend_mensili[c].values)
                           for c in rend_mensili.columns])
    return {
        "tickers": list(prezzi_df.columns),
        "rend_annuo_asset": rend_annuo_asset,
        "vol_annua_asset": vol_annua_asset,
        "corr": corr,
        "media_mensile": media_mensile,
        "cholesky_mensile": L,
        "rend_portafoglio": rend_portafoglio,
        "vol_portafoglio": vol_portafoglio,
        "cagr_portafoglio": cagr_portafoglio,
        "cagr_asset": cagr_asset,
        "n_mesi_storico": len(rend_mensili),
        "prezzi_df": prezzi_df,
        "rend_mensili_pesato": rend_mensili_pesato,
    }


@st.cache_data(show_spinner=False)
def genera_rendimenti_portafoglio_gbm(media_mensile, cholesky_mensile, pesi,
                                      durata_anni: int, cagr_target=None,
                                      n: int = 200, seed: int = 13,
                                      nu: float = 0.0):
    """
    Traiettorie MENSILI del portafoglio (n x durata*12) con shock correlati
    (Cholesky). Se `cagr_target` e' dato (CAGR annuo composto, tipicamente
    dallo shrinkage verso l'ancora di lungo periodo), il drift viene
    TRASLATO di una costante uguale per tutti gli asset in modo che la
    traiettoria MEDIANA del portafoglio componga ~ a quel tasso.

    Correzione di Ito': il drift aritmetico mensile necessario e'
        mu_arit = (1+CAGR)^(1/12) - 1 + sigma_p_m^2 / 2
    perche' componendo rendimenti semplici con rumore, la mediana perde
    ~sigma^2/2 rispetto alla media aritmetica (volatility drag).
    `nu`>2 attiva code grasse (T di Student multivariata: il fattore di
    coda e' condiviso tra gli asset dello stesso mese).
    """
    rng = np.random.default_rng(seed)
    n_asset = len(pesi)
    mesi_tot = durata_anni * 12
    drift = media_mensile.copy()
    if cagr_target is not None:
        sig_p_m = float(np.linalg.norm(cholesky_mensile.T @ np.asarray(pesi)))
        target_m = (1 + cagr_target) ** (1 / 12) - 1 + 0.5 * sig_p_m ** 2
        attuale_m = float(np.dot(pesi, drift))
        drift = drift + (target_m - attuale_m)

    traiettorie = np.zeros((n, mesi_tot))
    for s in range(n):
        z = _shock_t_student(rng, (mesi_tot, n_asset), nu, condividi_righe=True)
        shock_mensili = z @ cholesky_mensile.T
        rend_mensili_asset = drift + shock_mensili
        traiettorie[s] = rend_mensili_asset @ pesi
    return traiettorie
