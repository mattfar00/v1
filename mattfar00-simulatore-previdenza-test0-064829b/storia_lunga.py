# ---------------------------------------------------------------------------
# STORIA LUNGA — livello dati comune per Fondo e PAC (Blocco B)
# ---------------------------------------------------------------------------
# Scarica e normalizza le serie storiche lunghe usate per caratterizzare gli
# ETF dell'anagrafica (data/anagrafica_etf.json) e, in prospettiva, i
# benchmark dei comparti del fondo. PRINCIPIO: solo esposizioni STATICHE
# (asset class larghe: azionario per area, obbligazionario aggregate/gov,
# oro). Gli ETF settoriali/tematici sono esclusi by design: la loro
# esposizione cambia nel tempo e non e' ricostruibile con un proxy fisso.
#
# Fonti (tutte gratuite, scaricate a runtime con cache; un CSV in
# data/storia_lunga/<serie>.csv ha SEMPRE la precedenza — formato: due
# colonne "date,value" con value = LIVELLO/prezzo, non rendimento):
#   - Shiller (mirror GitHub 'datasets/s-and-p-500'): S&P composite,
#     dividendi, CPI USA, tassi 10Y — mensile dal 1871.
#   - Yahoo Finance (yfinance): fondi comuni USA a storia lunga con NAV
#     total-return (VFINX 1976, VEURX 1990, VEIEX 1994, VBMFX 1986,
#     QQQ 1999, VUSXX 1992...).
#   - FRED (endpoint CSV): oro LBMA dal 1968, CPI Italia dal ~1955.
# Ogni funzione ritorna pd.Series MENSILE di RENDIMENTI SEMPLICI, indice
# fine-mese, oppure None se la fonte non risponde (mai eccezioni in UI).
# ---------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import streamlit as st

CARTELLA_LOCALE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "data", "storia_lunga")

URL_SHILLER = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
URL_FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

# Ticker Yahoo dei proxy a storia lunga (NAV aggiustato = total return).
PROXY_YAHOO = {
    "vfinx": "VFINX",   # S&P 500, dal 1976
    "veurx": "VEURX",   # Europa sviluppata (USD), dal 1990
    "veiex": "VEIEX",   # Mercati emergenti (USD), dal 1994
    "vbmfx": "VBMFX",   # Aggregate bond USA (USD), dal 1986
    "rpibx": "RPIBX",   # Bond internazionali NON coperti, dal 1986
    "qqq":   "QQQ",     # Nasdaq-100 TR, dal 1999
    "vusxx": "VUSXX",   # T-bill/cash, dal 1992
}
# Serie FRED: (id primario, eventuali fallback)
PROXY_FRED = {
    "gold_lbma": ("GOLDAMGBD228NLBM", ["GOLDPMGBD228NLBM"]),
    "cpi_it": ("ITACPIALLMINMEI", ["CPALTT01ITM661N"]),
}

# Pesi del composite azionario globale (esposizione STATICA dichiarata,
# ~MSCI ACWI odierno senza Pacifico). Rinormalizzati sulle serie disponibili
# in ciascun mese (pre-1994 niente EM, pre-1990 solo USA).
PESI_WORLD_COMPOSITE = {"vfinx": 0.60, "veurx": 0.30, "veiex": 0.10}


# ---------------------------------------------------------------------------
# UTILITÀ
# ---------------------------------------------------------------------------
def _da_livelli_a_rendimenti(livelli: pd.Series) -> pd.Series:
    livelli = livelli.dropna().astype(float)
    livelli = livelli.resample("ME").last().dropna()
    return livelli.pct_change().dropna()


def _csv_locale(nome: str):
    """Override manuale: data/storia_lunga/<nome>.csv con colonne date,value."""
    percorso = os.path.join(CARTELLA_LOCALE, f"{nome}.csv")
    if not os.path.isfile(percorso):
        return None
    df = pd.read_csv(percorso)
    df.columns = [c.strip().lower() for c in df.columns]
    serie = pd.Series(df["value"].values,
                      index=pd.to_datetime(df["date"]), name=nome)
    return _da_livelli_a_rendimenti(serie)


@st.cache_data(show_spinner=False, ttl=7 * 24 * 3600)
def carica_yahoo_max(nome: str):
    """Storico massimo mensile total-return di un proxy Yahoo. None se fallisce."""
    loc = _csv_locale(nome)
    if loc is not None:
        return loc
    try:
        import yfinance as yf
        data = yf.download(PROXY_YAHOO[nome], period="max", interval="1mo",
                           progress=False, auto_adjust=True, actions=False)
        if data is None or data.empty:
            return None
        col = data["Close"]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        return _da_livelli_a_rendimenti(col)
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=30 * 24 * 3600)
def carica_fred(nome: str):
    """Serie FRED (livelli) -> rendimenti/variazioni mensili. None se fallisce."""
    loc = _csv_locale(nome)
    if loc is not None:
        return loc
    sid, fallback = PROXY_FRED[nome]
    for s in [sid] + list(fallback):
        try:
            df = pd.read_csv(URL_FRED.format(sid=s))
            df.columns = ["date", "value"]
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            serie = pd.Series(df["value"].values,
                              index=pd.to_datetime(df["date"]), name=nome)
            out = _da_livelli_a_rendimenti(serie)
            if len(out) > 24:
                return out
        except Exception:
            continue
    # Fallback per l'ORO: le serie LBMA su FRED sono state dismesse (2024).
    # Yahoo: futures COMEX GC=F dal 2000, poi ETC GLD dal 2004.
    if nome == "gold_lbma":
        try:
            import yfinance as yf
            for tk in ("GC=F", "GLD"):
                data = yf.download(tk, period="max", interval="1mo",
                                   progress=False, auto_adjust=True,
                                   actions=False)
                if data is None or data.empty:
                    continue
                col = data["Close"]
                if isinstance(col, pd.DataFrame):
                    col = col.iloc[:, 0]
                out = _da_livelli_a_rendimenti(col)
                if len(out) > 24:
                    return out
        except Exception:
            pass
    return None


@st.cache_data(show_spinner=False, ttl=30 * 24 * 3600)
def carica_shiller():
    """
    Dataset Shiller (mensile dal 1871). Ritorna dict di pd.Series:
      sp500_tr   rendimenti mensili S&P TOTAL RETURN (prezzo + dividendo/12)
      cpi_us     variazione mensile CPI USA
      bond10_tr  rendimenti mensili RICOSTRUITI del decennale USA:
                 r ~ y/12 + Dmod*(y_prec - y_curr), Dmod = duration modificata.
                 E' una ricostruzione (no dati di prezzo reali): usarla per
                 ANCORE e sanity check, non come verita' mensile fine.
    None se la fonte non risponde.
    """
    try:
        loc = os.path.join(CARTELLA_LOCALE, "shiller.csv")
        df = pd.read_csv(loc) if os.path.isfile(loc) else pd.read_csv(URL_SHILLER)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()

        p = df["SP500"].astype(float)
        d = df["Dividend"].astype(float)
        tr = (p + d / 12.0) / p.shift(1) - 1.0
        sp500_tr = tr.dropna()
        sp500_tr.index = sp500_tr.index + pd.offsets.MonthEnd(0)

        cpi = df["Consumer Price Index"].astype(float)
        cpi_us = cpi.pct_change().dropna()
        cpi_us.index = cpi_us.index + pd.offsets.MonthEnd(0)

        y = df["Long Interest Rate"].astype(float) / 100.0
        dmod = (1.0 - (1.0 + y) ** -10) / y.replace(0, np.nan)
        bond = (y.shift(1) / 12.0 + dmod * (y.shift(1) - y)).dropna()
        bond.index = bond.index + pd.offsets.MonthEnd(0)

        return {"sp500_tr": sp500_tr, "cpi_us": cpi_us, "bond10_tr": bond}
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=7 * 24 * 3600)
def costruisci_world_composite():
    """
    Azionario globale a ESPOSIZIONE STATICA: 60% USA + 30% Europa + 10% EM,
    pesi rinormalizzati ogni mese sulle serie disponibili (pre-1994 senza EM,
    pre-1990 solo USA). Approssimazione dichiarata di MSCI World/ACWI; un
    data/storia_lunga/world_composite.csv (es. MSCI World ufficiale) la
    sostituisce integralmente.
    """
    loc = _csv_locale("world_composite")
    if loc is not None:
        return loc
    basi = {k: carica_yahoo_max(k) for k in PESI_WORLD_COMPOSITE}
    basi = {k: v for k, v in basi.items() if v is not None}
    if "vfinx" not in basi:
        return None
    df = pd.DataFrame(basi)
    pesi = pd.DataFrame(
        {k: np.where(df[k].notna(), PESI_WORLD_COMPOSITE[k], 0.0) for k in df},
        index=df.index)
    somma = pesi.sum(axis=1)
    out = (df.fillna(0.0) * pesi).sum(axis=1) / somma.replace(0, np.nan)
    return out.dropna()


# ---------------------------------------------------------------------------
# REGISTRO SERIE — punto d'accesso unico per i motori (Fondo e PAC)
# ---------------------------------------------------------------------------
def carica_tutte_le_serie():
    """
    Ritorna (serie, note): serie = dict nome -> pd.Series di rendimenti
    mensili (senza intersezione: ogni serie tiene la SUA lunghezza),
    note = dict nome -> stringa fonte/avviso. Include il CPI per i reali.
    """
    serie, note = {}, {}

    sh = carica_shiller()
    if sh is not None:
        serie["shiller_sp500"] = sh["sp500_tr"]
        serie["cpi_us"] = sh["cpi_us"]
        serie["bond10y_usa_ricostruito"] = sh["bond10_tr"]
        note["shiller_sp500"] = "Shiller/GitHub, TR dal 1871"
        note["cpi_us"] = "CPI USA (Shiller)"
        note["bond10y_usa_ricostruito"] = ("RICOSTRUZIONE dai tassi 10Y "
                                           "(solo ancore/sanity check)")
    else:
        note["shiller_sp500"] = "⚠️ fonte non raggiungibile"

    for nome in PROXY_YAHOO:
        s = carica_yahoo_max(nome)
        if s is not None:
            serie[nome] = s
            note[nome] = f"Yahoo {PROXY_YAHOO[nome]}, NAV total return"
        else:
            note[nome] = f"⚠️ Yahoo {PROXY_YAHOO[nome]} non scaricato"

    for nome in PROXY_FRED:
        s = carica_fred(nome)
        if s is not None:
            serie[nome] = s
            note[nome] = f"FRED {PROXY_FRED[nome][0]}"
        else:
            note[nome] = (f"⚠️ FRED {PROXY_FRED[nome][0]} non disponibile "
                          f"(per l'oro il fallback Yahoo GC=F/GLD è automatico) — "
                          f"drop-in: data/storia_lunga/{nome}.csv")

    wc = costruisci_world_composite()
    if wc is not None:
        serie["world_composite"] = wc
        note["world_composite"] = ("composite statico 60/30/10 "
                                   "USA/Europa/EM (rinormalizzato)")
    return serie, note


def serie_reale(rend: pd.Series, cpi: pd.Series) -> pd.Series:
    """Deflaziona rendimenti mensili con l'inflazione mensile allineata."""
    df = pd.concat([rend, cpi], axis=1, join="inner").dropna()
    return (1 + df.iloc[:, 0]) / (1 + df.iloc[:, 1]) - 1


def _cagr(s: pd.Series) -> float:
    if s is None or len(s) == 0:
        return np.nan
    return float(np.exp(np.log1p(s).sum() * 12 / len(s)) - 1)


# ---------------------------------------------------------------------------
# TAB DI VERIFICA DATI (Blocco B: nessun motore toccato)
# ---------------------------------------------------------------------------
def render_tab_dati():
    st.subheader("📚 Storia lunga — livello dati comune (verifica)")
    st.caption(
        "Serie proxy a **esposizione statica** per caratterizzare gli ETF "
        "dell'anagrafica e i benchmark dei comparti. Nessuna intersezione: "
        "ogni serie conserva la propria lunghezza. Gli ETF settoriali/"
        "tematici sono esclusi by design (esposizione non ricostruibile). "
        "Un CSV in `data/storia_lunga/<serie>.csv` (date,value in livelli) "
        "sostituisce la fonte remota. Questi dati NON alimentano ancora i "
        "motori: prima si verificano qui (Blocco B), poi mapping e beta "
        "(Blocco C), poi i motori (Blocco D)."
    )
    with st.spinner("Scarico/leggo le serie lunghe (cache 7-30 giorni)..."):
        serie, note = carica_tutte_le_serie()

    if not serie:
        st.error("Nessuna serie caricata: ambiente senza rete? Usa i drop-in "
                 "CSV in data/storia_lunga/.")
        return

    cpi_us = serie.get("cpi_us")
    righe = []
    for nome, s in serie.items():
        if nome == "cpi_us":
            continue
        reale = serie_reale(s, cpi_us) if cpi_us is not None else None
        ann = (1 + s) .rolling(12).apply(np.prod, raw=True) - 1
        righe.append({
            "Serie": nome,
            "Fonte": note.get(nome, ""),
            "Da": s.index.min().strftime("%Y-%m"),
            "A": s.index.max().strftime("%Y-%m"),
            "Mesi": len(s),
            "CAGR nom. (%)": round(_cagr(s) * 100, 2),
            "CAGR reale USD (%)": (round(_cagr(reale) * 100, 2)
                                    if reale is not None and len(reale) else np.nan),
            "Vol (%)": round(float(s.std(ddof=1)) * np.sqrt(12) * 100, 1),
            "Peggior mese (%)": round(float(s.min()) * 100, 1),
            "Peggior 12m (%)": (round(float(ann.min()) * 100, 1)
                                 if len(ann.dropna()) else np.nan),
        })
    st.dataframe(pd.DataFrame(righe), use_container_width=True, hide_index=True)

    problemi = [f"{k}: {v}" for k, v in note.items() if v.startswith("⚠️")]
    if problemi:
        st.warning("Fonti non disponibili — " + " · ".join(problemi))

    st.caption(
        "Sanity check consigliati: CAGR reale azionario USA ~6-7% dal 1871; "
        "oro reale ~1-2% dal 1968 con vol ~15-20%; bond10y ricostruito reale "
        "~1,5-2,5%. Se vedi numeri lontani da questi, la fonte o il "
        "drop-in ha un problema. Il CAGR 'reale' qui usa il CPI USA (serie "
        "in USD); la deflazione in euro reali italiani (CPI Italia) entra "
        "nei motori al Blocco D."
    )

