# ---------------------------------------------------------------------------
# MODELLO LUNGO — factor modeling, ricostruzione, validazione, stress test
# ---------------------------------------------------------------------------
# Blocco C+D+E. Usa il livello dati di storia_lunga.py e le anagrafiche
# (data/anagrafica_etf.json, data/fondi/*.json) per:
#   1. RICOSTRUIRE i comparti del fondo dal mix benchmark (splice: NAV reale
#      dove esiste, mix proxy alfa-aggiustato prima) con verifica RBSA.
#   2. ESTENDERE le classi del PAC coi proxy lunghi (beta su overlap).
#   3. BOOTSTRAP CONGIUNTO rendimenti+INFLAZIONE (stessi indici: correlazione
#      empirica asset/inflazione preservata) -> montanti in euro reali.
#   4. VALIDAZIONE OUT-OF-SAMPLE: stimi fino all'anno X, simuli, confronti
#      con quello che e' successo davvero (copertura banda P10-P90).
#   5. STRESS TEST DETERMINISTICI: riapplica episodi storici al mix corrente.
#   6. STIMA dei parametri di modello dai dati: nu (code), kappa (mean
#      reversion via variance ratio), peso bayesiano dello shrinkage.
# ---------------------------------------------------------------------------

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from storia_lunga import carica_tutte_le_serie, serie_reale
from pac_engine import cagr_da_mensili, ricentra_mensili, shrink_verso_ancora

_BASE = os.path.dirname(os.path.abspath(__file__))

# Mapping classe PAC -> serie lunga (per l'estensione delle classi)
CLASSE_TO_SERIE = {
    "Azionario": "world_composite",
    "Obbligazionario": "vbmfx",
    "Oro/Materie prime": "gold_lbma",
    # Immobiliare e Azioni singole: nessun proxy statico affidabile -> no ext.
}

EPISODI_STRESS = {
    "1973-74 shock petrolifero": ("1972-12", "1974-12"),
    "2000-02 dot-com": ("2000-03", "2002-12"),
    "2008-09 crisi finanziaria": ("2007-10", "2009-03"),
    "2022 shock tassi/inflazione": ("2021-12", "2022-12"),
}


# ---------------------------------------------------------------------------
# DATI DI BASE
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def rendimenti_csv_fondo(percorso_csv: str) -> pd.DataFrame:
    """Rendimenti mensili dei comparti dal CSV quote (indice datetime)."""
    df = pd.read_csv(percorso_csv)
    df["data"] = pd.to_datetime(dict(year=df["anno"], month=df["mese"], day=1))
    df = df.set_index("data").drop(columns=["anno", "mese"]).sort_index()
    df.index = df.index + pd.offsets.MonthEnd(0)
    return df.astype(float).pct_change()


def _serie_lunghe():
    serie, note = carica_tutte_le_serie()
    return serie, note


# ---------------------------------------------------------------------------
# RBSA-LITE E RICOSTRUZIONE
# ---------------------------------------------------------------------------
def stima_pesi_rbsa(y: pd.Series, X: pd.DataFrame):
    """
    Style analysis semplificata: minimi quadrati con intercetta, pesi
    troncati a >=0 e rinormalizzati (approssimazione della regressione
    vincolata alla Sharpe; adeguata per 2-5 regressori poco collineari).
    Ritorna (pesi_dict, alpha_mensile, r2, n_mesi_overlap).
    """
    df = pd.concat([y, X], axis=1, join="inner").dropna()
    if len(df) < 24:
        return None
    b = df.iloc[:, 0].values
    A = df.iloc[:, 1:].values
    A1 = np.column_stack([A, np.ones(len(b))])
    coef, *_ = np.linalg.lstsq(A1, b, rcond=None)
    w = np.clip(coef[:-1], 0.0, None)
    w = w / w.sum() if w.sum() > 0 else np.full(A.shape[1], 1.0 / A.shape[1])
    fitted = A @ w
    alpha = float(np.mean(b - fitted))
    resid = b - fitted - alpha
    r2 = 1.0 - float(np.var(resid)) / max(float(np.var(b)), 1e-12)
    return ({c: float(wi) for c, wi in zip(df.columns[1:], w)},
            alpha, r2, len(df))


def _mix_benchmark(benchmark: list, serie: dict):
    """Serie del mix dichiarato: somma pesata (pesi aggregati per serie)."""
    pesi = {}
    for voce in benchmark or []:
        pesi[voce["serie"]] = pesi.get(voce["serie"], 0.0) + float(voce["peso"])
    disponibili = {k: serie[k] for k in pesi if k in serie}
    if not disponibili:
        return None, pesi
    df = pd.DataFrame(disponibili).dropna()
    tot = sum(pesi[k] for k in disponibili)
    mix = sum(df[k] * (pesi[k] / tot) for k in disponibili)
    return mix, pesi


def ricostruisci_da_benchmark(rend_reale: pd.Series, benchmark: list,
                              serie: dict):
    """
    Splice: rendimenti REALI dove esistono; prima, mix benchmark corretto
    dell'alfa medio dell'overlap (cattura costi/tracking della gestione).
    Ritorna dict con serie estesa e diagnostica, o None.
    """
    mix, pesi_dich = _mix_benchmark(benchmark, serie)
    if mix is None:
        return None
    rend_reale = rend_reale.dropna()
    if len(rend_reale) < 24:
        # NESSUN NAV (o troppo poco): serie = MIX BENCHMARK PURO, alfa 0.
        # Serve per fondi appena aggiunti senza CSV quote: simulabili da
        # subito, con l'avvertenza che costi/tracking reali non sono inclusi
        # (inserire l'ISC in anagrafica quando disponibile).
        return {"estesa": mix, "giunzione": None, "alpha_annuo": 0.0,
                "pesi_dichiarati": pesi_dich, "rbsa": None,
                "mesi_reali": len(rend_reale), "mesi_estesi": len(mix)}
    X = pd.DataFrame({k: serie[k] for k in pesi_dich if k in serie})
    rbsa = stima_pesi_rbsa(rend_reale, X)

    overlap = pd.concat([rend_reale, mix], axis=1, join="inner").dropna()
    alpha_m = float((overlap.iloc[:, 0] - overlap.iloc[:, 1]).mean())
    inizio_reale = rend_reale.index.min()
    pre = ((1 + mix[mix.index < inizio_reale]) * (1 + alpha_m) - 1)
    estesa = pd.concat([pre, rend_reale]).sort_index()

    return {
        "estesa": estesa,
        "giunzione": inizio_reale,
        "alpha_annuo": (1 + alpha_m) ** 12 - 1,
        "pesi_dichiarati": pesi_dich,
        "rbsa": rbsa,          # (pesi_stimati, alpha, r2, n) o None
        "mesi_reali": len(rend_reale),
        "mesi_estesi": len(estesa),
    }


@st.cache_data(show_spinner=False)
def serie_estesa_comparto(percorso_csv: str, comparto: str, cfg_json: str):
    """
    Serie mensile ESTESA (np.array) del comparto se l'anagrafica lo consente
    (ricostruzione == 'consentita' e benchmark presente), altrimenti None.
    cfg_json: anagrafica del comparto serializzata (hashabilita' cache).
    """
    cfg = json.loads(cfg_json)
    if cfg.get("ricostruzione") != "consentita" or not cfg.get("benchmark"):
        return None
    rend = rendimenti_csv_fondo(percorso_csv)
    if comparto not in rend.columns:
        return None
    serie, _ = _serie_lunghe()
    ric = ricostruisci_da_benchmark(rend[comparto], cfg["benchmark"], serie)
    if ric is None:
        return None
    return ric["estesa"].values.astype(float)


def estendi_classi_pac(serie_classi: dict):
    """
    serie_classi: dict nome_classe -> pd.Series rendimenti mensili (ETF).
    Estende ogni classe col proxy lungo (beta+alfa su overlap). Le classi
    senza proxy restano invariate. Ritorna (dict esteso, diagnostica).
    """
    lunghe, _ = _serie_lunghe()
    out, diag = {}, {}
    for classe, s in serie_classi.items():
        proxy_nome = CLASSE_TO_SERIE.get(classe)
        proxy = lunghe.get(proxy_nome) if proxy_nome else None
        if proxy is None:
            out[classe] = s
            diag[classe] = "nessun proxy statico: solo storico ETF"
            continue
        ov = pd.concat([s, proxy], axis=1, join="inner").dropna()
        if len(ov) < 24:
            out[classe] = s
            diag[classe] = f"overlap col proxy troppo corto ({len(ov)} mesi)"
            continue
        beta = float(np.cov(ov.iloc[:, 0], ov.iloc[:, 1])[0, 1] /
                     max(np.var(ov.iloc[:, 1]), 1e-12))
        alpha = float(ov.iloc[:, 0].mean() - beta * ov.iloc[:, 1].mean())
        corr = float(np.corrcoef(ov.iloc[:, 0], ov.iloc[:, 1])[0, 1])
        pre = (alpha + beta * proxy[proxy.index < s.index.min()])
        out[classe] = pd.concat([pre, s]).sort_index()
        diag[classe] = (f"proxy {proxy_nome}: beta {beta:.2f}, corr {corr:.2f}, "
                        f"{len(out[classe])} mesi totali (reali: {len(s)})")
    return out, diag


# ---------------------------------------------------------------------------
# BOOTSTRAP CONGIUNTO RENDIMENTI + INFLAZIONE (euro reali)
# ---------------------------------------------------------------------------
def bootstrap_reale_con_inflazione(rend_nominali: pd.Series, cpi: pd.Series,
                                   mesi: int, n: int, block: int, seed: int,
                                   cagr_reale_target=None):
    """
    Ricampiona A BLOCCHI CONGIUNTI la coppia (rendimento reale, inflazione):
    stessi indici -> la correlazione empirica asset/inflazione e' preservata
    (es. bond che soffrono quando l'inflazione accelera). Se dato, il drift
    REALE viene ricentrato su cagr_reale_target (shrinkage a monte).
    Ritorna (paths_reali (n,mesi), paths_inflazione (n,mesi)).
    """
    reale = serie_reale(rend_nominali, cpi)
    df = pd.concat([reale, cpi], axis=1, join="inner").dropna()
    r = df.iloc[:, 0].values
    if cagr_reale_target is not None:
        r = ricentra_mensili(r, cagr_reale_target)
    infl = df.iloc[:, 1].values
    m = len(r)
    if m < block:
        raise ValueError(f"Servono almeno {block} mesi, disponibili {m}.")
    rng = np.random.default_rng(seed)
    n_bl = int(np.ceil(mesi / block))
    pr = np.empty((n, mesi))
    pi = np.empty((n, mesi))
    for s_ in range(n):
        start = rng.integers(0, m - block + 1, size=n_bl)
        idx = np.concatenate([np.arange(i, i + block) for i in start])[:mesi]
        pr[s_] = r[idx]
        pi[s_] = infl[idx]
    return pr, pi


# ---------------------------------------------------------------------------
# PARAMETRI SUGGERITI DAI DATI (nu, kappa, shrinkage)
# ---------------------------------------------------------------------------
def suggerisci_parametri():
    """
    Stime dai dati lunghi:
    - nu (code grasse): dalla curtosi in eccesso dei log-rendimenti mensili,
      per la T di Student vale k_ecc = 6/(nu-4)  =>  nu = 4 + 6/k_ecc.
    - kappa (mean reversion): dal Variance Ratio a 5 anni. VR<1 = mean
      reversion. Mappa empirica prudente: kappa ~ 0.5*(1-VR), cap 0.25.
    - shrinkage: peso bayesiano w* = tau^2/(tau^2 + se^2) con se = errore
      standard della media campionaria e tau = incertezza dell'ancora
      (~2 punti percentuali se usi una CMA seria). Confrontato con la
      regola operativa w = mesi/240.
    """
    serie, _ = _serie_lunghe()
    out = {"note": []}
    s = serie.get("shiller_sp500")
    if s is not None and len(s) > 600:
        x = np.log1p(s.values)
        z = (x - x.mean())
        k_ecc = float(np.mean(z ** 4) / (np.mean(z ** 2) ** 2)) - 3.0
        out["curtosi_eccesso"] = k_ecc
        out["nu_stimato"] = (4.0 + 6.0 / k_ecc) if k_ecc > 0.7 else 30.0
        roll = pd.Series(x).rolling(60).sum().dropna()
        vr = float(roll.var(ddof=1) / (60.0 * np.var(x, ddof=1)))
        out["vr_5anni"] = vr
        out["kappa_suggerito"] = float(np.clip(0.5 * (1.0 - vr), 0.0, 0.25))
    for nome, vol_tipica in (("azionario", 0.15), ("obbligazionario", 0.05)):
        righe = {}
        for mesi in (60, 120, 180, 240):
            se = vol_tipica / np.sqrt(mesi / 12.0)
            w_bayes = (0.02 ** 2) / (0.02 ** 2 + se ** 2)
            righe[mesi] = {"w_bayes": round(w_bayes, 2),
                           "w_regola_240": round(min(1.0, mesi / 240.0), 2)}
        out[f"shrinkage_{nome}"] = righe
    return out


# ---------------------------------------------------------------------------
# VALIDAZIONE OUT-OF-SAMPLE
# ---------------------------------------------------------------------------
def valida_out_of_sample(rend: pd.Series, anno_cutoff: int, anni_fwd: int,
                         ancora: float, block: int = 12, n: int = 500,
                         mesi_pieni: int = 240, seed: int = 77):
    """
    Stima su dati fino al 31/12/anno_cutoff (CAGR campione + shrinkage verso
    l'ancora), block-bootstrap in avanti per anni_fwd, confronto con il
    realizzato. Ritorna dict con percentili simulati, path reale, copertura
    della banda P10-P90 e rank percentile del montante finale realizzato.
    """
    rend = rend.dropna()
    prima = rend[rend.index.year <= anno_cutoff]
    dopo = rend[rend.index.year > anno_cutoff][: anni_fwd * 12]
    if len(prima) < max(60, block) or len(dopo) < 12:
        return None
    cagr_c = cagr_da_mensili(prima.values)
    target, w = shrink_verso_ancora(cagr_c, len(prima), ancora,
                                    mesi_pieni=mesi_pieni)
    base = ricentra_mensili(prima.values, target)
    rng = np.random.default_rng(seed)
    mesi = len(dopo)
    n_bl = int(np.ceil(mesi / block))
    m = len(base)
    monti = np.empty((n, mesi))
    for s_ in range(n):
        st_ = rng.integers(0, m - block + 1, size=n_bl)
        idx = np.concatenate([np.arange(i, i + block) for i in st_])[:mesi]
        monti[s_] = np.cumprod(1 + base[idx])
    reale_path = np.cumprod(1 + dopo.values)
    p10 = np.percentile(monti, 10, axis=0)
    p50 = np.percentile(monti, 50, axis=0)
    p90 = np.percentile(monti, 90, axis=0)
    dentro = float(np.mean((reale_path >= p10) & (reale_path <= p90)))
    rank = float(np.mean(monti[:, -1] <= reale_path[-1]))
    return {"indice": dopo.index, "p10": p10, "p50": p50, "p90": p90,
            "reale": reale_path, "copertura": dentro, "rank_finale": rank,
            "cagr_campione": cagr_c, "cagr_target": target, "peso_storico": w}


# ---------------------------------------------------------------------------
# STRESS TEST DETERMINISTICO
# ---------------------------------------------------------------------------
def stress_su_mix(pesi_serie: dict, episodio: str):
    """
    Applica la sequenza mensile REALE di un episodio storico al mix corrente
    (pesi statici, ribilanciamento mensile implicito). Ritorna drawdown,
    rendimento dell'episodio e mesi di recupero (sui dati successivi).
    """
    serie, _ = _serie_lunghe()
    disponibili = {k: serie[k] for k in pesi_serie if k in serie and pesi_serie[k] > 0}
    if not disponibili:
        return None
    df = pd.DataFrame(disponibili).dropna()
    tot = sum(pesi_serie[k] for k in disponibili)
    mix = sum(df[k] * (pesi_serie[k] / tot) for k in disponibili)
    ini, fine = EPISODI_STRESS[episodio]
    finestra = mix.loc[ini:fine]
    if len(finestra) < 6:
        return None
    cum = np.cumprod(1 + finestra.values)
    run_max = np.maximum.accumulate(cum)
    max_dd = float((cum / run_max - 1).min())
    tot_ep = float(cum[-1] - 1)
    dopo = mix.loc[mix.index > finestra.index.max()]
    livello = cum[-1]
    picco = float(run_max[-1])
    mesi_rec = None
    if len(dopo):
        cum_dopo = livello * np.cumprod(1 + dopo.values)
        sopra = np.nonzero(cum_dopo >= picco)[0]
        mesi_rec = int(sopra[0] + 1) if len(sopra) else None
    return {"finestra": finestra, "max_drawdown": max_dd,
            "rendimento_episodio": tot_ep, "mesi_recupero": mesi_rec,
            "n_mesi": len(finestra)}


# ---------------------------------------------------------------------------
# TAB UI — Modello & Validazione
# ---------------------------------------------------------------------------
def render_tab_modello(ctx):
    st.subheader("🔬 Modello & Validazione")
    st.caption(
        "Factor modeling su benchmark, validazione out-of-sample, stress "
        "test storici e parametri stimati dai dati. Tutto quello che c'è "
        "qui serve a VERIFICARE il modello prima di fidarsi dei montanti."
    )
    anag_fondi = ctx.get("anagrafica_fondi", {})
    percorsi = ctx.get("percorsi_csv", {})

    # ---- 1. Ricostruzione comparti --------------------------------------
    st.markdown("### 1 · Ricostruzione comparti dal benchmark (RBSA)")
    scelte = [(f, c) for f, cfg in anag_fondi.items()
              for c in (cfg.get("comparti") or {})
              if (cfg.get("comparti") or {}).get(c, {}).get("benchmark")]
    if not scelte:
        st.info("Nessun comparto con benchmark in anagrafica (data/fondi/).")
    else:
        et = [f"{f} · {c}" for f, c in scelte]
        sel = st.selectbox("Comparto", et, key="ml_comp")
        f_sel, c_sel = scelte[et.index(sel)]
        cfg_c = anag_fondi[f_sel]["comparti"][c_sel]
        percorso = percorsi.get(f_sel)
        if percorso is None:
            st.warning("CSV del fondo non trovato.")
        else:
            rend = rendimenti_csv_fondo(percorso)
            serie, _n = _serie_lunghe()
            ric = (ricostruisci_da_benchmark(rend[c_sel].dropna(),
                                             cfg_c["benchmark"], serie)
                   if c_sel in rend.columns else None)
            if ric is None:
                st.warning("Ricostruzione non calcolabile (dati/overlap insufficienti).")
            else:
                a, b_, c_ = st.columns(3)
                a.metric("Mesi reali → estesi",
                         f"{ric['mesi_reali']} → {ric['mesi_estesi']}")
                b_.metric("Alfa gestione vs mix", f"{ric['alpha_annuo']*100:+.2f}%/anno",
                          help="Differenza media NAV−mix sull'overlap: include "
                               "costi e tracking. Applicata al tratto ricostruito.")
                if ric["rbsa"]:
                    pesi_st, _al, r2, n_ov = ric["rbsa"]
                    c_.metric("R² style analysis", f"{r2:.2f}",
                              help=f"Su {n_ov} mesi di overlap. Sotto ~0.85 il mix "
                                   "spiega male il comparto: non abilitare la "
                                   "ricostruzione.")
                    st.caption("Pesi dichiarati: " +
                               ", ".join(f"{k} {v:.0%}" for k, v in ric["pesi_dichiarati"].items()) +
                               " · Pesi RBSA stimati: " +
                               ", ".join(f"{k} {v:.0%}" for k, v in pesi_st.items()))
                cum_e = (1 + ric["estesa"]).cumprod()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=cum_e.index, y=cum_e.values,
                                         name="Serie estesa", line=dict(width=2)))
                if ric["giunzione"] is not None:
                    fig.add_vline(x=ric["giunzione"], line_dash="dot",
                                  annotation_text="inizio NAV reale")
                else:
                    st.caption("ℹ️ Nessun NAV nel CSV: serie = mix benchmark "
                               "PURO (alfa 0, costi non inclusi).")
                fig.update_layout(height=300, yaxis_type="log",
                                  yaxis_title="Crescita di 1€ (log)")
                st.plotly_chart(fig, use_container_width=True)
                if cfg_c.get("ricostruzione") != "consentita":
                    st.caption("🔒 Di default questa serie NON alimenta i motori: "
                               "si attiva col pulsante per-comparto in sidebar "
                               "('Factor modeling' → 'Ricostruisci ...') dopo aver "
                               "verificato R² e pesi qui sopra. Per renderla "
                               "attiva di default: `\"ricostruzione\": "
                               "\"consentita\"` nel JSON del fondo.")
                if cfg_c.get("da_verificare"):
                    st.caption("⚠️ Da verificare: " + "; ".join(cfg_c["da_verificare"]))

    # ---- 2. Parametri suggeriti ------------------------------------------
    st.markdown("### 2 · Parametri suggeriti dai dati lunghi")
    par = suggerisci_parametri()
    if "nu_stimato" in par:
        c1, c2, c3 = st.columns(3)
        c1.metric("ν stimato (code grasse)", f"{par['nu_stimato']:.1f}",
                  help=f"Da curtosi in eccesso {par['curtosi_eccesso']:.1f} dei "
                       "log-rendimenti mensili S&P dal 1871 (k=6/(ν−4)). "
                       "Range sensato: 4–6. Default app: 5 ✓")
        c2.metric("Variance Ratio 5 anni", f"{par['vr_5anni']:.2f}",
                  help="<1 = mean reversion nei rendimenti di lungo periodo.")
        c3.metric("κ suggerito", f"{par['kappa_suggerito']:.2f}",
                  help="Mappa prudente κ≈0.5·(1−VR). Range sensato: 0.05–0.20. "
                       "Default slider: 0.10.")
    st.caption(
        "**Shrinkage — quanto pesare lo storico.** Peso bayesiano "
        "w\\*=τ²/(τ²+se²) con ancora da CMA (incertezza τ≈2 p.p.): "
        + " · ".join(
            f"{nome}: " + ", ".join(
                f"{m} mesi → w\\* {v['w_bayes']:.0%} (regola /240: {v['w_regola_240']:.0%})"
                for m, v in par[f"shrinkage_{nome}"].items())
            for nome in ("azionario", "obbligazionario")) +
        ". Lettura: per l'AZIONARIO la regola mesi/240 e' GENEROSA verso lo "
        "storico (10 anni: w 50% contro w* bayesiano ~15%): se la validazione "
        "sotto mostra realizzati spesso sotto P50, alza MESI_PIENA_FIDUCIA "
        "(es. 360) o usa un'ancora CMA aggiornata. Per l'obbligazionario le "
        "due regole quasi coincidono."
    )

    # ---- 3. Validazione out-of-sample ------------------------------------
    st.markdown("### 3 · Validazione out-of-sample")
    st.caption("Stimo il modello coi soli dati fino all'anno di taglio, simulo "
               "in avanti, confronto con il realizzato. Banda P10–P90 ben "
               "calibrata ⇒ copertura ~80% e rank del finale lontano da 0/1.")
    serie, _n = _serie_lunghe()
    candidati = {k: v for k, v in serie.items()
                 if k not in ("cpi_us", "cpi_it") and len(v) >= 240}
    if candidati:
        v1, v2, v3, v4 = st.columns(4)
        nome_s = v1.selectbox("Serie", sorted(candidati), key="ml_oos_s")
        anno_ct = v2.number_input("Taglio (anno)", 1950, 2020, 2010, key="ml_ct")
        anni_f = v3.number_input("Anni avanti", 3, 20, 10, key="ml_fw")
        anc = v4.number_input("Ancora (CAGR %)", 0.0, 12.0, 6.5, 0.1,
                              key="ml_anc") / 100
        oos = valida_out_of_sample(candidati[nome_s], int(anno_ct),
                                   int(anni_f), anc)
        if oos is None:
            st.warning("Dati insufficienti attorno al taglio scelto.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Copertura banda P10–P90", f"{oos['copertura']*100:.0f}%",
                      help="Quota di mesi realizzati dentro la banda. Target ~80%. "
                           "Molto meno = modello sovraconfidente (banda stretta).")
            m2.metric("Rank del montante finale", f"{oos['rank_finale']*100:.0f}%",
                      help="Percentile del realizzato tra i simulati. Vicino a "
                           "0% o 100% = drift mal calibrato.")
            m3.metric("Drift usato", f"{oos['cagr_target']*100:.2f}%",
                      help=f"CAGR campione {oos['cagr_campione']*100:.2f}%, peso "
                           f"storico {oos['peso_storico']*100:.0f}%.")
            fig = go.Figure()
            x = oos["indice"]
            fig.add_trace(go.Scatter(x=list(x) + list(x[::-1]),
                                     y=list(oos["p90"]) + list(oos["p10"][::-1]),
                                     fill="toself", fillcolor="rgba(42,120,214,0.12)",
                                     line=dict(color="rgba(0,0,0,0)"),
                                     name="P10–P90 simulata"))
            fig.add_trace(go.Scatter(x=x, y=oos["p50"], name="P50 simulata",
                                     line=dict(dash="dash")))
            fig.add_trace(go.Scatter(x=x, y=oos["reale"], name="Realizzato",
                                     line=dict(width=3)))
            fig.update_layout(height=320, yaxis_title="Crescita di 1€")
            st.plotly_chart(fig, use_container_width=True)

    # ---- 4. Inflazione correlata ------------------------------------------
    st.markdown("### 4 · Inflazione: correlazione empirica coi rendimenti")
    cpi = serie.get("cpi_it") if serie.get("cpi_it") is not None else serie.get("cpi_us")
    nome_cpi = "cpi_it" if serie.get("cpi_it") is not None else "cpi_us"
    if cpi is not None:
        righe = []
        cpi_a = (1 + cpi).rolling(12).apply(np.prod, raw=True) - 1
        for k in ("shiller_sp500", "vbmfx", "gold_lbma", "world_composite"):
            if k in serie:
                s_a = (1 + serie[k]).rolling(12).apply(np.prod, raw=True) - 1
                dfc = pd.concat([s_a, cpi_a], axis=1, join="inner").dropna()
                if len(dfc) > 60:
                    righe.append({"Serie": k,
                                  f"Corr. con inflazione 12m ({nome_cpi})":
                                      round(float(dfc.corr().iloc[0, 1]), 2)})
        if righe:
            st.dataframe(pd.DataFrame(righe), hide_index=True,
                         use_container_width=True)
        st.caption(
            "Il generatore `bootstrap_reale_con_inflazione` ricampiona a "
            "blocchi congiunti (rendimento reale, inflazione): queste "
            "correlazioni entrano nei percorsi GRATIS, senza modellarle. "
            "È il pezzo che i motori nominali con inflazione fissa non hanno: "
            "l'integrazione nei montanti è il passo successivo (Blocco D2)."
        )

    # ---- 5. Stress test deterministici ------------------------------------
    st.markdown("### 5 · Stress test storici sul mix corrente")
    s1, s2, s3 = st.columns(3)
    quota_az = s1.slider("Quota azionaria del mix (%)", 0, 100, 60, 5,
                         key="ml_st_az") / 100
    quota_oro = s2.slider("Quota oro (%)", 0, 50, 0, 5, key="ml_st_oro") / 100
    episodio = s3.selectbox("Episodio", list(EPISODI_STRESS), key="ml_st_ep")
    pesi = {"world_composite": quota_az * 0.999,
            "shiller_sp500": 0.0,
            "gold_lbma": quota_oro,
            "vbmfx": max(0.0, 1 - quota_az - quota_oro)}
    if episodio.startswith("1973") or episodio.startswith("2000"):
        # pre-1976/dati composite corti: usa S&P Shiller come azionario
        pesi["shiller_sp500"] = pesi.pop("world_composite")
    res = stress_su_mix(pesi, episodio)
    if res is None:
        st.warning("Serie insufficienti per questo episodio con questo mix.")
    else:
        t1, t2, t3 = st.columns(3)
        t1.metric("Max drawdown", f"{res['max_drawdown']*100:.1f}%")
        t2.metric("Rendimento episodio", f"{res['rendimento_episodio']*100:+.1f}%",
                  help=f"{res['n_mesi']} mesi, NOMINALE.")
        t3.metric("Mesi per recuperare il picco",
                  "—" if res["mesi_recupero"] is None else str(res["mesi_recupero"]),
                  help="Sul mix nominale, coi dati realmente seguiti all'episodio.")
        st.caption(
            "Uso: se il tuo P10 simulato su orizzonte pari all'episodio è "
            "MIGLIORE di questi numeri, il modello è troppo ottimista sulle "
            "code — alza ν o allunga il blocco del bootstrap."
        )
