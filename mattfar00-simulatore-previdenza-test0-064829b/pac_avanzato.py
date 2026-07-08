# ---------------------------------------------------------------------------
# PAC UNIFICATO — modello MULTI-ASSET (fino a 4 classi)
# ---------------------------------------------------------------------------
# Modulo autonomo, stesso pattern di backtest/ui.py: espone
#     render_pac_avanzato(ctx)
# da chiamare in un tab dell'app principale. NON tocca il motore esistente.
#
# SEZIONE PAC UNIFICATA: questo tab assorbe anche il vecchio "PAC semplice".
# Due fonti per i parametri (rendimento/rischio/correlazioni):
#   a) PORTAFOGLIO TICKER (preferita): OGNI ETF del catalogo e' ammesso.
#      I ticker vengono classificati in 4 classi (Azionario, Obbligazionario,
#      Oro/Materie prime, Immobiliare) con preselezione automatica dal
#      catalogo, correggibile. Per ogni classe presente si costruisce una
#      serie mensile equal-weight; da queste si stimano CAGR, sigma e la
#      matrice di correlazione. AVVISI automatici per volatilita' alta
#      (azioni singole, settoriali, leva). Drift corretto con shrinkage
#      verso ancore di lungo periodo per classe / manuale / solo storico.
#   b) PARAMETRI MANUALI a 2 classi (CAGR, vol, rho scelti dall'utente):
#      disponibile sempre, unica opzione se il portafoglio ticker manca.
#      In questa modalita' il motore e' solo GBM (il bootstrap richiede
#      serie storiche reali).
#
# Caratteristiche:
# - Motore Monte Carlo SELEZIONABILE:
#     a) Block-bootstrap storico CONGIUNTO: pesca blocchi contigui con gli
#        STESSI indici da tutte le serie di classe -> correlazioni empiriche
#        preservate per costruzione, code reali incluse.
#     b) GBM multivariato lognormale con CHOLESKY sulla matrice di
#        correlazione. Semantica CAGR: la mediana di ogni classe compone
#        al CAGR dichiarato (correzione di Ito' incorporata). Code grasse
#        opzionali (T di Student multivariata) e MEAN REVERSION opzionale
#        (richiamo O-U del log-rendimento cumulato verso il trend).
# - GLIDEPATH PER CLASSE: pesi iniziali e finali per ogni classe, rampa
#   lineare tra due anni scelti (derisking generalizzato).
# - RIBILANCIAMENTO ogni N mesi sul VALORE delle quote: vende i bucket
#   sovrappesati (tassa la plusvalenza realizzata pro-quota sul costo
#   medio), reinveste il netto nei bucket sottopesati.
# - COSTI ETF all'acquisto, TER, IMPOSTA DI BOLLO 0,2%/anno pro-rata.
# - DECUMULO: prelievo mensile lordo (opz. indicizzato), pro-quota sui
#   bucket, con probabilita' di successo.
# - DEFLAZIONE del montante (euro reali).
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pac_engine import (cagr_da_mensili, shrink_verso_ancora, ricentra_mensili,
                        _shock_t_student, MESI_PIENA_FIDUCIA,
                        CLASSI_ASSET, TICKER_TO_CLASSE)

# Ancore di lungo periodo di default per classe (CAGR nominali, modificabili
# in UI; in alternativa usare CMA aggiornate: JPMorgan LTCMA, Vanguard...).
ANCORE_DEFAULT = {
    "Azionario": 6.5,
    "Obbligazionario": 2.5,
    "Oro/Materie prime": 3.5,
    "Immobiliare": 5.5,
    "Azioni singole": 6.5,   # atteso ~mercato: l'extra-rischio e' idiosincratico
}
# Allocazione di default (peso iniziale, peso finale) per classe.
PESI_DEFAULT = {
    "Azionario": (80, 40),
    "Obbligazionario": (20, 60),
    "Oro/Materie prime": (0, 0),
    "Immobiliare": (0, 0),
    "Azioni singole": (0, 0),
}
# Parole chiave che identificano un ETF/ETC nel nome dello strumento.
_KEYWORDS_ETF = ("etf", "etc", "ucits", "ishares", "xtrackers", "vanguard",
                 "invesco", "wisdomtree", "amundi", "lyxor", "spdr", "index",
                 "msci", "ftse", "s&p", "stoxx", "acc", "dist")
SOGLIA_VOL_ALTA = 0.25   # avviso oltre il 25% di vol annua


# ---------------------------------------------------------------------------
# GENERATORI DI RENDIMENTI MENSILI (n scenari x mesi x K classi)
# ---------------------------------------------------------------------------
def genera_gbm_cholesky(mu_vec, sig_vec, corr, mesi, n, seed,
                        kappa=0.0, nu=0.0):
    """
    GBM lognormale mensile a K asset con shock correlati via Cholesky sulla
    matrice di correlazione e mean reversion opzionale (kappa/anno).

    `mu_vec` sono CAGR annui (rendimento composto): la traiettoria MEDIANA
    di ogni classe compone esattamente a quel tasso. Il volatility drag e'
    incorporato nella parametrizzazione lognormale (correzione di Ito').
    `nu`>2 attiva code grasse: T di Student multivariata con fattore di
    coda CONDIVISO dalle classi nello stesso mese (crash congiunti).

    Mean reversion: sul log-prezzo cumulato X_t di ciascuna classe,
        r_t = mu_log + (kappa/12) * (mu_log * t - X_t) + shock_t
    """
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu_vec, dtype=float)
    sig = np.asarray(sig_vec, dtype=float)
    K = mu.size

    mu_m = (1.0 + mu) ** (1.0 / 12.0) - 1.0
    sig_m = sig / np.sqrt(12.0)

    sigma_log = np.sqrt(np.log(1.0 + (sig_m ** 2) / (1.0 + mu_m) ** 2))
    mu_log = np.log(1.0 + mu_m)   # mediana mensile = (1+CAGR)^(1/12)

    corr = np.asarray(corr, dtype=float)
    L = np.linalg.cholesky(corr + np.eye(K) * 1e-10)

    k_m = float(kappa) / 12.0
    out = np.empty((n, mesi, K))
    X = np.zeros((n, K))
    for t in range(mesi):
        z = _shock_t_student(rng, (n, K), nu, condividi_righe=True) @ L.T
        r_log = (mu_log + k_m * (mu_log * t - X)) + z * sigma_log
        X += r_log
        out[:, t, :] = np.exp(r_log) - 1.0
    return out


def genera_bootstrap_congiunto(serie_mat, mesi, n, block, seed):
    """
    Block-bootstrap CONGIUNTO a K classi: pesca blocchi contigui di `block`
    mesi con gli STESSI indici da tutte le colonne di `serie_mat` (m x K,
    gia' allineate) -> le correlazioni empiriche tra le classi sono
    preservate per costruzione. Nessun wrap-around.
    """
    S = np.asarray(serie_mat, dtype=float)
    if S.ndim == 1:
        S = S[:, None]
    m = S.shape[0]
    if m < block:
        raise ValueError(f"Servono almeno {block} mesi comuni, disponibili {m}.")
    rng = np.random.default_rng(seed)
    n_blocchi = int(np.ceil(mesi / block))
    out = np.empty((n, mesi, S.shape[1]))
    for s in range(n):
        start = rng.integers(0, m - block + 1, size=n_blocchi)
        idx = np.concatenate([np.arange(i, i + block) for i in start])[:mesi]
        out[s] = S[idx]
    return out


# ---------------------------------------------------------------------------
# GLIDEPATH DI DERISKING — pesi per classe, mese per mese
# ---------------------------------------------------------------------------
def glidepath_pesi(mesi_tot, w_start_vec, w_end_vec, anno_inizio, anno_fine):
    """
    Matrice (mesi_tot x K) dei pesi target: costanti al vettore iniziale,
    rampa lineare tra anno_inizio e anno_fine, poi costanti al finale.
    Ogni riga e' normalizzata a somma 1.
    """
    ws = np.asarray(w_start_vec, dtype=float)
    we = np.asarray(w_end_vec, dtype=float)
    ws = ws / ws.sum() if ws.sum() > 0 else np.full_like(ws, 1.0 / ws.size)
    we = we / we.sum() if we.sum() > 0 else ws.copy()

    W = np.tile(ws, (mesi_tot, 1))
    m0 = max(0, (int(anno_inizio) - 1) * 12)
    m1 = min(max(m0 + 1, int(anno_fine) * 12), mesi_tot)
    if m0 < mesi_tot and not np.allclose(ws, we):
        rampa = np.linspace(0.0, 1.0, max(2, m1 - m0))[:, None]
        W[m0:m1] = ws + (we - ws) * rampa[: m1 - m0]
        if m1 < mesi_tot:
            W[m1:] = we
    W = np.clip(W, 0.0, 1.0)
    W /= np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    return W


# ---------------------------------------------------------------------------
# MOTORE DI SIMULAZIONE (vettorizzato su scenari e classi, loop sui mesi)
# ---------------------------------------------------------------------------
def simula_pac_avanzato(paths, p):
    """
    paths: (n, mesi_tot, K) rendimenti mensili per classe.
    p: dict di parametri (vedi render). Ritorna dict con storie e statistiche.

    Contabilita' per scenario (array shape (n, K)):
      V  valore di mercato dei bucket
      B  costo fiscale (basis), per la plusvalenza pro-quota
    """
    n, mesi_tot, K = paths.shape
    mesi_acc = p["mesi_acc"]
    W = p["w_target"]                       # (mesi_tot, K)
    aliq = np.asarray(p["aliq"], float)     # (K,)

    V = np.tile(p["cap_iniziale"] * W[0], (n, 1))
    B = V.copy()

    tasse_cum = np.zeros(n)
    costi_cum = np.zeros(n)
    bollo_cum = np.zeros(n)
    prelievi_netti_cum = np.zeros(n)
    fallito = np.zeros(n, dtype=bool)

    storia = np.empty((n, mesi_tot))
    ter_m = 1.0 - (1.0 - p["ter"]) ** (1.0 / 12.0)
    bollo_m = p["bollo_pct"] / 12.0

    def _vendi(V, B, importo):
        """Vende `importo` (n x K) dai bucket: aggiorna basis pro-quota,
        ritorna (V, B, netto (n x K), tassa (n x K))."""
        importo = np.minimum(importo, V)
        with np.errstate(divide="ignore", invalid="ignore"):
            gain_frac = np.where(V > 0, np.clip((V - B) / V, 0.0, 1.0), 0.0)
        tassa = importo * gain_frac * aliq          # broadcast (K,)
        quota = np.where(V > 0, importo / np.maximum(V, 1e-12), 0.0)
        B = B * (1.0 - quota)
        V = V - importo
        return V, B, importo - tassa, tassa

    for t in range(mesi_tot):
        w = W[t]
        in_acc = t < mesi_acc

        # 1) VERSAMENTO mensile (solo accumulo), al netto dei costi d'acquisto
        if in_acc:
            rata = p["rata_mensile"][t]
            if rata > 0:
                costo = p["costo_fisso_ordine"] + rata * p["costo_pct_ordine"]
                netto = max(0.0, rata - costo)
                costi_cum += (rata - netto)
                V += netto * w
                B += netto * w

        # 2) RENDIMENTO di mercato del mese
        V *= (1.0 + paths[:, t, :])
        V = np.maximum(V, 0.0)

        # 3) TER — costo ricorrente scaricato sul valore
        if ter_m > 0:
            V *= (1.0 - ter_m)

        # 4) IMPOSTA DI BOLLO 0,2%/anno pro-rata mensile, pro-quota sul valore
        if bollo_m > 0:
            tot = V.sum(axis=1)
            imposta = tot * bollo_m
            bollo_cum += imposta
            with np.errstate(divide="ignore", invalid="ignore"):
                frazioni = np.where(tot[:, None] > 0,
                                    V / np.maximum(tot[:, None], 1e-12), 0.0)
            V -= imposta[:, None] * frazioni

        # 5) RIBILANCIAMENTO ogni N mesi, in base al VALORE delle quote
        if p["reb_attivo"] and ((t + 1) % p["reb_ogni_mesi"] == 0):
            tot = V.sum(axis=1)
            delta = V - tot[:, None] * w          # >0: bucket sovrappesato
            attiva = np.abs(delta).max(axis=1) > tot * p["reb_soglia"]
            if attiva.any():
                vendite = np.where(attiva[:, None] & (delta > 0), delta, 0.0)
                V, B, netto, tassa = _vendi(V, B, vendite)
                tasse_cum += tassa.sum(axis=1)
                pool = netto.sum(axis=1)
                c = pool * p["costo_trans_pct"]
                costi_cum += c
                pool -= c
                bisogno = np.where(attiva[:, None] & (delta < 0), -delta, 0.0)
                btot = bisogno.sum(axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    frac = np.where(btot[:, None] > 0,
                                    bisogno / np.maximum(btot[:, None], 1e-12), 0.0)
                acquisto = pool[:, None] * frac
                V += acquisto
                B += acquisto

        # 6) DECUMULO: prelievo mensile lordo, pro-quota su tutti i bucket
        if not in_acc and p["decumulo"]:
            w_lordo = p["prelievo_mensile"][t - mesi_acc]
            tot = V.sum(axis=1)
            vivo = (tot > 0) & ~fallito
            richiesta = np.where(vivo, np.minimum(w_lordo, tot), 0.0)
            fallito |= vivo & (tot < w_lordo)
            with np.errstate(divide="ignore", invalid="ignore"):
                frazioni = np.where(tot[:, None] > 0,
                                    V / np.maximum(tot[:, None], 1e-12), 0.0)
            V, B, netto, tassa = _vendi(V, B, richiesta[:, None] * frazioni)
            tasse_cum += tassa.sum(axis=1)
            prelievi_netti_cum += netto.sum(axis=1)

        storia[:, t] = V.sum(axis=1)

    return {
        "storia": storia,
        "tasse_cum": tasse_cum,
        "costi_cum": costi_cum,
        "bollo_cum": bollo_cum,
        "prelievi_netti_cum": prelievi_netti_cum,
        "fallito": fallito,
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def render_pac_avanzato(ctx):
    st.subheader("🧪 PAC — multi-asset, derisking, ribilanciamento, decumulo")
    st.caption(
        "Sezione PAC unificata: parametri dal **portafoglio ticker** (dati "
        "storici Yahoo, tutte le classi del catalogo: azionario, "
        "obbligazionario, oro/materie prime, immobiliare — con correzione "
        "del drift e avvisi di volatilità) oppure **manuali** a 2 classi. "
        "Motore Monte Carlo selezionabile (bootstrap storico congiunto o "
        "GBM-Cholesky con mean reversion e code grasse), glidepath per "
        "classe, ribilanciamento periodico con tasse sul realizzato, bollo "
        "0,2%, costi ETF opzionali e fase di decumulo. Curve anche "
        "deflazionate (euro reali). Non e' consulenza finanziaria."
    )

    durata = int(ctx["durata"])
    vp_serie = list(ctx.get("vp_serie") or [0.0] * durata)
    cap_iniziale = float(ctx.get("cap_iniziale_pac", 0.0))
    seed = int(st.session_state.get("master_seed", 33))

    # --- FONTE DEI PARAMETRI: portafoglio ticker (dati storici) o manuale ----
    errore_download = ctx.get("portafoglio_errore")
    usa_portafoglio_ctx = ctx.get("usa_portafoglio", False)
    stima = classifica_e_stima(ctx)

    if stima is not None:
        fonte = st.radio(
            "Fonte dei parametri (rendimento, volatilità, correlazioni)",
            ["Portafoglio ticker (dati storici Yahoo)", "Parametri manuali (2 classi)"],
            index=0, horizontal=True, key="pav_fonte",
            help="Con i ticker, tutto è stimato dai prezzi reali (con "
                 "correzione del drift). In manuale scegli tu CAGR, "
                 "volatilità e correlazione di due classi.",
        )
        fonte_ticker = fonte.startswith("Portafoglio")
    else:
        fonte_ticker = False
        if usa_portafoglio_ctx and errore_download:
            st.error(
                f"⚠️ Portafoglio ticker configurato ma download da Yahoo "
                f"fallito: **{errore_download}** — procedo in modalità "
                f"parametrica manuale. (Cause tipiche: rete assente, rate "
                f"limit Yahoo, ticker errato, storico comune troppo corto.)"
            )
        elif usa_portafoglio_ctx:
            st.warning(
                "⚠️ Portafoglio ticker configurato ma non ancora elaborato: "
                "aspetta il ricalcolo della pagina principale, poi torna qui. "
                "Nel frattempo puoi usare la modalità parametrica manuale."
            )
        else:
            st.info(
                "ℹ️ Nessun portafoglio ticker in sidebar (**5. PAC (ETF)** → "
                "'Portafoglio ticker'): modalità **parametrica manuale** a "
                "2 classi. Coi ticker, parametri e correlazioni verrebbero "
                "stimati dai prezzi reali (tutte le classi del catalogo)."
            )

    if fonte_ticker:
        classi = stima["classi"]
        serie_mat = stima["serie_mat"]          # (m, K)
        mu = stima["mu"].copy()                 # CAGR per classe
        sig = stima["sig"]
        corr = stima["corr"]
        n_mesi = stima["n_mesi"]

        # --- FACTOR MODELING: estensione delle classi coi proxy lunghi -----
        estendi_lungo = st.checkbox(
            "Estendi le classi con la storia lunga (factor proxy)",
            value=False, key="pav_estendi",
            help="Ogni classe viene estesa all'indietro col suo proxy a "
                 "esposizione statica (Azionario→composite dal 1976, "
                 "Obbligazionario→aggregate USA dal 1986, Oro→LBMA dal 1968) "
                 "via beta stimato sull'overlap. Decenni di dati in più, con "
                 "1987, 2000, 2008 dentro. Immobiliare e azioni singole non "
                 "si estendono (nessun proxy statico): se presenti, lo "
                 "storico comune resta limitato dalla classe più corta.",
        )
        if estendi_lungo:
            try:
                from modello_lungo import estendi_classi_pac
                _base = {c: pd.Series(serie_mat[:, i], index=stima["date"])
                         for i, c in enumerate(classi)}
                _est, _diag = estendi_classi_pac(_base)
                _df_ext = pd.concat(_est, axis=1, join="inner").dropna()
                _df_ext = _df_ext[classi]
                if len(_df_ext) > n_mesi:
                    serie_mat = _df_ext.values
                    mu = np.array([cagr_da_mensili(_df_ext[c].values)
                                   for c in classi])
                    sig = _df_ext.std(ddof=1).values * np.sqrt(12.0)
                    corr = (np.corrcoef(serie_mat, rowvar=False)
                            if len(classi) > 1 else np.array([[1.0]]))
                    n_mesi = len(_df_ext)
                else:
                    _corta = min(_est, key=lambda c: len(_est[c]))
                    st.warning(
                        f"⚠️ Estensione SENZA effetto: lo storico congiunto "
                        f"resta {n_mesi} mesi perché la classe più corta "
                        f"(**{_corta}**, {len(_est[_corta])} mesi) vincola "
                        f"l'intersezione. Se una classe non ha proxy (vedi "
                        f"diagnostica sotto) limita tutte le altre: valuta "
                        f"di toglierla dal portafoglio o di accettare lo "
                        f"storico corto."
                    )
                st.caption("🔬 " + " · ".join(f"{c}: {d}" for c, d in _diag.items()))
            except Exception as e:
                st.warning(f"Estensione non riuscita ({e}): uso solo lo storico ETF.")

        st.caption(
            f"📊 Parametri stimati da **{n_mesi} mesi** di storico — " +
            " · ".join(f"{c}: {', '.join(stima['tickers_per_classe'][c])}"
                       for c in classi)
        )
        if stima["avvisi_vol"]:
            st.warning(
                "🎢 **Volatilità alta rilevata** (>" +
                f"{SOGLIA_VOL_ALTA*100:.0f}%/anno): " +
                "; ".join(stima["avvisi_vol"]) +
                ". Tipico di azioni singole, ETF settoriali, leva o materie "
                "prime: sono ammessi, ma allargano molto la banda P10–P90 e "
                "rendono la stima storica meno affidabile. Pesali con cautela."
            )
        cols = st.columns(len(classi) + 1)
        for i, c in enumerate(classi):
            cols[i].metric(f"CAGR {c}", f"{mu[i]*100:+.2f}%",
                           help=f"σ {sig[i]*100:.1f}% — rendimento composto "
                                "annuo del campione (geometrico).")
        with cols[-1]:
            with st.expander("ρ classi"):
                st.dataframe(pd.DataFrame(corr, index=classi, columns=classi)
                             .style.format("{:.2f}"), use_container_width=True)
    else:
        classi = ["Azionario", "Obbligazionario"]
        serie_mat = None
        n_mesi = None
        st.markdown("**Parametri manuali (CAGR = rendimento composto annuo; "
                    "la traiettoria mediana compone a quel tasso)**")
        m1, m2, m3, m4, m5 = st.columns(5)
        mu_e = m1.number_input("CAGR Azionario (%)", 0.0, 12.0, 6.5, 0.1,
                               key="pav_man_mue",
                               help="Riferimento: azionario globale ~6,5% "
                                    "nominale secolare (o una CMA aggiornata).") / 100
        sig_e = m2.number_input("σ Azionario (%)", 5.0, 30.0, 15.0, 0.5,
                                key="pav_man_sige") / 100
        mu_b = m3.number_input("CAGR Obbligaz. (%)", 0.0, 8.0, 2.5, 0.1,
                               key="pav_man_mub") / 100
        sig_b = m4.number_input("σ Obbligaz. (%)", 1.0, 15.0, 5.0, 0.5,
                                key="pav_man_sigb") / 100
        rho = m5.number_input("ρ azionario-obblig.", -0.9, 0.9, 0.10, 0.05,
                              key="pav_man_rho",
                              help="Storicamente oscilla tra circa -0,3 e +0,4.")
        mu = np.array([mu_e, mu_b])
        sig = np.array([sig_e, sig_b])
        corr = np.array([[1.0, rho], [rho, 1.0]])

    K = len(classi)
    c1, c2, c3 = st.columns(3)

    # --- Motore Monte Carlo -------------------------------------------------
    with c1:
        st.markdown("**Motore Monte Carlo**")
        if fonte_ticker:
            motore = st.radio(
                "Generatore dei rendimenti",
                ["GBM multivariato (Cholesky)", "Block-bootstrap storico"],
                key="pav_motore",
                help="**GBM-Cholesky**: genera rendimenti CASUALI da "
                     "CAGR/sigma/correlazioni stimati sopra (lognormale). "
                     "**Block-bootstrap**: RIPESCA a blocchi mesi realmente "
                     "accaduti dallo storico Yahoo, con gli stessi indici per "
                     "tutte le classi — nessuna distribuzione teorica, ma "
                     "limitato ai pattern già visti in passato.",
            )
            usa_bootstrap = motore.startswith("Block")
        else:
            st.caption("Motore: **GBM multivariato (Cholesky)**. Il "
                       "block-bootstrap richiede serie storiche reali: "
                       "disponibile solo con la fonte 'Portafoglio ticker'.")
            usa_bootstrap = False
        n_scen = st.slider(
            "Numero scenari", 200, 2000, 500, 100, key="pav_n",
            help="Più scenari = stime P10/P50/P90 più stabili ma calcolo più "
                 "lento. 500 è un buon compromesso.",
        )
        if usa_bootstrap:
            block = st.number_input(
                "Lunghezza blocco (mesi)", 3, 36, 12, 1, key="pav_block",
                help="Periodo del blocco contiguo ricampionato dallo storico. "
                     "Le serie sono pescate con gli STESSI indici: le "
                     "correlazioni tra le classi sono preservate.",
            )
            kappa = 0.0
        else:
            kappa = st.slider(
                "Mean reversion κ (per anno)", 0.0, 0.5, 0.10, 0.01, key="pav_kappa",
                help="0 = GBM puro. Valori > 0 richiamano il log-rendimento "
                     "cumulato verso il trend di lungo periodo (stile O-U): "
                     "riduce la probabilita' di scenari estremi persistenti.",
            )

    # --- Drift: shrinkage verso l'ancora / manuale / solo storico ------------
    with c2:
        st.markdown("**Drift (correzione realismo)**")
        if fonte_ticker:
            st.caption(
                "Lo storico disponibile e' spesso corto e coglie un solo "
                "regime: estrapolarne la media per decenni sovrastima. Lo "
                "**shrinkage** pesa lo storico per `w = mesi/240` e "
                "un'**ancora di lungo periodo** per classe per il resto "
                "(media secolare o CMA aggiornata: JPMorgan LTCMA, "
                "Vanguard...). Nel bootstrap le serie vengono ricentrate sul "
                "CAGR corretto preservando volatilita', sequenze e "
                "correlazioni. Vol e correlazioni restano SEMPRE dai dati."
            )
            modo_drift = st.radio(
                "Correzione del drift",
                ["Shrinkage verso le ancore", "Manuale", "Solo storico"],
                index=0, key="pav_drift",
            )
            w_camp = min(1.0, n_mesi / float(MESI_PIENA_FIDUCIA))
            if modo_drift.startswith("Shrinkage"):
                righe = []
                for i, c in enumerate(classi):
                    anc = st.number_input(
                        f"Ancora {c} (CAGR %)", 0.0, 12.0,
                        ANCORE_DEFAULT.get(c, 5.0), 0.1,
                        key=f"pav_anc_{i}") / 100
                    mu_corr, _ = shrink_verso_ancora(mu[i], n_mesi, anc)
                    righe.append(f"{c} {mu[i]*100:.2f}% → **{mu_corr*100:.2f}%**")
                    mu[i] = mu_corr
                st.caption(f"Peso storico **{w_camp*100:.0f}%** ({n_mesi} mesi). "
                           + " · ".join(righe))
            elif modo_drift.startswith("Manuale"):
                for i, c in enumerate(classi):
                    default_mu = min(12.0, max(-5.0, round(float(mu[i]) * 100, 1)))
                    mu[i] = st.slider(
                        f"CAGR {c} corretto (%)", -5.0, 12.0,
                        default_mu, 0.1, key=f"pav_muov_{i}",
                        help="Rendimento composto annuo della traiettoria "
                             "mediana della classe.",
                    ) / 100
            else:
                st.caption(
                    f"⚠️ CAGR del campione ({n_mesi} mesi) estrapolato per "
                    f"{durata}+ anni: rischio di forte sovrastima se lo "
                    f"storico copre un regime favorevole."
                )
        else:
            modo_drift = "Manuale"
            st.caption(
                "Fonte parametrica manuale: i CAGR che hai inserito sopra "
                "SONO il drift (nessuna correzione da applicare). Usa valori "
                "composti realistici: la media storica di un periodo "
                "favorevole non è un buon input."
            )
        if not usa_bootstrap:
            code_grasse = st.checkbox(
                "Code grasse (T di Student, ν=5)", value=True, key="pav_tstud",
                help="Shock a code grasse al posto della gaussiana: P10 più "
                     "severo e realistico. Il fattore di coda è condiviso "
                     "dalle classi nello stesso mese (crash congiunti). Il "
                     "bootstrap non ne ha bisogno: ripesca le code reali.",
            )
            nu_t = 5.0 if code_grasse else 0.0
        else:
            nu_t = 0.0

    # --- Allocazione & derisking: pesi iniziali e finali PER CLASSE ----------
    with c3:
        st.markdown("**Allocazione (pesi per classe)**")
        st.caption(
            "Quanto rischio prendere è una scelta di policy, non un fatto "
            "statistico. I pesi vengono normalizzati a somma 100%."
        )
        derisking_on = st.checkbox(
            "Attiva derisking (glidepath) — opzionale", value=False,
            key="pav_derisk_on",
            help="Se attivo, i pesi passano gradualmente dall'allocazione "
                 "iniziale a quella finale tra i due anni scelti: riduce il "
                 "*sequence-of-returns risk* (un crollo a ridosso del "
                 "decumulo costringe a vendere a prezzi bassi). Se spento, "
                 "l'allocazione resta costante per tutta la simulazione.",
        )
        w_start, w_end = [], []
        for i, c in enumerate(classi):
            d0, d1_ = PESI_DEFAULT.get(c, (0, 0))
            if derisking_on:
                cc0, cc1 = st.columns(2)
                w_start.append(cc0.number_input(f"{c} — iniziale (%)", 0, 100,
                                                int(d0), 5, key=f"pav_w0_{i}"))
                w_end.append(cc1.number_input(f"{c} — finale (%)", 0, 100,
                                              int(d1_), 5, key=f"pav_w1_{i}"))
            else:
                w_start.append(st.number_input(f"{c} — peso (%)", 0, 100,
                                               int(d0), 5, key=f"pav_w0_{i}"))
        if not derisking_on:
            w_end = list(w_start)
        s0, s1 = sum(w_start), sum(w_end)
        if s0 == 0:
            st.error("I pesi sono tutti 0: imposta almeno una classe.")
            return
        if s1 == 0:
            w_end = list(w_start)
            s1 = s0
        if abs(s0 - 100) > 0.5 or (derisking_on and abs(s1 - 100) > 0.5):
            st.caption(f"ℹ️ Somme: iniziale {s0}%" +
                       (f" · finale {s1}%" if derisking_on else "") +
                       " — normalizzate a 100%.")
        if derisking_on:
            a0 = st.number_input(
                "Derisking: dall'anno", 1, durata, max(1, durata - 10), key="pav_a0",
                help="Anno in cui INIZIA la transizione verso i pesi finali.",
            )
            a1 = st.number_input(
                "Derisking: fino all'anno", int(a0), durata, durata, key="pav_a1",
                help="Anno in cui la rampa TERMINA: da qui i pesi restano quelli "
                     "finali, anche durante il decumulo.",
            )
            _W_prev = glidepath_pesi(durata * 12, w_start, w_end, int(a0), int(a1))
            _fig_glide = go.Figure()
            _x = list(np.arange(1, durata * 12 + 1) / 12.0)
            for i, c in enumerate(classi):
                _fig_glide.add_trace(go.Scatter(
                    x=_x, y=_W_prev[:, i] * 100, name=c, stackgroup="w",
                    mode="lines",
                ))
            _fig_glide.update_layout(
                height=200, margin=dict(l=10, r=10, t=10, b=30),
                xaxis_title="Anni", yaxis_title="% allocazione",
                yaxis_range=[0, 100],
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(_fig_glide, use_container_width=True,
                            config={"displayModeBar": False})
        else:
            a0, a1 = 1, 1   # nessuna rampa: pesi costanti

    d1, d2, d3 = st.columns(3)

    with d1:
        st.markdown("**Ribilanciamento**")
        reb_attivo = st.checkbox(
            "Ribilancia periodicamente", True, key="pav_reb",
            help="Se attivo, ogni N mesi riporta i pesi al target del "
                 "glidepath, vendendo i bucket sovrappesati. Se disattivo, "
                 "il portafoglio 'deriva' liberamente (nessuna vendita, "
                 "nessuna tassa sul realizzato in accumulo).",
        )
        reb_ogni = st.number_input(
            "Ogni quanti mesi", 1, 60, 12, 1, key="pav_rebm",
            help="Riporta i pesi al target del glidepath in base al valore "
                 "corrente delle quote.", disabled=not reb_attivo,
        )
        reb_soglia = st.slider(
            "Soglia minima di scostamento (%)", 0.0, 10.0, 1.0, 0.5, key="pav_rebsoglia",
            help="Non ribilancia se lo scostamento massimo dal target e' "
                 "sotto questa % del portafoglio (evita micro-vendite tassate).",
            disabled=not reb_attivo,
        ) / 100
        costo_trans = st.number_input(
            "Costo transazione ribilanciamento (%)", 0.0, 2.0, 0.0, 0.05,
            key="pav_ctrans", help="Se non impostato vale 0.",
        ) / 100

    with d2:
        st.markdown("**Costi & fisco**")
        ter = st.number_input(
            "TER ETF (%/anno)", 0.0, 2.0, 0.0, 0.01, key="pav_ter",
            help="Costo ricorrente degli ETF. Default 0: se il rendimento che "
                 "usi e' gia' netto di TER (es. stimato dai prezzi), lascialo a 0 "
                 "per non contarlo due volte.",
        ) / 100
        costo_fisso_ord = st.number_input(
            "Costo d'acquisto fisso (€/ordine)", 0.0, 50.0, 0.0, 0.5, key="pav_cfix",
            help="Commissione fissa del broker su ogni rata mensile. Default 0.",
        )
        costo_pct_ord = st.number_input(
            "Costo d'acquisto (%/ordine)", 0.0, 2.0, 0.0, 0.05, key="pav_cpct",
            help="Commissione percentuale su ogni rata mensile. Default 0.",
        ) / 100
        bollo_on = st.checkbox(
            "Imposta di bollo 0,2%/anno", True, key="pav_bollo",
            help="Imposta di bollo italiana sui prodotti finanziari (0,2% "
                 "annuo sul controvalore, applicata pro-rata mensile).",
        )
        aliq_pct = st.slider(
            "Aliquota plusvalenze (%)", 0, 26, 26, key="pav_aliq",
            help="Aliquota ordinaria italiana sulle plusvalenze da ETF "
                 "armonizzati: 26%. Applicata solo alla parte di plusvalenza "
                 "quando vendi (ribilanciamento o decumulo). Nota: gli ETC "
                 "su oro/materie prime sono comunque al 26%.",
        )
        bond_125 = st.checkbox(
            "12,5% sul bucket obbligazionario", False, key="pav_b125",
            help="Approssimazione: tratta l'obbligazionario come titoli di "
                 "Stato/white list (aliquota ridotta). La quota esatta va "
                 "verificata sulla documentazione fiscale dello strumento.",
        )

    with d3:
        st.markdown("**Decumulo & inflazione**")
        decumulo = st.checkbox(
            "Aggiungi fase di decumulo", True, key="pav_dec",
            help="Dopo l'accumulo, prelievo mensile: niente più versamenti, "
                 "si vende per generare una rendita.",
        )
        anni_dec = st.number_input(
            "Anni di decumulo", 1, 40, 25, 1, key="pav_decanni",
            disabled=not decumulo,
            help="Durata della fase di prelievo.",
        )
        prelievo0 = st.number_input(
            "Prelievo mensile LORDO (€)", 0.0, 20000.0, 1500.0, 50.0,
            key="pav_prel", disabled=not decumulo,
            help="Prelievo prima della tassa sulla plusvalenza incorporata "
                 "(pro-quota sul costo medio). Il netto percepito e' riportato "
                 "nei risultati.",
        )
        prelievo_indicizzato = st.checkbox(
            "Indicizza il prelievo all'inflazione", True, key="pav_previdx",
            disabled=not decumulo,
        )
        inflazione = st.slider(
            "Inflazione per la deflazione (%)", 0.0, 5.0, 2.0, 0.1, key="pav_infl",
            help="Usata per esprimere il montante in euro REALI (potere "
                 "d'acquisto di oggi) e per indicizzare il prelievo.",
        ) / 100

    # -------------------------------------------------------------------------
    # COSTRUZIONE E SIMULAZIONE
    # -------------------------------------------------------------------------
    mesi_acc = durata * 12
    mesi_dec = (int(anni_dec) * 12) if decumulo else 0
    mesi_tot = mesi_acc + mesi_dec

    W = glidepath_pesi(mesi_tot, w_start, w_end, int(a0), int(a1))
    if mesi_dec:
        W[mesi_acc:] = W[mesi_acc - 1]   # in decumulo resta l'allocazione finale

    rata_mensile = np.repeat(np.asarray(vp_serie, dtype=float) / 12.0, 12)[:mesi_acc]

    if decumulo and mesi_dec:
        idx = np.arange(mesi_dec)
        fattore = (1 + inflazione) ** ((mesi_acc + idx) / 12.0) if prelievo_indicizzato else 1.0
        prelievo_serie = prelievo0 * (fattore if np.ndim(fattore) else np.ones(mesi_dec))
    else:
        prelievo_serie = np.zeros(0)

    try:
        if usa_bootstrap:
            # Il drift corretto (shrinkage/manuale) si applica RICENTRANDO le
            # serie storiche sul CAGR target per classe: vol, sequenze e
            # correlazioni restano reali, cambia solo il tasso di composizione.
            S = serie_mat
            if not modo_drift.startswith("Solo"):
                S = np.column_stack([ricentra_mensili(serie_mat[:, i], mu[i])
                                     for i in range(K)])
            paths = genera_bootstrap_congiunto(S, mesi_tot, n_scen,
                                               int(block), seed + 500)
        else:
            paths = genera_gbm_cholesky(mu, sig, corr, mesi_tot, n_scen,
                                        seed + 500, kappa=kappa, nu=nu_t)
    except (ValueError, np.linalg.LinAlgError) as e:
        st.error(f"Impossibile generare gli scenari: {e}")
        return

    aliq_vec = [0.125 if (c == "Obbligazionario" and bond_125)
                else aliq_pct / 100.0 for c in classi]

    p = dict(
        mesi_acc=mesi_acc, cap_iniziale=cap_iniziale, w_target=W,
        rata_mensile=rata_mensile,
        reb_attivo=reb_attivo, reb_ogni_mesi=int(reb_ogni), reb_soglia=reb_soglia,
        costo_trans_pct=costo_trans, ter=ter,
        costo_fisso_ordine=costo_fisso_ord, costo_pct_ordine=costo_pct_ord,
        bollo_pct=0.002 if bollo_on else 0.0,
        aliq=aliq_vec,
        decumulo=bool(decumulo and mesi_dec), prelievo_mensile=prelievo_serie,
    )
    res = simula_pac_avanzato(paths, p)

    # -------------------------------------------------------------------------
    # RISULTATI
    # -------------------------------------------------------------------------
    storia = res["storia"]
    mesi_x = np.arange(1, mesi_tot + 1) / 12.0
    defl = (1 + inflazione) ** (-mesi_x)     # deflatore (euro reali di oggi)

    p10 = np.percentile(storia, 10, axis=0)
    p50 = np.percentile(storia, 50, axis=0)
    p90 = np.percentile(storia, 90, axis=0)
    p50_reale = p50 * defl

    fine_acc = mesi_acc - 1
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Fine accumulo — P50 nominale", f"€ {p50[fine_acc]:,.0f}",
              help=f"P10 € {p10[fine_acc]:,.0f} · P90 € {p90[fine_acc]:,.0f}")
    k2.metric("Fine accumulo — P50 reale", f"€ {p50_reale[fine_acc]:,.0f}",
              help="Deflazionato: potere d'acquisto di oggi.")
    if p["decumulo"]:
        successo = 100.0 * (1.0 - res["fallito"].mean())
        k3.metric("Prob. successo decumulo", f"{successo:.1f}%",
                  help="Quota di scenari in cui il capitale copre TUTTI i "
                       "prelievi fino alla fine del decumulo.")
        k4.metric("Prelievi netti totali (P50)",
                  f"€ {np.median(res['prelievi_netti_cum']):,.0f}")
    else:
        k3.metric("Tasse ribilanciamento (P50)", f"€ {np.median(res['tasse_cum']):,.0f}")
        k4.metric("Bollo cumulato (P50)", f"€ {np.median(res['bollo_cum']):,.0f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(mesi_x) + list(mesi_x[::-1]), y=list(p90) + list(p10[::-1]),
        fill="toself", fillcolor="rgba(155,89,182,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="P10–P90", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=mesi_x, y=p50, name="P50 nominale",
                             line=dict(color="#9b59b6", width=3)))
    fig.add_trace(go.Scatter(x=mesi_x, y=p50_reale, name="P50 reale (deflazionato)",
                             line=dict(color="#2a78d6", width=2, dash="dash")))
    if p["decumulo"]:
        fig.add_vline(x=durata, line_dash="dot", line_color="#888",
                      annotation_text="fine accumulo / inizio decumulo")
    fig.update_layout(xaxis_title="Anni", yaxis_title="Montante PAC (€)",
                      yaxis_tickformat="€,.0f", hovermode="x unified", height=440,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    # Tabella costi/tasse cumulati mediani
    df_sintesi = pd.DataFrame({
        "Voce": ["Tasse su ribilanciamento/prelievi", "Imposta di bollo",
                 "Costi acquisto + transazione"],
        "Cumulato P50 (€)": [np.median(res["tasse_cum"]),
                             np.median(res["bollo_cum"]),
                             np.median(res["costi_cum"])],
    })
    st.dataframe(df_sintesi.style.format({"Cumulato P50 (€)": "€ {:,.0f}"}),
                 use_container_width=True, hide_index=True)
    st.caption(
        "La banda P10–P90 riflette solo l'incertezza dei rendimenti simulati. "
        "Le tasse sul realizzato usano la plusvalenza pro-quota sul costo "
        "medio: e' un'approssimazione del regime fiscale reale (che ragiona "
        "per lotti/minusvalenze compensabili). Stima illustrativa, non "
        "consulenza fiscale o finanziaria."
    )


# ---------------------------------------------------------------------------
# SERIE STORICHE E PARAMETRI STIMATI DAL PORTAFOGLIO TICKER (Yahoo Finance)
# ---------------------------------------------------------------------------
def classifica_e_stima(ctx):
    """
    Unica fonte di verita' per ENTRAMBI i motori Monte Carlo: tutto arriva
    dal portafoglio ticker gia' scaricato da Yahoo in sidebar.

    1) Ogni ticker viene assegnato a una delle 4 classi (Azionario,
       Obbligazionario, Oro/Materie prime, Immobiliare): preselezione
       automatica dal catalogo (TICKER_TO_CLASSE) o dal nome, correggibile.
    2) Per ogni classe presente, la serie mensile e' la media equal-weight
       dei rendimenti mensili dei suoi ticker.
    3) Da queste serie si stimano CAGR, sigma (annualizzati) e la matrice
       di correlazione tra le classi.

    Ritorna dict con: classi (presenti), serie_mat (m x K), mu, sig, corr,
    n_mesi, tickers_per_classe, avvisi_vol — oppure None se il portafoglio
    ticker non e' disponibile.
    """
    pi = ctx.get("portafoglio_info")
    if pi is None or "prezzi_df" not in pi:
        return None

    prezzi = pi["prezzi_df"]
    rend = prezzi.pct_change().dropna()
    tickers = list(rend.columns)
    nomi = ctx.get("ticker_to_nome", {})

    def classe_default(t):
        if t in TICKER_TO_CLASSE:
            return TICKER_TO_CLASSE[t]
        nl = nomi.get(t, t).lower()
        if any(k in nl for k in ("bond", "obblig", "aggregate", "government", "btp", "treasury")):
            return "Obbligazionario"
        if any(k in nl for k in ("gold", "oro", "commodit", "silver", "materie")):
            return "Oro/Materie prime"
        if any(k in nl for k in ("reit", "immobil", "property", "epra", "nareit")):
            return "Immobiliare"
        # Ticker manuale senza indizi da ETF nel nome -> quasi certamente
        # un'AZIONE SINGOLA (es. AAPL, ENI.MI): bucket dedicato.
        if not any(k in nl for k in _KEYWORDS_ETF):
            return "Azioni singole"
        return "Azionario"

    etichette = [f"{t} — {nomi.get(t, t)}" for t in tickers]
    st.markdown("**Classificazione dei ticker** (preselezione automatica, correggibile)")
    assegnazione = {}
    sel_cols = st.columns(min(4, max(1, len(tickers))))
    for i, (t, et) in enumerate(zip(tickers, etichette)):
        with sel_cols[i % len(sel_cols)]:
            assegnazione[t] = st.selectbox(
                et, CLASSI_ASSET,
                index=CLASSI_ASSET.index(classe_default(t)),
                key=f"pav_cls_{t}",
            )

    classi = [c for c in CLASSI_ASSET
              if any(assegnazione[t] == c for t in tickers)]
    if not classi:
        return None
    if len(classi) == 1:
        st.warning(
            f"⚠️ Tutti i ticker sono nella classe **{classi[0]}**: con un solo "
            f"bucket non c'è diversificazione da modellare — derisking e "
            f"ribilanciamento non avranno effetto."
        )

    if "Azioni singole" in classi:
        stock_list = [t for t in tickers if assegnazione[t] == "Azioni singole"]
        st.warning(
            "📌 **Azioni singole nel portafoglio**: " + ", ".join(stock_list) +
            ". Sono ammesse, ma ricorda: il rischio idiosincratico non si "
            "diversifica dentro il bucket (a meno di molti titoli), la stima "
            "storica su un singolo titolo è poco affidabile, e né il GBM né "
            "il bootstrap modellano il fallimento della singola azienda — "
            "il rischio reale è PEGGIORE di quello simulato. Tienile a peso "
            "contenuto."
        )

    tickers_per_classe = {c: [t for t in tickers if assegnazione[t] == c]
                          for c in classi}
    serie_mat = np.column_stack([
        rend[tickers_per_classe[c]].mean(axis=1).values for c in classi
    ])
    n_mesi = serie_mat.shape[0]

    mu = np.array([cagr_da_mensili(serie_mat[:, i]) for i in range(len(classi))])
    sig = serie_mat.std(axis=0, ddof=1) * np.sqrt(12.0)
    if len(classi) > 1:
        corr = np.corrcoef(serie_mat, rowvar=False)
    else:
        corr = np.array([[1.0]])

    # Avvisi di volatilita' alta: per singolo ticker e per classe.
    avvisi = []
    vol_ticker = rend.std(ddof=1) * np.sqrt(12.0)
    for t in tickers:
        if float(vol_ticker[t]) > SOGLIA_VOL_ALTA:
            avvisi.append(f"{t} (σ {float(vol_ticker[t])*100:.0f}%)")
    for i, c in enumerate(classi):
        if sig[i] > SOGLIA_VOL_ALTA:
            avvisi.append(f"classe {c} (σ {sig[i]*100:.0f}%)")

    return {
        "classi": classi, "serie_mat": serie_mat, "n_mesi": n_mesi,
        "mu": mu, "sig": sig, "corr": corr,
        "tickers_per_classe": tickers_per_classe,
        "avvisi_vol": avvisi,
        "date": rend.index,
    }
