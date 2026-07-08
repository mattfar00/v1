"""
backtest.ui.tab — rendering Streamlit del backtest storico.

Espone un'unica funzione render_backtest_tab(ctx: dict). Tutto ciò che serve
arriva dal main dentro `ctx` (dependency injection): niente import dal main,
niente stato globale. Streamlit e Plotly sono importati SOLO qui, così i
moduli metrics/data/engine restano usabili anche senza Streamlit installato.

Chiavi attese in ctx (le opzionali possono mancare):
    durata                : int, anni di simulazione
    storico_mensile       : dict  (STORICO_MENSILE del main)
    fondo                 : str
    comparto              : str
    usa_portafoglio       : bool
    portafoglio_info      : dict | None
    ticker_to_nome        : dict  (per etichette benchmark)  [opzionale]
    # per la modalità "via motore":
    simula_capitale       : callable
    fattori_mediani       : list[float]
    sched                 : list[dict]
    scal                  : dict
    vol_extra_serie       : list[float]
    vp_serie              : list[float]
    cap_iniziale_fondo    : float [opzionale, default 10000 per lump sum]
    cap_iniziale_pac      : float [opzionale]
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from ..metrics import scheda_completa, rolling_returns
from ..data import (
    serie_fondo_reale, serie_pac_reale, serie_singolo_asset,
    allinea_ultimi_mesi,
)
from ..engine import (
    backtest_lump_sum, backtest_con_contributi, backtest_via_motore,
)

# etichette leggibili delle metriche di scheda_completa
_ETICHETTE = {
    "CAGR": "CAGR",
    "Volatilita_annua": "Volatilità annua",
    "Downside_dev_annua": "Downside deviation",
    "Best_year": "Anno migliore",
    "Worst_year": "Anno peggiore",
    "Max_drawdown": "Max drawdown",
    "Sharpe": "Sharpe",
    "Sortino": "Sortino",
    "Calmar": "Calmar",
    "VaR_5pct_mensile": "VaR 5% (mese)",
    "CVaR_5pct_mensile": "CVaR 5% (mese)",
    "Mesi_positivi_pct": "% mesi positivi",
    "N_mesi": "N. mesi",
}
_PERC = {"CAGR", "Volatilita_annua", "Downside_dev_annua", "Best_year",
         "Worst_year", "Max_drawdown", "VaR_5pct_mensile", "CVaR_5pct_mensile",
         "Mesi_positivi_pct"}


def _fmt_scheda(nome_serie, scheda):
    righe = []
    for k, v in scheda.items():
        if k == "N_mesi":
            val = f"{int(v)}"
        elif k in _PERC:
            val = f"{v*100:+.2f}%" if k not in ("Volatilita_annua",
                   "Downside_dev_annua", "VaR_5pct_mensile",
                   "CVaR_5pct_mensile", "Mesi_positivi_pct") else f"{v*100:.2f}%"
        else:
            val = f"{v:.2f}"
        righe.append((_ETICHETTE[k], val))
    return pd.DataFrame(righe, columns=["Metrica", nome_serie]).set_index("Metrica")


def _tabella_rolling(serie):
    rr = rolling_returns(serie)
    if not rr:
        return None
    righe = [{"Finestra": f"{a} anni",
              "Media": f"{m*100:+.2f}%",
              "Migliore": f"{hi*100:+.2f}%",
              "Peggiore": f"{lo*100:+.2f}%"} for a, (m, hi, lo) in rr.items()]
    return pd.DataFrame(righe)


def _pannello_asset(nome, serie_reale, mesi_richiesti, rf_annuo,
                    modalita, capitale_iniziale, contributo_mensile, colore):
    """Disegna metriche + growth chart per un singolo asset (fondo/pac/bench)."""
    if serie_reale.size == 0:
        st.info(f"Serie storica non disponibile per **{nome}**.")
        return None

    serie, n_eff, troncata = allinea_ultimi_mesi(serie_reale, mesi_richiesti)
    if not troncata:
        st.caption(f"ℹ️ **{nome}**: storico più corto della durata scelta — "
                   f"backtest su {n_eff} mesi ({n_eff/12:.1f} anni) invece di "
                   f"{mesi_richiesti//12}.")
    if n_eff < 36:
        st.caption(f"⚠️ **{nome}**: meno di 3 anni di dati, metriche indicative.")

    if modalita == "Lump sum (capitale unico)":
        equity, _ = backtest_lump_sum(serie, capitale_iniziale)
        versato = capitale_iniziale
    else:  # DCA
        equity, _, versato = backtest_con_contributi(
            serie, contributo_mensile, capitale_iniziale)

    # Le metriche di rischio si calcolano SEMPRE sui rendimenti dell'asset,
    # non sul montante (i versamenti falserebbero drawdown e volatilità).
    scheda = scheda_completa(serie, rf_annuo=rf_annuo)
    return {"nome": nome, "serie": serie, "equity": equity, "versato": versato,
            "scheda": scheda, "colore": colore, "n_eff": n_eff}


def render_backtest_tab(ctx: dict):
    st.subheader("🔁 Backtest storico (traiettoria reale)")
    st.caption(
        "A differenza degli scenari Monte Carlo delle altre sezioni, qui la "
        "strategia gira su una **singola traiettoria: i rendimenti mensili "
        "effettivamente accaduti**. Risponde a «come si è comportata davvero "
        "questa allocazione», non «quanto è incerto il futuro». Le metriche di "
        "rischio (drawdown, volatilità, Sharpe…) sono calcolate sui rendimenti "
        "dell'asset; il grafico mostra la crescita del capitale."
    )

    durata = ctx["durata"]
    mesi_richiesti = durata * 12

    c1, c2, c3 = st.columns(3)
    modalita = c1.radio(
        "Modalità", ["Lump sum (capitale unico)", "DCA (versamento mensile)"],
        help="Lump sum = confronto 'pulito' fra allocazioni (stile Portfolio "
             "Visualizer). DCA = versamento costante: più realistico, ma "
             "attenua il drawdown.",
    )
    capitale_iniziale = c2.number_input(
        "Capitale iniziale (€)", min_value=0, value=10000, step=1000,
        help="Per lump sum è il capitale investito una tantum.",
    )
    contributo_mensile = c3.number_input(
        "Versamento mensile DCA (€)", min_value=0, value=300, step=50,
        disabled=modalita.startswith("Lump"),
    )
    rf_annuo = st.slider(
        "Tasso risk-free per Sharpe/Sortino (%)", 0.0, 5.0, 2.0, 0.1,
        help="BOT/Euribor medio del periodo. Metti 0 per ignorarlo.",
    ) / 100

    # --- serie reali ---
    serie_fondo = serie_fondo_reale(ctx["storico_mensile"], ctx["fondo"], ctx["comparto"])
    serie_pac = serie_pac_reale(ctx.get("portafoglio_info"))

    pannelli = []
    p = _pannello_asset(f"Fondo · {ctx['comparto']}", serie_fondo, mesi_richiesti,
                        rf_annuo, modalita, capitale_iniziale, contributo_mensile,
                        "#2a78d6")
    if p:
        pannelli.append(p)

    if ctx.get("usa_portafoglio") and serie_pac.size:
        p = _pannello_asset("PAC (portafoglio)", serie_pac, mesi_richiesti,
                            rf_annuo, modalita, capitale_iniziale,
                            contributo_mensile, "#9b59b6")
        if p:
            pannelli.append(p)
    else:
        st.caption("ℹ️ Backtest PAC disponibile solo in modalità 'Portafoglio "
                   "ticker' (serve lo storico reale scaricato).")

    # --- benchmark opzionale (un ticker già scaricato, es. SWDA/VWCE) ---
    pinfo = ctx.get("portafoglio_info")
    if pinfo and "prezzi_df" in pinfo:
        tickers_disp = list(pinfo["prezzi_df"].columns)
        bench = st.selectbox("Benchmark (opzionale)", ["—"] + tickers_disp)
        if bench != "—":
            serie_b = serie_singolo_asset(pinfo, bench)
            nome_b = ctx.get("ticker_to_nome", {}).get(bench, bench)
            p = _pannello_asset(f"Benchmark · {nome_b}", serie_b, mesi_richiesti,
                                rf_annuo, modalita, capitale_iniziale,
                                contributo_mensile, "#888888")
            if p:
                pannelli.append(p)

    if not pannelli:
        st.warning("Nessuna serie storica disponibile per il backtest.")
        return

    # --- grafico growth ---
    fig = go.Figure()
    for p in pannelli:
        x = list(range(1, p["equity"].size + 1))
        fig.add_trace(go.Scatter(
            x=x, y=p["equity"], name=p["nome"],
            line=dict(color=p["colore"], width=2.5)))
    fig.update_layout(
        xaxis_title="Mese", yaxis_title="Capitale (€)",
        yaxis_tickformat="€,.0f", hovermode="x unified", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)

    # --- tabella metriche affiancate ---
    st.markdown("**Metriche di performance (stile Portfolio Visualizer)**")
    schede = [_fmt_scheda(p["nome"], p["scheda"]) for p in pannelli]
    st.dataframe(pd.concat(schede, axis=1), use_container_width=True)

    # --- rolling returns per il primo pannello ---
    st.markdown("**Rolling returns (CAGR su finestre mobili)** — " + pannelli[0]["nome"])
    df_roll = _tabella_rolling(pannelli[0]["serie"])
    if df_roll is not None:
        st.dataframe(df_roll, use_container_width=True, hide_index=True)
    else:
        st.caption("Storico troppo corto per finestre rolling di 1+ anno.")

    st.caption(
        "⚠️ Le quote del fondo negoziale sono già al netto di imposta "
        "sostitutiva e costi di gestione; il PAC è netto-di-TER ma lordo di "
        "capital gain fino alla vendita. Backtest su storici brevi = pochi "
        "cicli di mercato: leggilo come indicativo, non predittivo."
    )


def render_backtest_via_motore(ctx: dict):
    """
    Variante che riusa il TUO simula_capitale con la traiettoria storica reale,
    mantenendo TFR/IRPEF/aliquote. Da chiamare separatamente se vuoi il montante
    netto 'completo' invece del growth semplice dell'asset.
    """
    st.subheader("🔁 Backtest via motore completo (con TFR/IRPEF)")
    durata = ctx["durata"]
    serie_fondo = serie_fondo_reale(ctx["storico_mensile"], ctx["fondo"], ctx["comparto"])
    serie_pac = serie_pac_reale(ctx.get("portafoglio_info"))

    sf, nf, _ = allinea_ultimi_mesi(serie_fondo, durata * 12)
    sp, npac, _ = allinea_ultimi_mesi(serie_pac, durata * 12)
    if nf < durata * 12:
        st.warning(f"Storico fondo insufficiente ({nf} mesi su {durata*12}): "
                   f"riduci la durata per il backtest completo.")
        return
    if npac < durata * 12:
        # se manca il PAC, riempi con zeri neutri per non bloccare il fondo
        sp = np.zeros(durata * 12)

    try:
        df = backtest_via_motore(
            ctx["simula_capitale"], ctx["fattori_mediani"], sf, sp,
            ctx["sched"], ctx["scal"], ctx["vol_extra_serie"], ctx["vp_serie"])
    except ValueError as e:
        st.error(str(e))
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Anno"], y=df["Fondo Netto (€)"],
                             name="Fondo (storico reale)", line=dict(color="#2a78d6", width=3)))
    if "PAC Netto (€)" in df:
        fig.add_trace(go.Scatter(x=df["Anno"], y=df["PAC Netto (€)"],
                                 name="PAC (storico reale)", line=dict(color="#9b59b6", width=2, dash="dash")))
    fig.update_layout(xaxis_title="Anno", yaxis_title="Netto (€)",
                      yaxis_tickformat="€,.0f", hovermode="x unified", height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True, height=380)
