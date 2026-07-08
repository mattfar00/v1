import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import csv
import json
from backtest.ui import render_backtest_tab, render_backtest_via_motore

# --- Logica PAC estratta in pac_engine.py (catalogo ETF, GBM, Cholesky, Yahoo) ---
from pac_engine import (
    CATALOGO_ETF, TICKER_TO_NOME, QUOTA_TITOLI_STATO_TICKER,
    classifica_ticker, mensili_ad_annui, rendimento_netto_pac,
    seleziona_traiettoria_per_percentile, genera_rendimenti_gbm,
    parse_ticker_pesi, scarica_prezzi_mensili, stima_parametri_portafoglio,
    genera_rendimenti_portafoglio_gbm,
    cagr_da_mensili, shrink_verso_ancora, ricentra_mensili, MESI_PIENA_FIDUCIA,
)
from pac_avanzato import render_pac_avanzato
from storia_lunga import render_tab_dati
from modello_lungo import render_tab_modello, serie_estesa_comparto

# ---------------------------------------------------------------------------
# CONFIGURAZIONE PAGINA
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Simulatore Previdenziale Pro", layout="wide")
st.title("🚀 Confronto Previdenziale: Fondo vs PAC + TFR")

# --- Gestione del Seed per ricalcolo casuale ---
if "master_seed" not in st.session_state:
    st.session_state.master_seed = 33

# ---------------------------------------------------------------------------
# DATI CCNL / FONDI NEGOZIALI — un file JSON per CCNL (data/ccnl/)
# ---------------------------------------------------------------------------
# Ogni CCNL/fondo negoziale vive in un proprio file JSON sotto data/ccnl/.
# Aggiungere un nuovo CCNL = copiare data/ccnl/_template.json, compilarlo e
# salvarlo con un nuovo nome: NON serve toccare questo script. I file che
# iniziano con "_" (come _template.json) vengono ignorati dal loader.
CARTELLA_CCNL_CANDIDATE = [
    "data/ccnl",
    "ccnl",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ccnl"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ccnl"),
]

def _trova_cartella_ccnl():
    for cartella in CARTELLA_CCNL_CANDIDATE:
        if os.path.isdir(cartella):
            return cartella
    return None

@st.cache_data
def carica_ccnl_preset():
    """
    Legge tutti i *.json in data/ccnl/ (esclusi quelli che iniziano con "_")
    e costruisce il dizionario CCNL_PRESET. Ritorna (preset, cartella, errori).
    """
    cartella = _trova_cartella_ccnl()
    if cartella is None:
        return {}, None, []

    campi_obbligatori = [
        "nome", "fondo", "contrib_lav_pct", "contrib_azienda_pct",
        "contrib_azienda_u35_pct", "tfr_pct", "costo_iniziale", "costo_fisso",
        "mensilita", "livelli", "scatto_ogni_anni", "scatti_max", "comparti",
    ]

    preset, errori = {}, []
    for fname in sorted(os.listdir(cartella)):
        if not fname.endswith(".json") or fname.startswith("_"):
            continue
        fpath = os.path.join(cartella, fname)
        try:
            with open(fpath, encoding="utf-8") as fh:
                cfg = json.load(fh)
        except Exception as e:
            errori.append(f"{fname}: JSON non valido ({e})")
            continue

        mancanti = [c for c in campi_obbligatori if c not in cfg]
        if mancanti:
            errori.append(f"{fname}: campi mancanti {mancanti}")
            continue

        comparti_raw = cfg["comparti"]
        if isinstance(comparti_raw, dict):
            lista_comparti = list(comparti_raw.keys())
        elif isinstance(comparti_raw, list):
            lista_comparti = list(comparti_raw)
        else:
            errori.append(f"{fname}: 'comparti' deve essere una lista di nomi")
            continue
        if not lista_comparti:
            errori.append(f"{fname}: 'comparti' è vuoto")
            continue

        nome = cfg["nome"]
        scatti_liv = cfg.get("scatti_valore_livello", {})
        livelli_senza_scatto = [l for l in cfg["livelli"] if l not in scatti_liv]
        if livelli_senza_scatto:
            errori.append(
                f"{fname}: livelli senza scatto specifico: {livelli_senza_scatto}. "
                f"Aggiungi tutti i livelli in 'scatti_valore_livello'."
            )
            continue

        preset[nome] = {
            "fondo": cfg["fondo"],
            "contrib_lav_pct": cfg["contrib_lav_pct"],
            "contrib_azienda_pct": cfg["contrib_azienda_pct"],
            "contrib_azienda_u35_pct": cfg["contrib_azienda_u35_pct"],
            "tfr_pct": cfg["tfr_pct"],
            "costo_iniziale": cfg["costo_iniziale"],
            "costo_fisso": cfg["costo_fisso"],
            "mensilita": cfg["mensilita"],
            "livelli": cfg["livelli"],
            "scatti_valore_livello": scatti_liv,
            "scatto_ogni_anni": cfg["scatto_ogni_anni"],
            "scatti_max": cfg["scatti_max"],
            "comparti": lista_comparti,
        }

    return preset, cartella, errori

CCNL_PRESET, _CARTELLA_CCNL, _ERRORI_CCNL = carica_ccnl_preset()

if not CCNL_PRESET:
    st.error(
        "**Nessun preset CCNL trovato.** Cercato in: "
        + ", ".join(f"`{c}`" for c in CARTELLA_CCNL_CANDIDATE) +
        ". Metti almeno un file .json (vedi data/ccnl/_template.json) nella "
        "cartella data/ccnl/ accanto allo script e ricarica la pagina."
    )
    st.stop()

if _ERRORI_CCNL:
    st.warning(
        "⚠️ Alcuni file CCNL in `" + str(_CARTELLA_CCNL) + "` sono stati "
        "ignorati per errori:\n\n" + "\n\n".join(f"- {e}" for e in _ERRORI_CCNL)
    )


# ---------------------------------------------------------------------------
# STORICO RENDIMENTI DEI COMPARTI — struttura MODULARE
# ---------------------------------------------------------------------------
# Ogni fondo e' dichiarato da un JSON in data/fondi/<nome>.json che indica il
# CSV delle quote (mai toccato dal codice: sola lettura) e l'anagrafica dei
# comparti (classe di rischio per le ancore, benchmark per la ricostruzione,
# garanzia, ISC). Aggiungere un fondo = aggiungere 1 CSV + 1 JSON, zero
# codice. Se data/fondi/ manca si usa il fallback storico hard-coded.
# ---------------------------------------------------------------------------
CARTELLE_DATI_CANDIDATE = [
    "data",
    ".",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
    os.path.dirname(os.path.abspath(__file__)),
]
_FALLBACK_FONDI = {"Cometa": "cometa.csv", "Fon.Te": "fonte.csv"}


@st.cache_data
def carica_anagrafiche_fondi():
    """
    Scansiona data/fondi/*.json. Ritorna dict nome_fondo -> anagrafica
    (chiavi: csv, comparti{...}). Fallback: dict minimale dai CSV noti.
    """
    anagrafiche = {}
    for cartella in CARTELLE_DATI_CANDIDATE:
        cartella_fondi = os.path.join(cartella, "fondi")
        if not os.path.isdir(cartella_fondi):
            continue
        for nome_file in sorted(os.listdir(cartella_fondi)):
            if not nome_file.endswith(".json"):
                continue
            try:
                with open(os.path.join(cartella_fondi, nome_file),
                          encoding="utf-8") as fh:
                    cfg = json.load(fh)
                if "nome" in cfg and "csv" in cfg:
                    anagrafiche[cfg["nome"]] = cfg
            except (json.JSONDecodeError, OSError) as e:
                st.warning(f"⚠️ Anagrafica fondo '{nome_file}' non leggibile: {e}")
        break   # usa la prima cartella data/ trovata
    if not anagrafiche:
        anagrafiche = {n: {"nome": n, "csv": c, "comparti": {}}
                       for n, c in _FALLBACK_FONDI.items()}
    return anagrafiche


ANAGRAFICA_FONDI = carica_anagrafiche_fondi()
FILE_STORICO_PER_FONDO = {n: cfg["csv"] for n, cfg in ANAGRAFICA_FONDI.items()}

def _trova_file_fondo(nome_file: str):
    for cartella in CARTELLE_DATI_CANDIDATE:
        candidato = os.path.join(cartella, nome_file)
        if os.path.isfile(candidato):
            return candidato
    return None

@st.cache_data
def carica_quote_storiche():
    """
    Legge un CSV largo per fondo (anno, mese, <comparto1>, ...) e costruisce:
    - STORICO_MENSILE / STORICO_ANNUALE per fondo/comparto
    - STORICO_MENSILE_ANNI / STORICO_ANNUALE_ANNI con l'anno di ogni osservazione
    Ritorna (mensile, annuale, mensile_anni, annuale_anni, percorsi, mancanti).
    """
    mensile, annuale = {}, {}
    mensile_anni, annuale_anni = {}, {}
    percorsi_trovati, fondi_mancanti = {}, []

    for fondo, nome_file in FILE_STORICO_PER_FONDO.items():
        percorso = _trova_file_fondo(nome_file)
        if percorso is None:
            fondi_mancanti.append((fondo, nome_file))
            continue
        percorsi_trovati[fondo] = percorso

        with open(percorso, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            comparti = [c for c in reader.fieldnames if c not in ("anno", "mese")]
            quote = {c: {} for c in comparti}
            for row in reader:
                anno = int(row["anno"]); mese = int(row["mese"])
                for c in comparti:
                    v = (row.get(c) or "").strip()
                    if v != "":
                        quote[c][(anno, mese)] = float(v)

        for comp, serie in quote.items():
            if len(serie) < 2:
                continue
            chiavi = sorted(serie)
            rend_m = [round(serie[chiavi[i]] / serie[chiavi[i-1]] - 1, 6)
                      for i in range(1, len(chiavi))]
            mensile.setdefault(fondo, {})[comp] = rend_m
            mensile_anni.setdefault(fondo, {})[comp] = [chiavi[i][0] for i in range(1, len(chiavi))]

            anni_dic = sorted({y for (y, m) in serie if m == 12})
            rend_a, anni_lista = [], []
            for y in anni_dic:
                if (y, 12) in serie and (y - 1, 12) in serie:
                    rend_a.append(round(serie[(y, 12)] / serie[(y - 1, 12)] - 1, 5))
                    anni_lista.append(y)
            annuale.setdefault(fondo, {})[comp] = rend_a
            annuale_anni.setdefault(fondo, {})[comp] = anni_lista

    return mensile, annuale, mensile_anni, annuale_anni, percorsi_trovati, fondi_mancanti

(STORICO_MENSILE, STORICO_ANNUALE, STORICO_MENSILE_ANNI, STORICO_ANNUALE_ANNI,
 _PERCORSI_TROVATI, _FONDI_MANCANTI) = carica_quote_storiche()

if _FONDI_MANCANTI:
    dettagli = "\n".join(
        f"- **{fondo}**: cercato `{nome_file}` in `data/` e nella cartella dello script"
        for fondo, nome_file in _FONDI_MANCANTI
    )
    st.error(
        "**File dati storici mancanti per uno o più fondi:**\n\n" + dettagli +
        "\n\nMetti i CSV (`cometa.csv`, `fonte.csv`) in una cartella `data/` "
        "accanto allo script e ricarica la pagina."
    )
    st.stop()

# --- VALIDAZIONE anagrafiche fondi vs CSV (struttura modulare) -------------
_avvisi_anagrafica = []
for _f, _comparti_csv in STORICO_MENSILE.items():
    _decl = set(ANAGRAFICA_FONDI.get(_f, {}).get("comparti", {}) or {})
    if not _decl:
        _avvisi_anagrafica.append(f"{_f}: nessuna anagrafica comparti (data/fondi/) — "
                                  f"ancore per parole chiave, ricostruzione off")
        continue
    for _c in sorted(set(_comparti_csv) - _decl):
        _avvisi_anagrafica.append(f"{_f}: '{_c}' nel CSV ma non in anagrafica")
    for _c in sorted(_decl - set(_comparti_csv)):
        _avvisi_anagrafica.append(f"{_f}: '{_c}' in anagrafica ma non nel CSV")
for _f, _cfg in ANAGRAFICA_FONDI.items():
    for _c, _info in (_cfg.get("comparti", {}) or {}).items():
        _b = _info.get("benchmark") or []
        if _b:
            _s = sum(float(x.get("peso", 0)) for x in _b)
            if abs(_s - 1.0) > 0.01:
                _avvisi_anagrafica.append(
                    f"{_f} · {_c}: pesi benchmark sommano a {_s:.2f} (≠1)")
if _avvisi_anagrafica:
    with st.expander("🧩 Validazione anagrafica fondi (data/fondi/)"):
        for _a in _avvisi_anagrafica:
            st.caption("• " + _a)


def mensile_disponibile(fondo: str, comparto: str, min_mesi: int = 24) -> bool:
    serie = STORICO_MENSILE.get(fondo, {}).get(comparto, [])
    return len(serie) >= min_mesi


def filtra_storico_da_anno(serie: list, anni: list, anno_inizio: int) -> list:
    """Mantiene solo le osservazioni con anno >= anno_inizio (mitiga il bias da punto di partenza)."""
    if not serie:
        return serie
    return [r for r, y in zip(serie, anni) if y >= anno_inizio]


_anni_min_per_serie = [min(anni) for fo in STORICO_ANNUALE_ANNI.values() for anni in fo.values() if anni]
_anni_max_per_serie = [max(anni) for fo in STORICO_ANNUALE_ANNI.values() for anni in fo.values() if anni]
ANNO_STORICO_MIN_GLOBALE = min(_anni_min_per_serie) if _anni_min_per_serie else 2000
ANNO_STORICO_MAX_GLOBALE = (max(_anni_max_per_serie) - 5) if _anni_max_per_serie else 2020

# ---------------------------------------------------------------------------
# COEFFICIENTI DI CRESCITA per tipo lavoratore (solo Operaio / Impiegato)
# ---------------------------------------------------------------------------
COEFF_LAVORATORE = {
    "Operaio":   0.88,
    "Impiegato": 1.08,
}

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.header("1. Contratto e Inquadramento")
ccnl_scelto = st.sidebar.selectbox("CCNL", list(CCNL_PRESET.keys()), index=0)
preset = CCNL_PRESET[ccnl_scelto]
mensilita = preset["mensilita"]
st.sidebar.caption(f"Fondo negoziale associato: **{preset['fondo']}**")

livello = st.sidebar.selectbox("Livello di inquadramento", list(preset["livelli"].keys()))
minimo_mensile = preset["livelli"][livello]
minimo_annuo = minimo_mensile * mensilita

scatto_valore_livello = preset["scatti_valore_livello"][livello]
comparti_base = list(preset["comparti"])

st.sidebar.caption(
    f"Minimo tabellare **{livello}**: {minimo_mensile:,.0f} €/mese × {mensilita} "
    f"mensilità = **{minimo_annuo:,.0f} €/anno**"
)
# --- Pulsante Ricalcolo ---
if st.sidebar.button("🎲 Ricalcola Scenari Casuali"):
    st.session_state.master_seed = int(np.random.randint(0, 100000))

st.sidebar.markdown("**Composizione della RAL**")
anni_anzianita_pregressi = st.sidebar.number_input(
    "Scatti di anzianità già maturati", min_value=0, max_value=preset["scatti_max"],
    value=0, step=1,
    help=f"Max {preset['scatti_max']} scatti, uno ogni {preset['scatto_ogni_anni']} anni. "
         f"Livello {livello}: {scatto_valore_livello:.1f} €/mese ciascuno",
)
superminimo_mensile = st.sidebar.number_input(
    "Superminimo (€/mese)", min_value=0, value=0, step=50,
    help="Voce individuale non prevista dal contratto. NON entra nella base "
         "di calcolo del contributo aziendale al fondo.",
)
premio_produzione_annuo = st.sidebar.number_input(
    "Premio di produzione (€/anno)", min_value=0, value=0, step=200,
    help="Premio di risultato variabile. NON entra nella base del contributo "
         "aziendale al fondo.",
)

st.sidebar.markdown("**Override manuale (opzionale)**")
ral_manuale = st.sidebar.number_input(
    "RAL effettiva a mano (€/anno, 0 = auto)", min_value=0, value=0, step=1000,
    help="Se la conosci, inserisci la tua RAL reale. Sostituisce quella calcolata "
         "e viene usata per TFR e IRPEF. Il contributo AZIENDA resta comunque "
         "calcolato sui minimi tabellari + scatti (come da contratto).",
)
capitale_iniziale_fondo = st.sidebar.number_input(
    "Capitale già presente nel fondo (€)", min_value=0, value=0, step=1000,
    help="Montante già accumulato se sei iscritto da tempo.",
)
capitale_iniziale_pac = st.sidebar.number_input(
    "Capitale già presente nel PAC (€)", min_value=0, value=0, step=1000,
    help="Montante ETF già accumulato, se il PAC è già avviato da tempo.",
)

st.sidebar.header("2. Profilo Lavoratore")
eta = st.sidebar.number_input("Età attuale", min_value=18, max_value=67, value=30, step=1)
tipo_lavoratore = st.sidebar.selectbox("Tipo di lavoratore", list(COEFF_LAVORATORE.keys()), index=1)
profilo_crescita = st.sidebar.selectbox(
    "Dinamismo di carriera",
    ["Moderata (2–5%/scatto)", "Media (3–7%/scatto)", "Spinta (6–10%/scatto)"],
    index=1,
)
crescita_base = st.sidebar.slider(
    "Crescita di base annua (inflazione + rinnovi CCNL) %", 0.0, 4.0, 2.0, 0.1,
    help="Adeguamento applicato ogni anno anche senza promozioni (~1,5–2,5% storico).",
) / 100

# --- Orizzonte spostato in alto: serve alle sezioni carriera/CCNL/disoccup. ---
durata = st.sidebar.slider("Anni di investimento", 1, 40, 25)

st.sidebar.markdown("**Passaggi di livello (promozioni pianificate)**")
usa_passaggi_livello = st.sidebar.checkbox(
    "Pianifica cambi di livello/mansione durante la carriera", value=False,
    help="Indica TU in quali anni futuri passerai a un livello superiore. Il "
         "nuovo minimo tabellare sostituisce la base da quell'anno; sopra continua "
         "la crescita simulata (scatti stocastici + inflazione).",
)
livelli_ccnl_lista = list(preset["livelli"].keys())
passaggi_livello = []  # lista di (anno_da, livello)
if usa_passaggi_livello:
    n_passaggi = st.sidebar.number_input(
        "Numero di passaggi di livello pianificati", min_value=1, max_value=10,
        value=1, step=1, key="n_passaggi_livello",
    )
    for i in range(int(n_passaggi)):
        pc1, pc2 = st.sidebar.columns([1, 2])
        anno_da = pc1.number_input(
            f"Anno #{i+1}", min_value=1, max_value=40, value=min(5 * (i + 1), 40),
            step=1, key=f"anno_passaggio_{i}",
        )
        livello_nuovo = pc2.selectbox(
            f"Nuovo livello #{i+1}", livelli_ccnl_lista,
            index=min(i + 1, len(livelli_ccnl_lista) - 1),
            key=f"livello_passaggio_{i}",
        )
        passaggi_livello.append((int(anno_da), livello_nuovo))
    passaggi_livello.sort(key=lambda x: x[0])

# --- Cambio CCNL / fondo durante la carriera --------------------------------
st.sidebar.markdown("**Cambio CCNL / fondo (cambio settore)**")
usa_cambio_ccnl = st.sidebar.checkbox(
    "Pianifica uno o più cambi di CCNL durante la carriera", value=False,
    help="Simula un cambio di settore/contratto: da un certo anno cambiano "
         "contributi, TFR, costi, comparto e minimi tabellari (nuovo fondo).",
)
cambi_ccnl = []  # lista di (anno_da, ccnl_name, livello, comparto)
if usa_cambio_ccnl:
    n_cambi = st.sidebar.number_input(
        "Numero di cambi CCNL pianificati", min_value=1, max_value=6,
        value=1, step=1, key="n_cambi_ccnl",
    )
    for i in range(int(n_cambi)):
        anno_c = st.sidebar.number_input(
            f"Cambio #{i+1} — anno", min_value=1, max_value=40,
            value=min(10 * (i + 1), 40), step=1, key=f"anno_ccnl_{i}",
        )
        ccnl_new = st.sidebar.selectbox(
            f"Cambio #{i+1} — nuovo CCNL", list(CCNL_PRESET.keys()),
            key=f"ccnl_new_{i}",
        )
        preset_new = CCNL_PRESET[ccnl_new]
        liv_new = st.sidebar.selectbox(
            f"Cambio #{i+1} — livello", list(preset_new["livelli"].keys()),
            key=f"liv_ccnl_{i}",
        )
        _cn = list(preset_new["comparti"])
        comp_new = st.sidebar.selectbox(
            f"Cambio #{i+1} — comparto", _cn,
            index=_cn.index("Azionario") if "Azionario" in _cn else len(_cn) - 1,
            key=f"comp_ccnl_{i}",
        )
        cambi_ccnl.append((int(anno_c), ccnl_new, liv_new, comp_new))
    cambi_ccnl.sort(key=lambda x: x[0])

    st.sidebar.markdown("**Contributi del datore al cambio fondo**")
    mantieni_contributi_azienda = st.sidebar.checkbox(
        "Mantieni nel nuovo fondo i contributi già versati dal VECCHIO datore",
        value=True,
        help="⚠️ Punto normativo NON del tutto pacifico. La posizione "
             "individuale nei fondi pensione negoziali è in linea generale "
             "portabile, ma la legge lascia margini di interpretazione su casi "
             "specifici. Attiva per simulare il trasferimento dell'intero "
             "montante; disattiva per lo scenario prudenziale in cui la quota "
             "del datore non si trasferisce. Verifica sempre lo statuto del "
             "fondo specifico.",
    )
else:
    mantieni_contributi_azienda = True  # nessun cambio CCNL pianificato: irrilevante

# --- Periodi di disoccupazione ------------------------------------------------
st.sidebar.markdown("**Periodi di disoccupazione**")
usa_disoccupazione = st.sidebar.checkbox(
    "Inserisci periodi senza reddito", value=False,
    help="Negli anni indicati: nessun contributo (TFR, azienda, tuo, PAC) e "
         "RAL a zero. I capitali già accumulati continuano comunque a rendere.",
)
anni_disoccupato = set()
if usa_disoccupazione:
    anni_disoccupato = set(st.sidebar.multiselect(
        "Anni di disoccupazione", list(range(1, durata + 1)),
        help="Anno 1 = primo anno di simulazione.",
    ))

st.sidebar.header("3. Fondo")
_idx_comp = comparti_base.index("Azionario") if "Azionario" in comparti_base else len(comparti_base) - 1
comparto = st.sidebar.selectbox("Comparto d'investimento", comparti_base, index=_idx_comp)

under35 = eta < 35
contrib_az_pct = preset["contrib_azienda_u35_pct"] if under35 else preset["contrib_azienda_pct"]

st.sidebar.caption(
    f"**{preset['fondo']} · {comparto}** — datore {contrib_az_pct*100:.2f}% "
    f"(sui minimi+scatti) · tu min {preset['contrib_lav_pct']*100:.2f}% · "
    f"TFR {preset['tfr_pct']*100:.2f}%. Il rendimento del comparto viene dallo "
    f"storico reale della quota (già netto di tasse e costi di gestione), non "
    f"da un'assunzione parametrica."
)

vers_vol_extra = st.sidebar.number_input(
    "Versamento volontario annuo (€)", min_value=0, value=1000, step=100,
    help="È il TUO flusso totale al fondo (oltre a TFR e contributo azienda). "
         "Deve coprire ALMENO il minimo CCNL (~0,5-0,7% dei minimi tabellari): "
         "sotto quella soglia perdi il diritto al contributo aziendale. "
         "Deducibile dall'IRPEF. Resta FISSO nel tempo a meno di variazioni "
         "pianificate qui sotto.",
)
usa_variazioni_vol_extra = st.sidebar.checkbox(
    "Pianifica variazioni di questo versamento nel tempo", value=False,
    key="usa_var_vol_extra",
    help="Es.: dall'anno 6 alzi a 1500€, dall'anno 12 abbassi a 800€.",
)
variazioni_vol_extra = []
if usa_variazioni_vol_extra:
    n_var_ve = st.sidebar.number_input(
        "Numero di variazioni", min_value=1, max_value=10, value=1, step=1,
        key="n_var_vol_extra",
    )
    for i in range(int(n_var_ve)):
        vc1, vc2 = st.sidebar.columns(2)
        anno_v = vc1.number_input(
            f"Var. #{i+1} — dall'anno", min_value=1, max_value=40,
            value=min(6 * (i + 1), 40), step=1, key=f"anno_var_ve_{i}",
        )
        importo_v = vc2.number_input(
            f"Var. #{i+1} — nuovo importo (€/anno)", min_value=0,
            value=1000, step=100, key=f"importo_var_ve_{i}",
        )
        variazioni_vol_extra.append((int(anno_v), float(importo_v)))

st.sidebar.header("4. Performance simulata (Fondo)")
st.sidebar.caption(
    "I rendimenti del fondo vengono SEMPRE dal ricampionamento (block-"
    "bootstrap) dello storico mensile reale della quota del comparto — mai "
    "da un'assunzione parametrica."
)
usa_mensile = True  # unico metodo disponibile: block-bootstrap mensile
block_mesi = st.sidebar.number_input(
    "Lunghezza blocco (mesi)", min_value=3, max_value=24, value=12, step=1,
    help="Dimensione del blocco contiguo ricampionato dallo storico mensile "
         "reale. 12 = un anno intero (preserva stagionalità e sequenze "
         "annuali); valori più piccoli mescolano più liberamente i mesi.",
)

st.sidebar.markdown("**Orizzonte storico usato per il resampling**")
anno_inizio_storico = st.sidebar.slider(
    "Escludi gli anni precedenti a...", ANNO_STORICO_MIN_GLOBALE, ANNO_STORICO_MAX_GLOBALE,
    ANNO_STORICO_MIN_GLOBALE, 1,
    help="Taglia via dallo storico usato per il resampling tutti gli anni "
         "precedenti a quello scelto (mitiga il bias da 'punto di partenza', "
         "es. rimbalzo post-2008). Default = usa tutta la storia disponibile.",
)
if anno_inizio_storico > ANNO_STORICO_MIN_GLOBALE:
    st.sidebar.caption(
        f"ℹ️ Storico troncato: verranno usate solo le osservazioni dal "
        f"{anno_inizio_storico} in poi, per ogni comparto che ha dati "
        f"precedenti a quell'anno."
    )

with st.sidebar.expander("🎯 Correzione realismo del drift (shrinkage)", expanded=False):
    st.caption(
        "La media di un campione corto è rumorosa e spesso riflette un solo "
        "regime di mercato (es. bull 2015-oggi). Il drift usato dal bootstrap "
        "viene corretto verso un'**ancora di lungo periodo**: "
        "`drift = w·storico + (1−w)·ancora`, con `w = mesi/240` (20 anni = "
        "piena fiducia allo storico). La correzione ricentra i rendimenti "
        "mensili ricampionati preservando volatilità e sequenze. "
        "Come ancora puoi usare la media secolare oppure una **Capital "
        "Market Assumption** aggiornata (JPMorgan LTCMA, Vanguard, "
        "BlackRock: pubbliche, riviste ogni anno)."
    )
    usa_shrinkage_fondo = st.checkbox("Applica shrinkage al fondo", value=True,
                                      key="shrink_fondo_on")
    mesi_piena_fiducia = st.number_input(
        "Mesi per piena fiducia allo storico", 120, 480, 240, 12,
        key="mesi_fiducia",
        help="w = mesi disponibili / questo valore (cap 100%). Con 240 (20 "
             "anni) i comparti con storico lungo NON vengono corretti (w=100%). "
             "Il confronto bayesiano nel tab '🔬 Modello & Validazione' "
             "suggerisce 300-360+ per l'azionario se l'ancora è una CMA "
             "seria. Vale per fondo E per il PAC a ticker.",
    )
    ancora_azionario = st.number_input(
        "Ancora comparti azionari (CAGR nominale %)", 0.0, 12.0, 6.5, 0.1,
        key="anc_az", disabled=not usa_shrinkage_fondo)
    ancora_bilanciato = st.number_input(
        "Ancora comparti bilanciati/dinamici (%)", 0.0, 10.0, 4.0, 0.1,
        key="anc_bil", disabled=not usa_shrinkage_fondo)
    ancora_prudente = st.number_input(
        "Ancora comparti garantiti/monetari (%)", 0.0, 8.0, 2.5, 0.1,
        key="anc_pru", disabled=not usa_shrinkage_fondo)


st.sidebar.markdown("**Factor modeling (storia lunga)**")
usa_ricostruite = st.sidebar.checkbox(
    "Usa serie RICOSTRUITE dal benchmark (factor modeling)", value=False,
    key="usa_ricostruite",
    help="COS'È: i comparti hanno pochi anni di quote (Garantito dal 2020, "
         "Fon.Te dal 2008...). La ricostruzione estende lo storico "
         "all'indietro col MIX DEL BENCHMARK dichiarato dal comparto "
         "(es. 60% azionario globale + 40% bond), costruito sulle serie "
         "lunghe (dal 1976-86) e corretto dell'alfa di gestione stimato "
         "sull'overlap. Il bootstrap poi ricampiona DECENNI (con dentro "
         "2008, 2000, ...) invece del solo regime recente. Qualità del "
         "mapping (R², pesi stimati vs dichiarati): tab '🔬 Modello & "
         "Validazione', sezione 1.",
)
override_ricostruzione = {}
if usa_ricostruite:
    st.sidebar.caption(
        "Attiva/disattiva PER COMPARTO (solo quelli con benchmark in "
        "anagrafica; i garantiti sono esclusi by design). Prima di fidarti: "
        "controlla l'R² nel tab Modello e i pesi sul DPI del fondo."
    )
    for _f_r, _cfg_r in ANAGRAFICA_FONDI.items():
        for _c_r, _i_r in (_cfg_r.get("comparti") or {}).items():
            if not _i_r.get("benchmark"):
                continue
            _default_r = _i_r.get("ricostruzione") == "consentita"
            override_ricostruzione[(_f_r, _c_r)] = st.sidebar.toggle(
                f"Ricostruisci {_f_r} · {_c_r}", value=_default_r,
                key=f"ric_{_f_r}_{_c_r}",
                help=("Benchmark: " +
                      " + ".join(f"{v['peso']:.0%} {v['serie']}"
                                 for v in _i_r["benchmark"]) +
                      (". ⚠️ Da verificare: " + "; ".join(_i_r["da_verificare"])
                       if _i_r.get("da_verificare") else "")),
            )

st.sidebar.markdown("**Glidepath di comparto (life-cycle)**")
usa_switch_comparto = st.sidebar.checkbox(
    "Pianifica cambi di comparto", value=False, key="usa_switch_comp",
    help="Come fanno i life-cycle reali: es. Azionario fino all'anno 25, "
         "poi Bilanciato, poi Garantito. Ogni cambio vale dall'anno "
         "indicato in poi (sul fondo attivo in quell'anno).",
)
cambi_comparto = []
if usa_switch_comparto:
    _comparti_disp = sorted(STORICO_MENSILE.get(preset["fondo"], {}).keys())
    n_sw = st.sidebar.number_input("Numero di cambi", 1, 5, 1, 1, key="n_sw_comp")
    for i in range(int(n_sw)):
        sc1, sc2 = st.sidebar.columns(2)
        anno_sw = sc1.number_input(f"Cambio #{i+1} — dall'anno", 1, 40,
                                   min(25 + 8 * i, 40), 1, key=f"sw_anno_{i}")
        comp_sw = sc2.selectbox(f"Cambio #{i+1} — comparto", _comparti_disp,
                                key=f"sw_comp_{i}")
        cambi_comparto.append((int(anno_sw), comp_sw))


def ancora_per_comparto(fondo: str, nome_comparto: str) -> float:
    """
    Ancora di lungo periodo del comparto: prima dalla classe_rischio
    dichiarata nell'anagrafica del fondo (data/fondi/<fondo>.json), poi
    fallback euristico sulle parole del nome.
    """
    info = ANAGRAFICA_FONDI.get(fondo, {}).get("comparti", {}).get(nome_comparto, {})
    classe = info.get("classe_rischio", "")
    if classe == "azionario":
        return ancora_azionario / 100.0
    if classe == "prudente":
        return ancora_prudente / 100.0
    if classe == "bilanciato":
        return ancora_bilanciato / 100.0
    nl = nome_comparto.lower()
    if any(k in nl for k in ("azion",)):
        return ancora_azionario / 100.0
    if any(k in nl for k in ("garant", "conserv", "monet", "prudent")):
        return ancora_prudente / 100.0
    return ancora_bilanciato / 100.0   # bilanciato/dinamico/sviluppo/crescita

st.sidebar.caption("Banda P10–P90 mostrata su tutte le curve (200 scenari).")
percentile_perf = st.sidebar.slider(
    "Percentile della linea centrale", 5, 95, 50, 5,
    help="P5 = scenario molto sfortunato · P50 = mediano · P95 = molto fortunato. "
         "La banda P10–P90 attorno resta sempre visibile.",
)

st.sidebar.header("5. PAC (ETF)")
versamento_pac = st.sidebar.number_input(
    "Versamento PAC Annuo (€)", min_value=0, value=3445, step=100,
    help="Resta FISSO nel tempo (nessuno scaling con la carriera) a meno di "
         "variazioni pianificate qui sotto.",
)
usa_variazioni_pac = st.sidebar.checkbox(
    "Pianifica variazioni del versamento PAC nel tempo", value=False,
    key="usa_var_pac",
    help="Es.: dall'anno 6 alzi a 5000€, dall'anno 12 abbassi a 2000€.",
)
variazioni_pac = []
if usa_variazioni_pac:
    n_var_pac = st.sidebar.number_input(
        "Numero di variazioni", min_value=1, max_value=10, value=1, step=1,
        key="n_var_pac",
    )
    for i in range(int(n_var_pac)):
        vc1, vc2 = st.sidebar.columns(2)
        anno_v = vc1.number_input(
            f"Var. #{i+1} — dall'anno", min_value=1, max_value=40,
            value=min(6 * (i + 1), 40), step=1, key=f"anno_var_pac_{i}",
        )
        importo_v = vc2.number_input(
            f"Var. #{i+1} — nuovo importo (€/anno)", min_value=0,
            value=3445, step=100, key=f"importo_var_pac_{i}",
        )
        variazioni_pac.append((int(anno_v), float(importo_v)))

modo_pac = st.sidebar.radio(
    "Modalità PAC",
    ["Semplice (parametri manuali)", "Portafoglio ticker (dati storici)"],
    index=0,
    help="Con i ticker, rendimenti/volatilità/correlazioni vengono stimati dallo "
         "storico Yahoo Finance e la simulazione usa asset correlati via Cholesky.",
)
usa_portafoglio = modo_pac.startswith("Portafoglio")

rend_medio_pac, vol_pac = 0.065, 0.15   # fallback (CAGR, non media aritmetica)
tickers_input = pesi_input = ""
anni_storico, override_rend, rend_override_val = 15, False, None
usa_shrinkage_pac, ancora_pac = True, 0.065
usa_bootstrap_pac = False
quota_ts_auto = 0.0
ticker_bond_non_stimabili = []

if usa_portafoglio:
    st.sidebar.markdown("**Catalogo ETF predefiniti (solo accumulazione UCITS)**")
    st.sidebar.caption(
        "Seleziona uno o più ETF dalla legenda, oppure aggiungine a mano. "
        "I ticker manuali NON in whitelist vengono segnalati (dist/acc/UCITS)."
    )

    selezione_catalogo = {}
    for categoria, etfs in CATALOGO_ETF.items():
        scelti = st.sidebar.multiselect(categoria, list(etfs.keys()), key=f"cat_{categoria}")
        for nome in scelti:
            selezione_catalogo[etfs[nome]] = nome

    tickers_manuali_str = st.sidebar.text_input(
        "Aggiungi ticker manuale (separati da virgola, opzionale)", value="",
        help="Per ETF non in catalogo o AZIONI SINGOLE (es. AAPL, ENI.MI). "
             "Le azioni sono ammesse ma molto più volatili di un ETF: l'app "
             "ti avviserà. Per gli ETF verifica che siano ad accumulo UCITS.",
    )
    tickers_manuali = [t.strip().upper() for t in tickers_manuali_str.split(",") if t.strip()]

    tickers_scelti = list(selezione_catalogo.keys())
    for t in tickers_manuali:
        if t not in tickers_scelti:
            tickers_scelti.append(t)

    # --- CONTROLLO ACCUMULAZIONE / UCITS sui ticker scelti ---
    avvisi_ticker = []
    for t in tickers_scelti:
        stato, nota = classifica_ticker(t)
        if stato == "warn":
            avvisi_ticker.append(f"⚠️ **{t}**: {nota}")
        elif stato == "sconosciuto":
            avvisi_ticker.append(f"❓ **{t}**: {nota}")
    if avvisi_ticker:
        st.sidebar.warning(
            "Controllo accumulazione/UCITS:\n\n" + "\n\n".join(avvisi_ticker)
        )

    if len(tickers_scelti) == 0:
        st.sidebar.warning("Nessun ticker selezionato: scegline almeno uno.")

    st.sidebar.markdown("**Pesi (%) per ciascun ticker selezionato**")
    pesi_dict = {}
    peso_default = round(100 / len(tickers_scelti), 1) if tickers_scelti else 0.0
    for t in tickers_scelti:
        etichetta = TICKER_TO_NOME.get(t, t)
        pesi_dict[t] = st.sidebar.number_input(
            f"Peso {etichetta}", min_value=0.0, max_value=100.0,
            value=peso_default, step=1.0, key=f"peso_{t}",
        )

    somma_pesi = sum(pesi_dict.values())
    if tickers_scelti:
        if abs(somma_pesi - 100.0) > 0.01:
            st.sidebar.caption(f"Somma pesi: {somma_pesi:.1f}% — normalizzata a 100%.")
        else:
            st.sidebar.caption(f"Somma pesi: {somma_pesi:.1f}% ✓")

    # Stima automatica della quota "titoli di Stato/white list" del portafoglio
    quota_ts_auto = 0.0
    ticker_bond_non_stimabili = []
    ticker_obbligazionari = set(CATALOGO_ETF.get("Obbligazionario", {}).values())
    if tickers_scelti and somma_pesi > 0:
        for t in tickers_scelti:
            peso_norm = pesi_dict[t] / somma_pesi
            quota_ts_auto += peso_norm * QUOTA_TITOLI_STATO_TICKER.get(t, 0.0)
            if t in ticker_obbligazionari and t not in QUOTA_TITOLI_STATO_TICKER:
                ticker_bond_non_stimabili.append(t)

    tickers_input = ", ".join(tickers_scelti)
    pesi_input = ", ".join(str(pesi_dict[t]) for t in tickers_scelti)

    with st.sidebar.expander("📖 Legenda completa ETF disponibili"):
        for categoria, etfs in CATALOGO_ETF.items():
            st.markdown(f"**{categoria}**")
            for nome, ticker in etfs.items():
                st.caption(f"`{ticker}` — {nome}")

    anni_storico = st.sidebar.slider(
        "Anni di storico per la stima", 5, 20, 15,
        help="Più anni = stima meno rumorosa e meno dipendente da un singolo "
             "regime. Limite pratico: molti UCITS quotati a Milano nascono "
             "dopo il 2015-2018, quindi lo storico comune può essere corto.",
    )
    motore_pac = st.sidebar.radio(
        "Motore Monte Carlo del PAC",
        ["GBM multivariato (Cholesky)", "Block-bootstrap storico"],
        index=0, key="pac_motore",
        help="Come nel tab PAC: **GBM** genera rendimenti casuali dai parametri "
             "stimati (lognormale, correlazioni via Cholesky); **bootstrap** "
             "ricampiona a blocchi la serie storica PESATA del portafoglio "
             "(blocco = quello impostato per il fondo, sezione 4). In entrambi "
             "i casi si applica il drift corretto qui sotto. Qui niente "
             "derisking/decumulo: accumulo puro a pesi fissi.",
    )
    usa_bootstrap_pac = motore_pac.startswith("Block")

    st.sidebar.markdown("**Drift del portafoglio (correzione realismo)**")
    correzione_drift_pac = st.sidebar.radio(
        "Come fissare il rendimento atteso",
        ["Shrinkage automatico verso l'ancora", "Manuale (CAGR)", "Solo storico (sconsigliato)"],
        index=0, key="corr_drift_pac",
        help="Lo storico breve è un cattivo predittore: con 10 anni di dati e "
             "vol 15% l'errore standard sulla media annua è ±4,7 punti. Lo "
             "shrinkage pesa lo storico per w = mesi/240 e l'ancora per il "
             "resto. Volatilità e correlazioni restano SEMPRE quelle storiche.",
    )
    override_rend = correzione_drift_pac.startswith("Manuale")
    usa_shrinkage_pac = correzione_drift_pac.startswith("Shrinkage")
    if override_rend:
        rend_override_val = st.sidebar.slider(
            "CAGR atteso portafoglio (%)", 1.0, 12.0, 6.0, 0.1,
            help="Rendimento composto annuo: la traiettoria MEDIANA compone "
                 "a questo tasso (correzione di Itô inclusa).") / 100
    ancora_pac = st.sidebar.number_input(
        "Ancora di lungo periodo portafoglio (CAGR %)", 0.0, 12.0, 6.5, 0.1,
        key="anc_pac", disabled=not usa_shrinkage_pac,
        help="Media secolare azionario mondiale ~6,5% nominale. In "
             "alternativa inserisci una CMA aggiornata (JPMorgan LTCMA, "
             "Vanguard), pesata per la tua allocazione azionario/bond.",
    ) / 100
else:
    rend_medio_pac = st.sidebar.slider(
        "Rendimento composto atteso PAC (CAGR, %)", 1.0, 12.0, 6.5, 0.1,
        help="CAGR nominale LORDO: la traiettoria mediana compone esattamente "
             "a questo tasso (il volatility drag è già incorporato — non "
             "inserire una media aritmetica). Riferimenti: azionario globale "
             "~6,5% secolare; 60/40 ~5%; bond ~2,5-3%.") / 100
    vol_pac        = st.sidebar.slider("Volatilità PAC (%)", 5.0, 25.0, 15.0, 0.5) / 100

code_grasse_pac = st.sidebar.checkbox(
    "Code grasse (T di Student, ν=5)", value=True, key="pac_tstudent",
    help="Shock a code grasse invece della gaussiana: i percentili bassi "
         "(P10) diventano più severi e realistici (i crash di mercato sono "
         "più frequenti di quanto preveda la normale). Vale per i motori "
         "GBM del PAC; il bootstrap del fondo ha già le code dei dati reali.",
)
NU_T_PAC = 5.0 if code_grasse_pac else 0.0

ter_pac          = st.sidebar.number_input("TER PAC (%)", value=0.20, step=0.01) / 100
tassa_uscita_pac = st.sidebar.slider("Tassazione Plusvalenze PAC (%)", 0, 26, 26)

st.sidebar.markdown("**Componente obbligazionaria — tassazione differenziata**")
usa_tassazione_ts_pac = st.sidebar.checkbox(
    "Applica aliquota ridotta (12,5%) sulla quota in titoli di Stato", value=False,
    help="Aliquota ridotta al 12,5% sulla quota riferibile a titoli di Stato "
         "italiani/white list. ⚠️ Modello SEMPLIFICATO (aliquota media pesata): "
         "la percentuale esatta va certificata dall'emittente. Verifica sempre "
         "con la documentazione ufficiale del tuo strumento.",
)
if usa_tassazione_ts_pac:
    default_quota = round(quota_ts_auto * 100, 1) if usa_portafoglio else 0.0
    quota_ts_pac_pct = st.sidebar.slider(
        "Quota del PAC in titoli di Stato/white list (%)", 0.0, 100.0,
        default_quota, 1.0,
        help="Pre-impostata stimando i soli ETF di titoli di Stato puri "
             "riconosciuti automaticamente. Per fondi obbligazionari MISTI o "
             "ticker manuali, correggi tu in base al KID/factsheet.",
    )
    if usa_portafoglio and ticker_bond_non_stimabili:
        st.sidebar.caption(
            "ℹ️ Nel portafoglio hai " + ", ".join(ticker_bond_non_stimabili) +
            ": è un obbligazionario MISTO (governativo+corporate), la quota "
            "titoli di Stato non è stimata automaticamente per questo ticker "
            "— il valore sopra la considera 0% a meno che tu non la corregga."
        )
    quota_ts_pac = quota_ts_pac_pct / 100
else:
    quota_ts_pac = 0.0

st.sidebar.header("6. TFR in Azienda")
rend_tfr  = st.sidebar.slider("Rendimento Annuo TFR in Azienda (%)", 0.0, 7.0, 2.5, 0.1,
                              help="Rivalutazione legale: 1,5% + 75% inflazione")/100
tassa_tfr = st.sidebar.slider("Tassazione TFR Uscita (%)", 23, 43, 27)

st.sidebar.header("7. Uscita dal fondo")
anni_gia_iscritto = st.sidebar.number_input(
    "Anni di adesione già maturati al fondo", min_value=0, max_value=40, value=0, step=1,
    help="Servono per l'aliquota di uscita agevolata (sconto dopo il 15° anno)",
)
motivo_uscita = st.sidebar.selectbox(
    "Motivo di uscita dal fondo",
    [
        "Prestazione pensionistica / causali agevolate (9–15%)",
        "Riscatto/anticipazione ordinaria (23%)",
    ],
    index=0,
)
uscita_ordinaria = motivo_uscita.startswith("Riscatto")
usa_entrambi = st.sidebar.checkbox("Uso sia Fondo che PAC (somma senza TFR)", value=True)


# ---------------------------------------------------------------------------
# IRPEF
# ---------------------------------------------------------------------------
LIMITE_DEDUCIBILITA = 5164.57

def aliquota_marginale(imponibile: float) -> float:
    if imponibile <= 28_000:
        return 0.23
    elif imponibile <= 50_000:
        return 0.35
    else:
        return 0.43

def calcola_irpef(imponibile: float) -> float:
    imponibile = max(0.0, imponibile)
    if imponibile <= 28_000:
        return imponibile * 0.23
    elif imponibile <= 50_000:
        return 28_000 * 0.23 + (imponibile - 28_000) * 0.35
    else:
        return 28_000 * 0.23 + 22_000 * 0.35 + (imponibile - 50_000) * 0.43


# ---------------------------------------------------------------------------
# ALIQUOTA DI USCITA DEL FONDO PENSIONE
# ---------------------------------------------------------------------------
def aliquota_uscita_fondo(anni_adesione_totali: int, ordinaria: bool = False) -> float:
    if ordinaria:
        return 0.23
    if anni_adesione_totali <= 15:
        return 0.15
    sconto = min(anni_adesione_totali - 15, 20) * 0.003
    return max(0.09, 0.15 - sconto)


# ---------------------------------------------------------------------------
# GENERAZIONE 1000 SIMULAZIONI DI CARRIERA
# ---------------------------------------------------------------------------
@st.cache_data
def genera_scenari(profilo: str, coeff: float, crescita_base: float,
                   n: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    range_profilo = {"Moderata": (0.02, 0.05), "Media": (0.03, 0.07), "Spinta": (0.06, 0.10)}
    profilo_key = profilo.split(" ")[0]
    r_min, r_max = range_profilo[profilo_key]
    boost_junior = 1.35 if profilo_key == "Spinta" else 1.55

    scenari = []
    for _ in range(n):
        molt = 1.0
        percorso = [1.0]
        attesa = 0
        target = rng.integers(1, 3)
        for anno in range(1, 40):
            attesa += 1
            base_anno = max(0.0, crescita_base + rng.normal(0, 0.004))
            molt *= (1.0 + base_anno)
            if anno <= 6:
                fase_molt, min_t, max_t = boost_junior, 1, 2
            elif anno <= 10:
                fase_molt, min_t, max_t = 1.05, 2, 3
            elif anno <= 18:
                fase_molt, min_t, max_t = 0.45, 3, 4
            else:
                fase_molt, min_t, max_t = 0.30, 4, 6
            prob_cambio = 0.15 if anno <= 10 else (0.08 if anno <= 18 else 0.04)
            cambio = (anno > 3) and (rng.random() < prob_cambio)
            if attesa >= target or cambio:
                amp = r_min + rng.random() * (r_max - r_min)
                amp *= fase_molt * coeff
                amp += rng.normal(0, 0.006)
                amp = max(0.0, amp)
                molt *= (1.0 + amp)
                attesa = 0
                target = rng.integers(min_t, max_t + 1)
            percorso.append(molt)
        scenari.append(percorso)
    return scenari


# ---------------------------------------------------------------------------
# BLOCK-BOOTSTRAP MENSILE DEL FONDO (resta qui: usa lo storico dei comparti)
# ---------------------------------------------------------------------------
@st.cache_data
def genera_rendimenti_block_bootstrap(serie_mensile: tuple, durata: int,
                                      block: int = 12, n: int = 200, seed: int = 33):
    """
    BLOCK-BOOTSTRAP MENSILE (Moving Block). Ricampiona blocchi CONTIGUI
    di `block` mesi dai rendimenti mensili storici reali del comparto
    (senza wrap-around) e li concatena fino a coprire `durata` anni.
    """
    serie = np.array(serie_mensile, dtype=float)
    m = serie.size

    if m < block:
        raise ValueError(f"Servono almeno {block} mesi, disponibili {m}.")

    rng = np.random.default_rng(seed)
    mesi_tot = durata * 12
    out = np.empty((n, mesi_tot))
    n_blocchi = int(np.ceil(mesi_tot / block))

    for s in range(n):
        start = rng.integers(0, m - block + 1, size=n_blocchi)
        path = np.concatenate([serie[st : st + block] for st in start])[:mesi_tot]
        out[s] = path

    return out


# ---------------------------------------------------------------------------
# COSTRUZIONE DELLO SCHEDULE ANNO-PER-ANNO (livello, CCNL, comparto, disoccup.)
# ---------------------------------------------------------------------------
def costruisci_serie_a_gradini(durata: int, base: float, variazioni: list) -> list:
    """Serie annua a gradini: parte da `base` e cambia negli anni di `variazioni`."""
    eventi = sorted(variazioni, key=lambda x: x[0])
    serie = []
    corrente = base
    idx_evento = 0
    for a in range(durata):
        anno = a + 1
        while idx_evento < len(eventi) and eventi[idx_evento][0] <= anno:
            corrente = eventi[idx_evento][1]
            idx_evento += 1
        serie.append(corrente)
    return serie


def costruisci_schedule(durata, ccnl_start, livello_start, comparto_start,
                        eta, anni_pregressi_scatti, superminimo_annuo,
                        premio_annuo, crescita_base, passaggi_livello,
                        cambi_ccnl, anni_disoccupato):
    """
    Lista lunga `durata`: CCNL/livello/comparto attivo in ogni anno e parametri
    contributivi derivati (gestisce cambi livello, cambi CCNL, disoccupazione).
    """
    eventi = []
    for anno_da, liv in passaggi_livello:
        eventi.append((anno_da, "livello", liv))
    for anno_da, ccnl_n, liv_n, comp_n in cambi_ccnl:
        eventi.append((anno_da, "ccnl", (ccnl_n, liv_n, comp_n)))
    eventi.sort(key=lambda x: x[0])

    sched = []
    for a in range(durata):
        anno = a + 1
        ccnl_att, liv_att, comp_att = ccnl_start, livello_start, comparto_start
        anno_ultimo_cambio_ccnl = 0
        for anno_da, tipo, payload in eventi:
            if anno_da <= anno:
                if tipo == "livello":
                    liv_att = payload
                else:
                    ccnl_att, liv_att, comp_att = payload
                    anno_ultimo_cambio_ccnl = anno_da
        preset_a = CCNL_PRESET[ccnl_att]
        if comp_att not in preset_a["comparti"]:
            comp_att = preset_a["comparti"][-1]

        mens_a = preset_a["mensilita"]
        minimo_mensile_a = preset_a["livelli"][liv_att]
        minimo_annuo_a = minimo_mensile_a * mens_a
        scatto_val_liv_a = preset_a["scatti_valore_livello"][liv_att]
        scatto_annuo_a = scatto_val_liv_a * mens_a
        freq_a = preset_a["scatto_ogni_anni"]
        max_a = preset_a["scatti_max"]

        eta_corrente = eta + a
        u35 = eta_corrente < 35
        ca_pct_a = preset_a["contrib_azienda_u35_pct"] if u35 else preset_a["contrib_azienda_pct"]
        lav_pct_a = preset_a["contrib_lav_pct"]
        tfr_pct_a = preset_a["tfr_pct"]
        costo_fisso_a = preset_a["costo_fisso"]

        # Scatti: un cambio CCNL azzera l'anzianità; un cambio livello no.
        if anno_ultimo_cambio_ccnl == 0:
            anni_servizio = anni_pregressi_scatti * freq_a + anno
        else:
            anni_servizio = anno - anno_ultimo_cambio_ccnl + 1
        scatti_maturati = min(max_a, anni_servizio // freq_a)
        base_teorica = minimo_annuo_a + scatti_maturati * scatto_annuo_a
        base_contrib_a = base_teorica * ((1 + crescita_base) ** a)

        scatti_in_ral = anni_pregressi_scatti if anno_ultimo_cambio_ccnl == 0 else 0
        ral_base_eff_a = (minimo_annuo_a + scatti_in_ral * scatto_annuo_a
                          + superminimo_annuo + premio_annuo)

        occupato = anno not in anni_disoccupato

        sched.append({
            "anno": anno,
            "ccnl": ccnl_att, "livello": liv_att, "comparto": comp_att,
            "fondo": preset_a["fondo"], "comparto_key": (preset_a["fondo"], comp_att),
            "mensilita": mens_a,
            "ca_pct": ca_pct_a, "lav_pct": lav_pct_a, "tfr_pct": tfr_pct_a,
            "costo_fisso_f": costo_fisso_a,
            "base_contrib": base_contrib_a if occupato else 0.0,
            "ral_base_eff": ral_base_eff_a,
            "scatti": scatti_maturati,
            "occupato": occupato,
            "cambio_ccnl_qui": (anno_ultimo_cambio_ccnl == anno),
        })
    return sched


# ---------------------------------------------------------------------------
# MOTORE DI SIMULAZIONE DEL CAPITALE (schedule-driven)
# ---------------------------------------------------------------------------
def simula_capitale(fattori, rend_fondo_mensili, rend_pac_mensili, sched, scal,
                    vol_extra_serie, vp_serie) -> pd.DataFrame:
    """
    Simula il montante MESE PER MESE (versamenti in 12 rate; ogni rata rende
    solo per i mesi residui). Vedi commenti nel corpo per il trattamento
    fiscale del fondo (quota già netta) vs PAC (TER + tassa in uscita).
    """
    ral_override = scal["ral_override"]
    ral_manuale = scal["ral_manuale"]
    ter_p = scal["ter_p"]
    ter_gia_incluso_pac = scal.get("ter_gia_incluso_pac", False)
    tp = scal["tp"] / 100
    quota_ts_pac = scal.get("quota_ts_pac", 0.0)
    tp_eff = tp * (1 - quota_ts_pac) + 0.125 * quota_ts_pac
    rt = scal["rt"]
    tt = scal["tt"] / 100
    anni_pregressi = scal["anni_pregressi"]
    uscita_ord = scal["uscita_ordinaria"]

    cap_fondo = float(scal.get("cap_iniziale_fondo", 0.0))
    cap_pac = float(scal.get("cap_iniziale_pac", 0.0))
    versato_pac_cum = float(scal.get("cap_iniziale_pac", 0.0))
    versato_fondo_cum = float(scal.get("cap_iniziale_fondo", 0.0))
    cap_tfr = 0.0
    mantieni_contributi_azienda = scal.get("mantieni_contributi_azienda", True)
    cap_fondo_azienda_ombra = 0.0
    rows = []

    for a, f in enumerate(fattori):
        s = sched[a]
        anno = a + 1
        occupato = s["occupato"]

        if s["cambio_ccnl_qui"] and not mantieni_contributi_azienda:
            cap_fondo = max(0.0, cap_fondo - cap_fondo_azienda_ombra)
            cap_fondo_azienda_ombra = 0.0

        if occupato:
            ral_curr = ral_manuale * f if ral_override else s["ral_base_eff"] * f
        else:
            ral_curr = 0.0

        base_contrib = s["base_contrib"]

        tfr_curr = ral_curr * s["tfr_pct"] if occupato else 0.0
        # Il minimo CCNL NON e' un flusso aggiuntivo: e' il requisito che dà
        # diritto al contributo aziendale. Il flusso del lavoratore e' SOLO
        # il versamento volontario impostato, che deve coprire il minimo.
        vol_min = base_contrib * s["lav_pct"]
        vf_curr = vol_extra_serie[a] if occupato else 0.0
        diritto_azienda = occupato and vf_curr >= vol_min - 1e-9
        ca_curr = base_contrib * s["ca_pct"] if diritto_azienda else 0.0
        vp_curr = vp_serie[a] if occupato else 0.0

        if occupato and (vf_curr + ca_curr) > 0:
            deducibile = min(vf_curr + ca_curr, LIMITE_DEDUCIBILITA)
            aliq_marg = aliquota_marginale(ral_curr)
            quota_lav = vf_curr / (vf_curr + ca_curr)
            risparmio_anno = deducibile * aliq_marg * quota_lav
        else:
            risparmio_anno = 0.0

        # Rendimenti fondo dalla quota reale: GIÀ netti di tassa annua e
        # costi di gestione (niente doppio conteggio); resta solo il costo
        # fisso amministrativo annuo in euro.
        rata_fondo = (vf_curr + tfr_curr + ca_curr) / 12.0
        rata_azienda = ca_curr / 12.0
        rata_pac = vp_curr / 12.0
        rata_tfr = tfr_curr / 12.0
        ter_p_m = 0.0 if ter_gia_incluso_pac else 1 - (1 - ter_p) ** (1 / 12)
        rt_m = (1 + rt) ** (1 / 12) - 1

        for mese in range(12):
            r_f = rend_fondo_mensili[a * 12 + mese]
            r_p = rend_pac_mensili[a * 12 + mese]

            cap_fondo += rata_fondo
            cap_fondo += cap_fondo * r_f

            cap_fondo_azienda_ombra += rata_azienda
            cap_fondo_azienda_ombra += cap_fondo_azienda_ombra * r_f

            versato_pac_cum += rata_pac
            cap_pac += rata_pac
            cap_pac *= (1 + r_p) * (1 - ter_p_m)

            cap_tfr += rata_tfr
            cap_tfr *= (1 + rt_m)

        cap_fondo = max(0.0, cap_fondo - s["costo_fisso_f"])

        # --- GARANZIA del comparto garantito: floor sul versato cumulato ---
        # Approssimazione: se il comparto dell'anno ha garanzia=true in
        # anagrafica (data/fondi/), il montante non scende sotto la somma
        # dei versamenti (valutato a fine anno, sull'intero capitale).
        versato_fondo_cum += vf_curr + tfr_curr + ca_curr
        _info_comp = (ANAGRAFICA_FONDI.get(s["comparto_key"][0], {})
                      .get("comparti", {}).get(s["comparto_key"][1], {}))
        if _info_comp.get("garanzia"):
            cap_fondo = max(cap_fondo, versato_fondo_cum)

        anni_adesione = anni_pregressi + anno
        aliq_uscita = aliquota_uscita_fondo(anni_adesione, ordinaria=uscita_ord)
        netto_fondo = cap_fondo * (1 - aliq_uscita)
        plusval_pac = max(0.0, cap_pac - versato_pac_cum)
        netto_pac = cap_pac - plusval_pac * tp_eff
        netto_tfr = cap_tfr * (1 - tt)

        rows.append({
            "Anno": anno,
            "CCNL": s["ccnl"], "Livello": s["livello"], "Comparto": s["comparto"],
            "Scatti": s["scatti"],
            "Occupato": "Sì" if occupato else "No",
            "RAL (€)": ral_curr,
            "Minimo CCNL richiesto (€)": vol_min,
            "Diritto contrib. azienda": "Sì" if diritto_azienda else ("—" if not occupato else "NO"),
            "Vers. Volontario (€)": vf_curr,
            "TFR al Fondo (€)": tfr_curr,
            "Contrib. Aziendale (€)": ca_curr,
            "Risparmio IRPEF (€)": risparmio_anno,
            "PAC annuo (€)": vp_curr,
            "Aliq. uscita fondo (%)": aliq_uscita * 100,
            "Fondo Netto (€)": netto_fondo,
            "PAC + TFR Netto (€)": netto_pac + netto_tfr,
            "PAC Netto (€)": netto_pac,
            "Fondo + PAC Netto (€)": netto_fondo + netto_pac,
        })
    return pd.DataFrame(rows)


def calcola_bande(fattori, rend_fondo_mat, rend_pac_mat, sched, scal,
                  vol_extra_serie, vp_serie, n_band=200):
    """P10/P50/P90 anno-per-anno per ogni curva (carriera fissata alla mediana)."""
    curve = ["Fondo Netto (€)", "PAC Netto (€)", "PAC + TFR Netto (€)", "Fondo + PAC Netto (€)"]
    acc = {c: [] for c in curve}
    m = min(n_band, rend_fondo_mat.shape[0], rend_pac_mat.shape[0])
    for i in range(m):
        d = simula_capitale(fattori, rend_fondo_mat[i], rend_pac_mat[i], sched, scal,
                            vol_extra_serie, vp_serie)
        for c in curve:
            acc[c].append(d[c].tolist())
    bande = {}
    for c in curve:
        arr = np.array(acc[c])
        bande[c] = {
            "p10": np.percentile(arr, 10, axis=0),
            "p50": np.percentile(arr, 50, axis=0),
            "p90": np.percentile(arr, 90, axis=0),
        }
    return bande


# ---------------------------------------------------------------------------
# ESECUZIONE
# ---------------------------------------------------------------------------
scatti_valore_annuo = anni_anzianita_pregressi * scatto_valore_livello * mensilita
base_contrib_iniziale = minimo_annuo + scatti_valore_annuo
superminimo_annuo = superminimo_mensile * mensilita
ral_auto = base_contrib_iniziale + superminimo_annuo + premio_produzione_annuo
ral_override = ral_manuale > 0
ral = ral_manuale if ral_override else ral_auto

coeff_totale = COEFF_LAVORATORE[tipo_lavoratore]
scenari = genera_scenari(profilo_crescita, coeff_totale, crescita_base, n=1000)

# --- Schedule anno-per-anno ---
sched = costruisci_schedule(
    durata, ccnl_scelto, livello, comparto, eta, anni_anzianita_pregressi,
    superminimo_annuo, premio_produzione_annuo, crescita_base,
    passaggi_livello if usa_passaggi_livello else [],
    cambi_ccnl if usa_cambio_ccnl else [],
    anni_disoccupato,
)

# --- Glidepath di comparto (life-cycle): override dal- l'anno indicato ---
if usa_switch_comparto and cambi_comparto:
    for anno_sw, comp_sw in sorted(cambi_comparto):
        for s in sched[anno_sw - 1:]:
            f_k = s["comparto_key"][0]
            if comp_sw in STORICO_MENSILE.get(f_k, {}):
                s["comparto_key"] = (f_k, comp_sw)
                s["comparto"] = comp_sw

N_BAND = 2000

# --- Traiettorie di rendimento del FONDO (per comparto, poi spliced) ---
comparto_keys = sorted({s["comparto_key"] for s in sched})
avvisi_corti = []
mancanti = []
mat_per_comparto = {}
SOGLIA_MESI_CORTI = 60
info_shrinkage_fondo = []
for ki, key in enumerate(comparto_keys):
    fondo_k, comp_k = key
    if mensile_disponibile(fondo_k, comp_k):
        serie_full = STORICO_MENSILE[fondo_k][comp_k]
        anni_full = STORICO_MENSILE_ANNI[fondo_k][comp_k]
        serie = tuple(filtra_storico_da_anno(serie_full, anni_full, anno_inizio_storico))
        if len(serie) < SOGLIA_MESI_CORTI:
            avvisi_corti.append(f"{fondo_k} · {comp_k} ({len(serie)} mesi)")
        # --- FACTOR MODELING: serie estesa dal benchmark, se consentita ----
        if usa_ricostruite:
            _cfg_c = ANAGRAFICA_FONDI.get(fondo_k, {}).get("comparti", {}).get(comp_k)
            if not _cfg_c or not _cfg_c.get("benchmark"):
                info_shrinkage_fondo.append(
                    f"{fondo_k} · {comp_k}: ricostruzione non disponibile "
                    f"(nessun benchmark in anagrafica — i garantiti sono "
                    f"esclusi by design)")
            elif not override_ricostruzione.get((fondo_k, comp_k), False):
                info_shrinkage_fondo.append(
                    f"{fondo_k} · {comp_k}: ricostruzione DISATTIVATA dal "
                    f"pulsante in sidebar")
            else:
                # Il pulsante in sidebar comanda: forza 'consentita' nel cfg
                _cfg_eff = dict(_cfg_c)
                _cfg_eff["ricostruzione"] = "consentita"
                _est = serie_estesa_comparto(
                    _PERCORSI_TROVATI.get(fondo_k, ""), comp_k,
                    json.dumps(_cfg_eff, sort_keys=True))
                if _est is not None:
                    serie = tuple(np.round(_est, 10))
                    info_shrinkage_fondo.append(
                        f"{fondo_k} · {comp_k}: serie RICOSTRUITA dal "
                        f"benchmark ({len(serie)} mesi; il taglio "
                        f"'escludi anni' non si applica)")
                else:
                    info_shrinkage_fondo.append(
                        f"{fondo_k} · {comp_k}: ricostruzione NON riuscita "
                        f"(serie proxy non scaricabili o overlap "
                        f"insufficiente — vedi tab Dati)")
        # --- SHRINKAGE: ricentra la serie verso l'ancora di lungo periodo ---
        if usa_shrinkage_fondo and serie:
            cagr_camp = cagr_da_mensili(serie)
            ancora_k = ancora_per_comparto(fondo_k, comp_k)
            cagr_corr, w_camp = shrink_verso_ancora(cagr_camp, len(serie), ancora_k,
                                                    mesi_pieni=int(mesi_piena_fiducia))
            serie = tuple(ricentra_mensili(serie, cagr_corr))
            info_shrinkage_fondo.append(
                f"{fondo_k} · {comp_k}: {len(serie)} mesi, CAGR storico "
                f"{cagr_camp*100:.2f}% → corretto {cagr_corr*100:.2f}% "
                f"(peso storico {w_camp*100:.0f}%, ancora {ancora_k*100:.1f}%)"
            )
        try:
            mat_per_comparto[key] = genera_rendimenti_block_bootstrap(
                serie, durata, block=int(block_mesi), n=N_BAND, seed=st.session_state.master_seed + ki)
        except ValueError as e:
            mancanti.append(f"{fondo_k} · {comp_k} (serie mensile troppo corta dopo il "
                            f"taglio all'anno {anno_inizio_storico}: {e})")
    else:
        # Nessun NAV nel CSV: il comparto e' simulabile SOLO via
        # ricostruzione dal benchmark (mix puro, alfa 0) se attivata.
        serie_mix = None
        if usa_ricostruite and override_ricostruzione.get((fondo_k, comp_k), False):
            _cfg_c = ANAGRAFICA_FONDI.get(fondo_k, {}).get("comparti", {}).get(comp_k)
            if _cfg_c and _cfg_c.get("benchmark"):
                _cfg_eff = dict(_cfg_c)
                _cfg_eff["ricostruzione"] = "consentita"
                _est = serie_estesa_comparto(
                    _PERCORSI_TROVATI.get(fondo_k, ""), comp_k,
                    json.dumps(_cfg_eff, sort_keys=True))
                if _est is not None:
                    serie_mix = tuple(np.round(_est, 10))
        if serie_mix is None:
            mancanti.append(
                f"{fondo_k} · {comp_k} (nessun NAV nel CSV — se il comparto "
                f"ha un benchmark in anagrafica, attiva 'Usa serie "
                f"RICOSTRUITE' e il pulsante del comparto in sidebar per "
                f"simularlo dal mix benchmark)")
        else:
            info_shrinkage_fondo.append(
                f"{fondo_k} · {comp_k}: NESSUN NAV — serie interamente dal "
                f"MIX BENCHMARK ({len(serie_mix)} mesi, alfa 0: costi reali "
                f"non inclusi)")
            if usa_shrinkage_fondo:
                cagr_camp = cagr_da_mensili(serie_mix)
                ancora_k = ancora_per_comparto(fondo_k, comp_k)
                cagr_corr, w_camp = shrink_verso_ancora(
                    cagr_camp, len(serie_mix), ancora_k,
                    mesi_pieni=int(mesi_piena_fiducia))
                serie_mix = tuple(ricentra_mensili(serie_mix, cagr_corr))
                info_shrinkage_fondo.append(
                    f"{fondo_k} · {comp_k}: CAGR mix {cagr_camp*100:.2f}% → "
                    f"corretto {cagr_corr*100:.2f}% (peso storico "
                    f"{w_camp*100:.0f}%, ancora {ancora_k*100:.1f}%)")
            try:
                mat_per_comparto[key] = genera_rendimenti_block_bootstrap(
                    serie_mix, durata, block=int(block_mesi), n=N_BAND,
                    seed=st.session_state.master_seed + ki)
            except ValueError as e:
                mancanti.append(f"{fondo_k} · {comp_k} (mix benchmark troppo corto: {e})")

if mancanti:
    st.error(
        "**Dati storici mancanti o insufficienti** per: " + "; ".join(mancanti) + ".\n\n"
        "I rendimenti provengono solo dal resampling dello storico reale. Se "
        "hai tagliato l'orizzonte storico (sezione '4. Performance simulata'), "
        "prova ad abbassare l'anno di inizio, oppure abbassa la lunghezza del "
        "blocco, oppure scegli un CCNL/comparto già coperto (es. Cometa)."
    )
    st.stop()

rend_fondo_mat = np.empty((N_BAND, durata * 12))
for a, s in enumerate(sched):
    rend_fondo_mat[:, a * 12:(a + 1) * 12] = \
        mat_per_comparto[s["comparto_key"]][:, a * 12:(a + 1) * 12]

# --- PAC: GBM parametrico / portafoglio ticker (da pac_engine) ---
portafoglio_info = None
errore_portafoglio = None
info_drift_pac = None
if usa_portafoglio:
    if not tickers_input.strip():
        errore_portafoglio = "Nessun ticker selezionato: usa il catalogo o inseriscine uno."
        rend_pac_mat = genera_rendimenti_gbm(0.065, 0.15, durata, n=N_BAND,
                                             seed=st.session_state.master_seed + 100, nu=NU_T_PAC)
    else:
        try:
            tickers, pesi = parse_ticker_pesi(tickers_input, pesi_input)
            prezzi_df = scarica_prezzi_mensili(tuple(tickers), anni_storico)
            portafoglio_info = stima_parametri_portafoglio(prezzi_df, pesi)
            # --- Drift: manuale, shrinkage verso l'ancora, o solo storico ---
            cagr_storico = portafoglio_info["cagr_portafoglio"]
            n_mesi = portafoglio_info["n_mesi_storico"]
            if override_rend:
                cagr_target = rend_override_val
                info_drift_pac = (f"Drift manuale: CAGR {cagr_target*100:.2f}% "
                                  f"(storico {cagr_storico*100:.2f}% su {n_mesi} mesi)")
            elif usa_shrinkage_pac:
                cagr_target, w_camp = shrink_verso_ancora(cagr_storico, n_mesi, ancora_pac,
                                                          mesi_pieni=int(mesi_piena_fiducia))
                info_drift_pac = (f"Shrinkage: CAGR storico {cagr_storico*100:.2f}% "
                                  f"({n_mesi} mesi, peso {w_camp*100:.0f}%) + ancora "
                                  f"{ancora_pac*100:.1f}% → target {cagr_target*100:.2f}%")
            else:
                cagr_target = None
                info_drift_pac = (f"⚠️ Solo storico: CAGR campione {cagr_storico*100:.2f}% "
                                  f"su {n_mesi} mesi estrapolato per {durata} anni — "
                                  f"rischio di forte sovrastima.")
            if usa_bootstrap_pac:
                # Block-bootstrap sulla serie storica PESATA del portafoglio,
                # ricentrata sul CAGR target (stessa tecnica del fondo):
                # vol, correlazioni e sequenze restano quelle reali.
                serie_pac = np.asarray(portafoglio_info["rend_mensili_pesato"], dtype=float)
                if cagr_target is not None:
                    serie_pac = ricentra_mensili(serie_pac, cagr_target)
                rend_pac_mat = genera_rendimenti_block_bootstrap(
                    tuple(np.round(serie_pac, 10)), durata, block=int(block_mesi),
                    n=N_BAND, seed=st.session_state.master_seed + 200)
            else:
                rend_pac_mat = genera_rendimenti_portafoglio_gbm(
                    portafoglio_info["media_mensile"], portafoglio_info["cholesky_mensile"],
                    pesi, durata, cagr_target=cagr_target, n=N_BAND,
                    seed=st.session_state.master_seed + 200, nu=NU_T_PAC,
                )
        except Exception as e:
            errore_portafoglio = str(e)
            rend_pac_mat = genera_rendimenti_gbm(0.065, 0.15, durata, n=N_BAND,
                                                 seed=st.session_state.master_seed + 100, nu=NU_T_PAC)
else:
    rend_pac_mat = genera_rendimenti_gbm(rend_medio_pac, vol_pac, durata, n=N_BAND,
                                         seed=st.session_state.master_seed + 100, nu=NU_T_PAC)

# --- Traiettorie centrali (per la tabella e la linea centrale) ---
rend_fondo_sel = seleziona_traiettoria_per_percentile(rend_fondo_mat, percentile_perf)
rend_pac_sel = seleziona_traiettoria_per_percentile(rend_pac_mat, percentile_perf)

# --- Versamenti FISSI ma modulabili a gradini nel tempo ---
vol_extra_serie = costruisci_serie_a_gradini(durata, vers_vol_extra, variazioni_vol_extra)
vp_serie = costruisci_serie_a_gradini(durata, versamento_pac, variazioni_pac)

# --- Parametri scalari (non variano con lo schedule) ---
scal = dict(
    ral_override=ral_override, ral_manuale=ral_manuale,
    ter_p=ter_pac, tp=tassa_uscita_pac, quota_ts_pac=quota_ts_pac,
    ter_gia_incluso_pac=usa_portafoglio,
    rt=rend_tfr, tt=tassa_tfr,
    anni_pregressi=anni_gia_iscritto, uscita_ordinaria=uscita_ordinaria,
    cap_iniziale_fondo=capitale_iniziale_fondo,
    cap_iniziale_pac=capitale_iniziale_pac,
    mantieni_contributi_azienda=mantieni_contributi_azienda,
)

fattori_mediani = [float(np.percentile([s[a] for s in scenari], 50)) for a in range(durata)]
df_main = simula_capitale(fattori_mediani, rend_fondo_sel, rend_pac_sel, sched, scal,
                          vol_extra_serie, vp_serie)

bande = calcola_bande(fattori_mediani, rend_fondo_mat, rend_pac_mat, sched, scal,
                      vol_extra_serie, vp_serie, n_band=N_BAND)
anni = list(range(1, durata + 1))


# ---------------------------------------------------------------------------
# INTESTAZIONE
# ---------------------------------------------------------------------------
motore_txt = f"Block-bootstrap mensile (blocco {int(block_mesi)} mesi)"
if usa_shrinkage_fondo:
    motore_txt += " + shrinkage del drift"
st.info(
    f"**Profilo:** {tipo_lavoratore} · {ccnl_scelto} · livello {livello} · comparto {comparto}  \n"
    f"Coefficiente crescita ×{coeff_totale:.2f} · crescita di base "
    f"{crescita_base*100:.1f}%/anno · rendimenti fondo: **{motore_txt}** (storico reale)  \n"
    f"Linea centrale **P{percentile_perf}** · banda **P10–P90** su tutte le curve "
    f"({N_BAND} scenari)  \n"
    f"*Valori nominali (includono l'inflazione, coerentemente con contributi e montante).*"
)

if avvisi_corti:
    st.caption(
        "ℹ️ Storico REALE ma relativamente breve (meno di 5 anni di mesi): "
        + ", ".join(avvisi_corti)
        + ". La banda P10–P90 per questi comparti dipende da un numero limitato "
          "di osservazioni sottostanti: trattala con più cautela."
    )

if info_shrinkage_fondo:
    with st.expander("🎯 Correzione del drift applicata (fondo)"):
        for riga in info_shrinkage_fondo:
            st.caption("• " + riga)
        st.caption(
            "Il CAGR del campione viene spostato verso l'ancora di lungo "
            "periodo in proporzione alla brevità dello storico (peso storico "
            f"= mesi/{MESI_PIENA_FIDUCIA}). Volatilità, correlazioni e "
            "sequenze dei mesi reali restano invariate."
        )
if info_drift_pac:
    _motore_pac_txt = ("block-bootstrap storico (blocco "
                       f"{int(block_mesi)} mesi)") if usa_bootstrap_pac else "GBM-Cholesky"
    st.caption("🎯 **PAC (portafoglio):** " + info_drift_pac +
               f" · motore: {_motore_pac_txt}")

_anni_senza_azienda = df_main.loc[df_main["Diritto contrib. azienda"] == "NO", "Anno"].tolist()
if _anni_senza_azienda:
    st.warning(
        "⚠️ **Contributo aziendale PERSO negli anni: "
        + ", ".join(str(int(x)) for x in _anni_senza_azienda)
        + "** — il versamento volontario in quegli anni è sotto il minimo "
          "CCNL richiesto (vedi colonne 'Minimo CCNL richiesto' e 'Diritto "
          "contrib. azienda' in tabella). Alza il versamento almeno al "
          "minimo per non lasciare soldi dell'azienda sul tavolo."
    )

if usa_cambio_ccnl and cambi_ccnl:
    st.caption(
        "🔁 **Cambi CCNL pianificati:** partenza da **" + ccnl_scelto + "** → "
        + " → ".join([f"anno {a}: **{c}** ({l}, {comp})" for a, c, l, comp in cambi_ccnl])
    )
if usa_disoccupazione and anni_disoccupato:
    st.caption(
        "⏸️ **Anni di disoccupazione:** "
        + ", ".join(str(x) for x in sorted(anni_disoccupato))
        + " — nessun contributo, i capitali continuano a rendere."
    )

# --- Composizione della RAL iniziale ---
st.subheader("🧱 Composizione della RAL (Anno 1)")
if ral_override:
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("RAL inserita a mano", f"€ {ral:,.0f}")
    rc2.metric("Base contributiva fondo", f"€ {base_contrib_iniziale:,.0f}")
    rc3.metric("RAL auto (confronto)", f"€ {ral_auto:,.0f}")
else:
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Minimo tabellare", f"€ {minimo_annuo:,.0f}")
    rc2.metric("Scatti anzianità", f"€ {scatti_valore_annuo:,.0f}")
    rc3.metric("Superminimo + premio", f"€ {superminimo_annuo + premio_produzione_annuo:,.0f}")
    rc4.metric("RAL totale", f"€ {ral:,.0f}")

st.caption(
    f"Il contributo aziendale ({contrib_az_pct*100:.2f}%) e il tuo minimo "
    f"({preset['contrib_lav_pct']*100:.2f}%) si calcolano sulla base contributiva "
    f"(minimi + scatti), non su superminimo/premio. Il TFR "
    f"({preset['tfr_pct']*100:.2f}%) è sull'intera retribuzione."
)
st.divider()


# ---------------------------------------------------------------------------
# SEZIONE COSTI DEL FONDO
# ---------------------------------------------------------------------------
st.subheader(f"💰 Struttura dei Costi — {preset['fondo']} ({comparto})")
cc1, cc2 = st.columns(2)
cc1.metric("Costo iniziale (una tantum)", f"€ {preset['costo_iniziale']:,.2f}")
cc2.metric("Costo fisso annuo", f"€ {preset['costo_fisso']:,.0f}")

costo_fisso_totale = preset["costo_fisso"] * durata + preset["costo_iniziale"]

with st.expander("📖 Come leggere i costi del fondo"):
    st.markdown(f"""
Il fondo pensione ha **due costi separati**, entrambi inclusi nella simulazione:

1. **Costo iniziale** — €{preset['costo_iniziale']:.2f} una tantum all'iscrizione.
2. **Costo fisso annuo** — €{preset['costo_fisso']:.0f}/anno di spese amministrative
   sulla posizione individuale. Su {durata} anni: ~€{costo_fisso_totale:,.0f}.

**Costi di gestione finanziaria e tassazione annua NON compaiono qui separatamente:**
il rendimento storico usato per la simulazione è quello della **quota reale**
pubblicata dal fondo per il comparto *{comparto}*, che è **già al netto**
dell'imposta sostitutiva annua e dei costi di gestione finanziaria. Applicare
un'ulteriore deduzione qui sarebbe un doppio conteggio.
""")
st.divider()


# ---------------------------------------------------------------------------
# RENDIMENTO PER ANNO DEL COMPARTO SCELTO
# ---------------------------------------------------------------------------
st.subheader(f"📗 Rendimento per anno — {comparto} ({preset['fondo']})")
st.caption("Rendimento della quota reale del comparto: già al netto di tasse "
           "e costi di gestione finanziaria. A sinistra lo storico reale (con "
           "lo stesso taglio di orizzonte impostato in sidebar, se attivo), a "
           "destra la previsione dal resampling.")

fondo_sel = preset["fondo"]
serie_ann_full = STORICO_ANNUALE.get(fondo_sel, {}).get(comparto, [])
anni_ann_full = STORICO_ANNUALE_ANNI.get(fondo_sel, {}).get(comparto, [])
serie_ann_sel = filtra_storico_da_anno(serie_ann_full, anni_ann_full, anno_inizio_storico)
anni_lbl = [y for y in anni_ann_full if y >= anno_inizio_storico]

col_a, col_b = st.columns(2)

st.markdown("**Storico reale (anno per anno)**")
if serie_ann_sel:
    df_stor = pd.DataFrame({
        "Anno": anni_lbl,
        "Rendimento (%)": [r * 100 for r in serie_ann_sel],
    })
    st.dataframe(
        df_stor.style.format({"Rendimento (%)": "{:+.2f}"}),
        use_container_width=True, hide_index=True, height=300,
    )
    cagr_l = float(np.prod([1 + r for r in serie_ann_sel])) ** (1 / len(serie_ann_sel)) - 1
    s1, s2 = st.columns(2)
    s1.metric("CAGR", f"{cagr_l*100:.2f}%")
    s2.metric("Anno peggiore", f"{min(serie_ann_sel)*100:+.1f}%")
    if len(serie_ann_sel) < 8:
        st.caption(f"⚠️ Solo {len(serie_ann_sel)} anni disponibili: statistiche indicative.")
    if anno_inizio_storico > ANNO_STORICO_MIN_GLOBALE and len(serie_ann_sel) < len(serie_ann_full):
        st.caption(f"✂️ Storico tagliato: {len(serie_ann_full) - len(serie_ann_sel)} "
                   f"anni esclusi (prima del {anno_inizio_storico}).")
else:
    st.info("Storico annuale non disponibile per questo comparto (o azzerato dal taglio impostato).")

# --- Dispersione simulata: 3 numeri di sintesi ---
key_sel = (fondo_sel, comparto)
if key_sel in mat_per_comparto:
    mat_annua = mensili_ad_annui(mat_per_comparto[key_sel])
    st.caption(f"**Dispersione dei rendimenti simulati** ({motore_txt}) — "
               f"in un anno qualsiasi, su tutti gli scenari:")
    d1, d2, d3 = st.columns(3)
    d1.metric("Mediano (P50)", f"{np.median(mat_annua)*100:+.2f}%")
    d2.metric("Sfortunato (P10)", f"{np.percentile(mat_annua, 10)*100:+.2f}%")
    d3.metric("Fortunato (P90)", f"{np.percentile(mat_annua, 90)*100:+.2f}%")
    st.caption("Non è una previsione anno-per-anno: è la forbice di rischio "
               "che alimenta la banda P10–P90 del grafico montante.")

st.divider()


# ---------------------------------------------------------------------------
# RENDIMENTO NETTO PER ANNO DEL PAC
# ---------------------------------------------------------------------------
st.subheader("📘 Rendimento netto per anno — PAC")
st.caption(
    "Netto = dopo TER. A differenza del fondo, il PAC in Italia NON tassa il "
    "rendimento anno per anno: la plusvalenza si tassa solo in uscita/vendita. "
    "A sinistra lo storico reale (solo col portafoglio a ticker), a destra la "
    "previsione simulata."
)
if usa_tassazione_ts_pac and quota_ts_pac > 0:
    tp_eff_display = tassa_uscita_pac * (1 - quota_ts_pac) + 12.5 * quota_ts_pac
    st.info(
        f"🏛️ Tassazione differenziata attiva: **{quota_ts_pac*100:.0f}%** del PAC "
        f"considerato titoli di Stato/white list (12,5%), il resto a "
        f"{tassa_uscita_pac}%. **Aliquota effettiva sulla plusvalenza: "
        f"{tp_eff_display:.2f}%**. Stima semplificata, non un calcolo fiscale "
        f"certificato."
    )

col_c, col_d = st.columns(2)

with col_c:
    st.markdown("**Storico reale del portafoglio (anno per anno)**")
    if usa_portafoglio and portafoglio_info is not None and not errore_portafoglio:
        rmp = portafoglio_info["rend_mensili_pesato"]
        n_anni_storico_pac = len(rmp) // 12
        if n_anni_storico_pac >= 1:
            blocchi = rmp[:n_anni_storico_pac * 12].reshape(n_anni_storico_pac, 12)
            annuali_storici = np.prod(1 + blocchi, axis=1) - 1
            netti_storici = rendimento_netto_pac(annuali_storici, ter_pac)
            df_stor_pac = pd.DataFrame({
                "Periodo (blocco 12 mesi)": list(range(1, n_anni_storico_pac + 1)),
                "Lordo (%)": annuali_storici * 100,
                "Netto TER (%)": np.asarray(netti_storici) * 100,
            })
            st.dataframe(
                df_stor_pac.style.format({"Lordo (%)": "{:+.2f}", "Netto TER (%)": "{:+.2f}"}),
                use_container_width=True, hide_index=True, height=300,
            )
            cagr_l_pac = float(np.prod(1 + annuali_storici)) ** (1 / n_anni_storico_pac) - 1
            cagr_n_pac = float(np.prod(1 + netti_storici)) ** (1 / n_anni_storico_pac) - 1
            sp1, sp2, sp3 = st.columns(3)
            sp1.metric("CAGR lordo", f"{cagr_l_pac*100:.2f}%")
            sp2.metric("CAGR netto TER", f"{cagr_n_pac*100:.2f}%")
            sp3.metric("Peggiore (netto)", f"{min(netti_storici)*100:+.1f}%")
            st.caption("Blocchi di 12 mesi consecutivi dall'inizio dello storico "
                       "scaricato (non anni solari): il blocco 1 è il più vecchio.")
        else:
            st.info("Meno di 12 mesi di storico: non è possibile comporre un anno.")
    else:
        st.info(
            "Storico reale non disponibile in modalità PAC 'Semplice' (solo "
            "assunzione parametrica GBM) oppure portafoglio non caricato/errato."
        )

with col_d:
    st.markdown("**Previsione simulata**")
    _ter_tabella_pac = 0.0 if usa_portafoglio else ter_pac
    net_mat_pac = rendimento_netto_pac(mensili_ad_annui(rend_pac_mat), _ter_tabella_pac)
    df_prev_pac = pd.DataFrame({
        "Anno": list(range(1, durata + 1)),
        "P10 netto (%)": np.percentile(net_mat_pac, 10, axis=0) * 100,
        "P50 netto (%)": np.percentile(net_mat_pac, 50, axis=0) * 100,
        "P90 netto (%)": np.percentile(net_mat_pac, 90, axis=0) * 100,
    })
    st.dataframe(
        df_prev_pac.style.format({c: "{:+.2f}" for c in df_prev_pac.columns if c != "Anno"}),
        use_container_width=True, hide_index=True, height=300,
    )
    st.caption(f"Rendimento netto annuo mediano simulato: "
               f"**{np.median(net_mat_pac)*100:+.2f}%** (su tutti anni e scenari).")

st.divider()


# ---------------------------------------------------------------------------
# SEZIONE TASSAZIONE IN USCITA
# ---------------------------------------------------------------------------
st.subheader("🏛️ Tassazione in Uscita")
anni_finali = anni_gia_iscritto + durata
aliq_uscita_finale = aliquota_uscita_fondo(anni_finali, ordinaria=uscita_ordinaria)
aliq_agevolata = aliquota_uscita_fondo(anni_finali, ordinaria=False)

tc1, tc2, tc3 = st.columns(3)
tc1.metric("Anni di adesione a fine periodo", f"{anni_finali}")
tc2.metric("Aliquota uscita applicata", f"{aliq_uscita_finale*100:.1f}%")
irpef_equiv = aliquota_marginale(df_main["RAL (€)"].iloc[-1]) * 100
tc3.metric("IRPEF ordinaria (confronto)", f"{irpef_equiv:.0f}%")

if uscita_ordinaria:
    st.warning(
        f"Riscatto/anticipazione ordinaria: ritenuta **23%**. Con uscita agevolata "
        f"pagheresti **{aliq_agevolata*100:.1f}%** — differenza di circa "
        f"**€ {df_main['Fondo Netto (€)'].iloc[-1] * (0.23 - aliq_agevolata) / (1 - 0.23):,.0f}** "
        f"sul montante finale netto."
    )
st.divider()


# ---------------------------------------------------------------------------
# COSTO MENSILE NETTO
# ---------------------------------------------------------------------------
st.subheader("💳 Costo Mensile Effettivo (Anno 1)")
r0 = df_main.iloc[0]
vers_vol_anno1 = r0["Vers. Volontario (€)"]   # il minimo CCNL non è un flusso aggiuntivo
risparmio_anno1 = r0["Risparmio IRPEF (€)"]
ca_anno1 = r0["Contrib. Aziendale (€)"]
pac_anno1 = r0["PAC annuo (€)"]
costo_netto_fondo_anno1 = max(0.0, vers_vol_anno1 - risparmio_anno1)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Costo netto fondo/mese", f"€ {costo_netto_fondo_anno1/mensilita:,.0f}")
m2.metric("Costo PAC/mese", f"€ {pac_anno1/12:,.0f}")
m3.metric("Totale investito/mese", f"€ {(costo_netto_fondo_anno1/mensilita + pac_anno1/12):,.0f}")
m4.metric("Contributo azienda (gratis)/anno", f"€ {ca_anno1:,.0f}")
if (usa_variazioni_vol_extra and variazioni_vol_extra) or (usa_variazioni_pac and variazioni_pac):
    st.caption("ℹ️ Hai pianificato variazioni dei versamenti nel tempo: questi "
               "importi valgono solo per l'Anno 1, guarda la tabella anno per "
               "anno per gli anni successivi.")
st.divider()

# ---------------------------------------------------------------------------
# USCITA IN RENDITA (non solo capitale)
# ---------------------------------------------------------------------------
st.subheader("👴 Uscita in rendita — fondo vs prelievo dal PAC")
st.caption(
    "Per legge, dal fondo pensione almeno il 50% del montante va in RENDITA "
    "vitalizia (100% capitale solo se la rendita ottenibile dal 70% del "
    "montante è sotto ~metà dell'assegno sociale, o per vecchi iscritti). "
    "Qui la conversione usa un coefficiente semplificato: quello vero dipende "
    "da convenzione assicurativa del fondo, basi demografiche e tasso tecnico "
    "— controlla il documento sulle rendite del tuo fondo. Stima illustrativa."
)
r_c1, r_c2, r_c3 = st.columns(3)
eta_pens = r_c1.number_input(
    "Età al pensionamento", 55, 75, int(min(75, max(55, eta + durata))), 1,
    help="Default: età attuale + durata simulazione.")
_coeff_default = {55: 3.6, 60: 4.0, 62: 4.2, 65: 4.6, 67: 5.0, 70: 5.6, 75: 6.6}
_eta_vicina = min(_coeff_default, key=lambda k: abs(k - eta_pens))
coeff_rendita = r_c2.number_input(
    "Coefficiente di conversione (%/anno)", 2.0, 9.0,
    float(_coeff_default[_eta_vicina]), 0.1,
    help="Rendita annua lorda per 100€ di montante. Preimpostato da tabella "
         "indicativa (rendita vitalizia semplice, senza reversibilità); i "
         "coefficienti reali del fondo possono differire sensibilmente.")
quota_rendita = r_c3.slider("Quota del montante fondo in rendita (%)", 50, 100, 50, 5) / 100

_r_fin = df_main.iloc[-1]
_fondo_fin = float(_r_fin["Fondo Netto (€)"])
_pac_fin = float(_r_fin["PAC Netto (€)"])
rendita_annua = _fondo_fin * quota_rendita * coeff_rendita / 100.0
capitale_residuo = _fondo_fin * (1 - quota_rendita)
prelievo_pac_4pct = _pac_fin * 0.04

r_m1, r_m2, r_m3, r_m4 = st.columns(4)
r_m1.metric("Rendita fondo (mensile)", f"€ {rendita_annua/12:,.0f}",
            help=f"Vitalizia, dal {quota_rendita*100:.0f}% del montante netto "
                 f"di € {_fondo_fin:,.0f}. Non si estingue mai (rischio "
                 f"longevità in capo all'assicurazione).")
r_m2.metric("Capitale fondo residuo", f"€ {capitale_residuo:,.0f}")
r_m3.metric("Prelievo PAC 'regola 4%' (mensile)", f"€ {prelievo_pac_4pct/12:,.0f}",
            help=f"4%/anno dal PAC netto di € {_pac_fin:,.0f}: NON garantito a "
                 f"vita — la probabilità di successo la vedi nel tab PAC col "
                 f"decumulo Monte Carlo.")
r_m4.metric("Rendita + 4% PAC (mensile)", f"€ {(rendita_annua + prelievo_pac_4pct)/12:,.0f}")
st.caption(
    "Confronto chiave: la rendita è ASSICURATA a vita ma il capitale è ceduto "
    "(niente eredità sulla quota convertita); il prelievo dal PAC mantiene il "
    "capitale ereditabile ma può esaurirsi. Il tab PAC (decumulo) quantifica "
    "quel rischio."
)
st.divider()


# ---------------------------------------------------------------------------
# SEZIONE PORTAFOGLIO A TICKER (se attivo)
# ---------------------------------------------------------------------------
if usa_portafoglio:
    st.subheader("📈 Portafoglio PAC a Ticker")
    if not tickers_input.strip():
        st.warning("Nessun ticker selezionato. Uso GBM di fallback (7% / 15%).")
    elif errore_portafoglio:
        st.error(f"Impossibile scaricare/stimare il portafoglio: {errore_portafoglio}. "
                 f"Uso GBM di fallback (7% / 15%).")
    else:
        pi = portafoglio_info
        note_acc = []
        for t in pi["tickers"]:
            stato, nota = classifica_ticker(t)
            if stato != "ok":
                note_acc.append(f"{'⚠️' if stato=='warn' else '❓'} **{t}** — {nota}")
        if note_acc:
            st.warning("Verifica accumulazione/UCITS:\n\n" + "\n\n".join(note_acc))

        SOGLIA_VOL_ALTA = 0.25
        vol_alte = [
            f"**{t}** ({v*100:.0f}%/anno)"
            for t, v in zip(pi["tickers"], pi["vol_annua_asset"])
            if v > SOGLIA_VOL_ALTA
        ]
        if vol_alte:
            st.warning(
                "🎢 **Alta volatilità rilevata** per: " + ", ".join(vol_alte) +
                f" (soglia {SOGLIA_VOL_ALTA*100:.0f}%). Tipico di azioni singole "
                "o ETF settoriali/leva: valuta con prudenza la banda P10–P90."
            )

        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("CAGR storico (composto)", f"{pi['cagr_portafoglio']*100:.2f}%",
                   help=f"Media aritmetica annualizzata: {pi['rend_portafoglio']*100:.2f}% "
                        "(più alta del CAGR per il volatility drag ~σ²/2). "
                        "Il drift della simulazione parte dal CAGR.")
        pc2.metric("Volatilità storica annua", f"{pi['vol_portafoglio']*100:.2f}%")
        pc3.metric("Asset nel portafoglio", f"{len(pi['tickers'])}")

        if override_rend:
            st.info(f"Drift corretto a mano: CAGR mediano {rend_override_val*100:.1f}% "
                    f"(volatilità/correlazioni restano storiche).")

        nomi_leggibili = [TICKER_TO_NOME.get(t, t) for t in pi["tickers"]]
        df_asset = pd.DataFrame({
            "Nome": nomi_leggibili, "Ticker": pi["tickers"],
            "Peso (%)": (pesi * 100).round(1),
            "CAGR storico (%)": (pi["cagr_asset"] * 100).round(2),
            "Volatilità annua (%)": (pi["vol_annua_asset"] * 100).round(2),
        })
        st.dataframe(df_asset, use_container_width=True, hide_index=True)

        with st.expander("🔗 Matrice di correlazione (sui rendimenti mensili)"):
            df_corr = pd.DataFrame(pi["corr"], index=pi["tickers"], columns=pi["tickers"])
            st.dataframe(df_corr, use_container_width=True)

        st.caption("⚠️ Volatilità/correlazioni storiche sono stime ragionevoli; il "
                   "rendimento medio storico molto meno. Meglio correggerlo a mano.")
    st.divider()


# ---------------------------------------------------------------------------
# KPI + GRAFICO
# ---------------------------------------------------------------------------
st.subheader(f"📊 Andamento Capitale Netto — linea P{percentile_perf} · banda P10–P90")
st.caption("Linea = percentile scelto (carriera mediana). Banda = P10–P90 sulla "
           "variabilità dei RENDIMENTI ({} scenari).".format(N_BAND))

last = df_main.iloc[-1]
b_fondo = bande["Fondo Netto (€)"]
b_pac = bande["PAC Netto (€)"]
b_pactfr = bande["PAC + TFR Netto (€)"]
b_fpac = bande["Fondo + PAC Netto (€)"]

cols = st.columns(4 if usa_entrambi else 3)
cols[0].metric("Fondo Netto", f"€ {last['Fondo Netto (€)']:,.0f}",
               help=f"P10: € {b_fondo['p10'][-1]:,.0f} — P90: € {b_fondo['p90'][-1]:,.0f}")
cols[1].metric("PAC + TFR Netto", f"€ {last['PAC + TFR Netto (€)']:,.0f}",
               help=f"P10: € {b_pactfr['p10'][-1]:,.0f} — P90: € {b_pactfr['p90'][-1]:,.0f}")
cols[2].metric("RAL Finale", f"€ {last['RAL (€)']:,.0f}",
               help=f"× {last['RAL (€)']/ral:.2f} vs partenza" if ral else "")
if usa_entrambi:
    cols[3].metric("Fondo + PAC (senza TFR)", f"€ {last['Fondo + PAC Netto (€)']:,.0f}",
                   help=f"P10: € {b_fpac['p10'][-1]:,.0f} — P90: € {b_fpac['p90'][-1]:,.0f}")

fig = go.Figure()

def aggiungi_banda(b, colore_fill, nome):
    fig.add_trace(go.Scatter(
        x=anni + anni[::-1],
        y=list(b["p90"]) + list(b["p10"])[::-1],
        fill="toself", fillcolor=colore_fill,
        line=dict(color="rgba(0,0,0,0)"), name=nome, hoverinfo="skip",
        showlegend=True,
    ))

aggiungi_banda(b_fondo,  "rgba(42,120,214,0.12)", "Fondo P10–P90")
aggiungi_banda(b_pactfr, "rgba(27,175,122,0.12)", "PAC+TFR P10–P90")
aggiungi_banda(b_pac,    "rgba(155,89,182,0.10)", "Solo PAC P10–P90")
if usa_entrambi:
    aggiungi_banda(b_fpac, "rgba(237,161,0,0.10)", "Fondo+PAC P10–P90")

fig.add_trace(go.Scatter(x=anni, y=df_main["Fondo Netto (€)"], name="Fondo Pensione",
                         line=dict(color="#2a78d6", width=3)))
fig.add_trace(go.Scatter(x=anni, y=df_main["PAC + TFR Netto (€)"], name="PAC + TFR",
                         line=dict(color="#1baf7a", width=3)))
fig.add_trace(go.Scatter(x=anni, y=df_main["PAC Netto (€)"], name="Solo PAC",
                         line=dict(color="#9b59b6", width=2, dash="dash")))
if usa_entrambi:
    fig.add_trace(go.Scatter(x=anni, y=df_main["Fondo + PAC Netto (€)"],
                             name="Fondo + PAC (senza TFR)",
                             line=dict(color="#eda100", width=3, dash="dot")))

fig.update_layout(xaxis_title="Anno", yaxis_title="Capitale Netto (€)",
                  yaxis_tickformat="€,.0f", hovermode="x unified",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                  height=460)
st.plotly_chart(fig, use_container_width=True)


st.divider()
st.subheader(f"🔎 Rendimento annuo del fondo nello scenario scelto — P{percentile_perf}")
st.caption(
    "Rendimento del fondo anno per anno lungo la traiettoria P"
    f"{percentile_perf} (la linea centrale del grafico). Già netto "
    "dell'imposta sostitutiva annua; NON ancora applicata la tassa di USCITA."
)

_rend_annui_fondo = np.prod(1 + rend_fondo_sel.reshape(durata, 12), axis=1) - 1
_indice_cum = 100 * np.cumprod(1 + _rend_annui_fondo)

df_rend_perc = pd.DataFrame({
    "Anno": list(range(1, durata + 1)),
    "Rendimento annuo (%)": _rend_annui_fondo * 100,
    "Indice (base 100)": _indice_cum,
})
st.dataframe(
    df_rend_perc.style.format({
        "Rendimento annuo (%)": "{:+.2f}",
        "Indice (base 100)": "{:,.1f}",
    }),
    use_container_width=True, hide_index=True, height=380,
)

_cagr_perc = (_indice_cum[-1] / 100) ** (1 / durata) - 1
_pc1, _pc2, _pc3 = st.columns(3)
_pc1.metric("CAGR di questo scenario", f"{_cagr_perc*100:.2f}%")
_pc2.metric("Anno migliore", f"{_rend_annui_fondo.max()*100:+.1f}%")
_pc3.metric("Anno peggiore", f"{_rend_annui_fondo.min()*100:+.1f}%")
st.caption(
    "💡 Anche uno scenario complessivamente fortunato (P90) contiene anni "
    "negativi, e uno sfortunato (P10) contiene anni positivi: il percentile "
    "descrive il risultato *cumulato*, non ogni singolo anno."
)


# ---------------------------------------------------------------------------
# TABELLA ANNO PER ANNO
# ---------------------------------------------------------------------------
st.subheader("📋 Dettaglio Anno per Anno")
st.caption("Montanti = linea centrale P{}. Contributi e RAL crescono con carriera/inflazione.".format(percentile_perf))

cols_show = ["Anno", "CCNL", "Livello", "Comparto", "Scatti", "Occupato", "RAL (€)",
             "Minimo CCNL richiesto (€)", "Diritto contrib. azienda",
             "Vers. Volontario (€)", "TFR al Fondo (€)",
             "Contrib. Aziendale (€)", "Risparmio IRPEF (€)", "PAC annuo (€)",
             "Aliq. uscita fondo (%)", "Fondo Netto (€)", "PAC + TFR Netto (€)", "PAC Netto (€)"]
if usa_entrambi:
    cols_show.append("Fondo + PAC Netto (€)")

fmt = {c: "€ {:,.0f}" for c in cols_show
       if c not in ("Anno", "CCNL", "Livello", "Comparto", "Scatti", "Occupato",
                    "Diritto contrib. azienda", "Aliq. uscita fondo (%)")}
fmt["Aliq. uscita fondo (%)"] = "{:.1f}%"
st.dataframe(df_main[cols_show].style.format(fmt), use_container_width=True, height=420)

st.caption(
    "⚠️ Stima illustrativa. Crescita salariale su dati ISTAT; contributi CCNL "
    "Cometa/Fon.Te; rendimenti del fondo dal ricampionamento dello storico "
    "reale della quota; PAC simulato con GBM o block-bootstrap della serie "
    "pesata del portafoglio ticker (drift corretto via shrinkage). "
    "Non è consulenza finanziaria o previdenziale."
)
st.divider()

_ctx_backtest = {
    "durata": durata,
    "storico_mensile": STORICO_MENSILE,
    "fondo": preset["fondo"],
    "comparto": comparto,
    "usa_portafoglio": usa_portafoglio,
    "portafoglio_info": portafoglio_info,
    "portafoglio_errore": errore_portafoglio,
    "ticker_to_nome": TICKER_TO_NOME,
    "simula_capitale": simula_capitale,
    "fattori_mediani": fattori_mediani,
    "sched": sched,
    "scal": scal,
    "vol_extra_serie": vol_extra_serie,
    "vp_serie": vp_serie,
    "cap_iniziale_fondo": capitale_iniziale_fondo,
    "cap_iniziale_pac": capitale_iniziale_pac,
}

_ctx_backtest["anagrafica_fondi"] = ANAGRAFICA_FONDI
_ctx_backtest["percorsi_csv"] = _PERCORSI_TROVATI

_tab_pv, _tab_motore, _tab_pav, _tab_dati, _tab_mod = st.tabs(
    ["📈 Backtest (growth asset)", "🏦 Backtest via motore (TFR/IRPEF)",
     "🧪 PAC (ticker o manuale)", "📚 Dati storia lunga",
     "🔬 Modello & Validazione"]
)
with _tab_pv:
    render_backtest_tab(_ctx_backtest)
with _tab_motore:
    render_backtest_via_motore(_ctx_backtest)
with _tab_pav:
    render_pac_avanzato(_ctx_backtest)
with _tab_dati:
    render_tab_dati()
with _tab_mod:
    render_tab_modello(_ctx_backtest)
