"""Verfahrensvergleich: welches Verfahren, gemessen woran? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Sechzehntes und letztes Stück der Linie Inverse Probleme und Regularisierung. Auf der Welt aus Stück 1 bis 15 (lineare Entfaltung, Gauß-Unschärfe) werden acht Verfahren (Tikhonov, TSVD, CGLS, L1, TV,
Plug-and-Play, GP-Posterior-Mittel, Blockzahl-Mischung) auf sechs Wahrheitsfamilien verglichen: mit ehrlicher Parameterwahl gegen Orakel, nach Datenrest, nach Energie in den Richtungen, die die Daten
nicht sehen, nach Abschnittszahl, nach Abdeckung der Unsicherheitsbänder und nach der Evidenz der Priors.
Kern: vfv_model.py, Generator: inverse_generator.py (wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import vfv_constants as C
from vfv_evaluation import (
    Settings,
    analyse,
    calibration_experiment,
    de,
    evidence_experiment,
    kinds_experiment,
    matrix_experiment,
    noise_label,
    pct,
    row_of,
    verdict_calibration,
    verdict_evidence,
    verdict_instance,
    verdict_kinds,
    verdict_matrix,
)
from vfv_pdf_export import build_pdf
from vfv_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from vfv_visualization import (
    build_band_chart,
    build_calibration_chart,
    build_evidence_chart,
    build_evidence_experiment_chart,
    build_feature_chart,
    build_matrix_chart,
    build_methods_chart,
    build_null_chart,
    build_recon_chart,
    build_segments_chart,
    build_spectrum_chart,
    build_table,
    style_table,
)

st.set_page_config(page_title="Verfahrensvergleich – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Alle acht Verfahren rechnen auf den Wahrheiten der Einstellung (2 bis 5 Sekunden) …")
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Experiment rechnet über feste Zufallsströme (10 bis 30 Sekunden) …")
def _matrix():
    return matrix_experiment()


@st.cache_data(show_spinner="Experiment rechnet über feste Zufallsströme (10 bis 30 Sekunden) …")
def _kinds():
    return kinds_experiment()


@st.cache_data(show_spinner="Experiment rechnet über feste Zufallsströme (10 bis 30 Sekunden) …")
def _calibration():
    return calibration_experiment()


@st.cache_data(show_spinner="Experiment rechnet über feste Zufallsströme (10 bis 30 Sekunden) …")
def _evidence():
    return evidence_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("⚖️ Verfahrensvergleich: welches Verfahren, gemessen woran?")
st.markdown(
    """
**Es gibt kein bestes Verfahren, nur eines, dessen Annahme zur Welt passt, und ob sie passt, sagt weder der Datenrest noch die Abdeckung, sondern nur der Vergleich mit der Wahrheit, die man in der Praxis nicht hat.**
Dies ist das sechzehnte und letzte Stück der Linie **Inverse Probleme und Regularisierung**: dieselbe Welt (lineare Entfaltung, Gauß-Unschärfe), **acht Verfahren** aus den Stücken davor und **sechs
Wahrheitsfamilien** (Spitzen, Blöcke, Wellen, Phantom, GP-Ziehung, Blockzahl-Mischung). Gefragt wird, **wer wo gewinnt** und was die **ehrliche Parameterwahl** gegenüber dem Orakel kostet, **ob der Datenrest
die Verfahren unterscheidet**, **ob ein Verfahren löscht oder erfindet** (Energie in den Richtungen, die die Daten nicht sehen, und Abschnittszahl), **wann die Unsicherheitsbänder kalibriert sind** und **ob die
Evidenz die Welt erkennt**. Die Welt ist simuliert; die Wahrheit ist bekannt und dient nur als Maßstab.

Weiter unten: vier Experimente (**🔬**, je mit eigenem Rechenknopf), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("Wie der Vergleich gemessen wird", expanded=True):
    st.markdown(
        r"""
1. **Acht Verfahren.** Tikhonov, abgeschnittene SVD, CGLS mit Abbruch, L1 (Weichschwelle), Total Variation, Plug-and-Play mit Haar-Weichschwelle, das **GP-Posterior-Mittel** und das **Posterior-Mittel der Blockzahl-Mischung** (Stück 15, exakt).
2. **Parameterwahl.** *Ehrlich* heißt ohne Wahrheit: **Diskrepanzprinzip** (Rest höchstens $\sigma\sqrt n$, am stärksten regularisiert), beim GP die **Evidenz**, bei Tikhonov und TSVD zusätzlich **GCV**. Das **Orakel** wählt den
   Parameter mit der Wahrheit und ist die Messlatte, nicht ein Verfahren.
3. **Datenrest.** $\|A\hat x - b\| / (\sigma\sqrt n)$: etwa 1, wenn das Modell die Daten erklärt.
4. **Stumme Richtungen.** Richtung $i$ der SVD ist stumm, wenn das Signal dort unter dem Rauschen liegt, $s_i\,|v_i^\top x| < \sigma$. Energie des Schätzers dort kommt nicht aus den Daten, sondern aus dem Prior.
5. **Abschnittszahl.** Zusammenhängende Stücke über 50 % des Maximums: **weniger** als die Wahrheit heißt gelöscht, **mehr** heißt erfunden.
6. **Bänder.** 90-%-Band aus GP-Posterior, Mischungs-Posterior oder **konformal** (Quantil der Fehler auf 60 Ziehungen derselben Familie); gemessen wird die Abdeckung der Wahrheit, mit der Breite.
7. **Evidenz.** $p(b)$ unter GP, weißem Prior und Mischung: das einzige Maß, das ohne Wahrheit auskommt.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Welt**")
    family = st.selectbox("Wahrheitsfamilie", C.FAMILIES, key="family_select", format_func=lambda v: C.FAMILY_LABELS[v], help="Aus welcher Verteilung die Wahrheiten gezogen werden (feste Ströme).")
    noise = st.select_slider("Rauschen (relativ)", options=C.NOISE_CHOICES, key="noise_select", format_func=noise_label, help="Standardabweichung des weißen Rauschens relativ zum Effektivwert der exakten Daten.")
    count = st.select_slider("Zahl der Wahrheiten", options=C.COUNT_CHOICES, key="count_select", help="Über so viele Wahrheiten der Familie werden Fehler und Kennzahlen gemittelt (Median); gezeigt wird die erste.")
    st.markdown("**Verfahren**")
    method = st.selectbox("Verfahren für Bild und Band", C.METHODS, key="method_select", format_func=lambda v: C.METHOD_LABELS[v], help="Welches Verfahren in Rekonstruktion, Nullraum und Abschnitten hervorgehoben wird; die Tabelle zeigt alle.")
    rule = st.selectbox("Parameterregel", C.RULES, key="rule_select", format_func=lambda v: C.RULE_LABELS[v], help="Ehrlich: Diskrepanzprinzip, beim GP die Evidenz. Orakel: der Parameter mit dem kleinsten Fehler, geht nur mit bekannter Wahrheit.")
    band = st.selectbox("Unsicherheitsband", C.BANDS, key="band_select", format_func=lambda v: C.BAND_LABELS[v], help="Wie das 90-%-Band gebildet wird; konformal wird auf 60 Ziehungen derselben Familie kalibriert.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Legt Wahrheiten, Rauschen und Kalibrierungsziehungen fest.")
    st.button("🎲 Neu würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"family_select": family, "noise_select": float(noise), "method_select": method, "rule_select": rule, "band_select": band, "count_select": int(count), "seed_input": int(seed)})

settings = Settings(family, float(noise), method, rule, band, int(count), int(seed))

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
a = _analysis(settings)
win = row_of(a.rows, a.winner_honest)
sel = row_of(a.rows, settings.method)
errs = [r["honest"] for r in a.rows]
res = [r["resid"] for r in a.rows]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ehrlicher Sieger", C.METHOD_SHORT[a.winner_honest], delta=f"Fehler {pct(win['honest'])}", delta_color="off", help="Verfahren mit dem kleinsten Median-Fehler bei ehrlicher Parameterwahl, ohne Wahrheit gewählt.")
c2.metric(C.METHOD_SHORT[settings.method], pct(sel["honest"] if settings.rule == "honest" else sel["oracle"]), delta=f"Orakel {pct(sel['oracle'])}", delta_color="off",
          help="Median-Fehler des gewählten Verfahrens nach der gewählten Parameterregel; darunter mit Orakel-Parameter.")
c3.metric("Größter Datenrest", f"{de(max(res), 1)} σ", delta=f"Fehler {de(100 * min(errs), 0)}–{de(100 * max(errs), 0)} %", delta_color="off",
          help="Größter Datenrest |A x̂ − b| / (σ√n) über alle acht Verfahren (ehrlich); darunter die Spanne ihrer Fehler: der Rest unterscheidet sie nicht.")
c4.metric("Abdeckung", pct(a.band_cover_batch), delta=f"Soll 90 %, Breite {de(a.band_width, 2)}", delta_color="off", help="Anteil der Koordinaten der Wahrheit im 90-%-Band, gemittelt über die Wahrheiten; Breite relativ zum Effektivwert der Wahrheit.")
st.caption(f"Wahrheitsfamilie: {C.FAMILY_LABELS[settings.family]}; {settings.count} Wahrheiten, Rauschen {noise_label(settings.noise)}; Sieger mit Orakel-Parameter: {C.METHOD_SHORT[a.winner_oracle]} ({pct(row_of(a.rows, a.winner_oracle)['oracle'])}). "
           f"Stumme Richtungen der ersten Wahrheit: {a.silent} von {C.N}.")

tab_rec, tab_meth, tab_spec, tab_seg, tab_band, tab_ev, tab_tab = st.tabs(["Rekonstruktion", "Verfahren", "Nullraum", "Abschnitte", "Band", "Evidenz", "Tabelle"])
with tab_rec:
    st.plotly_chart(build_recon_chart(a), width="stretch", key="recon_chart")
    st.caption("Die erste Wahrheit (grün), das gewählte Verfahren (farbig) und, wenn es nicht der Sieger ist, der ehrliche Sieger (orange gestrichelt); die grauen Punkte sind die unscharfen, verrauschten Daten.")
with tab_meth:
    st.plotly_chart(build_methods_chart(a), width="stretch", key="methods_chart")
    st.dataframe([{"Verfahren": C.METHOD_SHORT[r["method"]], "Fehler ehrlich": pct(r["honest"]), "Fehler Orakel": pct(r["oracle"]), "Fehler GCV": "–" if r["gcv"] is None else pct(r["gcv"]),
                   "Datenrest (σ)": de(r["resid"], 2), "Energie in stummen Richtungen": pct(r["null"]), "Kosinus zur Wahrheit dort": de(r["cos"], 2)} for r in a.rows], hide_index=True, width="stretch")
    st.caption("Median über die Wahrheiten der Einstellung. Ehrlich: ohne Wahrheit gewählt (Diskrepanzprinzip, beim GP Evidenz, bei der Mischung gibt es keinen freien Parameter); Orakel: der beste Parameter je Wahrheit; GCV nur bei Tikhonov und TSVD.")
with tab_spec:
    st.plotly_chart(build_spectrum_chart(a), width="stretch", key="spectrum_chart")
    st.caption("Koeffizienten des Signals in den Richtungen der Messung (logarithmisch). Unter der gestrichelten Schwelle σ / sᵢ liegt das Signal unter dem Rauschen: diese Richtungen sind stumm. Liegt der Schätzer (x) dort weit über der "
               "Wahrheit (grün), stammt das nicht aus den Daten, sondern aus dem Verfahren.")
with tab_seg:
    st.plotly_chart(build_segments_chart(a), width="stretch", key="segments_chart")
    st.caption("Zahl der zusammenhängenden Abschnitte über 50 % des Maximums, für die erste Wahrheit: weniger als die Wahrheit heißt gelöscht, mehr heißt erfunden. Eine Einzelprobe; Experiment 2 misst es über viele Wahrheiten.")
with tab_band:
    st.plotly_chart(build_band_chart(a), width="stretch", key="band_chart")
    st.caption(f"90-%-Band ({C.BAND_LABELS[settings.band]}) für die erste Wahrheit: deckt {pct(a.band_cover)} der Koordinaten ab; über alle {settings.count} Wahrheiten {pct(a.band_cover_batch)} (Soll {pct(C.LEVEL)}).")
with tab_ev:
    st.plotly_chart(build_evidence_chart(a), width="stretch", key="evidence_chart")
    st.caption("Wie oft welcher Prior die größte Evidenz p(b) hat, ohne dass die Wahrheit benutzt wird. Zur Auswahl stehen ein GP mit exponentieller Korrelation, ein weißer Prior und die Blockzahl-Mischung.")
with tab_tab:
    st.dataframe(style_table(build_table(a)), hide_index=True, width="stretch", height=560)
    st.caption("Die ersten Punkte der ersten Wahrheit: Daten, Wahrheit, alle acht Verfahren (nach der gewählten Regel) und das Band.")

if st.button("📄 PDF der aktuellen Instanz erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="verfahrensvergleich.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Passt das Verfahren zur Welt?")
verdict(verdict_instance(a))
st.caption("Das Urteil vergleicht das gewählte Verfahren mit dem ehrlichen Sieger (höchstens 1,1-faches: gehört zu den besten; ab 2-fachem: die Annahme passt nicht), nennt die Spanne von Datenrest und Fehler über alle Verfahren, "
           "die Energie in den stummen Richtungen, ob das Verfahren Abschnitte löscht oder erfindet, ob das Band kalibriert ist (Abdeckung 85 bis 95 %) und welcher Prior die größte Evidenz hat.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------


def experiment(key, run, label):
    """Rechenknopf je Experiment: das Ergebnis liegt im Zwischenspeicher und bleibt in der Sitzung."""
    if st.button(label, key=f"{key}_run"):
        st.session_state[key] = run()
    return st.session_state.get(key)


st.subheader("🔬 Experiment 1: Wer gewinnt wo?")
res1 = experiment("exp1", _matrix, "🧪 Experiment 1 rechnen (rund 15 Sekunden)")
st.caption(f"Sechs Familien × acht Verfahren × zwei Rauschstufen, je {C.EXP_COUNT} Wahrheiten, ehrliche Parameterwahl; der Sieger jeder Zeile ist unterstrichen. Die Tabelle zeigt auch das Orakel und GCV.")
if res1 is None:
    st.info("Noch nicht gerechnet: der Knopf rechnet zwölf Zellen mit je 40 Wahrheiten.")
else:
    for nz in C.NOISE_CHOICES:
        st.plotly_chart(build_matrix_chart(res1, nz), width="stretch", key=f"matrix_chart_{nz}")
    verdict(verdict_matrix(res1))
    st.dataframe([{"Rauschen": noise_label(c["noise"]), "Familie": C.FAMILY_SHORT[c["family"]], "Sieger ehrlich": f"{C.METHOD_SHORT[c['winner']]} ({pct(row_of(c['rows'], c['winner'])['honest'])})",
                   "Sieger Orakel": f"{C.METHOD_SHORT[c['winner_oracle']]} ({pct(row_of(c['rows'], c['winner_oracle'])['oracle'])})",
                   **{f"{C.METHOD_SHORT[m]} ehrlich / Orakel": f"{pct(row_of(c['rows'], m)['honest'])} / {pct(row_of(c['rows'], m)['oracle'])}" for m in C.METHODS}} for c in res1["cells"]],
                 hide_index=True, width="stretch")
    st.caption("Jede Familie hat ihren eigenen Sieger, und der Gewinn der passenden Annahme (L1 bei Spitzen, TV bei Blöcken, die Mischung in ihrer Welt) ist größer als jede Parameterwahl.")

st.subheader("🔬 Experiment 2: Löschen oder erfinden?")
res2 = experiment("exp2", _kinds, "🧪 Experiment 2 rechnen (rund 15 Sekunden)")
st.caption("Teil 1: Energie in den stummen Richtungen (Rauschen 10 %), je Familie. Teil 2: Blockzahl-Ziehungen mit einem oder drei Blöcken: zeigt das Verfahren so viele Abschnitte wie die Wahrheit, mehr (erfindet) oder weniger (löscht)?")
if res2 is None:
    st.info("Noch nicht gerechnet: der Knopf rechnet sechs Familien und vier Klassen von Blockzahl-Ziehungen.")
else:
    st.plotly_chart(build_null_chart(res2), width="stretch", key="null_chart")
    verdict(verdict_kinds(res2))
    for nz in C.FEATURE_NOISES:
        st.plotly_chart(build_feature_chart(res2, nz, 3), width="stretch", key=f"feature3_chart_{nz}")
        st.plotly_chart(build_feature_chart(res2, nz, 1), width="stretch", key=f"feature1_chart_{nz}")
    st.dataframe([{"Familie": C.FAMILY_SHORT[r["family"]], "stumme Richtungen (Median)": de(r["silent"], 0), "Wahrheit": pct(r["truth"]),
                   **{f"{C.METHOD_SHORT[m]} (Energie / Kosinus)": f"{pct(v['null'])} / {de(v['cos'], 2)}" for m, v in r["methods"].items()}} for r in res2["null"]], hide_index=True, width="stretch")
    st.caption("Glatte Verfahren löschen (ein Block bleibt, drei Blöcke werden zu einem), L1 erfindet Spitzen und legt viel Energie in Richtungen ab, die die Daten nicht sehen. Unschärfe ist ehrliches Nichtwissen, Schärfe kann Erfindung sein; "
               "die Mischung trifft nur, wenn die Welt ihre ist, und eine einzelne Probe aus ihrem Posterior zeigt bei mehrdeutigen Daten auch falsche Blockzahlen.")

st.subheader("🔬 Experiment 3: Wann ist ein Band kalibriert?")
res3 = experiment("exp3", _calibration, "🧪 Experiment 3 rechnen (rund 10 Sekunden)")
st.caption(f"Rauschen 10 %, je Familie {C.EXP_COUNT} Wahrheiten: Anteil der Koordinaten der Wahrheit im 90-%-Band für das GP-Posterior (τ fest und aus der Evidenz), den Mischungs-Posterior und konformale Bänder "
           f"(auf {C.EXP_CAL} Ziehungen derselben Familie und auf Wellen kalibriert).")
if res3 is None:
    st.info("Noch nicht gerechnet: der Knopf rechnet sechs Familien mit fünf Bändern.")
else:
    st.plotly_chart(build_calibration_chart(res3), width="stretch", key="calibration_chart")
    verdict(verdict_calibration(res3))
    st.dataframe([{"Familie": C.FAMILY_SHORT[r["family"]], **{f"{lab} (Abdeckung / Breite)": f"{pct(r[key]['cover'])} / {de(r[key]['width'], 2)}" for key, lab in
                   (("gp_fixed", "GP τ = 0,5"), ("gp", "GP Evidenz"), ("mixture", "Mischung"), ("conformal", "konformal gleiche Familie"), ("conformal_waves", "konformal auf Wellen"))}} for r in res3["rows"]],
                 hide_index=True, width="stretch")
    st.caption("Ein Band ist kalibriert, wenn sein Prior zur Welt passt oder wenn auf der Welt kalibriert wurde, in der es gilt. Die Abdeckung allein sagt wenig: das GP-Band deckt Wellen zu 98 % ab, weil es dort zu breit ist, "
               "und das konformale Band verliert bei Familienwechsel.")

st.subheader("🔬 Experiment 4: Erkennt die Evidenz die Welt?")
res4 = experiment("exp4", _evidence, "🧪 Experiment 4 rechnen (rund 10 Sekunden)")
st.caption(f"Je Rauschstufe und Familie {C.EXP_COUNT} Wahrheiten: wie oft GP, weißer Prior oder Blockzahl-Mischung die größte Evidenz p(b) hat. Die Wahrheit wird nicht benutzt.")
if res4 is None:
    st.info("Noch nicht gerechnet: der Knopf rechnet zwölf Zellen mit drei Priors.")
else:
    for nz in C.NOISE_CHOICES:
        st.plotly_chart(build_evidence_experiment_chart(res4, nz), width="stretch", key=f"evidence_exp_chart_{nz}")
    verdict(verdict_evidence(res4))
    st.dataframe([{"Rauschen": noise_label(r["noise"]), "Familie": C.FAMILY_SHORT[r["family"]], "GP": pct(r["gp"]), "weißer Prior": pct(r["white"]), "Mischung": pct(r["mixture"]),
                   "log-Evidenz Mischung − GP": de(r["diff"], 1)} for r in res4["rows"]], hide_index=True, width="stretch")
    st.caption("Die Evidenz wählt unter den angebotenen Priors fast immer den richtigen. Sie sagt aber nichts über Welten, für die kein Kandidat passt (Blöcke, Wellen, Phantom), und nichts über den Fehler eines Verfahrens, das mit dem Prior nichts zu tun hat (TV).")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Wahrheit ist bekannt** | In der Praxis nicht: Orakel, Abdeckung, Abschnittszahl und Energie in stummen Richtungen brauchen sie. Ohne Wahrheit bleiben Datenrest (unterscheidet nichts) und Evidenz (nur unter den Kandidaten). | – |
| **Der Vergleich ist fair** | Die Einstellungen sind gewählt (FISTA mit 150 Schritten je λ, ADMM mit 400, Haar ohne Zyklenverschiebung, Diskrepanzprinzip mit τ = 1); andere Einstellungen ändern Zahlen, und 40 Wahrheiten geben Richtwerte. | – |
| **Die Welt ist eine der sechs** | n = 48, eine Dimension, Unschärfe 0,03, ein Operator; keine Bilder, kein trainiertes Netz, kein MCMC, keine Ensemble-Kalman-Inversion, kein Diffusionssampler mit TMM oder DPS. | – |
| **Halluzination ist messbar** | Hier nur als zwei Maße in dieser Welt (Energie in stummen Richtungen, Abschnittszahl); keine Aussage über Bilder oder Anwender. | – |
| **Konformal löst die Kalibrierung** | Nur wenn Kalibrierung und Anwendung austauschbar sind; bei Familienwechsel bricht die Abdeckung, mit 60 Ziehungen und Koordinatenquantilen sind die Bänder breit. | – |
| **Die Evidenz entscheidet** | Nur unter den angebotenen Priors; für Blöcke, Wellen und Phantom ist keiner der drei richtig, und das beste Verfahren (TV) ist keiner davon. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** $b = A x + e$, $e \sim \mathcal N(0, \sigma^2 I)$, $A$ die Gauß-Faltungsmatrix ($n = 48$, $w = 0{,}03$), $A = U \Sigma V^\top$, $\sigma$ je Wahrheit gleich der Rauschstufe mal dem Effektivwert von $A x$.

**Verfahren.** Tikhonov: $\hat x_\lambda = V\,\mathrm{diag}\bigl(s_i/(s_i^2+\lambda)\bigr) U^\top b$. TSVD: die ersten $k$ Komponenten. CGLS: die $k$-te Krylov-Iterierte. L1: $\min \tfrac12\|Ax-b\|^2 + \lambda\|x\|_1$ (FISTA).
TV: $\min \tfrac12\|Ax-b\|^2 + \lambda\|Dx\|_1$ (ADMM, $D$ Vorwärtsdifferenz). Plug-and-Play: ISTA mit der Weichschwelle der Haar-Koeffizienten, $x \leftarrow W^\top \operatorname{soft}\bigl(W(x - tA^\top(Ax - b)), t\lambda\bigr)$.
GP: $x \sim \mathcal N(0, \tau^2 C)$, $C_{ij} = e^{-|s_i - s_j|/\ell}$, Mittel $\tau^2 C A^\top(\tau^2 A C A^\top + \sigma^2 I)^{-1} b$. Mischung: $\sum_k \pi_k \mathcal N(\mu_k, \Sigma_k)$ (Stück 15), Posterior-Mittel exakt.

**Parameterwahl.** Orakel: $\lambda^\star = \arg\min \|\hat x_\lambda - x\|$. Diskrepanzprinzip: der am stärksten regularisierte Kandidat mit $\|A\hat x - b\| \le \sigma\sqrt n$. GCV: $\min_\lambda n\|r_\lambda\|^2/(n - \operatorname{tr} F_\lambda)^2$.
Evidenz (ML-II): $\hat\tau = \arg\max_\tau \log \mathcal N(b; 0, \tau^2 A C A^\top + \sigma^2 I)$.

**Stumme Richtungen.** $x = \sum_i c_i v_i$ mit $c_i = v_i^\top x$; Richtung $i$ ist stumm, wenn $s_i |c_i| < \sigma$. Energieanteil $\sum_{i\ \mathrm{stumm}} \hat c_i^2 / \sum_i \hat c_i^2$ und Kosinus der stummen Anteile von Schätzung und Wahrheit.

**Konformal.** Fehler $|\hat x_j^{(l)} - x_j^{(l)}|$ auf $m$ Kalibrierungsziehungen; Halbbreite je Koordinate der $\lceil (m+1)(1-\alpha) \rceil$-kleinste Wert. Bei austauschbaren Ziehungen gilt $\Pr(x_j \in [\hat x_j \pm q_j]) \ge 1 - \alpha$ je Koordinate.

**Literatur.** Das Beispiel ist erfunden. Diskrepanzprinzip: V. A. Morozov (1966); GCV: G. H. Golub, M. Heath, G. Wahba (Technometrics 1979); Evidenz: D. J. C. MacKay (Neural Computation 1992); konforme Vorhersage: V. Vovk, A. Gammerman, G. Shafer (2005),
Überblick A. N. Angelopoulos und S. Bates (2021); Halluzination in Rekonstruktionen: S. Bhadra u. a. (IEEE TMI 2021) (alles aus dem Gedächtnis, nicht geprüft; die Zahlen hier sind gemessen, nicht der Literatur entnommen).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Inverse Probleme: vom Rauschen zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-inverse-probleme.html)."
)
