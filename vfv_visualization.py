"""Plotly-Figuren und die Tabelle der Instanz. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt.
Referenzlinien auf logarithmischen Achsen werden als Spur gezeichnet (add_hline mit Beschriftung setzt die Beschriftung dort an die falsche Höhe)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import vfv_constants as C
from vfv_evaluation import Analysis, de, noise_label, pct, row_of

HOVER = "s %{x:.3f}<br>x %{y:.3f}<extra></extra>"


def _fixed(fig: go.Figure, height: int = 360, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_table(a: Analysis, rows: int = C.TABLE_ROWS) -> pd.DataFrame:
    k = min(rows, C.N)
    cols = {"Punkt": [f"{i + 1}" for i in range(k)], "Ort s": a.s[:k], "Daten b": a.b[:k], "Wahrheit": a.truth[:k]}
    for m in C.METHODS:
        cols[C.METHOD_SHORT[m]] = a.est[m][:k]
    cols["Band unten"], cols["Band oben"] = a.band_lo[:k], a.band_hi[:k]
    return pd.DataFrame(cols)


def style_table(df: pd.DataFrame):
    return df.style.format({c: (lambda v: de(v, 4)) for c in df.columns if c != "Punkt"})


def build_recon_chart(a: Analysis) -> go.Figure:
    """Wahrheit, gewähltes Verfahren und (wenn verschieden) der ehrliche Sieger; die grauen Punkte sind die Daten b."""
    st = a.settings
    lo, hi = C.SHOW_RANGE[st.family]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.s, y=a.b, mode="markers", name="Daten b", marker=dict(color=C.COLOR_DATA, size=5), hovertemplate="s %{x:.3f}<br>b %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.s, y=a.truth, mode="lines", name="Wahrheit", line=dict(color=C.COLOR_TRUTH, width=3), hoverinfo="skip"))
    if a.winner_honest != st.method:
        fig.add_trace(go.Scatter(x=a.s, y=a.est_honest[a.winner_honest], mode="lines", name=f"ehrlicher Sieger: {C.METHOD_SHORT[a.winner_honest]}", line=dict(color=C.COLOR_ORACLE, width=2, dash="dash"), hovertemplate=HOVER))
    fig.add_trace(go.Scatter(x=a.s, y=a.est[st.method], mode="lines", name=f"{C.METHOD_SHORT[st.method]} ({'ehrlich' if st.rule == 'honest' else 'Orakel'})", line=dict(color=C.METHOD_COLORS[st.method], width=2.5), hovertemplate=HOVER))
    fig.update_xaxes(title="Ort s")
    fig.update_yaxes(title="Wert", range=[lo, hi])
    return _fixed(fig, 400)


def build_methods_chart(a: Analysis) -> go.Figure:
    """Median-Fehler aller Verfahren über die Wahrheiten der Einstellung: ehrliche Parameterwahl und Orakel; Tikhonov und TSVD zusätzlich mit GCV (Raute)."""
    labels = [C.METHOD_SHORT[m] for m in C.METHODS]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[100 * r["honest"] for r in a.rows], name="ehrlich", marker_color=[C.METHOD_COLORS[r["method"]] for r in a.rows], text=[pct(r["honest"]) for r in a.rows], textposition="outside"))
    fig.add_trace(go.Bar(x=labels, y=[100 * r["oracle"] for r in a.rows], name="Orakel", marker_color="rgba(214,138,46,0.55)", marker_line=dict(color=C.COLOR_ORACLE, width=1.5)))
    gcv = [(C.METHOD_SHORT[r["method"]], 100 * r["gcv"]) for r in a.rows if r["gcv"] is not None]
    fig.add_trace(go.Scatter(x=[g[0] for g in gcv], y=[g[1] for g in gcv], mode="markers", name="GCV", marker=dict(symbol="diamond", size=11, color="#222222")))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Verfahren", type="category")
    fig.update_yaxes(title="Median-Fehler (%)", range=[0, 100])
    return _fixed(fig, 380)


def build_spectrum_chart(a: Analysis) -> go.Figure:
    """Koeffizienten |vᵢᵀ x| in den Richtungen der Messung (logarithmisch): Wahrheit und gewähltes Verfahren; die gestrichelte Kurve σ / sᵢ trennt hörbare von stummen Richtungen (darunter liegt das Signal unter dem Rauschen)."""
    idx = np.arange(1, C.N + 1)
    floor = 1e-8
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=a.sig / a.sv, mode="lines", name="Schwelle σ / sᵢ", line=dict(color=C.COLOR_DATA, width=2, dash="dash"), hovertemplate="Richtung %{x}<br>Schwelle %{y:.2e}<extra></extra>"))
    fig.add_trace(go.Scatter(x=idx, y=np.maximum(a.coef_truth, floor), mode="markers", name="Wahrheit", marker=dict(color=C.COLOR_TRUTH, size=7), hovertemplate="Richtung %{x}<br>%{y:.2e}<extra></extra>"))
    fig.add_trace(go.Scatter(x=idx, y=np.maximum(a.coef_est, floor), mode="markers", name=C.METHOD_SHORT[a.settings.method], marker=dict(color=C.METHOD_COLORS[a.settings.method], size=7, symbol="x"), hovertemplate="Richtung %{x}<br>%{y:.2e}<extra></extra>"))
    fig.update_xaxes(title="Richtung i (absteigende Singulärwerte)")
    fig.update_yaxes(title="Koeffizient", type="log", exponentformat="power", range=[-6, 3])
    return _fixed(fig, 360)


def build_segments_chart(a: Analysis) -> go.Figure:
    """Zahl der Abschnitte (über 50 % des Maximums) je Verfahren für die erste Wahrheit; die Linie ist die Zahl der Wahrheit."""
    labels = [C.METHOD_SHORT[m] for m in C.METHODS]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[a.seg_est[m] for m in C.METHODS], marker_color=[C.METHOD_COLORS[m] for m in C.METHODS], name="Schätzung"))
    fig.add_trace(go.Scatter(x=labels, y=[a.seg_truth] * len(labels), mode="lines", name="Wahrheit", line=dict(color=C.COLOR_TRUTH, width=3, dash="dash")))
    fig.update_xaxes(title="Verfahren (ehrlich oder Orakel, wie gewählt)", type="category")
    fig.update_yaxes(title="Abschnitte", rangemode="tozero", dtick=1)
    return _fixed(fig, 320)


def build_band_chart(a: Analysis) -> go.Figure:
    """90-%-Band des gewählten Bandtyps mit seiner Mitte, der Wahrheit und den Daten."""
    st = a.settings
    lo, hi = C.SHOW_RANGE[st.family]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.s, y=a.band_hi, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.band_lo, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=C.COLOR_BAND, name="Band 5–95 %", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.band_mid, mode="lines", name="Mitte", line=dict(color=C.COLOR_EST, width=2.5), hovertemplate=HOVER))
    fig.add_trace(go.Scatter(x=a.s, y=a.truth, mode="lines", name="Wahrheit", line=dict(color=C.COLOR_TRUTH, width=3), hoverinfo="skip"))
    fig.update_xaxes(title="Ort s")
    fig.update_yaxes(title="Wert", range=[lo, hi])
    return _fixed(fig, 380)


def build_evidence_chart(a: Analysis) -> go.Figure:
    """Anteil der Wahrheiten der Einstellung, bei denen GP, weißer Prior oder Mischung die größte Evidenz hat."""
    names = ["GP (exponentiell)", "weißer Prior", "Blockzahl-Mischung"]
    vals = [a.evid["gp"], a.evid["white"], a.evid["mixture"]]
    fig = go.Figure(go.Bar(x=names, y=vals, marker_color=[C.METHOD_COLORS["gp"], C.COLOR_DATA, C.METHOD_COLORS["mixture"]], text=[pct(v) for v in vals], textposition="outside"))
    fig.update_xaxes(title=f"Prior (Wahrheiten: {C.FAMILY_SHORT[a.settings.family]})", type="category")
    fig.update_yaxes(title="Anteil mit größter Evidenz", range=[0, 1.15])
    return _fixed(fig, 320, showlegend=False)


# ----------------------------------------------------------------------------- Experimente

def _cell(cells: list, noise: float, family: str) -> dict:
    return next(c for c in cells if c["noise"] == noise and c["family"] == family)


def build_matrix_chart(res: dict, noise: float) -> go.Figure:
    """Median-Fehler (ehrliche Parameterwahl) je Familie und Verfahren bei einer Rauschstufe als Wärmebild; der Sieger jeder Zeile ist unterstrichen."""
    z, text = [], []
    for fam in C.FAMILIES:
        cell = _cell(res["cells"], noise, fam)
        errs = [row_of(cell["rows"], m)["honest"] for m in C.METHODS]
        z.append([min(100 * e, 100.0) for e in errs])
        text.append([(f"<u>{pct(e)}</u>" if m == cell["winner"] else pct(e)) for m, e in zip(C.METHODS, errs)])
    fig = go.Figure(go.Heatmap(z=z, x=[C.METHOD_SHORT[m] for m in C.METHODS], y=[C.FAMILY_SHORT[f] for f in C.FAMILIES], text=text, texttemplate="%{text}", textfont=dict(size=13), colorscale="YlOrRd", zmin=0, zmax=100,
                               colorbar=dict(title="Fehler (%)"), hovertemplate="%{y}, %{x}: %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="Wahrheitsfamilie")
    fig.update_xaxes(title=f"Verfahren (Rauschen {noise_label(noise)})", type="category")
    return _fixed(fig, 380, showlegend=False)


def build_null_chart(res: dict) -> go.Figure:
    """Energieanteil in den stummen Richtungen (Rauschen 10 %) je Familie: die Wahrheit und die Verfahren (ehrlich); unten Kosinus zur Wahrheit steht in der Tabelle."""
    fams = [C.FAMILY_SHORT[r["family"]] for r in res["null"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=fams, y=[100 * r["truth"] for r in res["null"]], name="Wahrheit", marker_color=C.COLOR_TRUTH))
    for m in ("tikhonov", "cgls", "l1", "tv", "pnp", "gp", "mixture"):
        fig.add_trace(go.Bar(x=fams, y=[100 * r["methods"][m]["null"] for r in res["null"]], name=C.METHOD_SHORT[m], marker_color=C.METHOD_COLORS[m]))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Wahrheitsfamilie", type="category")
    fig.update_yaxes(title="Energie in stummen Richtungen (%)", range=[0, 60])
    return _fixed(fig, 400)


def build_feature_chart(res: dict, noise: float, blocks: int) -> go.Figure:
    """Anteil der Schätzer mit gleich vielen, mehr oder weniger Abschnitten als die Wahrheit (Blockzahl-Ziehungen mit `blocks` Blöcken)."""
    f = next(x for x in res["features"] if x["noise"] == noise and x["blocks"] == blocks)
    names = list(f["kinds"])
    label = {**{m: C.METHOD_SHORT[m] for m in C.METHODS}, "sample": "Mischung (Probe)", "map": "Mischung (MAP)"}
    fig = go.Figure()
    for key, name, color in (("less", "weniger (löscht)", "#8fb3d9"), ("same", "gleich viele", "#2F6B65"), ("more", "mehr (erfindet)", "#c0392b")):
        fig.add_trace(go.Bar(x=[label[n] for n in names], y=[f["kinds"][n][key] for n in names], name=name, marker_color=color))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title=f"Verfahren (Wahrheit: {blocks} Block/Blöcke, Rauschen {noise_label(noise)})", type="category")
    fig.update_yaxes(title="Anteil der Läufe", range=[0, 1.0])
    return _fixed(fig, 360)


def build_calibration_chart(res: dict) -> go.Figure:
    """Abdeckung des 90-%-Bands je Familie für GP, Mischung und konformal (gleiche Familie, auf Wellen kalibriert); die Linie ist das Soll."""
    fams = [C.FAMILY_SHORT[r["family"]] for r in res["rows"]]
    fig = go.Figure()
    for key, name, color in (("gp_fixed", "GP (τ = 0,5)", C.METHOD_COLORS["gp"]), ("gp", "GP (τ aus der Evidenz)", "#7fb3a8"), ("mixture", "Mischung", C.METHOD_COLORS["mixture"]),
                             ("conformal", "konformal (gleiche Familie)", "#4c78a8"), ("conformal_waves", "konformal (auf Wellen kalibriert)", "#D68A2E")):
        fig.add_trace(go.Bar(x=fams, y=[r[key]["cover"] for r in res["rows"]], name=name, marker_color=color))
    fig.add_hline(y=C.LEVEL, line=dict(color="#222222", width=1.5, dash="dot"), annotation_text="Soll 0,90", annotation_position="top left")
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Wahrheitsfamilie", type="category")
    fig.update_yaxes(title="Abdeckung der Wahrheit", range=[0, 1.1])
    return _fixed(fig, 400)


def build_evidence_experiment_chart(res: dict, noise: float) -> go.Figure:
    """Anteil der Läufe mit größter Evidenz je Familie (gestapelt: GP, weißer Prior, Mischung)."""
    rows = [r for r in res["rows"] if r["noise"] == noise]
    fams = [C.FAMILY_SHORT[r["family"]] for r in rows]
    fig = go.Figure()
    for key, name, color in (("gp", "GP", C.METHOD_COLORS["gp"]), ("white", "weißer Prior", C.COLOR_DATA), ("mixture", "Blockzahl-Mischung", C.METHOD_COLORS["mixture"])):
        fig.add_trace(go.Bar(x=fams, y=[r[key] for r in rows], name=name, marker_color=color))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title=f"Wahrheitsfamilie (Rauschen {noise_label(noise)})", type="category")
    fig.update_yaxes(title="Anteil mit größter Evidenz", range=[0, 1.0])
    return _fixed(fig, 340)
