"""PDF-Export der aktuellen Instanz (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import vfv_constants as C
from vfv_evaluation import Analysis, de, noise_label, pct

_REPLACE = {"→": "->", "–": "-", "—": "-", "σ": "sigma", "λ": "lambda", "τ": "tau", "×": "x", "−": "-", "€": "EUR", "≥": ">=", "·": "*", "‖": "|", "ᵀ": "^T", "⁻": "-",
            "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
MAX_ROWS = 40


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Verfahrensvergleich"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Wahrheitsfamilie {C.FAMILY_SHORT[s.family]}, Rauschen {noise_label(s.noise)} des Effektivwerts, {s.count} Wahrheiten, gewaehlt {C.METHOD_SHORT[s.method]} "
                            f"({'ehrliche Parameterwahl' if s.rule == 'honest' else 'Orakel-Parameter'}), Band: {C.BAND_LABELS[s.band]}, Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    head, widths = ["Verfahren", "ehrlich", "Orakel", "GCV", "Datenrest", "Nullraum"], [48, 24, 24, 24, 26, 26]
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(head, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for r in a.rows:
        row = [C.METHOD_SHORT[r["method"]], pct(r["honest"]), pct(r["oracle"]), "-" if r["gcv"] is None else pct(r["gcv"]), f"{de(r['resid'], 2)} sigma", pct(r["null"])]
        for v, w in zip(row, widths):
            pdf.cell(w, 5.5, _s(v), border=1)
        pdf.ln()
    pdf.ln(2)
    lines = [
        f"Sieger (ehrlich) {C.METHOD_SHORT[a.winner_honest]}, mit Orakel {C.METHOD_SHORT[a.winner_oracle]}. Energie der Wahrheit in den stummen Richtungen {pct(a.null_truth)}.",
        f"Abdeckung des Bands: erste Wahrheit {pct(a.band_cover)}, im Mittel {pct(a.band_cover_batch)} (Soll {pct(C.LEVEL)}), relative Breite {de(a.band_width, 2)}.",
        f"Evidenz: GP {pct(a.evid['gp'])}, weisser Prior {pct(a.evid['white'])}, Mischung {pct(a.evid['mixture'])} der Laeufe.",
    ]
    for line in lines:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    head2 = ["Punkt", "Ort s", "Daten b", "Wahrheit"] + [C.METHOD_SHORT[m] for m in ("tikhonov", "l1", "tv")]
    widths2 = [12, 20, 24, 24, 24, 24, 24]
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(head2, widths2):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for i in range(min(C.N, MAX_ROWS)):
        row = [f"{i + 1}", de(a.s[i], 3), de(a.b[i], 3), de(a.truth[i], 3)] + [de(a.est[m][i], 3) for m in ("tikhonov", "l1", "tv")]
        for v, w in zip(row, widths2):
            pdf.cell(w, 5.5, _s(v), border=1)
        pdf.ln()
    if C.N > MAX_ROWS:
        pdf.ln(1)
        pdf.cell(0, 5, _s(f"... {C.N - MAX_ROWS} weitere Zeilen nicht gezeigt."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Verfahrensvergleich: acht Verfahren, Parameterwahl, Nullraum, Abdeckung und Evidenz. Quelle: verfahrensvergleich-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
