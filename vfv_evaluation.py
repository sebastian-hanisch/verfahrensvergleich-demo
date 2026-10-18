"""Auswertung: Formatierung, Vergleich aller Verfahren auf einer Einstellung, die vier Experimente über feste Ströme und die Verdikte. Nur numpy, kein Streamlit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import vfv_constants as C
import vfv_model as model
from inverse_generator import grid

_SUPER = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
PATH_METHODS = ("tikhonov", "tsvd", "cgls", "l1", "tv", "pnp")
HONEST_KEY = {"tikhonov": "discrepancy", "tsvd": "discrepancy", "cgls": "discrepancy", "l1": "discrepancy", "tv": "discrepancy", "pnp": "discrepancy", "gp": "evidence", "mixture": "prior"}
PRIOR_NAMES = {"gp": "das GP", "white": "den weißen Prior", "mixture": "die Blockzahl-Mischung"}
FITTING_PRIOR = {"spikes": "white", "blocks": "gp", "waves": "gp", "phantom": "gp", "gauss": "gp", "blockzahl": "mixture"}   # welcher der drei Priors die größte Evidenz hat (Messreihe)


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,5·10⁹."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def pct(v: float) -> str:
    """Anteil als Prozent: unter 99,5 Prozent mit einer Stelle, darüber ohne; ab dem Zehnfachen als Vielfaches."""
    if not math.isfinite(v):
        return "unendlich"
    if v >= 10.0:
        return f"{sci(v)}-fach"
    x = 100.0 * v
    return f"{de(x, 0 if x >= 99.5 else 1)} %"


def noise_label(v: float) -> str:
    """Rauschstufe als Prozent des Effektivwerts, z. B. 0,1 %."""
    return de(100.0 * v, 4).rstrip("0").rstrip(",") + " %" if v > 0.0 else "0 %"


def _by(rows: list, **kw) -> dict:
    return next(r for r in rows if all(r[k] == v for k, v in kw.items()))


# ----------------------------------------------------------------------------- Alle Verfahren auf einer Menge von Wahrheiten

@lru_cache(maxsize=2)
def gp_cmat(n: int = C.N) -> np.ndarray:
    """Korrelationsmatrix des GP-Priors (tau = 1, exponentiell, l = PRIOR_ELL)."""
    return model.exp_cov(n, 1.0, C.PRIOR_ELL)


def solve_all(a: np.ndarray, B: np.ndarray, sig: np.ndarray, X: np.ndarray) -> dict:
    """{Verfahren: {Regel: Schätzungen (m, n)}} für alle acht Verfahren: die Pfadverfahren mit `discrepancy`, `gcv` (Tikhonov, TSVD) und `oracle`; GP mit `evidence` (tau per ML-II), `fixed`
    (tau = GP_FIXED_TAU) und `oracle` (bester tau des Gitters); die Mischung mit `prior` (ohne freien Parameter)."""
    out = {}
    for name in PATH_METHODS:
        cands, res, gcv = model.PATHS[name](a, B, sig)
        out[name] = model.choose(cands, res, sig, X, gcv)
    cm = gp_cmat(a.shape[1])
    tau_eb, _ = model.gp_best_tau(a, B, sig, cm)
    cand = np.array([model.gp_mean(a, B, sig, t, cm) for t in model.gp_taus()])
    out["gp"] = {"evidence": model.gp_mean(a, B, sig, tau_eb, cm), "fixed": model.gp_mean(a, B, sig, C.GP_FIXED_TAU, cm),
                 "oracle": cand[model.rel_err(cand, X[None]).argmin(axis=0), np.arange(len(B))], "tau": tau_eb}
    mix = model.blockzahl_mixture(a.shape[1])
    out["mixture"] = {"prior": np.array([mix.posterior(a, B[i], sig[i]).mean() for i in range(len(B))])}
    return out


def honest(sol: dict, method: str) -> np.ndarray:
    return sol[method][HONEST_KEY[method]]


def oracle(sol: dict, method: str) -> np.ndarray:
    return sol[method]["oracle"] if "oracle" in sol[method] else sol[method]["prior"]


def method_rows(a: np.ndarray, B: np.ndarray, sig: np.ndarray, X: np.ndarray, sol: dict) -> list:
    """Je Verfahren: Median über die Wahrheiten von Fehler (ehrlich, Orakel, GCV), Datenrest (ehrlich), Energieanteil und Kosinus in den stummen Richtungen (ehrlich)."""
    mask = model.silent_mask(a, X, sig)
    rows = []
    for name in C.METHODS:
        h = honest(sol, name)
        rows.append({
            "method": name, "honest": float(np.median(model.rel_err(h, X))), "oracle": float(np.median(model.rel_err(oracle(sol, name), X))),
            "gcv": float(np.median(model.rel_err(sol[name]["gcv"], X))) if "gcv" in sol[name] else None,
            "resid": float(np.median(model.data_residual(a, h, B, sig))),
            "null": float(np.median(model.null_share(a, h, mask))), "cos": float(np.median(model.null_cosine(a, h, X, mask))),
        })
    return rows


# ----------------------------------------------------------------------------- Bänder

def calibration_halfwidth(a: np.ndarray, family: str, noise: float, seed: int, count: int = C.CONFORMAL_CAL) -> np.ndarray:
    """Halbbreite je Koordinate (konformal, Niveau LEVEL) aus den absoluten Fehlern von Tikhonov mit Diskrepanzprinzip auf `count` Ziehungen der Familie."""
    Xc = model.draw(family, count, seed)
    Bc, sc = model.make_data(Xc, noise, seed + 1, a)
    cands, res, _ = model.tikhonov_path(a, Bc, sc)
    est = model.choose(cands, res, sc)["discrepancy"]
    return model.conformal_halfwidth(np.abs(est - Xc))


def make_band(kind: str, a: np.ndarray, B: np.ndarray, sig: np.ndarray, tik: np.ndarray, half: np.ndarray, seed: int):
    """(Mitte, untere, obere Grenze) je Zeile (m, n) des 90-%-Bands: `gp` Posterior mit tau per Evidenz, `mixture` Perzentile von MIX_DRAWS exakten Posterior-Ziehungen, `conformal` Tikhonov +- half."""
    m = len(B)
    if kind == "conformal":
        return tik, tik - half[None, :], tik + half[None, :]
    mid, lo, hi = np.zeros_like(B), np.zeros_like(B), np.zeros_like(B)
    if kind == "gp":
        cm = gp_cmat(a.shape[1])
        tau, _ = model.gp_best_tau(a, B, sig, cm)
        for i in range(m):
            mean, cov = model.gp_posterior(a, B[i], sig[i], tau[i], cm)
            sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
            mid[i], lo[i], hi[i] = mean, mean - C.Z90 * sd, mean + C.Z90 * sd
        return mid, lo, hi
    mix = model.blockzahl_mixture(a.shape[1])
    for i in range(m):
        post = mix.posterior(a, B[i], sig[i])
        draws, _ = post.sample(C.MIX_DRAWS, seed + i)
        mid[i], lo[i], hi[i] = post.mean(), np.percentile(draws, 5, axis=0), np.percentile(draws, 95, axis=0)
    return mid, lo, hi


def band_stats(X: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """(mittlere Abdeckung, Median der mittleren Bandbreite relativ zum Effektivwert der Wahrheit) über die Zeilen."""
    cov = [model.coverage(X[i], lo[i], hi[i]) for i in range(len(X))]
    rms = np.sqrt(np.mean(X ** 2, axis=1))
    return float(np.mean(cov)), float(np.median((hi - lo).mean(axis=1) / rms))


def evidences(a: np.ndarray, B: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """(3, m) Log-Evidenz von GP (tau per ML-II), weißem Prior (tau per ML-II) und Blockzahl-Mischung."""
    _, eg = model.gp_best_tau(a, B, sig, gp_cmat(a.shape[1]))
    _, ew = model.gp_best_tau(a, B, sig, np.eye(a.shape[1]))
    mix = model.blockzahl_mixture(a.shape[1])
    return np.array([eg, ew, [mix.log_evidence(a, B[i], sig[i]) for i in range(len(B))]])


# ----------------------------------------------------------------------------- Einstellung und Analyse

@dataclass(frozen=True)
class Settings:
    family: str = "blocks"
    noise: float = C.DEFAULT_NOISE
    method: str = "tv"
    rule: str = "honest"
    band: str = "gp"
    count: int = C.DEFAULT_COUNT
    seed: int = C.DEFAULT_SEED


@dataclass
class Analysis:
    settings: Settings
    s: np.ndarray
    truth: np.ndarray
    b: np.ndarray
    sig: float
    est: dict                  # Schätzung je Verfahren für die erste Wahrheit nach der gewählten Regel
    est_honest: dict
    est_oracle: dict
    rows: list                 # Tabelle je Verfahren (Mediane über die Wahrheiten)
    winner_honest: str
    winner_oracle: str
    null_truth: float          # Median des Energieanteils der Wahrheit in den stummen Richtungen
    silent: int                # Zahl stummer Richtungen der ersten Wahrheit
    sv: np.ndarray             # Singulärwerte
    coef_truth: np.ndarray     # |v_i^T x| der ersten Wahrheit
    coef_est: np.ndarray       # |v_i^T x_hat| des gewählten Verfahrens
    seg_truth: int
    seg_est: dict
    band_mid: np.ndarray
    band_lo: np.ndarray
    band_hi: np.ndarray
    band_cover: float          # Abdeckung der ersten Wahrheit
    band_cover_batch: float    # mittlere Abdeckung über die Wahrheiten
    band_width: float
    evid: dict                 # Anteil der Läufe mit größter Evidenz je Prior und Mittel der Differenz Mischung - GP


def analyse(settings: Settings) -> Analysis:
    st = settings
    a = model.op()
    seed = st.seed
    X = model.draw(st.family, st.count, seed + C.SEED_TRUTH)
    B, sig = model.make_data(X, st.noise, seed + C.SEED_NOISE, a)
    sol = solve_all(a, B, sig, X)
    rows = method_rows(a, B, sig, X, sol)
    pick = honest if st.rule == "honest" else oracle
    est = {m: pick(sol, m)[0] for m in C.METHODS}
    mask = model.silent_mask(a, X, sig)
    _, sv, vt = np.linalg.svd(a)
    tik = honest(sol, "tikhonov")
    half = calibration_halfwidth(a, st.family, st.noise, seed + C.SEED_CAL)
    mid, lo, hi = make_band(st.band, a, B, sig, tik, half, seed + C.SEED_DRAW)
    cover_b, width = band_stats(X, lo, hi)
    ev = evidences(a, B, sig)
    best = ev.argmax(axis=0)
    return Analysis(
        settings=st, s=grid(C.N), truth=X[0], b=B[0], sig=float(sig[0]), est=est, est_honest={m: honest(sol, m)[0] for m in C.METHODS}, est_oracle={m: oracle(sol, m)[0] for m in C.METHODS},
        rows=rows, winner_honest=min(rows, key=lambda r: r["honest"])["method"], winner_oracle=min(rows, key=lambda r: r["oracle"])["method"],
        null_truth=float(np.median(model.null_share(a, X, mask))), silent=int(mask[0].sum()), sv=sv, coef_truth=np.abs(vt @ X[0]), coef_est=np.abs(vt @ est[st.method]),
        seg_truth=model.n_segments(X[0]), seg_est={m: model.n_segments(est[m]) for m in C.METHODS}, band_mid=mid[0], band_lo=lo[0], band_hi=hi[0],
        band_cover=model.coverage(X[0], lo[0], hi[0]), band_cover_batch=cover_b, band_width=width,
        evid={"gp": float(np.mean(best == 0)), "white": float(np.mean(best == 1)), "mixture": float(np.mean(best == 2)), "diff": float(np.mean(ev[2] - ev[0]))},
    )


def row_of(rows: list, method: str) -> dict:
    return _by(rows, method=method)


# ----------------------------------------------------------------------------- Experiment 1: Fehlermatrix

def matrix_experiment() -> dict:
    """Je Rauschstufe und Familie EXP_COUNT Wahrheiten: Median des Fehlers aller Verfahren mit ehrlicher Regel, mit Orakel und (Tikhonov, TSVD) mit GCV, dazu der Datenrest; Sieger je Zelle."""
    a = model.op()
    cells = []
    for noise in C.NOISE_CHOICES:
        for fi, fam in enumerate(C.FAMILIES):
            X = model.draw(fam, C.EXP_COUNT, C.SEED_BASE + 7 * fi)
            B, sig = model.make_data(X, noise, C.SEED_BASE + 500 + 7 * fi, a)
            rows = method_rows(a, B, sig, X, solve_all(a, B, sig, X))
            cells.append({"noise": noise, "family": fam, "rows": rows, "winner": min(rows, key=lambda r: r["honest"])["method"], "winner_oracle": min(rows, key=lambda r: r["oracle"])["method"]})
    return {"cells": cells}


# ----------------------------------------------------------------------------- Experiment 2: Fehlerarten

def kinds_experiment() -> dict:
    """Teil 1: Energieanteil in den stummen Richtungen (Rauschen 10 %), Median je Familie und Verfahren (ehrlich), mit Kosinus zur Wahrheit. Teil 2: Merkmalstest an Blockzahl-Ziehungen mit
    ein und drei Blöcken: Anteil der Schätzer mit gleich vielen, mehr oder weniger Abschnitten als die Wahrheit selbst (Rauschen 1 und 10 %), dazu eine Probe und die wahrscheinlichste Komponente
    des exakten Mischungs-Posterior."""
    a = model.op()
    null = []
    for fi, fam in enumerate(C.FAMILIES):
        X = model.draw(fam, C.EXP_COUNT, C.SEED_BASE + 100 + 7 * fi)
        B, sig = model.make_data(X, 1e-1, C.SEED_BASE + 600 + 7 * fi, a)
        rows = method_rows(a, B, sig, X, solve_all(a, B, sig, X))
        mask = model.silent_mask(a, X, sig)
        null.append({"family": fam, "silent": float(np.median(mask.sum(axis=1))), "truth": float(np.median(model.null_share(a, X, mask))),
                     "methods": {r["method"]: {"null": r["null"], "cos": r["cos"]} for r in rows}})
    mix = model.blockzahl_mixture()
    feats = []
    for noise in C.FEATURE_NOISES:
        draws, comp = mix.sample(3 * C.EXP_COUNT, C.SEED_BASE + 200)
        for cls, blocks in ((0, 1), (2, 3)):
            X = draws[comp == cls][:C.EXP_COUNT]
            B, sig = model.make_data(X, noise, C.SEED_BASE + 700 + cls, a)
            sol = solve_all(a, B, sig, X)
            ests = {m: honest(sol, m) for m in C.METHODS}
            samples, maps = [], []
            for i in range(len(X)):
                post = mix.posterior(a, B[i], sig[i])
                samples.append(post.sample(1, C.SEED_BASE + 800 + i)[0][0])
                maps.append(post.means[int(np.argmax(post.w))])
            ests["sample"], ests["map"] = np.array(samples), np.array(maps)
            seg_true = np.array([model.n_segments(x) for x in X])
            kinds = {}
            for m, est in ests.items():
                seg = np.array([model.n_segments(x) for x in est])
                kinds[m] = {"same": float(np.mean(seg == seg_true)), "more": float(np.mean(seg > seg_true)), "less": float(np.mean(seg < seg_true))}
            feats.append({"noise": noise, "blocks": blocks, "kinds": kinds})
    return {"null": null, "features": feats}


# ----------------------------------------------------------------------------- Experiment 3: Kalibrierung

CAL_LABELS = ("gp_fixed", "gp", "mixture", "conformal", "conformal_waves")


def calibration_experiment() -> dict:
    """Rauschen 10 %, EXP_COUNT Wahrheiten je Familie: Abdeckung des 90-%-Bands und relative Breite für das GP-Posterior (tau fest und per Evidenz), den Mischungs-Posterior und konformale Bänder
    (Tikhonov, kalibriert auf EXP_CAL Ziehungen derselben Familie und auf Wellen)."""
    a = model.op()
    cm = gp_cmat()
    half_waves = calibration_halfwidth(a, C.WAVES_CAL, 1e-1, C.SEED_BASE + 900, C.EXP_CAL)
    rows = []
    for fi, fam in enumerate(C.FAMILIES):
        X = model.draw(fam, C.EXP_COUNT, C.SEED_BASE + 300 + 7 * fi)
        B, sig = model.make_data(X, 1e-1, C.SEED_BASE + 1000 + 7 * fi, a)
        cands, res, _ = model.tikhonov_path(a, B, sig)
        tik = model.choose(cands, res, sig)["discrepancy"]
        half = calibration_halfwidth(a, fam, 1e-1, C.SEED_BASE + 950 + 7 * fi, C.EXP_CAL)
        out = {}
        for label, taus in (("gp_fixed", np.full(len(B), C.GP_FIXED_TAU)), ("gp", model.gp_best_tau(a, B, sig, cm)[0])):
            lo, hi = np.zeros_like(B), np.zeros_like(B)
            for i in range(len(B)):
                mean, cov = model.gp_posterior(a, B[i], sig[i], taus[i], cm)
                sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
                lo[i], hi[i] = mean - C.Z90 * sd, mean + C.Z90 * sd
            out[label] = band_stats(X, lo, hi)
        _, lo, hi = make_band("mixture", a, B, sig, tik, half, C.SEED_BASE + 1100)
        out["mixture"] = band_stats(X, lo, hi)
        out["conformal"] = band_stats(X, tik - half[None, :], tik + half[None, :])
        out["conformal_waves"] = band_stats(X, tik - half_waves[None, :], tik + half_waves[None, :])
        rows.append({"family": fam, **{k: {"cover": v[0], "width": v[1]} for k, v in out.items()}})
    return {"rows": rows}


# ----------------------------------------------------------------------------- Experiment 4: Welt erkennen

def evidence_experiment() -> dict:
    """Je Rauschstufe und Familie EXP_COUNT Wahrheiten: Anteil der Läufe, in denen GP (G), weißer Prior (W) oder Mischung (M) die größte Evidenz hat, und Mittel der Log-Evidenz-Differenz Mischung - GP."""
    a = model.op()
    rows = []
    for noise in C.NOISE_CHOICES:
        for fi, fam in enumerate(C.FAMILIES):
            X = model.draw(fam, C.EXP_COUNT, C.SEED_BASE + 400 + 7 * fi)
            B, sig = model.make_data(X, noise, C.SEED_BASE + 1200 + 7 * fi, a)
            ev = evidences(a, B, sig)
            best = ev.argmax(axis=0)
            rows.append({"noise": noise, "family": fam, "gp": float(np.mean(best == 0)), "white": float(np.mean(best == 1)), "mixture": float(np.mean(best == 2)), "diff": float(np.mean(ev[2] - ev[0]))})
    return {"rows": rows}


# ----------------------------------------------------------------------------- Verdikte: (Art, Text)

def _segments(k: int) -> str:
    return f"{k} Abschnitt" if k == 1 else f"{k} Abschnitte"


def _seg_text(truth: int, est: int) -> str:
    if est > truth:
        return f"zeigt {_segments(est)}, die Wahrheit {truth}: es erfindet"
    if est < truth:
        return f"zeigt {_segments(est)}, die Wahrheit {truth}: es löscht"
    return f"zeigt wie die Wahrheit {_segments(truth)}"


def verdict_instance(a: Analysis) -> tuple[str, str]:
    """Wie gut ist das gewählte Verfahren in dieser Welt? Fehler gegen den ehrlichen Sieger (WIN_TOL, BAD_FACTOR), Spanne von Datenrest und Fehler über alle Verfahren, Energie in den stummen
    Richtungen, Abschnittszahl, Abdeckung des Bands (COVER_LO bis COVER_HI), Evidenz."""
    st = a.settings
    rows = a.rows
    sel = row_of(rows, st.method)
    win = row_of(rows, a.winner_honest)
    err_sel = sel["honest"] if st.rule == "honest" else sel["oracle"]
    err_win = win["honest"]
    ratio = err_sel / err_win
    errs = [r["honest"] for r in rows]
    res = [r["resid"] for r in rows]
    lead = (f"{C.FAMILY_SHORT[st.family]}, Rauschen {noise_label(st.noise)}, {st.count} Wahrheiten: mit ehrlicher Parameterwahl gewinnt {C.METHOD_SHORT[a.winner_honest]} ({pct(err_win)}), "
            f"mit Orakel-Parameter {C.METHOD_SHORT[a.winner_oracle]} ({pct(row_of(rows, a.winner_oracle)['oracle'])}); {C.METHOD_SHORT[st.method]} liegt bei {pct(err_sel)} "
            f"({'ehrlich' if st.rule == 'honest' else 'Orakel'}). Alle acht Verfahren treffen die Daten auf {de(min(res), 1)} bis {de(max(res), 1)} σ, ihre Fehler liegen zwischen {pct(min(errs))} und {pct(max(errs))}: "
            f"der Datenrest unterscheidet sie nicht.")
    if ratio <= C.WIN_TOL:
        kind, text = "success", f"{lead} {C.METHOD_SHORT[st.method]} gehört in dieser Welt zu den besten."
    elif ratio >= C.BAD_FACTOR:
        kind, text = "warning", f"{lead} {C.METHOD_SHORT[st.method]} ist in dieser Welt {de(ratio, 1)}-mal so schlecht wie der Sieger: seine Annahme passt nicht zur Wahrheit."
    else:
        kind, text = "info", f"{lead} {C.METHOD_SHORT[st.method]} liegt beim {de(ratio, 1)}-Fachen des Siegers."
    text += (f" Es legt {pct(sel['null'])} seiner Energie in Richtungen, die die Daten nicht sehen (die Wahrheit {pct(a.null_truth)}; Kosinus der beiden Anteile {de(sel['cos'], 2)}), und "
             f"{_seg_text(a.seg_truth, a.seg_est[st.method])} (erste Wahrheit).")
    ok = C.COVER_LO <= a.band_cover_batch <= C.COVER_HI
    text += (f" Das Band ({C.BAND_LABELS[st.band]}) deckt im Mittel {pct(a.band_cover_batch)} der Wahrheit ab (Soll {pct(C.LEVEL)}, relative Breite {de(a.band_width, 2)}): "
             f"{'kalibriert' if ok else 'nicht kalibriert'}.")
    fit = FITTING_PRIOR[st.family]
    top = max(("gp", "white", "mixture"), key=lambda k: a.evid[k])
    text += f" Die Evidenz wählt unter GP, weißem Prior und Mischung in {pct(a.evid[top])} der Läufe {PRIOR_NAMES[top]}{' (passend zur Familie)' if top == fit else ''}."
    return kind, text


def _cell(cells: list, noise: float, family: str) -> dict:
    return next(c for c in cells if c["noise"] == noise and c["family"] == family)


def verdict_matrix(res: dict) -> tuple[str, str]:
    cells = res["cells"]
    parts = []
    for fam in C.FAMILIES:
        c = _cell(cells, 1e-2, fam)
        parts.append(f"{C.FAMILY_SHORT[fam]} {C.METHOD_SHORT[c['winner']]} ({pct(row_of(c['rows'], c['winner'])['honest'])})")
    sp = _cell(cells, 1e-2, "spikes")
    bz = _cell(cells, 1e-2, "blockzahl")
    wv = _cell(cells, 1e-2, "waves")
    l1_spikes = row_of(sp["rows"], "l1")["honest"]
    others = [r["honest"] for r in sp["rows"] if r["method"] != "l1"]
    mix_own = row_of(bz["rows"], "mixture")["honest"]
    best_other = min(r["oracle"] for r in bz["rows"] if r["method"] != "mixture")
    mix_wave = row_of(wv["rows"], "mixture")["honest"]
    tik_wave = row_of(wv["rows"], "tikhonov")["honest"]
    return "warning", (f"Sieger bei Rauschen 1 % mit ehrlicher Parameterwahl: {', '.join(parts)}. Auf Spitzen liegt L1 bei {pct(l1_spikes)}, alle anderen Verfahren zwischen {pct(min(others))} und {pct(max(others))}. "
                       f"Die Blockzahl-Mischung erreicht in ihrer eigenen Welt {pct(mix_own)} (bestes anderes Verfahren mit Orakel-Parameter: {pct(best_other)}), auf Wellen aber {pct(mix_wave)} gegen {pct(tik_wave)} bei Tikhonov. "
                       f"Kein Verfahren gewinnt überall, und der Gewinn der passenden Annahme ist größer als der jeder Parameterwahl.")


def verdict_kinds(res: dict) -> tuple[str, str]:
    nulls = res["null"]
    l1_null = [r["methods"]["l1"]["null"] for r in nulls if r["family"] != "spikes"]
    l1_cos = [r["methods"]["l1"]["cos"] for r in nulls if r["family"] != "spikes"]
    tik_null = [r["methods"]["tikhonov"]["null"] for r in nulls]
    truth_null = [r["truth"] for r in nulls if r["family"] != "spikes"]
    f3 = _by([f for f in res["features"] if f["noise"] == 1e-1], blocks=3)["kinds"]
    f1 = _by([f for f in res["features"] if f["noise"] == 1e-1], blocks=1)["kinds"]
    smooth = [f3[m]["less"] for m in ("tikhonov", "cgls", "tv", "pnp", "gp")]
    return "warning", (f"In den Richtungen, die die Daten nicht sehen (Rauschen 10 %), legt L1 außer auf Spitzen {pct(min(l1_null))} bis {pct(max(l1_null))} seiner Energie ab (Wahrheit {pct(min(truth_null))} bis "
                       f"{pct(max(truth_null))}; Kosinus {de(min(l1_cos), 2)} bis {de(max(l1_cos), 2)}), Tikhonov {pct(min(tik_null))} bis {pct(max(tik_null))}. Bei drei Blöcken zeigen Tikhonov, CGLS, TV, PnP und GP in "
                       f"{pct(min(smooth))} bis {pct(max(smooth))} der Läufe zu wenige Abschnitte (Auslöschen), L1 in {pct(f3['l1']['more'])} zu viele (Erfinden); bei einem Block zeigt L1 in {pct(f1['l1']['more'])} "
                       f"zu viele und eine Probe aus dem exakten Posterior in {pct(f1['sample']['more'])}. Unschärfe ist ehrliches Nichtwissen, Schärfe kann Erfindung sein.")


def verdict_calibration(res: dict) -> tuple[str, str]:
    rows = res["rows"]
    gp = {r["family"]: r["gp_fixed"]["cover"] for r in rows}
    mx = {r["family"]: r["mixture"]["cover"] for r in rows}
    cw = {r["family"]: r["conformal_waves"]["cover"] for r in rows}
    cs = {r["family"]: r["conformal"]["cover"] for r in rows}
    return "warning", (f"Abdeckung des 90-%-Bands bei Rauschen 10 %: das GP-Posterior (τ = {de(C.GP_FIXED_TAU, 1)}) {pct(gp['gauss'])} auf Prior-Ziehungen, {pct(gp['spikes'])} auf Spitzen und {pct(gp['waves'])} "
                       f"auf Wellen (zu breit); die Blockzahl-Mischung {pct(mx['blockzahl'])} in der eigenen Welt und {pct(min(v for k, v in mx.items() if k != 'blockzahl'))} bis "
                       f"{pct(max(v for k, v in mx.items() if k != 'blockzahl'))} in den anderen. Konformale Bänder halten die Abdeckung für die Familie, auf der sie kalibriert wurden "
                       f"({pct(min(cs.values()))} bis {pct(max(cs.values()))}), auf Wellen kalibriert fällt sie auf Spitzen auf {pct(cw['spikes'])} und auf Blockzahl auf {pct(cw['blockzahl'])}. "
                       f"Die Abdeckung gilt nur für die Welt, zu der das Verfahren passt oder auf der es kalibriert wurde.")


def verdict_evidence(res: dict) -> tuple[str, str]:
    rows = res["rows"]
    own = [r["white"] if r["family"] == "spikes" else (r["mixture"] if r["family"] == "blockzahl" else r["gp"]) for r in rows]
    bz = _by(rows, noise=1e-1, family="blockzahl")
    others = [r["diff"] for r in rows if r["family"] != "blockzahl"]
    return "info", (f"Unter GP, weißem Prior und Blockzahl-Mischung hat in {pct(min(own))} bis {pct(max(own))} der Läufe der zur Familie passende Prior die größte Evidenz (Spitzen: weißer Prior, Blockzahl: Mischung, "
                    f"sonst GP). Die Differenz der Log-Evidenz Mischung minus GP liegt bei Blockzahl bei {de(bz['diff'], 1)} (Rauschen 10 %) und bei den anderen Familien zwischen {de(min(others), 1)} und "
                    f"{de(max(others), 1)}. Die Evidenz erkennt die Welt ohne Wahrheit, aber nur unter den angebotenen Kandidaten: für Blöcke, Wellen und Phantom hat keiner der drei den richtigen Prior.")
