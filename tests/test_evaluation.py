"""Einheiten der Auswertung: Formatierung, Verfahren auf Wahrheitsmengen, Bänder, Evidenz, Analyse, Schwellen und Verdikte."""

import dataclasses

import numpy as np
import pytest

import vfv_constants as C
import vfv_evaluation as ev
import vfv_model as model


def test_german_number_formats():
    assert ev.de(1.234, 2) == "1,23" and ev.de(-1.5, 1) == "−1,5"
    assert ev.sci(2.5e9) == "2,5·10⁹" and ev.sci(9.96e3) == "1,0·10⁴" and ev.sci(4.5e-10) == "4,5·10⁻¹⁰" and ev.sci(0.0) == "0,0" and ev.sci(float("inf")) == "unendlich"
    assert ev.pct(0.1234) == "12,3 %" and ev.pct(0.0) == "0,0 %" and ev.pct(0.998) == "100 %" and ev.pct(12.0) == "1,2·10¹-fach" and ev.pct(float("inf")) == "unendlich"
    assert ev.noise_label(0.1) == "10 %" and ev.noise_label(0.01) == "1 %" and ev.noise_label(0.001) == "0,1 %" and ev.noise_label(0.0) == "0 %"


@pytest.fixture(scope="module")
def small():
    a = model.op()
    X = model.draw("blocks", 6, 11)
    B, sig = model.make_data(X, 0.05, 12, a)
    return a, X, B, sig, ev.solve_all(a, B, sig, X)


def test_solve_all_has_every_method_and_rule(small):
    a, X, B, sig, sol = small
    assert set(sol) == set(C.METHODS)
    for m in ("tikhonov", "tsvd"):
        assert set(sol[m]) == {"discrepancy", "gcv", "oracle"}
    for m in ("cgls", "l1", "tv", "pnp"):
        assert set(sol[m]) == {"discrepancy", "oracle"}
    assert set(sol["gp"]) == {"evidence", "fixed", "oracle", "tau"} and set(sol["mixture"]) == {"prior"}
    for m in C.METHODS:
        assert ev.honest(sol, m).shape == (6, C.N) and ev.oracle(sol, m).shape == (6, C.N)


def test_the_oracle_is_never_worse_than_the_honest_rule_per_problem(small):
    a, X, B, sig, sol = small
    for m in C.METHODS:
        assert (model.rel_err(ev.oracle(sol, m), X) <= model.rel_err(ev.honest(sol, m), X) + 1e-12).all(), m


def test_honest_and_oracle_pick_the_documented_keys(small):
    a, X, B, sig, sol = small
    assert ev.honest(sol, "gp") is sol["gp"]["evidence"] and ev.honest(sol, "mixture") is sol["mixture"]["prior"] and ev.honest(sol, "tv") is sol["tv"]["discrepancy"]
    assert ev.oracle(sol, "mixture") is sol["mixture"]["prior"] and ev.oracle(sol, "l1") is sol["l1"]["oracle"]


def test_gp_with_evidence_is_the_mean_at_the_best_tau(small):
    a, X, B, sig, sol = small
    cm = ev.gp_cmat()
    assert sol["gp"]["evidence"] == pytest.approx(model.gp_mean(a, B, sig, sol["gp"]["tau"], cm), abs=1e-10)
    assert sol["gp"]["fixed"] == pytest.approx(model.gp_mean(a, B, sig, C.GP_FIXED_TAU, cm), abs=1e-10)
    assert set(np.round(sol["gp"]["tau"], 10)) <= set(np.round(model.gp_taus(), 10))


def test_method_rows_are_medians_and_gcv_only_for_the_svd_methods(small):
    a, X, B, sig, sol = small
    rows = ev.method_rows(a, B, sig, X, sol)
    assert [r["method"] for r in rows] == list(C.METHODS)
    for r in rows:
        assert r["honest"] == pytest.approx(np.median(model.rel_err(ev.honest(sol, r["method"]), X))) and r["oracle"] == pytest.approx(np.median(model.rel_err(ev.oracle(sol, r["method"]), X)))
        assert (r["gcv"] is not None) == (r["method"] in ("tikhonov", "tsvd"))
        assert r["resid"] == pytest.approx(np.median(model.data_residual(a, ev.honest(sol, r["method"]), B, sig)))
    assert ev.row_of(rows, "tv")["method"] == "tv"


def test_calibration_halfwidth_is_the_conformal_quantile_of_tikhonov_errors():
    a = model.op()
    half = ev.calibration_halfwidth(a, "blocks", 0.1, 5, count=30)
    Xc = model.draw("blocks", 30, 5)
    Bc, sc = model.make_data(Xc, 0.1, 6, a)
    cands, res, _ = model.tikhonov_path(a, Bc, sc)
    est = model.choose(cands, res, sc)["discrepancy"]
    assert half.shape == (C.N,) and (half > 0).all() and half == pytest.approx(model.conformal_halfwidth(np.abs(est - Xc)))


def test_make_band_variants(small):
    a, X, B, sig, sol = small
    tik = ev.honest(sol, "tikhonov")
    half = np.full(C.N, 0.3)
    mid, lo, hi = ev.make_band("conformal", a, B, sig, tik, half, 1)
    assert mid is tik and lo == pytest.approx(tik - 0.3) and hi == pytest.approx(tik + 0.3)
    mid, lo, hi = ev.make_band("gp", a, B, sig, tik, half, 1)
    assert (lo < mid).all() and (mid < hi).all() and mid == pytest.approx(ev.honest(sol, "gp"), abs=1e-6)
    mid, lo, hi = ev.make_band("mixture", a, B, sig, tik, half, 1)
    assert (lo <= hi).all() and mid == pytest.approx(ev.honest(sol, "mixture"), abs=1e-8)
    assert np.array_equal(ev.make_band("mixture", a, B, sig, tik, half, 1)[1], lo)


def test_band_stats_by_hand():
    X = np.array([[0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0]])
    lo = np.array([[0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    hi = np.array([[1.0, 1.0, 4.0, 2.0], [2.0, 2.0, 2.0, 2.0]])
    cov, width = ev.band_stats(X, lo, hi)
    assert cov == pytest.approx(0.75)                                  # (2 von 4 und 4 von 4) / 2 = 0,75
    rms = np.sqrt(np.mean(X ** 2, axis=1))
    assert width == pytest.approx(np.median((hi - lo).mean(axis=1) / rms))


def test_evidences_have_one_row_per_prior(small):
    a, X, B, sig, sol = small
    e = ev.evidences(a, B, sig)
    assert e.shape == (3, 6) and np.isfinite(e).all()
    assert e[0] == pytest.approx(model.gp_best_tau(a, B, sig, ev.gp_cmat())[1]) and e[1] == pytest.approx(model.gp_best_tau(a, B, sig, np.eye(C.N))[1])
    assert e[2][0] == pytest.approx(model.blockzahl_mixture().log_evidence(a, B[0], sig[0]))


def test_analysis_fields_are_consistent():
    st = ev.Settings(count=20)
    a = ev.analyse(st)
    X = model.draw(st.family, 20, st.seed + C.SEED_TRUTH)
    assert np.array_equal(a.truth, X[0]) and a.s.shape == (C.N,) and a.sig > 0
    assert [r["method"] for r in a.rows] == list(C.METHODS) and a.winner_honest == min(a.rows, key=lambda r: r["honest"])["method"] and a.winner_oracle == min(a.rows, key=lambda r: r["oracle"])["method"]
    assert set(a.est) == set(C.METHODS) and a.band_lo.shape == a.band_hi.shape == a.band_mid.shape == (C.N,)
    assert a.coef_truth.shape == a.coef_est.shape == a.sv.shape == (C.N,) and (a.sv[:-1] >= a.sv[1:]).all()
    assert a.seg_truth == model.n_segments(a.truth) and a.seg_est["tv"] == model.n_segments(a.est["tv"])
    assert 0 <= a.band_cover_batch <= 1 and 0 <= a.band_cover <= 1
    assert a.evid["gp"] + a.evid["white"] + a.evid["mixture"] == pytest.approx(1.0) and 0 <= a.silent <= C.N
    assert a.est_honest["gp"] == pytest.approx(a.est["gp"]) and a.est_oracle["l1"].shape == (C.N,)


def test_the_rule_selects_honest_or_oracle_estimates():
    h = ev.analyse(ev.Settings(count=20, rule="honest"))
    o = ev.analyse(ev.Settings(count=20, rule="oracle"))
    assert np.array_equal(h.est["l1"], h.est_honest["l1"]) and np.array_equal(o.est["l1"], o.est_oracle["l1"]) and not np.array_equal(h.est["l1"], o.est["l1"])
    assert np.array_equal(h.est_honest["l1"], o.est_honest["l1"])


def test_analysis_is_deterministic_and_depends_on_the_settings():
    st = ev.Settings(count=20)
    a, b = ev.analyse(st), ev.analyse(st)
    assert np.array_equal(a.truth, b.truth) and np.array_equal(a.est["tv"], b.est["tv"]) and a.rows == b.rows
    assert not np.array_equal(a.truth, ev.analyse(dataclasses.replace(st, seed=st.seed + 1)).truth)
    assert not np.array_equal(a.truth, ev.analyse(dataclasses.replace(st, family="spikes")).truth)
    assert not np.array_equal(a.b, ev.analyse(dataclasses.replace(st, noise=0.01)).b)
    assert not np.array_equal(a.band_lo, ev.analyse(dataclasses.replace(st, band="conformal")).band_lo)
    assert np.array_equal(a.band_lo, ev.analyse(dataclasses.replace(st, method="l1")).band_lo)       # das Band hängt nicht vom hervorgehobenen Verfahren ab


def test_segment_text_by_hand():
    assert ev._seg_text(1, 1) == "zeigt wie die Wahrheit 1 Abschnitt" and ev._seg_text(3, 3) == "zeigt wie die Wahrheit 3 Abschnitte"
    assert ev._seg_text(1, 3) == "zeigt 3 Abschnitte, die Wahrheit 1: es erfindet" and ev._seg_text(3, 1) == "zeigt 1 Abschnitt, die Wahrheit 3: es löscht"


def _crafted(**kw):
    """Die Standard-Analyse mit künstlich gesetzten Fehlern: Sieger tv mit 1,0 (zur Veranschaulichung), gewählt tikhonov mit kw['sel']."""
    a = ev.analyse(ev.Settings(count=20, method="tikhonov"))
    rows = []
    for r in a.rows:
        r = dict(r)
        r["honest"] = r["oracle"] = 1.0 if r["method"] == "tv" else 5.0
        if r["method"] == "tikhonov":
            r["honest"] = kw.get("sel", 1.0)
        rows.append(r)
    return dataclasses.replace(a, rows=rows, winner_honest="tv", winner_oracle="tv", **{k: v for k, v in kw.items() if k != "sel"})


def test_verdict_thresholds_are_inclusive_at_the_limits():
    assert ev.verdict_instance(_crafted(sel=1.1))[0] == "success" and "gehört in dieser Welt zu den besten" in ev.verdict_instance(_crafted(sel=1.1))[1]
    assert ev.verdict_instance(_crafted(sel=1.1000001))[0] == "info" and "beim 1,1-Fachen des Siegers" in ev.verdict_instance(_crafted(sel=1.1000001))[1]
    kind, text = ev.verdict_instance(_crafted(sel=2.0))
    assert kind == "warning" and "2,0-mal so schlecht wie der Sieger" in text and "seine Annahme passt nicht zur Wahrheit" in text
    assert ev.verdict_instance(_crafted(sel=1.9999))[0] == "info"


def test_verdict_names_the_ranges_the_band_and_the_evidence():
    text = ev.verdict_instance(_crafted(sel=1.0))[1]
    assert "mit ehrlicher Parameterwahl gewinnt TV (100,0 %)" not in text and "Alle acht Verfahren treffen die Daten auf" in text and "der Datenrest unterscheidet sie nicht" in text
    assert "Es legt" in text and "Richtungen, die die Daten nicht sehen" in text and "(erste Wahrheit)" in text and "Die Evidenz wählt unter GP, weißem Prior und Mischung in" in text


def test_the_band_is_calibrated_inside_the_inclusive_range():
    for cover, ok in ((C.COVER_LO, True), (C.COVER_LO - 1e-4, False), (C.COVER_HI, True), (C.COVER_HI + 1e-4, False), (0.9, True)):
        text = ev.verdict_instance(_crafted(band_cover_batch=cover))[1]
        assert (": kalibriert." in text) is ok and (": nicht kalibriert." in text) is (not ok), cover


def test_the_evidence_phrase_marks_the_matching_prior():
    fit = ev.FITTING_PRIOR["blocks"]
    assert fit == "gp"
    good = ev.verdict_instance(_crafted(evid={"gp": 0.9, "white": 0.0, "mixture": 0.1, "diff": -5.0}))[1]
    bad = ev.verdict_instance(_crafted(evid={"gp": 0.1, "white": 0.0, "mixture": 0.9, "diff": 5.0}))[1]
    assert "das GP (passend zur Familie)" in good and "in 90,0 % der Läufe das GP" in good
    assert "in 90,0 % der Läufe die Blockzahl-Mischung." in bad and "(passend zur Familie)" not in bad


def test_fitting_prior_table_covers_every_family():
    assert set(ev.FITTING_PRIOR) == set(C.FAMILIES) and ev.FITTING_PRIOR["spikes"] == "white" and ev.FITTING_PRIOR["blockzahl"] == "mixture"
    assert set(ev.PRIOR_NAMES) == {"gp", "white", "mixture"} and set(ev.HONEST_KEY) == set(C.METHODS)
