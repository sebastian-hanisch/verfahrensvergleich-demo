"""Behauptungen der vier Experimente (Strukturen und Bänder statt letzter Stellen; feste Ströme, 40 Wahrheiten je Zelle)."""

import pytest

import vfv_constants as C
import vfv_evaluation as ev


@pytest.fixture(scope="module")
def e1():
    return ev.matrix_experiment()


@pytest.fixture(scope="module")
def e2():
    return ev.kinds_experiment()


@pytest.fixture(scope="module")
def e3():
    return ev.calibration_experiment()


@pytest.fixture(scope="module")
def e4():
    return ev.evidence_experiment()


def _cell(res, noise, fam):
    return next(c for c in res["cells"] if c["noise"] == noise and c["family"] == fam)


def _r(cell, method, key="honest"):
    return ev.row_of(cell["rows"], method)[key]


# ----------------------------------------------------------------------------- Experiment 1

def test_experiment_1_has_every_cell_and_every_method(e1):
    assert len(e1["cells"]) == len(C.NOISE_CHOICES) * len(C.FAMILIES)
    for c in e1["cells"]:
        assert [r["method"] for r in c["rows"]] == list(C.METHODS)
        assert {r["method"] for r in c["rows"] if r["gcv"] is not None} == {"tikhonov", "tsvd"}


def test_experiment_1_every_family_has_its_own_winner_class(e1):
    for noise in C.NOISE_CHOICES:
        assert _cell(e1, noise, "spikes")["winner"] == "l1" and _cell(e1, noise, "blocks")["winner"] == "tv" and _cell(e1, noise, "blockzahl")["winner"] == "mixture"
        for fam in ("waves", "phantom", "gauss"):
            assert _cell(e1, noise, fam)["winner"] in ("tikhonov", "tsvd", "cgls", "gp"), (noise, fam)
    assert len({_cell(e1, 1e-2, f)["winner"] for f in C.FAMILIES}) >= 4


def test_experiment_1_l1_wins_spikes_by_a_wide_margin_and_loses_elsewhere(e1):
    for noise, other_min in ((1e-2, 0.3), (1e-1, 0.6)):
        c = _cell(e1, noise, "spikes")
        assert _r(c, "l1") < (0.02 if noise == 1e-2 else 0.12) and min(_r(c, m) for m in C.METHODS if m != "l1") > other_min
    for fam in ("blocks", "waves", "phantom", "gauss", "blockzahl"):
        for noise in C.NOISE_CHOICES:
            assert _r(_cell(e1, noise, fam), "l1") > 0.4, (fam, noise)


def test_experiment_1_tv_wins_blocks(e1):
    c1, c2 = _cell(e1, 1e-2, "blocks"), _cell(e1, 1e-1, "blocks")
    assert _r(c1, "tv") < 0.05 and _r(c1, "tikhonov") > 3 * _r(c1, "tv") and _r(c2, "tv") < 0.25 < _r(c2, "tikhonov") + 0.02 and _r(c2, "tv") < _r(c2, "tikhonov")


def test_experiment_1_the_mixture_wins_its_own_world_even_against_the_oracle_and_loses_elsewhere(e1):
    for noise, bound in ((1e-2, 0.16), (1e-1, 0.25)):
        c = _cell(e1, noise, "blockzahl")
        assert _r(c, "mixture") < bound and _r(c, "mixture") < min(_r(c, m, "oracle") for m in C.METHODS if m != "mixture")
    assert _r(_cell(e1, 1e-2, "waves"), "mixture") > 1.5 * _r(_cell(e1, 1e-2, "waves"), "tikhonov")
    assert _r(_cell(e1, 1e-1, "waves"), "mixture") > 1.8 * _r(_cell(e1, 1e-1, "waves"), "cgls")
    assert _r(_cell(e1, 1e-1, "phantom"), "mixture") > 2.0 * _r(_cell(e1, 1e-1, "phantom"), "cgls") and _r(_cell(e1, 1e-1, "spikes"), "mixture") > 0.5


def test_experiment_1_the_oracle_is_never_worse_and_the_honest_rule_costs_up_to_a_factor_two(e1):
    gaps = []
    for c in e1["cells"]:
        for r in c["rows"]:
            assert r["oracle"] <= r["honest"] + 1e-12
            gaps.append(r["honest"] / r["oracle"])
    assert max(gaps) > 1.5 and max(gaps) < 2.5
    assert _r(_cell(e1, 1e-2, "spikes"), "pnp") > 1.5 * _r(_cell(e1, 1e-2, "spikes"), "pnp", "oracle") and _r(_cell(e1, 1e-2, "blocks"), "tv") > 1.4 * _r(_cell(e1, 1e-2, "blocks"), "tv", "oracle")
    for c in e1["cells"]:
        assert _r(c, "tikhonov") < 1.3 * _r(c, "tikhonov", "oracle")


def test_experiment_1_the_data_residual_does_not_separate_the_methods(e1):
    for c in e1["cells"]:
        for r in c["rows"]:
            assert 0.5 < r["resid"] < 1.25, (c["noise"], c["family"], r["method"])
    c = _cell(e1, 1e-1, "spikes")
    errs = [r["honest"] for r in c["rows"]]
    res = [r["resid"] for r in c["rows"]]
    assert max(errs) / min(errs) > 8 and max(res) < 1.0 and min(res) > 0.7


def test_experiment_1_verdict(e1):
    kind, text = ev.verdict_matrix(e1)
    assert kind == "warning" and "Sieger bei Rauschen 1 %" in text and "Spitzen L1" in text and "Blöcke TV" in text and "Kein Verfahren gewinnt überall" in text


# ----------------------------------------------------------------------------- Experiment 2

def _null(e2, fam):
    return next(r for r in e2["null"] if r["family"] == fam)


def _feat(e2, noise, blocks):
    return next(f for f in e2["features"] if f["noise"] == noise and f["blocks"] == blocks)["kinds"]


def test_experiment_2_l1_puts_energy_where_the_data_are_silent(e2):
    for fam in ("blocks", "waves", "phantom", "gauss", "blockzahl"):
        r = _null(e2, fam)
        assert r["methods"]["l1"]["null"] > 0.35 and r["truth"] < 0.09 and r["methods"]["l1"]["cos"] < 0.2, fam
    sp = _null(e2, "spikes")
    assert sp["methods"]["l1"]["null"] == pytest.approx(sp["truth"], abs=0.08) and sp["methods"]["l1"]["cos"] > 0.99 and sp["truth"] > 0.3


def test_experiment_2_smooth_methods_stay_quiet_in_the_silent_directions(e2):
    for r in e2["null"]:
        assert r["methods"]["tikhonov"]["null"] < 0.015 and r["methods"]["cgls"]["null"] < 0.005 and r["methods"]["tsvd"]["null"] < 0.005 and r["methods"]["gp"]["null"] < 0.02, r["family"]
        assert 0.025 < r["methods"]["tv"]["null"] < 0.07 and 0.04 < r["methods"]["pnp"]["null"] < 0.13 and 0.06 < r["methods"]["mixture"]["null"] < 0.13
    assert _null(e2, "blocks")["methods"]["tv"]["cos"] > 0.5 and _null(e2, "spikes")["methods"]["tv"]["cos"] < -0.3
    assert _null(e2, "blockzahl")["methods"]["mixture"]["cos"] > 0.8 and all(_null(e2, f)["methods"]["mixture"]["cos"] < 0.1 for f in ("spikes", "blocks", "waves", "phantom", "gauss"))
    assert 24 <= _null(e2, "spikes")["silent"] <= 26 and _null(e2, "waves")["silent"] >= 34


def test_experiment_2_smooth_methods_delete_features_l1_invents_them(e2):
    f3 = _feat(e2, 1e-1, 3)
    for m in ("tikhonov", "tsvd", "cgls", "tv", "pnp", "gp"):
        assert f3[m]["less"] >= 0.8 and f3[m]["more"] <= 0.02, m
    assert f3["tv"]["less"] >= 0.95 and f3["l1"]["more"] >= 0.25 and f3["l1"]["same"] >= 0.4
    assert f3["mixture"]["same"] >= 0.85 and f3["sample"]["same"] >= 0.7 and f3["map"]["same"] >= 0.8
    f1 = _feat(e2, 1e-1, 1)
    assert f1["l1"]["more"] >= 0.8 and f1["sample"]["more"] > f1["tv"]["more"] + 0.1 and f1["sample"]["more"] > f1["tikhonov"]["more"] + 0.05 and f1["tikhonov"]["less"] > f1["tikhonov"]["more"]
    f3lo = _feat(e2, 1e-2, 3)
    assert f3lo["tv"]["less"] >= 0.6 and f3lo["mixture"]["same"] >= 0.85


def test_experiment_2_each_row_of_a_feature_table_sums_to_one(e2):
    for f in e2["features"]:
        for m, k in f["kinds"].items():
            assert k["same"] + k["more"] + k["less"] == pytest.approx(1.0), m
    assert len(e2["features"]) == 4 and len(e2["null"]) == 6


def test_experiment_2_verdict(e2):
    kind, text = ev.verdict_kinds(e2)
    assert kind == "warning" and "(Auslöschen)" in text and "(Erfinden)" in text and "Unschärfe ist ehrliches Nichtwissen, Schärfe kann Erfindung sein" in text


# ----------------------------------------------------------------------------- Experiment 3

def _cal(e3, fam):
    return next(r for r in e3["rows"] if r["family"] == fam)


def test_experiment_3_conformal_bands_keep_their_coverage_on_the_calibration_family_only(e3):
    for r in e3["rows"]:
        assert 0.85 <= r["conformal"]["cover"] <= 0.95, r["family"]
    assert 0.85 <= _cal(e3, "waves")["conformal_waves"]["cover"] <= 0.95 and 0.85 <= _cal(e3, "phantom")["conformal_waves"]["cover"] <= 0.95
    assert _cal(e3, "spikes")["conformal_waves"]["cover"] < 0.82 and _cal(e3, "blockzahl")["conformal_waves"]["cover"] < 0.75 and _cal(e3, "gauss")["conformal_waves"]["cover"] < 0.75
    assert _cal(e3, "spikes")["conformal"]["width"] > 2 * _cal(e3, "spikes")["gp_fixed"]["width"]


def test_experiment_3_the_gp_band_is_calibrated_only_for_its_prior(e3):
    assert 0.85 <= _cal(e3, "gauss")["gp_fixed"]["cover"] <= 0.95 and _cal(e3, "spikes")["gp_fixed"]["cover"] < 0.85
    assert _cal(e3, "waves")["gp_fixed"]["cover"] > 0.95 and _cal(e3, "phantom")["gp_fixed"]["cover"] > 0.95
    assert 0.85 <= _cal(e3, "blocks")["gp"]["cover"] <= 0.95 and _cal(e3, "waves")["gp"]["cover"] > 0.95


def test_experiment_3_the_mixture_band_is_calibrated_only_in_its_own_world_and_narrower(e3):
    own = _cal(e3, "blockzahl")
    assert 0.85 <= own["mixture"]["cover"] <= 0.95 and own["mixture"]["width"] < 0.8 * own["gp_fixed"]["width"]
    others = [_cal(e3, f)["mixture"]["cover"] for f in C.FAMILIES if f != "blockzahl"]
    assert max(others) < 0.9 and min(others) < 0.7


def test_experiment_3_verdict(e3):
    kind, text = ev.verdict_calibration(e3)
    assert kind == "warning" and "auf Wellen kalibriert fällt sie auf Spitzen" in text and "Die Abdeckung gilt nur für die Welt" in text


# ----------------------------------------------------------------------------- Experiment 4

def test_experiment_4_the_evidence_recognizes_the_world_among_the_candidates(e4):
    for r in e4["rows"]:
        fit = ev.FITTING_PRIOR[r["family"]]
        assert r[fit] >= 0.85, (r["noise"], r["family"])
        assert r["gp"] + r["white"] + r["mixture"] == pytest.approx(1.0)
    assert len(e4["rows"]) == len(C.NOISE_CHOICES) * len(C.FAMILIES)


def test_experiment_4_the_log_evidence_difference_of_mixture_and_gp(e4):
    for r in e4["rows"]:
        if r["family"] == "blockzahl":
            assert r["diff"] > 5
        else:
            assert r["diff"] < -10
    assert next(r for r in e4["rows"] if r["noise"] == 1e-2 and r["family"] == "spikes")["diff"] < -50


def test_experiment_4_verdict(e4):
    kind, text = ev.verdict_evidence(e4)
    assert kind == "info" and "Die Evidenz erkennt die Welt ohne Wahrheit, aber nur unter den angebotenen Kandidaten" in text
