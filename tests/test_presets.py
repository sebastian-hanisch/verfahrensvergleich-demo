"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Reglergrenzen und die dokumentierten Ausgänge."""

import dataclasses

import pytest

import vfv_constants as C
import vfv_evaluation as ev
import vfv_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_on_the_choice_lists():
    for name, p in C.PRESETS.items():
        assert p["family"] in C.FAMILIES and p["noise"] in C.NOISE_CHOICES and p["method"] in C.METHODS and p["rule"] in C.RULES and p["band"] in C.BANDS, name
        assert p["count"] in C.COUNT_CHOICES and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standard"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_every_default_is_a_valid_widget_value():
    for key, spec in P.SETTING_SPECS.items():
        if spec.choices is not None:
            assert spec.default in spec.choices, key
        else:
            assert spec.lo <= spec.default <= spec.hi, key


def test_settings_fields_follow_the_preset_keys():
    assert [f.name for f in dataclasses.fields(ev.Settings)] == list(P.PRESET_KEYS)


def test_label_tables_cover_every_choice():
    assert set(C.FAMILY_LABELS) == set(C.FAMILY_SHORT) == set(C.FAMILIES) and set(C.METHOD_LABELS) == set(C.METHOD_SHORT) == set(C.METHOD_COLORS) == set(C.METHODS)
    assert set(C.RULE_LABELS) == set(C.RULES) and set(C.BAND_LABELS) == set(C.BANDS) and set(C.SHOW_RANGE) == set(C.FAMILIES)


@pytest.fixture(scope="module")
def outcomes():
    return {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}


def _ratio(a):
    return ev.row_of(a.rows, a.settings.method)["honest"] / ev.row_of(a.rows, a.winner_honest)["honest"]


def test_standard_preset(outcomes):
    a = outcomes["Standard"]
    assert ev.verdict_instance(a)[0] == "success" and a.winner_honest == "tv" and a.winner_oracle == "tv" and ev.row_of(a.rows, "tv")["honest"] == pytest.approx(0.21, abs=0.05)
    assert 0.85 <= a.band_cover_batch <= 0.95 and a.seg_truth == a.seg_est["tv"] == 1 and ": kalibriert." in ev.verdict_instance(a)[1]


def test_spikes_preset(outcomes):
    a = outcomes["Spitzen"]
    assert ev.verdict_instance(a)[0] == "success" and a.winner_honest == "l1" and ev.row_of(a.rows, "l1")["honest"] < 0.15
    assert min(r["honest"] for r in a.rows if r["method"] != "l1") > 0.6 and a.band_cover_batch < 0.85 and ": nicht kalibriert." in ev.verdict_instance(a)[1]
    assert ev.row_of(a.rows, "l1")["null"] == pytest.approx(a.null_truth, abs=0.1)


def test_waves_preset(outcomes):
    a = outcomes["Wellen"]
    assert ev.verdict_instance(a)[0] == "success" and a.winner_honest in ("cgls", "tikhonov", "tsvd", "gp") and ev.row_of(a.rows, "l1")["honest"] > 4 * ev.row_of(a.rows, a.winner_honest)["honest"]
    assert a.band_cover_batch > 0.95 and ev.row_of(a.rows, "cgls")["null"] < 0.01


def test_blockzahl_preset(outcomes):
    a = outcomes["Blockzahl"]
    assert ev.verdict_instance(a)[0] == "success" and a.winner_honest == "mixture" and 0.85 <= a.band_cover_batch <= 0.95 and a.seg_truth == a.seg_est["mixture"] == 3
    assert a.evid["mixture"] > 0.9 and a.evid["diff"] > 5


def test_wrong_world_preset(outcomes):
    a = outcomes["Falsche Welt"]
    text = ev.verdict_instance(a)[1]
    assert ev.verdict_instance(a)[0] == "warning" and _ratio(a) > 2.0 and "seine Annahme passt nicht zur Wahrheit" in text
    assert a.band_cover_batch < 0.85 and ": nicht kalibriert." in text and a.evid["gp"] > 0.9 and a.evid["diff"] < -5


def test_low_noise_preset(outcomes):
    a, std = outcomes["Wenig Rauschen"], outcomes["Standard"]
    assert ev.verdict_instance(a)[0] == "success" and a.winner_honest == "tv"
    for m in C.METHODS:
        assert ev.row_of(a.rows, m)["honest"] < ev.row_of(std.rows, m)["honest"], m


def test_conformal_preset(outcomes):
    a = outcomes["Konformal"]
    assert 0.85 <= a.band_cover_batch <= 0.95 and ": kalibriert." in ev.verdict_instance(a)[1] and ev.verdict_instance(a)[0] in ("success", "info")
    half = (a.band_hi - a.band_lo) / 2
    assert a.band_mid == pytest.approx(a.est_honest["tikhonov"]) and a.band_lo == pytest.approx(a.band_mid - half)


def test_experiment_constants_are_consistent():
    assert C.N == 48 and C.W == 0.03 and C.DEFAULT_COUNT in C.COUNT_CHOICES and C.DEFAULT_NOISE in C.NOISE_CHOICES and set(C.FEATURE_NOISES) <= set(C.NOISE_CHOICES)
    assert (C.LEVEL, C.COVER_LO, C.COVER_HI, C.WIN_TOL, C.BAD_FACTOR, C.SEG_FRAC, C.DISCREPANCY_TAU) == (0.9, 0.85, 0.95, 1.10, 2.0, 0.5, 1.0)
    assert C.CONFORMAL_CAL == 60 and C.EXP_COUNT <= max(C.COUNT_CHOICES) and C.WAVES_CAL in C.FAMILIES and C.GP_FIXED_TAU == 0.5 and C.Z90 == pytest.approx(1.6449)
    assert C.N % (2 ** C.HAAR_LEVELS) == 0 and len(C.FAMILIES) == 6 and len(C.METHODS) == 8
