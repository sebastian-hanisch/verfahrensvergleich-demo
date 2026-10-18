"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py, klickt jeden Preset-Button, setzt jeden Regler auf jede Auswahl, rechnet jedes Experiment einmal
und prüft, dass kein Python-Fehler auftritt (insbesondere StreamlitDuplicateElementId bei st.plotly_chart ohne eindeutiges key=)."""

import os

from streamlit.testing.v1 import AppTest

import vfv_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 900
SLIDERS = ("noise_select", "count_select")


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _set(at: AppTest, key, value):
    (at.select_slider(key=key) if key in SLIDERS else at.selectbox(key=key)).set_value(value)


def test_app_loads_with_metrics_tabs_and_a_verdict():
    at = _fresh_app()
    assert len(at.metric) == 4 and sum("🔬" in s.value for s in at.subheader) == 4
    assert any("gehört in dieser Welt zu den besten" in s.value for s in at.success)
    assert len(at.tabs) == 7 and any("Noch nicht gerechnet" in i.value for i in at.info)


def test_presets_and_dice_buttons_do_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels
    for index, label in enumerate(labels):
        if "Experiment" in label or "PDF" in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_option_of_every_widget_does_not_raise():
    for key, options in (("family_select", C.FAMILIES), ("method_select", C.METHODS), ("rule_select", C.RULES), ("band_select", C.BANDS), ("noise_select", C.NOISE_CHOICES), ("count_select", C.COUNT_CHOICES)):
        for value in options:
            at = _fresh_app()
            _set(at, key, value)
            at.run(timeout=TIMEOUT)
            assert not at.exception, (key, value, [str(e) for e in at.exception])


def test_every_preset_is_applied_to_the_widgets():
    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name, preset in C.PRESETS.items():
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert at.selectbox(key="family_select").value == preset["family"] and at.select_slider(key="noise_select").value == preset["noise"]
        assert at.selectbox(key="method_select").value == preset["method"] and at.selectbox(key="rule_select").value == preset["rule"] and at.selectbox(key="band_select").value == preset["band"]
        assert at.select_slider(key="count_select").value == preset["count"] and at.number_input(key="seed_input").value == preset["seed"]


def test_the_wrong_world_preset_warns_and_the_spikes_preset_praises_l1():
    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    at = _fresh_app()
    at.button[labels["Falsche Welt"]].click().run(timeout=TIMEOUT)
    assert any("seine Annahme passt nicht zur Wahrheit" in w.value for w in at.warning)
    at = _fresh_app()
    at.button[labels["Spitzen"]].click().run(timeout=TIMEOUT)
    assert any("gewinnt L1" in s.value for s in at.success)


def test_changing_a_setting_recomputes_the_metrics():
    at = _fresh_app()
    first = [m.value for m in at.metric]
    _set(at, "family_select", "spikes")
    at.run(timeout=TIMEOUT)
    assert not at.exception and [m.value for m in at.metric] != first


def test_pdf_is_created_only_after_the_click():
    at = _fresh_app()
    assert "pdf_made" not in at.session_state
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    settings, data = at.session_state["pdf_made"]
    assert data[:5] == b"%PDF-" and len(data) > 1000


def test_pdf_of_every_band_is_created():
    for band in C.BANDS:
        at = _fresh_app()
        _set(at, "band_select", band)
        at.run(timeout=TIMEOUT)
        at.button(key="pdf_make").click().run(timeout=TIMEOUT)
        assert not at.exception, (band, [str(e) for e in at.exception])
        assert at.session_state["pdf_made"][1][:5] == b"%PDF-"


def test_each_experiment_runs_once_and_shows_its_verdict():
    at = _fresh_app()
    for key in ("exp1_run", "exp2_run", "exp3_run", "exp4_run"):
        at.button(key=key).click().run(timeout=TIMEOUT)
        assert not at.exception, (key, [str(e) for e in at.exception])
    texts = [e.value for e in [*at.info, *at.warning]]
    for marker in ("Sieger bei Rauschen 1 %", "(Auslöschen)", "Konformale Bänder halten die Abdeckung", "Die Evidenz erkennt die Welt ohne Wahrheit"):
        assert any(marker in t for t in texts), marker
    assert not any("Noch nicht gerechnet" in i.value for i in at.info)


def test_permalink_values_are_clipped_and_unknown_choices_ignored():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["family"] = "kreis"
    at.query_params["r"] = "0.5"
    at.query_params["method"] = "magic"
    at.query_params["rule"] = "zufall"
    at.query_params["band"] = "bootstrap"
    at.query_params["count"] = "7"
    at.query_params["seed"] = "99999999"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.selectbox(key="family_select").value == "blocks" and at.select_slider(key="noise_select").value == 0.1 and at.selectbox(key="method_select").value == "tv"
    assert at.selectbox(key="rule_select").value == "honest" and at.selectbox(key="band_select").value == "gp" and at.select_slider(key="count_select").value == 40
    assert at.number_input(key="seed_input").value == 999999


def test_valid_permalink_values_are_taken_over():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["family"] = "spikes"
    at.query_params["method"] = "l1"
    at.query_params["rule"] = "oracle"
    at.query_params["band"] = "conformal"
    at.query_params["count"] = "20"
    at.query_params["seed"] = "7"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.selectbox(key="family_select").value == "spikes" and at.selectbox(key="method_select").value == "l1" and at.selectbox(key="rule_select").value == "oracle"
    assert at.selectbox(key="band_select").value == "conformal" and at.select_slider(key="count_select").value == 20 and at.number_input(key="seed_input").value == 7


def test_the_address_bar_mirrors_the_settings():
    at = _fresh_app()
    at.selectbox(key="method_select").set_value("l1").run(timeout=TIMEOUT)
    assert at.query_params["method"] == ["l1"] or at.query_params["method"] == "l1"
