"""Einheiten des Kerns an der Mini-Instanz (Handrechnung in conftest.py) und an kleinen Sonderfällen."""

import math

import numpy as np
import pytest

import vfv_constants as C
import vfv_model as model
from inverse_generator import SplitMix64


def test_soft_threshold():
    assert model.soft([-2.0, -0.5, 0.0, 0.5, 2.0], 1.0).tolist() == [-1.0, 0.0, 0.0, 0.0, 1.0]


def test_rel_err_by_hand():
    assert model.rel_err(np.array([[3.0, 4.0]]), np.array([[0.0, 0.0 + 5.0]]))[0] == pytest.approx(math.sqrt(9 + 1) / 5)


def test_tikhonov_path_by_hand(mini):
    a, b, sig = mini["a"], mini["b"], mini["sig"]
    cands, res, gcv = model.tikhonov_path(a, b, sig)
    assert cands.shape == (C.LAMBDA_COUNT, 1, 3) and res.shape == gcv.shape == (C.LAMBDA_COUNT, 1)
    # lambda = 1 · 10^(-1) liegt bei Index 4 (Schrittweite 0,25 im Exponenten); b hier = A x, also ist beta = (2,3094; 0; -0,08165)
    assert np.logspace(0, -C.LAMBDA_SPAN, C.LAMBDA_COUNT)[4] == pytest.approx(0.1)
    assert cands[4, 0, 0] == pytest.approx(cands[4, 0, 2], abs=1e-12)
    f = np.array([1.0, 0.25, 0.01])
    f = f / (f + 0.1)
    beta = np.array([4 / math.sqrt(3), 0.0, -0.2 / math.sqrt(6)])
    r2 = float((((1 - f) * beta) ** 2).sum())
    assert res[4, 0] == pytest.approx(math.sqrt(r2), abs=1e-12) and res[4, 0] == pytest.approx(0.2227, abs=2e-4)
    assert gcv[4, 0] == pytest.approx(3 * r2 / (3 - f.sum()) ** 2, abs=1e-12) and gcv[4, 0] == pytest.approx(0.0900, abs=2e-4)
    assert cands[-1, 0] == pytest.approx([1.0, 2.0, 1.0], abs=1e-5)       # lambda = 1e-10 · 1: fast die Inverse
    # lambda = 1 (= s_max²): 0,5 · 2,3094 · (1,1,1)/sqrt(3) und 0,1/1,01 · (-0,08165) · (1,-2,1)/sqrt(6) ergeben (0,6634; 0,6733; 0,6634)
    assert cands[0, 0] == pytest.approx([0.66337, 0.67327, 0.66337], abs=1e-4)


def test_tikhonov_values_by_hand(mini):
    """lambda = 0,1 (Index 4): x_1 = 2,3094 / 1,1 · (1,1,1)/sqrt(3) + (-0,08165 · 0,1 / 0,11) · (1,-2,1)/sqrt(6) = (1,1818; 1,2727; 1,1818); lambda = 0,01 (Index 8) wie in Stück 14: (1,153465; 1,653465; 1,153465)."""
    cands, _, _ = model.tikhonov_path(mini["a"], mini["b"], mini["sig"])
    assert cands[4, 0] == pytest.approx([1.181818, 1.272727, 1.181818], abs=1e-5)
    assert np.logspace(0, -C.LAMBDA_SPAN, C.LAMBDA_COUNT)[8] == pytest.approx(0.01)
    assert cands[8, 0] == pytest.approx([1.153465, 1.653465, 1.153465], abs=1e-5)


def test_tsvd_path_by_hand(mini):
    a, b, sig = mini["a"], mini["b"], mini["sig"]
    cands, res, gcv = model.tsvd_path(a, b, sig)
    assert cands.shape == (3, 1, 3)
    assert cands[0, 0] == pytest.approx([4 / 3] * 3, abs=1e-12) and cands[1, 0] == pytest.approx([4 / 3] * 3, abs=1e-12) and cands[2, 0] == pytest.approx([1.0, 2.0, 1.0], abs=1e-12)
    assert res[:, 0] == pytest.approx([0.08165, 0.08165, 0.0], abs=1e-5)
    assert gcv[0, 0] == pytest.approx(0.005, abs=1e-6) and gcv[1, 0] == pytest.approx(0.02, abs=1e-6) and math.isinf(gcv[2, 0])


def test_choose_by_hand(mini):
    a, b, sig, x = mini["a"], mini["b"], mini["sig"], mini["x"]
    cands, res, gcv = model.tsvd_path(a, b, sig)
    out = model.choose(cands, res, sig, x, gcv)
    assert set(out) == {"discrepancy", "gcv", "oracle"}
    assert out["discrepancy"][0] == pytest.approx([4 / 3] * 3) and out["gcv"][0] == pytest.approx([4 / 3] * 3) and out["oracle"][0] == pytest.approx([1.0, 2.0, 1.0])
    assert set(model.choose(cands, res, sig)) == {"discrepancy"}


def test_discrepancy_rule_picks_the_most_regularized_candidate_below_the_threshold_and_the_last_if_none():
    cands = np.arange(4.0)[:, None, None] * np.ones((4, 1, 2))
    sig = np.array([1.0])
    thr = sig[0] * math.sqrt(2)
    res = np.array([[thr + 0.1], [thr], [thr - 0.1], [0.0]])
    assert model.choose(cands, res, sig)["discrepancy"][0, 0] == 1.0          # gleich der Schwelle zählt (<=)
    assert model.choose(cands, res + 10, sig)["discrepancy"][0, 0] == 3.0
    assert model.choose(cands, res, sig, tau=0.5)["discrepancy"][0, 0] == 3.0   # Schwelle 0,5 sigma sqrt(n): nur der Rest 0


def test_cgls_by_hand(mini):
    a, b, sig = mini["a"], mini["b"], mini["sig"]
    cands, res, gcv = model.cgls_path(a, b, sig)
    assert cands.shape == (3, 1, 3) and gcv is None
    assert cands[0, 0] == pytest.approx([1.33, 1.34, 1.33], abs=1e-4)
    assert cands[2, 0] == pytest.approx([1.0, 2.0, 1.0], abs=1e-6)
    assert res[2, 0] == pytest.approx(0.0, abs=1e-6) and res[0, 0] == pytest.approx(np.linalg.norm(b[0] - a @ cands[0, 0]), abs=1e-12)


def test_haar_matrix_by_hand():
    w = model.haar_matrix(4, 2)
    r = math.sqrt(0.5)
    assert w == pytest.approx(np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, -0.5, -0.5], [r, -r, 0, 0], [0, 0, r, -r]]), abs=1e-12)
    with pytest.raises(ValueError):
        model.haar_matrix(6, 2)
    assert model.haar_matrix(3, 0) == pytest.approx(np.eye(3))


def test_l1_path_by_hand(mini):
    a, b, sig = mini["a"], mini["b"], mini["sig"]
    cands, res, _ = model.l1_path(a, b, sig)
    assert cands.shape == (C.SOFT_LAMBDAS, 1, 3)
    assert cands[0, 0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)                       # lambda = |A^T b|_inf: die Null
    assert cands[-1, 0] == pytest.approx([1.0, 2.0, 1.0], abs=0.02)                        # lambda = 1e-4 |A^T b|_inf: fast die Inverse
    assert res[0, 0] == pytest.approx(np.linalg.norm(b[0]), abs=1e-12)


def test_pnp_with_one_level_free_dimension_equals_l1(mini):
    """n = 3 hat keine Haar-Ebene (n nicht durch 2 teilbar): die Matrix ist die Einheitsmatrix, Plug-and-Play = L1."""
    a, b, sig = mini["a"], mini["b"], mini["sig"]
    assert np.array_equal(model.pnp_path(a, b, sig)[0], model.l1_path(a, b, sig)[0])


def test_tv_path_by_hand(mini):
    a, b, sig = mini["a"], mini["b"], mini["sig"]
    cands, res, _ = model.tv_path(a, b, sig)
    assert cands.shape == (C.TV_LAMBDAS, 1, 3)
    assert cands[-1, 0] == pytest.approx([1.0, 2.0, 1.0], abs=0.02)
    assert res[-1, 0] < res[0, 0]


def test_gp_by_hand_for_one_coordinate():
    a, cm = np.array([[1.0]]), np.array([[1.0]])
    b, sig = np.array([[1.0]]), np.array([0.5])
    assert model.gp_log_evidence(a, b, sig, 2.0, cm)[0] == pytest.approx(-0.5 / 4.25 - 0.5 * math.log(4.25), abs=1e-12)
    assert model.gp_mean(a, b, sig, 2.0, cm)[0, 0] == pytest.approx(4 / 4.25, abs=1e-12)
    mean, cov = model.gp_posterior(a, b[0], 0.5, 2.0, cm)
    assert mean[0] == pytest.approx(4 / 4.25, abs=1e-12) and cov[0, 0] == pytest.approx(4 - 16 / 4.25, abs=1e-12)
    tau, ev = model.gp_best_tau(a, b, sig, cm)
    assert tau[0] == pytest.approx(math.sqrt(0.75), rel=0.2)         # tau² + sigma² = b² gibt tau = 0,866; das Gitter ist grob
    assert ev[0] == pytest.approx(-0.5 - 0.5 * math.log(1.0), abs=0.01)


def test_gp_taus_grid():
    t = model.gp_taus()
    assert len(t) == 40 and t[0] == pytest.approx(0.01) and t[-1] == pytest.approx(10 ** 1.5)


def test_data_residual_by_hand(mini):
    r = model.data_residual(mini["a"], np.zeros((1, 3)), mini["b"], mini["sig"])
    assert r[0] == pytest.approx(math.sqrt(1.69 + 1.96 + 1.69) / (0.1 * math.sqrt(3)), abs=1e-12) and r[0] == pytest.approx(13.34, abs=0.01)
    assert model.data_residual(mini["a"], mini["x"], mini["b"], mini["sig"])[0] == pytest.approx(0.0, abs=1e-12)


def test_silent_directions_by_hand(mini):
    a, x, sig = mini["a"], mini["x"], mini["sig"]
    mask = model.silent_mask(a, x, sig)
    assert mask[0].tolist() == [False, True, True]
    assert model.null_share(a, x, mask)[0] == pytest.approx(0.6667 / 6, abs=1e-4) and model.null_share(a, x, mask)[0] == pytest.approx(1 / 9, abs=1e-12)
    assert model.null_cosine(a, x, x, mask)[0] == pytest.approx(1.0)


def test_null_cosine_of_an_inverted_silent_part_is_minus_one(mini):
    a, x, sig = mini["a"], mini["x"], mini["sig"]
    _, _, vt = np.linalg.svd(a)
    mask = model.silent_mask(a, x, sig)
    c = x @ vt.T
    flipped = np.where(mask, -c, c) @ vt
    assert model.null_cosine(a, flipped, x, mask)[0] == pytest.approx(-1.0, abs=1e-9)
    assert model.null_share(a, flipped, mask)[0] == pytest.approx(1 / 9, abs=1e-12)
    only_audible = np.where(mask, 0.0, c) @ vt
    assert model.null_share(a, only_audible, mask)[0] == pytest.approx(0.0, abs=1e-12)


def test_n_segments_by_hand():
    assert model.n_segments([0, 1, 1, 0, 0, 1]) == 2 and model.n_segments([0, 1, 0, 1, 0]) == 2 and model.n_segments([1, 1, 1]) == 1
    assert model.n_segments([1, 0, 0, 1]) == 2 and model.n_segments([0.4, 1, 0.4]) == 1 and model.n_segments([0.5, 1, 0.5]) == 1
    assert model.n_segments([0.6, 1, 0.6], frac=0.7) == 1 and model.n_segments([1, 0.6, 1], frac=0.7) == 2


def test_coverage_by_hand():
    assert model.coverage(np.array([0.0, 1.0, 2.0, 3.0]), np.array([0, 0, 3, 0]), np.array([1, 1, 4, 2])) == 0.5
    assert model.coverage(np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 1.0


def test_conformal_halfwidth_by_hand():
    s4 = np.array([[0.1], [0.3], [0.2], [0.4]])
    assert model.conformal_halfwidth(s4)[0] == pytest.approx(0.4)                       # ceil(5 · 0,9) = 5 > 4: der größte Wert
    assert model.conformal_halfwidth(np.arange(1.0, 20.0)[:, None])[0] == 18.0          # m = 19: ceil(20 · 0,9) = 18
    assert model.conformal_halfwidth(np.arange(1.0, 10.0)[:, None])[0] == 9.0           # m = 9: ceil(10 · 0,9) = 9
    assert model.conformal_halfwidth(np.arange(1.0, 20.0)[:, None], level=0.5)[0] == 10.0   # ceil(20 · 0,5) = 10


def test_mixture_posterior_and_evidence_by_hand():
    mix = model.Mixture([0.5, 0.5], [[-1.0], [1.0]], [[[0.04]], [[0.04]]])
    a, b = np.array([[1.0]]), np.array([0.1])
    post = mix.posterior(a, b, 0.2)
    assert post.w == pytest.approx([0.07586, 0.92414], abs=1e-5) and post.means.ravel() == pytest.approx([-0.45, 0.55]) and post.mean()[0] == pytest.approx(0.47414, abs=1e-5)
    e1, e2 = math.exp(-1.21 / 0.16) / math.sqrt(0.08), math.exp(-0.81 / 0.16) / math.sqrt(0.08)
    assert mix.log_evidence(a, b, 0.2) == pytest.approx(math.log(0.5 * e1 + 0.5 * e2), abs=1e-12)
    assert mix.mean()[0] == 0.0


def test_mixture_sample_follows_the_fixed_stream_exactly():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    g = model.Mixture([1.0], [[1.0, 2.0]], [cov])
    ev, q = np.linalg.eigh(cov)
    rng = SplitMix64(9)
    rng.uniform(3)
    z = rng.normal(6).reshape(3, 2)
    x, comp = g.sample(3, 9)
    assert comp.tolist() == [0, 0, 0] and x == pytest.approx(np.array([1.0, 2.0]) + (z * np.sqrt(ev)) @ q.T, abs=1e-12)


def test_posterior_weights_survive_far_data():
    mix = model.Mixture([0.5, 0.5], [[-1.0], [1.0]], [[[0.04]], [[0.04]]])
    post = mix.posterior(np.array([[1.0]]), np.array([1000.0]), 0.2)
    assert np.isfinite(post.w).all() and post.w == pytest.approx([0.0, 1.0], abs=1e-12)


def test_posterior_weights_with_different_covariances_by_hand():
    g = model.Mixture([0.5, 0.5], [[-1.0], [1.0]], [[[0.04]], [[0.16]]])
    post = g.posterior(np.array([[1.0]]), np.array([0.1]), 0.2)
    e1 = math.exp(-1.1 ** 2 / (2 * 0.08)) / math.sqrt(0.08)
    e2 = math.exp(-0.9 ** 2 / (2 * 0.20)) / math.sqrt(0.20)
    assert post.w == pytest.approx([e1 / (e1 + e2), e2 / (e1 + e2)], abs=1e-12)
    assert [c[0, 0] for c in post.cov] == pytest.approx([0.04 - 0.04 ** 2 / 0.08, 0.16 - 0.16 ** 2 / 0.20], abs=1e-12)


def test_templates_masses_and_cells():
    t = model.templates()
    assert {k: float(v.sum()) for k, v in t.items()} == {"one": 8.0, "two": 8.0, "three": 8.0}
    assert np.flatnonzero(t["one"]).tolist() == list(range(20, 28)) and np.flatnonzero(t["two"]).tolist() == [20, 21, 22, 25, 26, 27] and np.flatnonzero(t["three"]).tolist() == [20, 21, 23, 24, 26, 27]


def test_exp_cov_by_hand():
    c = model.exp_cov(4, 0.3, 0.12)
    assert np.diag(c) == pytest.approx([0.09 + 1e-10] * 4, rel=0, abs=1e-14)
    assert c[0, 1] == pytest.approx(0.09 * math.exp(-0.25 / 0.12), rel=0, abs=1e-14) and np.allclose(c, c.T)


def test_blockzahl_mixture_has_three_equal_components():
    mix = model.blockzahl_mixture()
    assert mix.w == pytest.approx([1 / 3] * 3) and mix.n == C.N and mix is model.blockzahl_mixture()


def test_draw_is_deterministic_and_family_specific():
    for fam in C.FAMILIES:
        x = model.draw(fam, 6, 3)
        assert x.shape == (6, C.N) and np.array_equal(x, model.draw(fam, 6, 3)) and not np.array_equal(x, model.draw(fam, 6, 4)) and np.isfinite(x).all(), fam
    assert (model.draw("spikes", 50, 1) >= 0).all() and (model.draw("blocks", 50, 1) >= 0).all() and model.draw("waves", 5, 1).max() <= 2.0
    spikes = model.draw("spikes", 200, 1)
    assert 1 <= np.min((spikes > 0).sum(axis=1)) and np.max((spikes > 0).sum(axis=1)) <= 4 and (spikes > 0).sum(axis=1).max() == 4
    assert model.draw("blockzahl", 4, 7) == pytest.approx(model.blockzahl_mixture().sample(4, 7)[0])


def test_gauss_family_has_the_prior_covariance():
    x = model.draw("gauss", 4000, 5)
    assert x.std() == pytest.approx(C.PRIOR_TAU, rel=0.05)
    assert np.corrcoef(x[:, 20], x[:, 21])[0, 1] == pytest.approx(math.exp(-(1 / C.N) / C.PRIOR_ELL), abs=0.04)


def test_make_data_uses_a_relative_sigma_and_the_fixed_stream():
    a = model.op()
    X = model.draw("blocks", 3, 5)
    B, sig = model.make_data(X, 0.1, 6, a)
    clean = X @ a.T
    assert sig == pytest.approx(0.1 * np.sqrt(np.mean(clean ** 2, axis=1)), rel=1e-12)
    assert (B - clean).ravel() == pytest.approx((sig[:, None] * SplitMix64(6).normal(clean.size).reshape(clean.shape)).ravel(), abs=1e-12)


def test_operator_is_cached_and_read_only():
    a = model.op()
    assert a is model.op() and a.shape == (C.N, C.N) and not a.flags.writeable and model.op(48, 0.05) is not a


def test_conformal_index_uses_m_plus_one():
    """m = 10, Niveau 0,9: ceil(11 · 0,9) = 10 (nicht ceil(10 · 0,9) = 9): der größte von zehn Werten."""
    assert model.conformal_halfwidth(np.arange(1.0, 11.0)[:, None])[0] == 10.0
    assert model.conformal_halfwidth(np.arange(1.0, 12.0)[:, None])[0] == 11.0           # m = 11: ceil(12 · 0,9) = ceil(10,8) = 11
    assert model.conformal_halfwidth(np.arange(1.0, 6.0)[:, None], level=0.6)[0] == 4.0   # m = 5: ceil(6 · 0,6) = ceil(3,6) = 4


def test_haar_size_check_names_the_divisibility():
    with pytest.raises(ValueError, match="teilbar"):
        model.haar_matrix(6, 2)
    assert model.haar_matrix(8, 3).shape == (8, 8)
    with pytest.raises(ValueError, match="teilbar"):
        model.haar_matrix(12, 3)


def test_segments_use_a_strict_threshold():
    """Ein Wert genau auf 50 % des Maximums gehört nicht zu einem Abschnitt: (1; 0,5; 1) hat zwei Abschnitte, (0,5; 1; 0,5) einen."""
    assert model.n_segments([1.0, 0.5, 1.0]) == 2 and model.n_segments([0.5, 1.0, 0.5]) == 1 and model.n_segments([1.0, 0.75, 1.0], frac=0.75) == 2


def test_a_direction_exactly_at_the_noise_level_is_audible():
    """Diagonale Matrix (1; 0,5), x = (0, 1): Richtung 1 hat c = 0 (stumm), Richtung 2 hat s |c| = 0,5; bei sigma = 0,5 ist sie nicht stumm (strikt kleiner), bei 0,5000001 schon."""
    a = np.diag([1.0, 0.5])
    x = np.array([[0.0, 1.0]])
    assert model.silent_mask(a, x, np.array([0.5]))[0].tolist() == [True, False]
    assert model.silent_mask(a, x, np.array([0.5000001]))[0].tolist() == [True, True]


def _fista_reference(a, b, rels, steps):
    """FISTA (Beck und Teboulle) mit Warmstart über absteigende lambda, Impulszähler läuft über alle lambda weiter: unabhängig aufgeschrieben, ein Problem."""
    t = 1.0 / np.linalg.norm(a, 2) ** 2
    lam_ref = np.abs(a.T @ b).max()
    x = np.zeros_like(b)
    z, tk = x.copy(), 1.0
    out = []
    for rel in rels:
        for _ in range(steps):
            xn = model.soft(z - t * (a.T @ (a @ z - b)), t * rel * lam_ref)
            tn = (1.0 + math.sqrt(1.0 + 4.0 * tk * tk)) / 2.0
            z = xn + (tk - 1.0) / tn * (xn - x)
            x, tk = xn, tn
        out.append(x.copy())
    return np.array(out)


def test_l1_path_follows_fista_with_momentum_and_warm_starts(mini):
    a, b, sig = mini["a"], mini["b"], mini["sig"]
    cands, _, _ = model.l1_path(a, b, sig, steps=5)
    rels = np.logspace(0, -C.SOFT_SPAN, C.SOFT_LAMBDAS)
    ref = _fista_reference(a, b[0], rels[:6], 5)
    assert cands[:6, 0] == pytest.approx(ref, abs=1e-12)
    assert not np.allclose(ref[3], _fista_reference(a, b[0], rels[3:4], 5)[0])           # der Warmstart zählt


def test_tv_path_runs_exactly_the_given_number_of_admm_steps(mini):
    """Ein Schritt ADMM aus Z = U = 0: x = (A^T A + rho D^T D)^-1 A^T b (rho = 1)."""
    a, b, sig = mini["a"], mini["b"], mini["sig"]
    dm = np.diff(np.eye(3), axis=0)
    one, _, _ = model.tv_path(a, b, sig, steps=1, count=1)
    assert one[0, 0] == pytest.approx(np.linalg.solve(a.T @ a + dm.T @ dm, a.T @ b[0]), abs=1e-12)
    two, _, _ = model.tv_path(a, b, sig, steps=2, count=1)
    assert not np.allclose(one, two)


def test_mixture_evidence_is_stable_for_far_data():
    mix = model.Mixture([0.5, 0.5], [[-1.0], [1.0]], [[[0.04]], [[0.04]]])
    ev = mix.log_evidence(np.array([[1.0]]), np.array([1000.0]), 0.2)
    s = 0.04 + 0.04
    expected = math.log(0.5) - 0.5 * (999.0 ** 2) / s - 0.5 * math.log(s)
    assert math.isfinite(ev) and ev == pytest.approx(expected, rel=1e-12)


def test_wave_family_by_hand_including_the_phase_sign():
    """n = 4: s = (0,125; 0,375; 0,625; 0,875), 3 Perioden (u = 0), Phase 0,25: (3 s + 0,25) mod 1 = (0,625; 0,375; 0,125; 0,875), |· - 0,5| · 2 = (0,25; 0,25; 0,75; 0,75)."""
    assert model._waves(np.array([0.0, 0.25]), 4) == pytest.approx([0.25, 0.25, 0.75, 0.75], abs=1e-12)
    assert model._waves(np.array([0.0, 0.25]), 4).tolist() != model._waves(np.array([0.0, 0.0]), 4).tolist()
