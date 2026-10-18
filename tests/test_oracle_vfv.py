"""Orakel-Tests: Verfahren und Kennzahlen gegen unabhängige Rechnungen (Normalgleichung, Krylov-Raum, exakte L1-Lösung, KKT-Bedingungen, direkte Dichte, verteilungsfreie Garantie)."""

import math

import numpy as np
import pytest

import vfv_constants as C
import vfv_model as model


def _data(family="blocks", noise=0.05, count=3, seed=4):
    a = model.op()
    X = model.draw(family, count, seed)
    B, sig = model.make_data(X, noise, seed + 1, a)
    return a, X, B, sig


def test_tikhonov_matches_the_normal_equations():
    a, X, B, sig = _data()
    cands, res, _ = model.tikhonov_path(a, B, sig)
    lams = np.logspace(0, -C.LAMBDA_SPAN, C.LAMBDA_COUNT) * np.linalg.norm(a, 2) ** 2
    for k in (2, 10, 20, 30):
        for i in range(len(B)):
            x = np.linalg.solve(a.T @ a + lams[k] * np.eye(C.N), a.T @ B[i])
            assert cands[k, i] == pytest.approx(x, abs=1e-8 * max(1.0, np.abs(x).max()), rel=1e-6)
            assert res[k, i] == pytest.approx(np.linalg.norm(a @ x - B[i]), rel=1e-6, abs=1e-8)


def test_gcv_equals_the_explicit_influence_matrix_formula():
    a, X, B, sig = _data()
    _, _, gcv = model.tikhonov_path(a, B, sig)
    lams = np.logspace(0, -C.LAMBDA_SPAN, C.LAMBDA_COUNT) * np.linalg.norm(a, 2) ** 2
    for k in (3, 12, 25):
        infl = a @ np.linalg.solve(a.T @ a + lams[k] * np.eye(C.N), a.T)
        for i in range(len(B)):
            r = (np.eye(C.N) - infl) @ B[i]
            assert gcv[k, i] == pytest.approx(C.N * (r @ r) / (C.N - np.trace(infl)) ** 2, rel=1e-6)


def test_tsvd_with_all_components_is_the_inverse_and_matches_the_pseudoinverse_in_between():
    a, X, B, sig = _data()
    cands, res, _ = model.tsvd_path(a, B, sig)
    for i in range(len(B)):
        assert cands[-1, i] == pytest.approx(np.linalg.solve(a, B[i]), abs=1e-6)
        u, s, vt = np.linalg.svd(a)
        k = 12
        assert cands[k - 1, i] == pytest.approx(vt[:k].T @ ((u[:, :k].T @ B[i]) / s[:k]), abs=1e-9)
        assert res[k - 1, i] == pytest.approx(np.linalg.norm(a @ cands[k - 1, i] - B[i]), rel=1e-6, abs=1e-9)


def test_cgls_iterate_is_the_least_squares_solution_over_the_krylov_space():
    a, X, B, sig = _data("waves", 0.02, 2)
    cands, res, _ = model.cgls_path(a, B, sig, iterations=4)
    for i in range(len(B)):
        v = a.T @ B[i]
        basis = [v]
        for _ in range(3):
            basis.append(a.T @ (a @ basis[-1]))
        for k in (1, 2, 3, 4):
            kmat = np.array(basis[:k]).T
            coef, *_ = np.linalg.lstsq(a @ kmat, B[i], rcond=None)
            assert cands[k - 1, i] == pytest.approx(kmat @ coef, rel=1e-5, abs=1e-6 * np.abs(kmat @ coef).max())


def _lasso_cd(a, b, lam, sweeps=30000):
    """Koordinatenabstieg für min 1/2 |A x - b|² + lam |x|_1 (Referenz, bis auf Maschinengenauigkeit konvergiert)."""
    n = a.shape[1]
    x = np.zeros(n)
    col2 = (a ** 2).sum(axis=0)
    r = b - a @ x
    for _ in range(sweeps):
        old = x.copy()
        for j in range(n):
            rho = a[:, j] @ r + col2[j] * x[j]
            new = np.sign(rho) * max(abs(rho) - lam, 0.0) / col2[j]
            r += a[:, j] * (x[j] - new)
            x[j] = new
        if np.abs(x - old).max() < 1e-13:
            break
    return x


def test_l1_matches_the_exact_lasso_on_a_well_posed_matrix():
    a = np.eye(8) + 0.3 * (np.eye(8, k=1) + np.eye(8, k=-1))
    rng = np.random.default_rng(3)
    b = (a @ np.array([0, 2.0, 0, 0, -1.5, 0, 0.5, 0]) + 0.02 * rng.standard_normal(8))[None, :]
    cands, _, _ = model.l1_path(a, b, np.array([0.02]), steps=1500)
    rels = np.logspace(0, -C.SOFT_SPAN, C.SOFT_LAMBDAS)
    for k in (6, 12, 18):
        lam = rels[k] * np.abs(a.T @ b[0]).max()
        assert cands[k, 0] == pytest.approx(_lasso_cd(a, b[0], lam), abs=1e-6)


def test_tv_solution_satisfies_the_kkt_conditions():
    a = np.eye(8) + 0.2 * (np.eye(8, k=1) + np.eye(8, k=-1))
    rng = np.random.default_rng(5)
    b = (a @ np.array([0, 0, 1.0, 1.0, 1.0, 0, 0, 0]) + 0.05 * rng.standard_normal(8))[None, :]
    cands, _, _ = model.tv_path(a, b, np.array([0.05]), steps=4000, count=15)
    lam_rel = np.logspace(-C.TV_START, -C.TV_SPAN, 15)
    dm = np.diff(np.eye(8), axis=0)
    for k in (3, 7, 11):
        x = cands[k, 0]
        lam = lam_rel[k] * np.abs(b[0] @ a).max()
        r = a.T @ (a @ x - b[0])
        g, *_ = np.linalg.lstsq(dm.T, -r / lam, rcond=None)
        assert np.abs(dm.T @ g + r / lam).max() < 2e-3                    # Dᵀ g = -r / lam lösbar (die Summe von r verschwindet)
        u = dm @ x
        assert np.abs(g).max() <= 1.0 + 2e-3
        active = np.abs(u) > 1e-3
        assert np.allclose(g[active], np.sign(u[active]), atol=2e-3)


def test_haar_matrix_is_orthonormal_and_pnp_is_the_haar_analysis_prox():
    w = model.haar_matrix()
    assert w @ w.T == pytest.approx(np.eye(C.N), abs=1e-12)
    x = np.random.default_rng(1).standard_normal(C.N)
    assert np.linalg.norm(w @ x) == pytest.approx(np.linalg.norm(x))
    a, X, B, sig = _data()
    step = 1.0 / np.linalg.norm(a, 2) ** 2
    lam = np.abs(B @ a).max(axis=1)[:, None] * 1.0
    y = step * (B @ a)
    first = model.soft(y @ w.T, step * lam) @ w
    one, _, _ = model.pnp_path(a, B, sig, steps=1)
    assert one[0] == pytest.approx(first, abs=1e-12)


def _dense_logdensity(b, cov):
    return -0.5 * b @ np.linalg.solve(cov, b) - 0.5 * np.linalg.slogdet(cov)[1]


def test_gp_evidence_and_mean_match_the_dense_formulas():
    a, X, B, sig = _data("gauss", 0.1, 2)
    cm = model.exp_cov(C.N, 1.0, C.PRIOR_ELL)
    tau = 0.4
    ev = model.gp_log_evidence(a, B, sig, tau, cm)
    mean = model.gp_mean(a, B, sig, tau, cm)
    for i in range(len(B)):
        s_ = tau ** 2 * a @ cm @ a.T + sig[i] ** 2 * np.eye(C.N)
        assert ev[i] == pytest.approx(_dense_logdensity(B[i], s_), rel=1e-9)
        assert mean[i] == pytest.approx(tau ** 2 * cm @ a.T @ np.linalg.solve(s_, B[i]), abs=1e-8)
        m2, cov = model.gp_posterior(a, B[i], sig[i], tau, cm)
        assert m2 == pytest.approx(mean[i], abs=1e-8) and np.allclose(cov, cov.T, atol=1e-10) and np.linalg.eigvalsh(0.5 * (cov + cov.T)).min() > -1e-9


def test_a_one_component_mixture_is_the_gp_posterior_and_has_its_evidence():
    a, X, B, sig = _data("gauss", 0.1, 2)
    tau = 0.5
    cm = model.exp_cov(C.N, 1.0, C.PRIOR_ELL)
    mix = model.Mixture([1.0], [np.zeros(C.N)], [tau ** 2 * cm])
    for i in range(len(B)):
        post = mix.posterior(a, B[i], sig[i])
        mean, cov = model.gp_posterior(a, B[i], sig[i], tau, cm)
        assert post.mean() == pytest.approx(mean, abs=1e-8) and post.cov[0] == pytest.approx(cov, abs=1e-8)
        assert mix.log_evidence(a, B[i], sig[i]) == pytest.approx(model.gp_log_evidence(a, B[i:i + 1], sig[i:i + 1], tau, cm)[0], rel=1e-9)


def test_mixture_evidence_is_the_log_of_the_weighted_densities():
    a, X, B, sig = _data("blockzahl", 0.1, 2)
    mix = model.blockzahl_mixture()
    for i in range(len(B)):
        parts = []
        for k in range(3):
            s_ = a @ mix.cov[k] @ a.T + sig[i] ** 2 * np.eye(C.N)
            parts.append(math.log(1 / 3) + _dense_logdensity(B[i] - a @ mix.means[k], s_))
        assert mix.log_evidence(a, B[i], sig[i]) == pytest.approx(math.log(sum(math.exp(p - max(parts)) for p in parts)) + max(parts), rel=1e-9)


def test_conformal_halfwidth_gives_the_distribution_free_guarantee_and_breaks_under_shift():
    rng = np.random.default_rng(7)
    trials, m = 20000, 19
    cal = np.abs(rng.standard_normal((m, trials)))
    test = np.abs(rng.standard_normal(trials))
    q = np.sort(cal, axis=0)[math.ceil((m + 1) * 0.9) - 1]
    assert np.mean(test <= q) == pytest.approx(0.9, abs=0.01)                     # exakt 18/20 = 0,9
    # die Halbbreite aus model.conformal_halfwidth ist dieselbe Ordnungsstatistik
    scores = np.abs(rng.standard_normal((m, 1)))
    assert model.conformal_halfwidth(scores)[0] == np.sort(scores[:, 0])[17]
    shifted = 2.0 * np.abs(rng.standard_normal(trials))                           # Familienwechsel: doppelt so große Fehler
    assert np.mean(shifted <= q) < 0.75


def test_silent_directions_for_unit_vectors_of_the_svd():
    a = model.op()
    _, s, vt = np.linalg.svd(a)
    sig = np.array([1e-3])
    loud, quiet = vt[0][None, :], vt[-1][None, :]
    assert not model.silent_mask(a, loud, sig)[0, 0] and model.silent_mask(a, quiet, sig)[0, -1]
    mask = model.silent_mask(a, np.vstack([loud, quiet]), np.array([1e-3, 1e-3]))
    assert model.null_share(a, np.vstack([loud, quiet]), mask) == pytest.approx([0.0, 1.0], abs=1e-12)
    assert s[-1] < 1e-3 * 1.0 and s[0] == pytest.approx(1.0, abs=0.01)


def test_the_discrepancy_principle_leaves_a_residual_of_about_one_sigma():
    """Kalibrierung des Prinzips an der Welt: Datenrest von Tikhonov mit Diskrepanz liegt im Median zwischen 0,7 und 1,05 sigma (1 % und 10 % Rauschen)."""
    a = model.op()
    for noise in (0.01, 0.1):
        X = model.draw("phantom", 40, 8)
        B, sig = model.make_data(X, noise, 9, a)
        cands, res, _ = model.tikhonov_path(a, B, sig)
        est = model.choose(cands, res, sig)["discrepancy"]
        assert 0.7 < np.median(model.data_residual(a, est, B, sig)) <= 1.05
