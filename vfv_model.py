"""Kern: Wahrheitsfamilien, acht Verfahren mit ihren Parameterregeln (Orakel, Diskrepanzprinzip, GCV, Evidenz), Kennzahlen (Nullraum, Abschnittszahl, Abdeckung) und Unsicherheitsbänder
(GP-Posterior, Mischungs-Posterior, konformal). Nur numpy, kein Streamlit, kein scipy.

Jede Funktion ist eine kleine, einzeln prüfbare Einheit (DEMO-PLAYBOOK §1b): `tests/test_model.py` rechnet sie an der 3 x 3-Mini-Instanz von Hand, `tests/test_oracle_vfv.py` gegen
Normalgleichung, Krylov-Raum, KKT-Bedingungen, die exakte L1-Lösung, die direkte Evidenz und die verteilungsfreie Garantie der konformalen Vorhersage.

Zeilenkonvention: B (m, n) Daten, X (m, n) Schätzungen und Wahrheiten, A (n, n) quadratisch. Jeder Pfad (`*_path`) gibt Kandidaten (K, m, n) von **stark nach schwach regularisiert**, die
Reste |A x - b| (K, m) und, wenn es eine geschlossene Form gibt, die GCV-Werte (K, m) zurück; `choose` wählt daraus nach Regel.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

import vfv_constants as C
from inverse_generator import SplitMix64, grid, operator

_TINY = 1e-300


def soft(v, t):
    """Weiche Schwelle sign(v) max(|v| - t, 0)."""
    v = np.asarray(v, dtype=float)
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


@lru_cache(maxsize=4)
def _op(n: int, w: float) -> np.ndarray:
    a = operator(n, w)
    a.flags.writeable = False
    return a


def op(n: int = C.N, w: float = C.W) -> np.ndarray:
    """Der Operator des Generators, zwischengespeichert."""
    return _op(int(n), round(float(w), 6))


def rel_err(X, T):
    """Relativer Fehler |x - t| / |t| je Zeile."""
    return np.linalg.norm(np.asarray(X) - T, axis=-1) / np.linalg.norm(T, axis=-1)


# ----------------------------------------------------------------------------- Wahrheitsfamilien und Daten

def exp_cov(n: int, tau: float, ell: float) -> np.ndarray:
    """tau² exp(-|s_i - s_j| / ell) + 1e-10 I auf dem Gitter (der Zuschlag hält die Zerlegungen stabil)."""
    s = grid(n)
    return tau ** 2 * np.exp(-np.abs(s[:, None] - s[None, :]) / ell) + 1e-10 * np.eye(n)


def _spikes(r, n):
    x = np.zeros(n)
    for j in range(C.SPIKE_COUNT[0] + int(r[0] * C.SPIKE_COUNT[1])):
        x[int(r[1 + 2 * j] * n) % n] += C.SPIKE_HEIGHT[0] + C.SPIKE_HEIGHT[1] * r[2 + 2 * j]
    return x


def _blocks(r, n):
    x = np.zeros(n)
    for j in range(C.BLOCK_COUNT[0] + int(r[0] * C.BLOCK_COUNT[1])):
        w = C.BLOCK_WIDTH[0] + int(r[1 + 3 * j] * C.BLOCK_WIDTH[1])
        left = int(r[2 + 3 * j] * (n - w))
        x[left:left + w] += C.BLOCK_HEIGHT[0] + C.BLOCK_HEIGHT[1] * r[3 + 3 * j]
    return x


def _waves(r, n):
    s = grid(n)
    per = C.WAVE_PERIODS[0] + int(r[0] * C.WAVE_PERIODS[1])
    lo, hi = C.WAVE_RANGE
    return np.abs((per * s + r[1]) % 1.0 - 0.5) * 2.0 * ((s > lo) & (s < hi))


def _phantom(r, n):
    s = grid(n)
    x = np.zeros(n)
    for j in range(2):
        c = C.PHANTOM_CENTER[0] + C.PHANTOM_CENTER[1] * r[3 * j]
        wd = C.PHANTOM_WIDTH[0] + C.PHANTOM_WIDTH[1] * r[3 * j + 1]
        h = C.PHANTOM_HEIGHT[0] + C.PHANTOM_HEIGHT[1] * r[3 * j + 2]
        x += h * np.exp(-((s - c) ** 2) / (2.0 * wd * wd))
    left = C.PHANTOM_BOX_LEFT[0] + C.PHANTOM_BOX_LEFT[1] * r[6]
    return x + (C.PHANTOM_BOX_HEIGHT[0] + C.PHANTOM_BOX_HEIGHT[1] * r[7]) * ((s >= left) & (s < left + C.PHANTOM_BOX_WIDTH))


_DRAW = {"spikes": _spikes, "blocks": _blocks, "waves": _waves, "phantom": _phantom}


def templates(n: int = C.N) -> dict:
    """Die drei Schablonen der Blockzahl (alle Masse 8): ein Block (Zellen 20 bis 27), zwei Blöcke (3 + Lücke 2 + 3), drei Blöcke (2 + 1 + 2 + 1 + 2)."""
    one, two, three = np.zeros(n), np.zeros(n), np.zeros(n)
    one[C.ONE_BLOCK[0]:C.ONE_BLOCK[1]] = C.MIX_MASS / (C.ONE_BLOCK[1] - C.ONE_BLOCK[0])
    for lo, hi in C.TWO_BLOCKS:
        two[lo:hi] = C.MIX_MASS / C.BLOCK_CELLS_MULTI
    for lo, hi in C.THREE_BLOCKS:
        three[lo:hi] = C.MIX_MASS / C.BLOCK_CELLS_MULTI
    return {"one": one, "two": two, "three": three}


class Mixture:
    """Gaußmischung sum_k w_k N(mean_k, cov_k) als Prior: exakter Posterior bei linearer Messung (Kalman je Komponente, Gewichte nach Evidenz), Evidenz, Ziehungen."""

    def __init__(self, weights, means, covs):
        self.w = np.asarray(weights, dtype=float)
        self.means = np.atleast_2d(np.asarray(means, dtype=float))
        self.cov = [np.atleast_2d(np.asarray(c, dtype=float)) for c in covs]
        self.n = self.means.shape[1]

    def mean(self) -> np.ndarray:
        return (self.w[:, None] * self.means).sum(axis=0)

    def posterior(self, a: np.ndarray, b: np.ndarray, sig: float) -> "Mixture":
        """Posterior bei b = A x + e, e ~ N(0, sig² I): Gewichte nach Evidenz N(b; A mean_k, A cov_k A^T + sig² I), Mittel und Kovarianz nach Kalman."""
        lw, means, covs = [], [], []
        for k in range(len(self.w)):
            c = self.cov[k]
            sm = a @ c @ a.T + sig ** 2 * np.eye(a.shape[0])
            r = b - a @ self.means[k]
            lw.append(math.log(self.w[k]) - 0.5 * float(r @ np.linalg.solve(sm, r)) - 0.5 * float(np.linalg.slogdet(sm)[1]))
            gain = c @ a.T @ np.linalg.inv(sm)
            means.append(self.means[k] + gain @ r)
            g = c - gain @ a @ c
            covs.append(0.5 * (g + g.T))
        lw = np.array(lw)
        w = np.exp(lw - lw.max())
        return Mixture(w / w.sum(), means, covs)

    def log_evidence(self, a: np.ndarray, b: np.ndarray, sig: float) -> float:
        """log p(b) = log sum_k w_k N(b; A mean_k, A cov_k A^T + sig² I) (bis auf die Konstante -n/2 log 2 pi, die nur Vergleiche zwischen Priors kürzt)."""
        lw = []
        for k in range(len(self.w)):
            sm = a @ self.cov[k] @ a.T + sig ** 2 * np.eye(a.shape[0])
            r = b - a @ self.means[k]
            lw.append(math.log(self.w[k]) - 0.5 * float(r @ np.linalg.solve(sm, r)) - 0.5 * float(np.linalg.slogdet(sm)[1]))
        mx = max(lw)
        return mx + math.log(sum(math.exp(v - mx) for v in lw))

    def sample(self, count: int, seed: int):
        """(count, n) Ziehungen und ihre Komponenten: erst die Komponente (kumulierte Gewichte), dann Normalverteilte über die Eigenzerlegung der Kovarianz."""
        rng = SplitMix64(seed)
        comp = np.minimum(np.searchsorted(np.cumsum(self.w), rng.uniform(count), side="right"), len(self.w) - 1)
        z = rng.normal(count * self.n).reshape(count, self.n)
        out = np.zeros((count, self.n))
        for k in range(len(self.w)):
            idx = comp == k
            ev, q = np.linalg.eigh(self.cov[k])
            out[idx] = self.means[k] + (z[idx] * np.sqrt(np.maximum(ev, 0.0))) @ q.T
        return out, comp


@lru_cache(maxsize=2)
def blockzahl_mixture(n: int = C.N) -> Mixture:
    """Drei Komponenten mit gleichen Gewichten (ein, zwei, drei Blöcke), Kovarianz tau² C (Stück 15)."""
    t = templates(n)
    cov = exp_cov(n, C.MIX_TAU, C.MIX_ELL)
    return Mixture([1.0 / 3.0] * 3, [t["one"], t["two"], t["three"]], [cov, cov, cov])


def draw(family: str, count: int, seed: int, n: int = C.N) -> np.ndarray:
    """(count, n) Wahrheiten der Familie aus dem festen Strom `seed`."""
    if family == "blockzahl":
        return blockzahl_mixture(n).sample(count, seed)[0]
    rng = SplitMix64(seed)
    if family == "gauss":
        return rng.normal(count * n).reshape(count, n) @ np.linalg.cholesky(exp_cov(n, C.PRIOR_TAU, C.PRIOR_ELL)).T
    u = rng.uniform(count * C.DRAW_WIDTH).reshape(count, C.DRAW_WIDTH)
    return np.array([_DRAW[family](u[i], n) for i in range(count)])


def make_data(X: np.ndarray, noise: float, seed: int, a: np.ndarray | None = None):
    """(B, sigma): Daten B = X A^T + e, absolutes Rauschen je Signal sigma_i = noise mal Effektivwert der exakten Daten."""
    a = op() if a is None else a
    clean = X @ a.T
    sig = noise * np.sqrt(np.mean(clean ** 2, axis=1))
    return clean + SplitMix64(seed).normal(clean.size).reshape(clean.shape) * sig[:, None], sig


# ----------------------------------------------------------------------------- Verfahren: Kandidaten (K, m, n) von stark nach schwach regularisiert

def tikhonov_path(a, B, sig):
    """Tikhonov per SVD: lambda = s_max² · 10^(0 ... -LAMBDA_SPAN); gibt (Kandidaten, Reste, GCV) mit GCV(lambda) = n r² / (n - sum f)², f = s² / (s² + lambda)."""
    u, s, vt = np.linalg.svd(a)
    n = len(s)
    beta = B @ u
    c = beta / s
    cands, res, gcv = [], [], []
    for lam in s[0] ** 2 * np.logspace(0, -C.LAMBDA_SPAN, C.LAMBDA_COUNT):
        f = s ** 2 / (s ** 2 + lam)
        cands.append((c * f) @ vt)
        r2 = (((1 - f) * beta) ** 2).sum(axis=1)
        res.append(np.sqrt(r2))
        gcv.append(n * r2 / (n - f.sum()) ** 2)
    return np.array(cands), np.array(res), np.array(gcv)


def tsvd_path(a, B, sig):
    """Abgeschnittene SVD mit k = 1 ... n Komponenten; GCV(k) = n r² / (n - k)² (unendlich bei k = n)."""
    u, s, vt = np.linalg.svd(a)
    n = len(s)
    beta = B @ u
    c = beta / s
    cands, res, gcv = [], [], []
    for k in range(1, n + 1):
        cands.append(c[:, :k] @ vt[:k])
        r2 = (beta[:, k:] ** 2).sum(axis=1)
        res.append(np.sqrt(r2))
        gcv.append(n * r2 / (n - k) ** 2 if k < n else np.full(len(B), np.inf))
    return np.array(cands), np.array(res), np.array(gcv)


def cgls_path(a, B, sig, iterations: int | None = None):
    """CGLS (Start 0): die Iterierten 1 ... K; der Abbruch ist die Regularisierung."""
    iterations = min(C.CGLS_ITERATIONS, a.shape[1]) if iterations is None else iterations
    x = np.zeros_like(B)
    r = B.copy()
    s = r @ a
    p = s.copy()
    gam = (s ** 2).sum(axis=1)
    cands, res = [], []
    for _ in range(iterations):
        q = p @ a.T
        alpha = gam / np.maximum((q ** 2).sum(axis=1), _TINY)
        x = x + alpha[:, None] * p
        r = r - alpha[:, None] * q
        s = r @ a
        gnew = (s ** 2).sum(axis=1)
        p = s + (gnew / np.maximum(gam, _TINY))[:, None] * p
        gam = gnew
        cands.append(x.copy())
        res.append(np.linalg.norm(r, axis=1))
    return np.array(cands), np.array(res), None


def haar_matrix(n: int = C.N, levels: int = C.HAAR_LEVELS) -> np.ndarray:
    """Orthonormale Haar-Transformation W (n, n) mit `levels` Ebenen (n durch 2^levels teilbar): Koeffizienten = W x."""
    if n % (2 ** levels):
        raise ValueError("n muss durch 2^levels teilbar sein")

    def fwd(v):
        out = v.copy()
        length = n
        for _ in range(levels):
            a = (out[:length:2] + out[1:length:2]) / math.sqrt(2.0)
            d = (out[:length:2] - out[1:length:2]) / math.sqrt(2.0)
            out[:length // 2], out[length // 2:length] = a, d
            length //= 2
        return out
    return np.array([fwd(e) for e in np.eye(n)]).T


def _soft_path(a, B, transform, steps, count=C.SOFT_LAMBDAS):
    """Weichschwelle (ggf. im Transformationsbereich) mit FISTA, warm gestartet über absteigende lambda (relativ zu |A^T b|_inf, 10^(0 ... -SOFT_SPAN))."""
    lam_ref = np.abs(B @ a).max(axis=1)
    t = 1.0 / float(np.linalg.norm(a, 2) ** 2)
    X = np.zeros_like(B)
    Z, tk = X.copy(), 1.0
    cands, res = [], []
    for rel in np.logspace(0, -C.SOFT_SPAN, count):
        lam = (rel * lam_ref)[:, None]
        for _ in range(steps):
            Y = Z - t * ((Z @ a.T - B) @ a)
            Xn = soft(Y, t * lam) if transform is None else soft(Y @ transform.T, t * lam) @ transform
            tn = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * tk * tk))
            Z = Xn + ((tk - 1.0) / tn) * (Xn - X)
            X, tk = Xn, tn
        cands.append(X.copy())
        res.append(np.linalg.norm(X @ a.T - B, axis=1))
    return np.array(cands), np.array(res), None


def l1_path(a, B, sig, steps: int = C.SOFT_STEPS):
    return _soft_path(a, B, None, steps)


def pnp_path(a, B, sig, steps: int = C.SOFT_STEPS):
    """Plug-and-Play mit dem Haar-Denoiser (Weichschwelle der Haar-Koeffizienten, ohne Zyklenverschiebung)."""
    return _soft_path(a, B, haar_matrix(a.shape[1], min(C.HAAR_LEVELS, int(math.log2(a.shape[1] & -a.shape[1])))), steps)


def tv_path(a, B, sig, steps: int = C.TV_STEPS, count: int = C.TV_LAMBDAS, rho: float = C.TV_RHO):
    """Total Variation min 1/2 |A x - b|² + lambda |D x|_1 (D = Vorwärtsdifferenz) per ADMM über absteigende lambda (relativ zu |A^T b|_inf, 10^(-TV_START ... -TV_SPAN))."""
    n = a.shape[1]
    dm = np.diff(np.eye(n), axis=0)
    minv = np.linalg.inv(a.T @ a + rho * dm.T @ dm)
    lam_ref = np.abs(B @ a).max(axis=1)
    atb = B @ a
    X = np.zeros_like(B)
    Zv = np.zeros((len(B), n - 1))
    Uv = np.zeros_like(Zv)
    cands, res = [], []
    for rel in np.logspace(-C.TV_START, -C.TV_SPAN, count):
        lam = (rel * lam_ref)[:, None]
        for _ in range(steps):
            X = (atb + rho * (Zv - Uv) @ dm) @ minv.T
            dx = X @ dm.T
            Zv = soft(dx + Uv, lam / rho)
            Uv = Uv + dx - Zv
        cands.append(X.copy())
        res.append(np.linalg.norm(X @ a.T - B, axis=1))
    return np.array(cands), np.array(res), None


PATHS = {"tikhonov": tikhonov_path, "tsvd": tsvd_path, "cgls": cgls_path, "l1": l1_path, "tv": tv_path, "pnp": pnp_path}


def choose(cands, res, sig, X=None, gcv=None, tau: float = C.DISCREPANCY_TAU):
    """Wählt aus den Kandidaten (K, m, n): `discrepancy` den am stärksten regularisierten mit Rest <= tau sigma sqrt(n) (keinen: den schwächsten), `gcv` das Minimum der GCV-Werte
    (nur wenn gegeben), `oracle` den mit dem kleinsten Fehler gegen die Wahrheit X (nur wenn gegeben)."""
    K, m, n = cands.shape
    idx = np.arange(m)
    out = {}
    ok = res <= tau * sig[None, :] * math.sqrt(n)
    first = np.where(ok.any(axis=0), ok.argmax(axis=0), K - 1)
    out["discrepancy"] = cands[first, idx]
    if gcv is not None:
        out["gcv"] = cands[gcv.argmin(axis=0), idx]
    if X is not None:
        out["oracle"] = cands[rel_err(cands, X[None]).argmin(axis=0), idx]
    return out


# ----------------------------------------------------------------------------- GP-Prior (exponentielle Korrelation) mit Evidenz

def gp_taus() -> np.ndarray:
    lo, hi, count = C.GP_TAU_GRID
    return np.logspace(lo, hi, int(count))


def gp_log_evidence(a, B, sig, tau, cmat):
    """Log-Evidenz je Zeile (bis auf die Konstante): -1/2 b^T S^-1 b - 1/2 log det S, S = tau² A C A^T + sig² I; tau skalar oder je Zeile."""
    ge, gq = np.linalg.eigh(a @ cmat @ a.T)
    ge = np.maximum(ge, 0.0)
    beta = B @ gq
    tau = np.broadcast_to(np.asarray(tau, dtype=float), (len(B),))
    var = tau[:, None] ** 2 * ge[None, :] + sig[:, None] ** 2
    return -0.5 * (beta ** 2 / var).sum(axis=1) - 0.5 * np.log(var).sum(axis=1)


def gp_best_tau(a, B, sig, cmat):
    """tau mit der größten Evidenz (Maximum-Likelihood der Hyperparameter, ML-II) je Zeile und diese Evidenz."""
    taus = gp_taus()
    ev = np.array([gp_log_evidence(a, B, sig, t, cmat) for t in taus])
    k = ev.argmax(axis=0)
    return taus[k], ev[k, np.arange(len(B))]


def gp_mean(a, B, sig, tau, cmat):
    """Posterior-Mittel tau² C A^T (tau² A C A^T + sig² I)^-1 b je Zeile."""
    ge, gq = np.linalg.eigh(a @ cmat @ a.T)
    ge = np.maximum(ge, 0.0)
    beta = B @ gq
    tau = np.broadcast_to(np.asarray(tau, dtype=float), (len(B),))
    w = tau[:, None] ** 2 / (tau[:, None] ** 2 * ge[None, :] + sig[:, None] ** 2)
    return ((beta * w) @ gq.T) @ (a @ cmat)


def gp_posterior(a, b, sig, tau, cmat):
    """(Mittel, Kovarianz) des Posterior für eine Zeile."""
    s_ = tau ** 2 * (a @ cmat @ a.T) + sig ** 2 * np.eye(a.shape[0])
    gain = tau ** 2 * cmat @ a.T @ np.linalg.inv(s_)
    return gain @ b, tau ** 2 * cmat - gain @ a @ (tau ** 2 * cmat)


# ----------------------------------------------------------------------------- Kennzahlen

def data_residual(a, X, B, sig):
    """|A x - b| / (sig sqrt(n)) je Zeile: bei einer Probe aus dem Posterior etwa 1."""
    return np.linalg.norm(X @ a.T - B, axis=1) / (sig * math.sqrt(a.shape[0]))


def silent_mask(a, X, sig):
    """(m, n) Wahrheitswert: Richtung i (rechter Singularvektor) ist stumm, wenn das Signal dort unter dem Rauschen liegt, s_i |v_i^T x| < sig."""
    _, s, vt = np.linalg.svd(a)
    return s[None, :] * np.abs(X @ vt.T) < sig[:, None]


def null_share(a, X, mask):
    """Anteil der Energie von X in den stummen Richtungen (Maske je Zeile) an der gesamten Energie."""
    _, _, vt = np.linalg.svd(a)
    c = X @ vt.T
    return (np.where(mask, c, 0.0) ** 2).sum(axis=1) / np.maximum((c ** 2).sum(axis=1), _TINY)


def null_cosine(a, X, T, mask):
    """Kosinus der Anteile von Schätzung X und Wahrheit T in den stummen Richtungen."""
    _, _, vt = np.linalg.svd(a)
    ce, ct = np.where(mask, X @ vt.T, 0.0), np.where(mask, T @ vt.T, 0.0)
    return (ce * ct).sum(axis=1) / np.maximum(np.linalg.norm(ce, axis=1) * np.linalg.norm(ct, axis=1), _TINY)


def n_segments(x, frac: float = C.SEG_FRAC) -> int:
    """Zahl der zusammenhängenden Abschnitte, in denen x über frac mal Maximum liegt."""
    above = np.asarray(x) > frac * np.max(x)
    return int(above[0]) + int(np.sum(above[1:] & ~above[:-1]))


def coverage(x_true, lo, hi) -> float:
    """Anteil der Koordinaten der Wahrheit im Band [lo, hi]."""
    return float(np.mean((x_true >= lo) & (x_true <= hi)))


def conformal_halfwidth(scores: np.ndarray, level: float = C.LEVEL) -> np.ndarray:
    """Halbbreite je Koordinate aus den absoluten Fehlern (m, n) der Kalibrierung: der ceil((m + 1) level)-kleinste Wert (ist das größer als m: der größte)."""
    m = len(scores)
    k = math.ceil((m + 1) * level)
    srt = np.sort(scores, axis=0)
    return srt[-1] if k > m else srt[k - 1]
