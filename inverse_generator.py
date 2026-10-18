"""Gemeinsamer Generator der Linie "Inverse Probleme und Regularisierung" (WORTGLEICHE Kopie in allen Repos der Linie, kein Modul-Präfix).

Version 1 (2026-10-14). Nur numpy und math, kein Streamlit, kein scipy. Änderungen an diesem Modul betreffen alle Repos der Linie: Schnittstelle,
Namen und Reihenfolge des Zufallsverbrauchs nur ändern, wenn alle Kopien nachgezogen werden (tools/portfolio_audit.py, SHARED_GROUPS).

Vorwärtsmodell: eine Größe x(s) auf [0, 1] wird mit einem Gauß-Kern der Breite w unscharf gemessen,
    g(t) = Integral k(t - s) x(s) ds,    k(u) = exp(-u^2 / (2 w^2)) / (w sqrt(2 pi)),
und an n Stellen t_i = (i + 0.5) / n mit weißem Rauschen abgelesen. Das Modell auf dem Gitter ist A = h K (Mittelpunktregel, h = 1/n).
Das Phantom (zwei Gauß-Hügel und ein Kasten) liegt in [0.1, 0.9]; seine unscharfe Version ist **analytisch** bekannt (Hügel bleiben Hügel mit der Breite
sqrt(b^2 + w^2), der Kasten wird zur Differenz zweier Normalverteilungsfunktionen). Daher gibt es zwei Arten von Daten:
    - "Verbrechen" (inverse crime): b = A x_n mit derselben Matrix, die danach invertiert wird;
    - "ehrlich": b = g(t_i), die vom Gitter unabhängige analytische Antwort.
Das Rauschen ist in beiden Fällen dasselbe (derselbe Strom), mit Standardabweichung `noise` mal dem Effektivwert der exakten Daten.

Zufall nur über einen übergebenen `SplitMix64`. Plattformhinweis: Normalverteilte laufen über `log` und `cos` der C-Bibliothek und können sich
zwischen Plattformen in der letzten Stelle unterscheiden; Tests dürfen Float-Werte daher nur mit Toleranz prüfen.
"""

import math

import numpy as np

GENERATOR_VERSION = 1

# Phantom: (Mitte, Breite, Höhe) der Hügel, (links, rechts, Höhe) des Kastens
BUMPS = ((0.30, 0.050, 1.0), (0.58, 0.030, 0.7))
BOX = (0.72, 0.84, 0.5)

_GAMMA = 0x9E3779B97F4A7C15
_MASK = (1 << 64) - 1
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB
_ERF = np.frompyfunc(math.erf, 1, 1)


class SplitMix64:
    """SplitMix64-Strom (Steele, Lea, Flood 2014). Ausgabe k hängt nur von (Seed, k) ab: plattform- und versionsunabhängig."""

    def __init__(self, seed: int):
        self.state = int(seed) & _MASK

    def u64(self, size: int) -> np.ndarray:
        """Die nächsten `size` 64-Bit-Werte; der Strom rückt um `size` Schritte weiter."""
        idx = np.arange(1, size + 1, dtype=np.uint64)
        with np.errstate(over="ignore"):
            z = np.uint64(self.state) + idx * np.uint64(_GAMMA)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(_M1)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(_M2)
            z = z ^ (z >> np.uint64(31))
        self.state = (self.state + size * _GAMMA) & _MASK
        return z

    def uniform(self, size: int) -> np.ndarray:
        """Gleichverteilte Werte in [0, 1) mit 53 Bit."""
        return (self.u64(size) >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))

    def normal(self, size: int) -> np.ndarray:
        """Standardnormalverteilte Werte (Box-Muller, nur der Kosinus-Zweig); verbraucht 2 * size Gleichverteilte."""
        u1 = self.uniform(size)
        u2 = self.uniform(size)
        return np.sqrt(-2.0 * np.log1p(-u1)) * np.cos(2.0 * np.pi * u2)


def normal_cdf(z) -> np.ndarray:
    """Standardnormal-Verteilungsfunktion über math.erf."""
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.asarray(_ERF(z / math.sqrt(2.0)), dtype=float))


def grid(n: int) -> np.ndarray:
    """Mittelpunkte der n Zellen von [0, 1]."""
    return (np.arange(n) + 0.5) / n


def phantom(s) -> np.ndarray:
    """Die Wahrheit x(s): zwei Gauß-Hügel und ein Kasten (unstetig, daher ein echter Modellfehler auf dem Gitter)."""
    s = np.asarray(s, dtype=float)
    out = np.zeros_like(s)
    for c, width, height in BUMPS:
        out = out + height * np.exp(-((s - c) ** 2) / (2.0 * width * width))
    left, right, height = BOX
    return out + height * ((s >= left) & (s < right))


def blurred_exact(t, w: float) -> np.ndarray:
    """g(t): die analytische unscharfe Version des Phantoms, unabhängig von jedem Gitter."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    for c, width, height in BUMPS:
        v = width * width + w * w
        out = out + height * (width / math.sqrt(v)) * np.exp(-((t - c) ** 2) / (2.0 * v))
    left, right, height = BOX
    return out + height * (normal_cdf((t - left) / w) - normal_cdf((t - right) / w))


def operator(n: int, w: float) -> np.ndarray:
    """A = h K auf dem Gitter mit n Punkten (Mittelpunktregel); die Zeilensumme liegt nahe 1."""
    s = grid(n)
    d = s[:, None] - s[None, :]
    return np.exp(-(d ** 2) / (2.0 * w * w)) / (w * math.sqrt(2.0 * math.pi)) / n


def noise_sigma(n: int, w: float, noise: float) -> float:
    """Absolute Standardabweichung des Rauschens: `noise` mal Effektivwert der exakten Daten."""
    g = blurred_exact(grid(n), w)
    return float(noise) * float(np.sqrt(np.mean(g * g)))


def draw_data(n: int, w: float, noise: float, rng: SplitMix64, crime: bool, reps: int = 1, sigma: float | None = None) -> np.ndarray:
    """(reps, n) Messwerte. crime=True: b = A x_n + e, crime=False: b = g(t_i) + e; das Rauschen e ist in beiden Fällen gleich.
    `sigma` überschreibt die absolute Standardabweichung (gleiches absolutes Rauschen bei wechselndem Gitter)."""
    s = grid(n)
    base = operator(n, w) @ phantom(s) if crime else blurred_exact(s, w)
    sd = noise_sigma(n, w, noise) if sigma is None else float(sigma)
    e = rng.normal(n * reps).reshape(reps, n) * sd
    return base[None, :] + e
