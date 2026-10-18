"""Generator der Linie (wortgleich in allen Repos): Strom, Normalverteilung, Gitter, Phantom, analytische Unschärfe, Operator, Daten."""

import math

import numpy as np
import pytest

import inverse_generator as gen

MASK = (1 << 64) - 1


def splitmix_reference(seed, count):
    """Unabhängige Python-Fassung des SplitMix64-Stroms (Steele, Lea, Flood 2014)."""
    state, out = seed & MASK, []
    for _ in range(count):
        state = (state + 0x9E3779B97F4A7C15) & MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        out.append(z ^ (z >> 31))
    return out


@pytest.mark.parametrize("seed", [0, 1, 35, 123456789, 2**63 + 5])
def test_splitmix_stream_matches_the_reference(seed):
    assert gen.SplitMix64(seed).u64(25).tolist() == splitmix_reference(seed, 25)


def test_the_stream_does_not_depend_on_how_it_is_drawn():
    a, b = gen.SplitMix64(7), gen.SplitMix64(7)
    assert np.concatenate([a.uniform(3), a.uniform(4)]).tolist() == b.uniform(7).tolist()


def test_uniform_is_in_the_half_open_unit_interval():
    u = gen.SplitMix64(3).uniform(50000)
    assert u.min() >= 0.0 and u.max() < 1.0 and abs(u.mean() - 0.5) < 0.01


def test_normal_has_mean_zero_and_unit_variance():
    z = gen.SplitMix64(11).normal(200000)
    assert abs(z.mean()) < 0.01 and abs(z.var() - 1.0) < 0.02


def test_normal_cdf_known_values_and_scalar_input():
    assert float(gen.normal_cdf(0.0)) == pytest.approx(0.5)
    assert float(gen.normal_cdf(1.959964)) == pytest.approx(0.975, abs=1e-6)
    assert float(gen.normal_cdf(-1.0)) == pytest.approx(0.15865525393145707, abs=1e-12)
    assert gen.normal_cdf(np.array([0.0, 1.0])).shape == (2,)


def test_grid_is_the_midpoint_grid():
    assert gen.grid(4).tolist() == [0.125, 0.375, 0.625, 0.875]


def test_phantom_values_at_hand_points():
    # Mitte des ersten Hügels: nur er trägt (Höhe 1,0); der zweite Hügel und der Kasten liegen weit weg
    assert float(gen.phantom(0.30)) == pytest.approx(1.0 + 0.7 * math.exp(-(0.28 ** 2) / (2 * 0.03 ** 2)), abs=1e-12)
    # im Kasten: Höhe 0,5 plus die Ausläufer der Hügel
    box = float(gen.phantom(0.78))
    assert box == pytest.approx(0.5 + 0.7 * math.exp(-(0.20 ** 2) / (2 * 0.03 ** 2)) + math.exp(-(0.48 ** 2) / (2 * 0.05 ** 2)), abs=1e-12)
    # links des Kastens ohne Kasten, die Kante gehört zum Kasten
    assert float(gen.phantom(0.72)) - float(gen.phantom(0.7199999)) == pytest.approx(0.5, abs=1e-3)
    # an den Rändern praktisch null
    assert float(gen.phantom(0.0)) == pytest.approx(math.exp(-18.0), rel=1e-9) and float(gen.phantom(1.0)) < 1e-12


def test_operator_rows_sum_to_about_one_inside_and_are_symmetric():
    a = gen.operator(64, 0.03)
    assert np.allclose(a, a.T)
    assert a[32].sum() == pytest.approx(1.0, abs=1e-9)
    assert a[0].sum() == pytest.approx(float(gen.normal_cdf(0.5 / 64 / 0.03)), abs=0.005)   # am Rand fehlt ein Teil der Glocke


def test_blurred_exact_of_a_single_bump_is_a_wider_bump():
    w = 0.04
    bump = gen.blurred_exact(np.array([0.30]), w)[0]
    # Beitrag des ersten Hügels in der Mitte: 1,0 * 0,05 / sqrt(0,05^2 + w^2); die anderen Beiträge sind kleiner als 1e-3 und dazu positiv
    assert bump >= 0.05 / math.sqrt(0.05 ** 2 + w ** 2) - 1e-12
    assert bump < 0.05 / math.sqrt(0.05 ** 2 + w ** 2) + 1e-3


def test_noise_sigma_is_the_relative_noise_times_the_rms_of_the_exact_data():
    g = gen.blurred_exact(gen.grid(48), 0.03)
    assert gen.noise_sigma(48, 0.03, 1e-3) == pytest.approx(1e-3 * math.sqrt(np.mean(g ** 2)), rel=1e-12)
    assert gen.noise_sigma(48, 0.03, 0.0) == 0.0


def test_the_two_kinds_of_data_share_the_same_noise_stream():
    n, w, r = 40, 0.03, 1e-2
    crime = gen.draw_data(n, w, r, gen.SplitMix64(5), crime=True)[0]
    honest = gen.draw_data(n, w, r, gen.SplitMix64(5), crime=False)[0]
    base_c = gen.operator(n, w) @ gen.phantom(gen.grid(n))
    base_h = gen.blurred_exact(gen.grid(n), w)
    assert np.allclose(crime - honest, base_c - base_h, atol=1e-15)


def test_repetitions_consume_one_block_and_the_explicit_sigma_wins():
    d = gen.draw_data(24, 0.03, 0.5, gen.SplitMix64(9), crime=False, reps=3, sigma=0.0)
    assert d.shape == (3, 24)
    assert np.allclose(d, gen.blurred_exact(gen.grid(24), 0.03)[None, :])


def test_the_box_is_half_open_and_ends_exactly_at_its_right_edge():
    just_inside = float(gen.phantom(0.8399999))
    at_edge = float(gen.phantom(0.84))
    assert just_inside - at_edge == pytest.approx(0.5, abs=1e-3)
