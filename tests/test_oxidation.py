import numpy as np
import pytest

from fabsim.oxidation import (
    integrate_growth,
    oxide_thickness,
    rate_constants,
    time_to_thickness,
)
from fabsim.reference import DEAL_GROVE_1965


def test_thickness_satisfies_deal_grove_equation():
    rates = rate_constants(1000, "wet")
    t = np.array([0.1, 1.0, 5.0])
    x = oxide_thickness(t, rates, x_initial=0.01)
    tau = (0.01**2 + rates.A * 0.01) / rates.B
    np.testing.assert_allclose(x**2 + rates.A * x, rates.B * (t + tau), rtol=1e-12)


def test_zero_time_returns_initial_oxide():
    rates = rate_constants(1100, "dry")
    assert oxide_thickness(0.0, rates, x_initial=0.025) == pytest.approx(0.025, rel=1e-12)


def test_time_to_thickness_inverts_thickness():
    rates = rate_constants(950, "wet")
    x = oxide_thickness(2.5, rates, x_initial=0.02)
    assert time_to_thickness(float(x), rates, x_initial=0.02) == pytest.approx(2.5, rel=1e-10)


def test_limiting_regimes():
    rates = rate_constants(1000, "dry")
    # thin oxide: linear growth at rate B/A
    t_short = 1e-4 * rates.A**2 / rates.B
    assert oxide_thickness(t_short, rates) == pytest.approx(rates.B_over_A * t_short, rel=1e-3)
    # thick oxide: parabolic growth, x -> sqrt(B t)
    t_long = 1e6 * rates.A**2 / rates.B
    assert oxide_thickness(t_long, rates) == pytest.approx(np.sqrt(rates.B * t_long), rel=1e-3)


def test_rk4_integration_matches_closed_form():
    rates = rate_constants(1000, "wet")
    t, x = integrate_growth(4.0, rates, x_initial=0.005, steps=400)
    np.testing.assert_allclose(x, oxide_thickness(t, rates, x_initial=0.005), rtol=1e-6)


def test_100_orientation_is_slower_than_111_only_in_linear_rate():
    r100 = rate_constants(1000, "dry", "100")
    r111 = rate_constants(1000, "dry", "111")
    assert r100.B == r111.B
    assert r111.B_over_A / r100.B_over_A == pytest.approx(1.68)


@pytest.mark.parametrize("row", DEAL_GROVE_1965, ids=lambda r: f"{r.ambient}-{r.temp_c}C")
def test_arrhenius_fits_track_tabulated_constants(row):
    """The fits should stay within 25 % of the tabulated (111) constants."""
    rates = rate_constants(row.temp_c, row.ambient, "111")
    assert rates.B == pytest.approx(row.B, rel=0.25)
    assert rates.B_over_A == pytest.approx(row.B / row.A, rel=0.25)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        rate_constants(1000, "steamy")
    with pytest.raises(ValueError):
        oxide_thickness(-1.0, rate_constants(1000))
    with pytest.raises(ValueError):
        time_to_thickness(0.01, rate_constants(1000), x_initial=0.02)
