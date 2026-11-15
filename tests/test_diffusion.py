import numpy as np
import pytest

from fabsim.diffusion import (
    Grid,
    crank_nicolson,
    dose,
    erfc_profile,
    fair_diffusivity,
    ftcs,
    implant_profile,
    intrinsic_diffusivity,
    junction_depth,
)

D = 1.0e-13  # cm^2/s, order of boron at ~1100 C
T = 3600.0  # s
GRID = Grid(length=4e-4, n=400)  # 4 um, 10 nm cells


def _relative_l2(numerical, exact):
    return float(np.linalg.norm(numerical - exact) / np.linalg.norm(exact))


def test_predeposition_matches_erfc():
    cs = 1e20
    c = crank_nicolson(np.zeros(GRID.n), GRID, D, T, dt=10.0, surface=cs)
    assert _relative_l2(c, erfc_profile(GRID.x, cs, D, T)) < 2e-3


def test_implant_anneal_matches_gaussian_broadening():
    q, rp, drp = 1e14, 0.3e-4, 0.07e-4
    c0 = implant_profile(GRID.x, q, rp, drp)
    c = crank_nicolson(c0, GRID, D, T, dt=10.0)
    assert _relative_l2(c, implant_profile(GRID.x, q, rp, drp, D, T)) < 1e-3


def test_drive_in_conserves_dose():
    c0 = implant_profile(GRID.x, 1e14, 0.3e-4, 0.07e-4)
    c = crank_nicolson(c0, GRID, D, 10 * T, dt=30.0)
    assert dose(c, GRID) == pytest.approx(dose(c0, GRID), rel=1e-12)


def test_explicit_and_implicit_solvers_agree():
    c0 = implant_profile(GRID.x, 1e14, 0.3e-4, 0.07e-4)
    dt = 0.4 * GRID.dx**2 / D
    a = ftcs(c0, GRID, D, T, dt=dt)
    b = crank_nicolson(c0, GRID, D, T, dt=dt)
    assert _relative_l2(a, b) < 1e-3


def test_ftcs_rejects_unstable_step():
    with pytest.raises(ValueError, match="unstable"):
        ftcs(np.zeros(GRID.n), GRID, D, T, dt=0.6 * GRID.dx**2 / D)


def test_crank_nicolson_is_stable_far_beyond_explicit_limit():
    c0 = implant_profile(GRID.x, 1e14, 0.3e-4, 0.07e-4)
    c = crank_nicolson(c0, GRID, D, T, dt=50 * GRID.dx**2 / D)
    assert np.all(np.isfinite(c)) and c.min() > -1e-6 * c.max()


def test_spatial_convergence_is_second_order():
    q, rp, drp = 1e14, 0.3e-4, 0.07e-4
    errors = []
    for n in (50, 100, 200):
        g = Grid(4e-4, n)
        c = crank_nicolson(implant_profile(g.x, q, rp, drp), g, D, T, dt=1.0)
        errors.append(_relative_l2(c, implant_profile(g.x, q, rp, drp, D, T)))
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(orders > 1.8), orders


def test_concentration_dependent_d_reduces_to_constant_at_low_doping():
    d_of_c = fair_diffusivity("boron", 1000)
    d_i = intrinsic_diffusivity("boron", 1000)
    c0 = implant_profile(GRID.x, 1e10, 0.3e-4, 0.07e-4)  # peak far below n_i
    a = crank_nicolson(c0, GRID, d_of_c, T, dt=10.0)
    b = crank_nicolson(c0, GRID, d_i, T, dt=10.0)
    assert _relative_l2(a, b) < 1e-3


def test_high_doping_diffuses_deeper_than_intrinsic():
    d_of_c = fair_diffusivity("boron", 1000)
    d_i = intrinsic_diffusivity("boron", 1000)
    kw = {"time_s": T, "dt": 5.0, "surface": 3e20}
    deep = crank_nicolson(np.zeros(GRID.n), GRID, d_of_c, **kw)
    shallow = crank_nicolson(np.zeros(GRID.n), GRID, d_i, **kw)
    assert junction_depth(deep, GRID, 1e16) > 1.5 * junction_depth(shallow, GRID, 1e16)


def test_junction_depth_of_known_profile():
    c = erfc_profile(GRID.x, 1e20, D, T)
    xj = junction_depth(c, GRID, 1e16)
    from scipy.special import erfcinv

    assert xj == pytest.approx(2 * np.sqrt(D * T) * erfcinv(1e-4), rel=1e-3)
