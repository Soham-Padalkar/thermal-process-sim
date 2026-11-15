"""1-D dopant diffusion in silicon.

Solves  dC/dt = d/dx ( D(C) dC/dx )  on 0 <= x <= L with

* surface (x = 0): either a fixed concentration (constant-source
  predeposition) or zero flux (drive-in under a capping oxide),
* bulk (x = L): zero flux.

Two time integrators share one finite-volume spatial discretisation:

* ``ftcs``            - explicit forward Euler, the simple baseline. Stable only
                        for D dt / dx^2 <= 1/2.
* ``crank_nicolson``  - implicit, second order in time, unconditionally stable.
                        Concentration-dependent D is handled by lagging D one
                        step (evaluated at the start of each step).

Units: cm, s, cm^-3, cm^2/s.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import erfc

from .constants import arrhenius, intrinsic_carrier_density

Diffusivity = float | Callable[[np.ndarray], np.ndarray]

# Fair's vacancy model: D = D0 + D1 * (n/ni) [+ D2 * (n/ni)^2], each term Arrhenius.
# (prefactor cm^2/s, activation eV). Textbook values; treat as approximate.
_FAIR = {
    "boron": ((0.037, 3.46), (0.41, 3.46), None),
    "arsenic": ((0.066, 3.44), (12.0, 4.05), None),
    "phosphorus": ((3.85, 3.66), (4.44, 4.00), (44.2, 4.37)),
}


def fair_diffusivity(dopant: str, temp_c: float) -> Callable[[np.ndarray], np.ndarray]:
    """Return D(C) for ``dopant`` at ``temp_c`` using Fair's charged-vacancy model.

    The carrier ratio follows from charge neutrality with full ionisation:
    n/ni = C/(2 ni) + sqrt((C/(2 ni))^2 + 1).
    """
    if dopant not in _FAIR:
        raise ValueError(f"dopant must be one of {sorted(_FAIR)}, got {dopant!r}")
    terms = _FAIR[dopant]
    d0, d1 = (arrhenius(*terms[0], temp_c), arrhenius(*terms[1], temp_c))
    d2 = arrhenius(*terms[2], temp_c) if terms[2] else 0.0
    ni = intrinsic_carrier_density(temp_c)

    def d_of_c(conc: np.ndarray) -> np.ndarray:
        r = np.maximum(conc, 0.0) / (2.0 * ni)
        ratio = r + np.sqrt(r * r + 1.0)
        return d0 + d1 * ratio + d2 * ratio * ratio

    return d_of_c


def intrinsic_diffusivity(dopant: str, temp_c: float) -> float:
    """Diffusivity (cm^2/s) in the low-concentration limit (n = ni)."""
    return float(fair_diffusivity(dopant, temp_c)(np.array([0.0]))[0])


@dataclass(frozen=True)
class Grid:
    """Uniform cell-centred grid on [0, length]."""

    length: float
    n: int

    @property
    def dx(self) -> float:
        return self.length / self.n

    @property
    def x(self) -> np.ndarray:
        return (np.arange(self.n) + 0.5) * self.dx


def _face_diffusivity(d: Diffusivity, conc: np.ndarray) -> tuple[np.ndarray, float]:
    """Diffusivity at the n-1 interior faces and at the surface cell."""
    if callable(d):
        dc = d(conc)
        # harmonic mean keeps the flux continuous across steep D gradients
        return 2.0 * dc[:-1] * dc[1:] / (dc[:-1] + dc[1:]), float(dc[0])
    return np.full(conc.size - 1, float(d)), float(d)


def _operator(d: Diffusivity, conc: np.ndarray, dx: float, surface: float | None):
    """Tridiagonal L and source s such that dC/dt = L C + s."""
    n = conc.size
    df, d_surf = _face_diffusivity(d, conc)
    w = df / dx**2
    lower = np.zeros(n)
    upper = np.zeros(n)
    diag = np.zeros(n)
    lower[1:] = w
    upper[:-1] = w
    diag[:-1] -= w
    diag[1:] -= w
    src = np.zeros(n)
    if surface is not None:  # Dirichlet value half a cell from the first node
        ws = 2.0 * d_surf / dx**2
        diag[0] -= ws
        src[0] = ws * surface
    return lower, diag, upper, src


def _apply(lower, diag, upper, c):
    out = diag * c
    out[1:] += lower[1:] * c[:-1]
    out[:-1] += upper[:-1] * c[1:]
    return out


def ftcs(conc0, grid: Grid, d: Diffusivity, time_s: float, dt: float, surface: float | None = None):
    """Explicit baseline. Raises ValueError if dt violates the stability limit."""
    conc = np.array(conc0, dtype=float)
    d_max = float(np.max(d(conc))) if callable(d) else float(d)
    if surface is not None and callable(d):
        d_max = max(d_max, float(d(np.array([surface]))[0]))
    # the Dirichlet half-cell at the surface tightens the usual dx^2/(2D) bound
    factor = 1.0 / 3.0 if surface is not None else 0.5
    limit = factor * grid.dx**2 / d_max
    if dt > limit * (1 + 1e-12):
        raise ValueError(f"FTCS unstable: dt={dt:.3g} s exceeds the limit {limit:.3g} s")
    steps = max(1, int(np.ceil(time_s / dt)))
    dt = time_s / steps
    for _ in range(steps):
        lower, diag, upper, src = _operator(d, conc, grid.dx, surface)
        conc = conc + dt * (_apply(lower, diag, upper, conc) + src)
    return conc


def crank_nicolson(conc0, grid: Grid, d: Diffusivity, time_s: float, dt: float, surface: float | None = None):
    """Implicit trapezoidal integrator; any dt is stable."""
    conc = np.array(conc0, dtype=float)
    n = conc.size
    steps = max(1, int(np.ceil(time_s / dt)))
    dt = time_s / steps
    ab = np.zeros((3, n))
    constant = not callable(d)
    if constant:
        lower, diag, upper, src = _operator(d, conc, grid.dx, surface)
    for _ in range(steps):
        if not constant:
            lower, diag, upper, src = _operator(d, conc, grid.dx, surface)
        rhs = conc + 0.5 * dt * _apply(lower, diag, upper, conc) + dt * src
        ab[0, 1:] = -0.5 * dt * upper[:-1]
        ab[1] = 1.0 - 0.5 * dt * diag
        ab[2, :-1] = -0.5 * dt * lower[1:]
        conc = solve_banded((1, 1), ab, rhs)
    return conc


# ---------------------------------------------------------------- analytical


def erfc_profile(x, surface: float, d: float, time_s: float):
    """Constant-surface-concentration (predeposition) solution, semi-infinite Si."""
    return surface * erfc(np.asarray(x) / (2.0 * np.sqrt(d * time_s)))


def gaussian_drive_in(x, dose: float, d: float, time_s: float):
    """Limited-source solution: dose Q (cm^-2) initially at the surface."""
    dt = d * time_s
    return dose / np.sqrt(np.pi * dt) * np.exp(-(np.asarray(x) ** 2) / (4.0 * dt))


def implant_profile(x, dose: float, rp: float, drp: float, d: float = 0.0, time_s: float = 0.0):
    """Gaussian implant (range rp, straggle drp) after annealing for ``time_s``.

    The surface is treated as reflecting (image source at -rp), so the dose in
    x > 0 is conserved up to the tail that the as-implanted Gaussian places at
    x < 0. With time_s = 0 this is the as-implanted profile.
    """
    x = np.asarray(x)
    var = drp**2 + 2.0 * d * time_s
    norm = dose / np.sqrt(2.0 * np.pi * var)
    return norm * (np.exp(-((x - rp) ** 2) / (2 * var)) + np.exp(-((x + rp) ** 2) / (2 * var)))


def dose(conc, grid: Grid) -> float:
    """Integrated dopant dose (cm^-2)."""
    return float(np.sum(conc) * grid.dx)


def junction_depth(conc, grid: Grid, background: float) -> float:
    """Depth (cm) where the profile first falls to ``background``, log-interpolated."""
    below = np.nonzero(conc < background)[0]
    if below.size == 0:
        raise ValueError("profile never falls below the background; enlarge the domain")
    i = int(below[0])
    if i == 0:
        return 0.0
    x = grid.x
    f = (np.log(conc[i - 1]) - np.log(background)) / (np.log(conc[i - 1]) - np.log(conc[i]))
    return float(x[i - 1] + f * (x[i] - x[i - 1]))
