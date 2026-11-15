"""Deal-Grove model of thermal oxidation of silicon.

The oxide thickness x (um) after time t (hours) satisfies

    x^2 + A x = B (t + tau)

where B is the parabolic rate constant (um^2/h), B/A the linear rate constant
(um/h) and tau accounts for an initial oxide x_i.

Rate constants are Arrhenius fits for (111) silicon as tabulated in standard
process textbooks (e.g. Plummer, Deal & Griffin, "Silicon VLSI Technology",
ch. 6). The linear rate constant on (100) silicon is lower by a factor 1.68.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import arrhenius

# ambient -> (C1 [um^2/h], E1 [eV], C2 [um/h], E2 [eV]) for (111) Si
_RATE_FITS = {
    "dry": (7.72e2, 1.23, 6.23e6, 2.00),
    "wet": (3.86e2, 0.78, 1.63e8, 2.05),
}
_ORIENTATION_FACTOR = {"111": 1.0, "100": 1.0 / 1.68}


@dataclass(frozen=True)
class RateConstants:
    """Deal-Grove rate constants at one temperature."""

    B: float  # parabolic rate constant, um^2/h
    B_over_A: float  # linear rate constant, um/h

    @property
    def A(self) -> float:
        return self.B / self.B_over_A


def rate_constants(temp_c: float, ambient: str = "dry", orientation: str = "100") -> RateConstants:
    """Return Deal-Grove rate constants for the given temperature and ambient."""
    if ambient not in _RATE_FITS:
        raise ValueError(f"ambient must be one of {sorted(_RATE_FITS)}, got {ambient!r}")
    if orientation not in _ORIENTATION_FACTOR:
        raise ValueError(f"orientation must be '100' or '111', got {orientation!r}")
    c1, e1, c2, e2 = _RATE_FITS[ambient]
    return RateConstants(
        B=arrhenius(c1, e1, temp_c),
        B_over_A=arrhenius(c2, e2, temp_c) * _ORIENTATION_FACTOR[orientation],
    )


def oxide_thickness(time_h, rates: RateConstants, x_initial: float = 0.0):
    """Closed-form oxide thickness (um) after ``time_h`` hours.

    ``x_initial`` is the oxide already present at t = 0 (um). For dry oxidation
    the Deal-Grove fit conventionally uses x_initial ~ 0.025 um to absorb the
    anomalously fast initial regime, which this model does not describe.
    """
    time_h = np.asarray(time_h, dtype=float)
    if np.any(time_h < 0):
        raise ValueError("time must be non-negative")
    a, b = rates.A, rates.B
    tau = (x_initial**2 + a * x_initial) / b
    return 0.5 * a * (np.sqrt(1.0 + 4.0 * b * (time_h + tau) / a**2) - 1.0)


def time_to_thickness(x_target: float, rates: RateConstants, x_initial: float = 0.0) -> float:
    """Hours needed to grow from ``x_initial`` to ``x_target`` (um)."""
    if x_target < x_initial:
        raise ValueError("target thickness is below the initial oxide thickness")
    a, b = rates.A, rates.B
    return ((x_target**2 + a * x_target) - (x_initial**2 + a * x_initial)) / b


def integrate_growth(time_h: float, rates: RateConstants, x_initial: float = 0.0, steps: int = 2000):
    """Numerically integrate dx/dt = B / (2x + A) with classic RK4.

    Independent of the closed form; used to cross-check it and as the starting
    point for models where B and A vary in time (temperature ramps).
    Returns (t, x) arrays of length ``steps + 1``.
    """
    a, b = rates.A, rates.B

    def f(x):
        return b / (2.0 * x + a)

    t = np.linspace(0.0, time_h, steps + 1)
    x = np.empty_like(t)
    x[0] = x_initial
    h = time_h / steps
    for i in range(steps):
        k1 = f(x[i])
        k2 = f(x[i] + 0.5 * h * k1)
        k3 = f(x[i] + 0.5 * h * k2)
        k4 = f(x[i] + h * k3)
        x[i + 1] = x[i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return t, x
