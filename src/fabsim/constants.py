"""Physical constants and Arrhenius helpers."""

from __future__ import annotations

import numpy as np

K_B_EV = 8.617333e-5  # Boltzmann constant, eV/K


def kelvin(temp_c: float) -> float:
    """Convert degrees Celsius to kelvin."""
    return temp_c + 273.15


def arrhenius(prefactor: float, activation_ev: float, temp_c: float) -> float:
    """Return prefactor * exp(-Ea / kT) at temperature ``temp_c`` (deg C)."""
    return prefactor * float(np.exp(-activation_ev / (K_B_EV * kelvin(temp_c))))


def intrinsic_carrier_density(temp_c: float) -> float:
    """Intrinsic carrier density of silicon in cm^-3 at process temperature.

    Uses the common high-temperature fit n_i = 3.9e16 T^1.5 exp(-0.605 eV / kT).
    """
    t = kelvin(temp_c)
    return 3.9e16 * t**1.5 * float(np.exp(-0.605 / (K_B_EV * t)))
