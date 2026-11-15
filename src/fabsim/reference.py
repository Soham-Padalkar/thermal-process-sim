"""Published reference values used for validation.

Rate constants for (111) silicon as tabulated by B. E. Deal and A. S. Grove,
"General relationship for the thermal oxidation of silicon", J. Appl. Phys. 36,
3770 (1965). The "wet" rows are for oxygen bubbled through 95 degC water.

The numbers were transcribed from secondary sources (textbook reproductions of
the original table); check them against the paper before citing them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DealGroveRow:
    ambient: str
    temp_c: float
    A: float  # um
    B: float  # um^2/h
    tau: float  # h


DEAL_GROVE_1965 = (
    DealGroveRow("dry", 1200, 0.040, 0.045, 0.027),
    DealGroveRow("dry", 1100, 0.090, 0.027, 0.076),
    DealGroveRow("dry", 1000, 0.165, 0.0117, 0.37),
    DealGroveRow("dry", 920, 0.235, 0.0049, 1.40),
    DealGroveRow("wet", 1200, 0.050, 0.720, 0.0),
    DealGroveRow("wet", 1100, 0.110, 0.510, 0.0),
    DealGroveRow("wet", 1000, 0.226, 0.287, 0.0),
    DealGroveRow("wet", 920, 0.500, 0.203, 0.0),
)
