"""Regenerate every figure and the results table in the README.

    python -m fabsim.figures            # writes docs/figures/*.png and results.json

All runs are deterministic (no random numbers).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import plotstyle as ps
from .diffusion import (
    Grid,
    crank_nicolson,
    erfc_profile,
    fair_diffusivity,
    ftcs,
    implant_profile,
    intrinsic_diffusivity,
    junction_depth,
)
from .oxidation import oxide_thickness, rate_constants
from .reference import DEAL_GROVE_1965

OUT = Path("docs/figures")
UM = 1e-4  # cm per micrometre


def _l2(num, exact):
    return float(np.linalg.norm(num - exact) / np.linalg.norm(exact))


def fig_oxidation(results: dict) -> None:
    """Growth curves and Arrhenius fits against the tabulated 1965 constants."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    t = np.logspace(-1.3, 1.3, 200)
    for ax, ambient, x0 in ((axes[0], "dry", 0.025), (axes[1], "wet", 0.0)):
        for color, temp in zip(ps.SERIES, (900, 1000, 1100, 1200)):
            x = oxide_thickness(t, rate_constants(temp, ambient, "100"), x_initial=x0)
            ax.loglog(t, x, color=color)
            ax.annotate(f"{temp} °C", (t[-1], x[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=8.5, color=ps.MUTED, va="center")  # fmt: skip
        ax.set(xlabel="time (h)", ylabel="oxide thickness (µm)", xlim=(t[0], t[-1] * 2.2),
               title=f"{ambient.capitalize()} oxidation, (100) Si")  # fmt: skip

    ax = axes[2]
    errors = []
    inv_t = np.linspace(0.66, 0.86, 50)
    temps = 1000.0 / inv_t - 273.15
    for color, ambient in ((ps.BLUE, "dry"), (ps.ORANGE, "wet")):
        fit = [rate_constants(tc, ambient, "111").B for tc in temps]
        ax.semilogy(inv_t, fit, color=color, label=f"{ambient}: Arrhenius fit")
        rows = [r for r in DEAL_GROVE_1965 if r.ambient == ambient]
        ax.semilogy([1000 / (r.temp_c + 273.15) for r in rows], [r.B for r in rows], "o", color=color,
                    markeredgecolor=ps.SURFACE, markeredgewidth=1.5, label=f"{ambient}: Deal & Grove table")  # fmt: skip
        for r in rows:
            rc = rate_constants(r.temp_c, ambient, "111")
            errors.append({"ambient": ambient, "temp_c": r.temp_c,
                           "B_error_pct": 100 * (rc.B / r.B - 1),
                           "BA_error_pct": 100 * (rc.B_over_A / (r.B / r.A) - 1)})  # fmt: skip
    ax.set(xlabel="1000 / T (1/K)", ylabel="parabolic rate constant B (µm²/h)",
           title="Fit vs. tabulated constants, (111) Si")  # fmt: skip
    ax.legend()
    fig.savefig(OUT / "oxidation.png")
    plt.close(fig)
    results["oxidation_fit_errors"] = errors
    results["oxidation_max_abs_B_error_pct"] = max(abs(e["B_error_pct"]) for e in errors)
    results["oxidation_max_abs_BA_error_pct"] = max(abs(e["BA_error_pct"]) for e in errors)


def fig_validation(results: dict) -> None:
    """Numerical profiles against the two classical closed-form solutions."""
    temp = 1100
    d = intrinsic_diffusivity("boron", temp)
    grid = Grid(2 * UM, 200)
    x_um = grid.x / UM
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex="col", height_ratios=(3, 1.2))

    cs = 1e18
    cases = []
    for color, minutes in zip(ps.SERIES, (10, 30, 90)):
        num = crank_nicolson(np.zeros(grid.n), grid, d, minutes * 60, dt=5.0, surface=cs)
        cases.append(
            (axes[0, 0], axes[1, 0], color, f"{minutes} min", num, erfc_profile(grid.x, cs, d, minutes * 60))
        )
    q, rp, drp = 1e13, 0.30 * UM, 0.07 * UM
    c0 = implant_profile(grid.x, q, rp, drp)
    for color, minutes in zip(ps.SERIES, (0, 30, 90)):
        num = crank_nicolson(c0, grid, d, minutes * 60, dt=5.0) if minutes else c0
        cases.append((axes[0, 1], axes[1, 1], color, f"{minutes} min", num,
                      implant_profile(grid.x, q, rp, drp, d, minutes * 60)))  # fmt: skip

    worst = 0.0
    for ax, ax_err, color, label, num, exact in cases:
        ax.semilogy(x_um, exact, color=color, label=label)
        ax.semilogy(x_um[::10], num[::10], "o", color=color, markersize=4.5,
                    markeredgecolor=ps.SURFACE, markeredgewidth=1.0)  # fmt: skip
        mask = exact > 1e-4 * exact.max()
        err = 100 * (num[mask] / exact[mask] - 1)
        ax_err.plot(x_um[mask], err, color=color, linewidth=1.5)
        worst = max(worst, float(np.max(np.abs(err))))
    axes[0, 0].set(ylim=(1e14, 3e18), ylabel="boron concentration (cm⁻³)",
                   title="Predeposition: solver (dots) vs. erfc solution (lines)")  # fmt: skip
    axes[0, 1].set(ylim=(1e14, 1e18), title="Implant anneal: solver (dots) vs. Gaussian solution (lines)")
    for ax in axes[0]:
        ax.legend(title=f"{temp} °C")
    for ax in axes[1]:
        ax.set(xlabel="depth (µm)", ylabel="error (%)")
    fig.savefig(OUT / "diffusion_validation.png")
    plt.close(fig)
    results["validation_max_pointwise_error_pct"] = worst


def fig_convergence(results: dict) -> None:
    """Grid and time-step convergence, and cost of explicit vs. implicit."""
    d, t_end = 1.0e-13, 3600.0
    q, rp, drp = 1e14, 0.30 * UM, 0.07 * UM
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    ns = np.array([25, 50, 100, 200, 400, 800])
    errs = []
    for n in ns:
        g = Grid(4 * UM, int(n))
        num = crank_nicolson(implant_profile(g.x, q, rp, drp), g, d, t_end, dt=0.5)
        errs.append(_l2(num, implant_profile(g.x, q, rp, drp, d, t_end)))
    dxs = 4 * UM / ns / 1e-7  # nm
    axes[0].loglog(dxs, errs, "o-", color=ps.BLUE, label="Crank-Nicolson")
    axes[0].loglog(dxs, errs[-1] * (dxs / dxs[-1]) ** 2, "--", color=ps.MUTED, linewidth=1, label="slope 2")
    axes[0].set(xlabel="cell size Δx (nm)", ylabel="relative L2 error", title="Spatial convergence")
    axes[0].legend()
    results["spatial_order"] = float(np.polyfit(np.log(dxs), np.log(errs), 1)[0])

    g = Grid(4 * UM, 1600)
    c0 = implant_profile(g.x, q, rp, drp)
    ref = crank_nicolson(c0, g, d, t_end, dt=0.5)
    dts = np.array([450.0, 225.0, 112.5, 56.25, 28.125])
    terr = [_l2(crank_nicolson(c0, g, d, t_end, dt=dt), ref) for dt in dts]
    axes[1].loglog(dts, terr, "o-", color=ps.BLUE, label="Crank-Nicolson")
    axes[1].loglog(dts, terr[-1] * (dts / dts[-1]) ** 2, "--", color=ps.MUTED, linewidth=1, label="slope 2")
    axes[1].set(xlabel="time step Δt (s)", ylabel="relative L2 error vs. Δt = 0.5 s",
                title="Temporal convergence")  # fmt: skip
    axes[1].legend()
    results["temporal_order"] = float(np.polyfit(np.log(dts), np.log(terr), 1)[0])

    cost = []
    for n in (100, 200, 400, 800):
        g = Grid(4 * UM, n)
        c0 = implant_profile(g.x, q, rp, drp)
        exact = implant_profile(g.x, q, rp, drp, d, t_end)
        row = {"n": n}
        for name, fn, dt in (("ftcs", ftcs, 0.45 * g.dx**2 / d), ("cn", crank_nicolson, 20.0)):
            t0 = time.perf_counter()
            num = fn(c0, g, d, t_end, dt=dt)
            row[f"{name}_ms"] = 1e3 * (time.perf_counter() - t0)
            row[f"{name}_err"] = _l2(num, exact)
            row[f"{name}_steps"] = int(np.ceil(t_end / dt))
        cost.append(row)
    axes[2].loglog([c["ftcs_err"] for c in cost], [c["ftcs_steps"] for c in cost], "o-", color=ps.ORANGE,
                   label="explicit FTCS (baseline)")  # fmt: skip
    axes[2].loglog([c["cn_err"] for c in cost], [c["cn_steps"] for c in cost], "o-", color=ps.BLUE,
                   label="Crank-Nicolson, Δt = 20 s")  # fmt: skip
    axes[2].set(xlabel="relative L2 error", ylabel="time steps for a 1 h anneal",
                title="Cost to reach a given accuracy")  # fmt: skip
    axes[2].legend()
    fig.savefig(OUT / "convergence.png")
    plt.close(fig)
    results["cost_comparison"] = cost


def fig_concentration_dependence(results: dict) -> None:
    """Extension beyond the textbook case: Fair's D(C) vs. constant D, plus sensitivity."""
    grid = Grid(2.0 * UM, 500)
    x_um = grid.x / UM
    t_end, background = 1800.0, 1e16
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    temp = 1000
    d_c, d_i = fair_diffusivity("boron", temp), intrinsic_diffusivity("boron", temp)
    ratios = []
    for color, cs in zip(ps.SERIES, (1e18, 1e19, 1e20, 3e20)):
        full = crank_nicolson(np.zeros(grid.n), grid, d_c, t_end, dt=2.0, surface=cs)
        const = crank_nicolson(np.zeros(grid.n), grid, d_i, t_end, dt=2.0, surface=cs)
        axes[0].semilogy(x_um, full, color=color, label=f"{cs:.0e} cm⁻³")
        axes[0].semilogy(x_um, const, color=color, linestyle=":", linewidth=1.3)
        xj_full, xj_const = (junction_depth(c, grid, background) / UM for c in (full, const))
        ratios.append({"surface_cm3": cs, "xj_fair_um": xj_full, "xj_constant_um": xj_const})
    axes[0].set(xlim=(0, 1.0), ylim=(1e15, 1e21), xlabel="depth (µm)", ylabel="boron concentration (cm⁻³)",
                title="Fair D(C) (solid) vs. constant D (dotted)")  # fmt: skip
    axes[0].legend(title=f"surface conc., {temp} °C, 30 min", loc="upper right")
    results["junction_depth_vs_surface_conc"] = ratios

    cs_axis = [r["surface_cm3"] for r in ratios]
    axes[1].semilogx(cs_axis, [r["xj_constant_um"] for r in ratios], "o-", color=ps.ORANGE,
                     label="constant D (baseline)")  # fmt: skip
    axes[1].semilogx(cs_axis, [r["xj_fair_um"] for r in ratios], "o-", color=ps.BLUE, label="Fair D(C)")
    axes[1].set(xlabel="surface concentration (cm⁻³)", ylabel="junction depth at 10¹⁶ cm⁻³ (µm)",
                title="What the constant-D model misses")  # fmt: skip
    axes[1].legend()

    temps = np.arange(950, 1101, 25)
    sens = []
    for cs, color in ((1e18, ps.AQUA), (1e20, ps.BLUE)):
        xj = []
        for tc in temps:
            c = crank_nicolson(np.zeros(grid.n), grid, fair_diffusivity("boron", float(tc)), t_end,
                               dt=2.0, surface=cs)  # fmt: skip
            xj.append(junction_depth(c, grid, background) / UM)
        axes[2].plot(temps, xj, "o-", color=color, label=f"{cs:.0e} cm⁻³")
        slope = np.gradient(np.log(xj), temps)
        sens.append({"surface_cm3": cs, "temps_c": temps.tolist(), "xj_um": xj,
                     "pct_per_degC_at_1000C": float(100 * slope[list(temps).index(1000)])})  # fmt: skip
    axes[2].set(xlabel="furnace temperature (°C)", ylabel="junction depth (µm)",
                title="Sensitivity to temperature, 30 min")  # fmt: skip
    axes[2].legend(title="surface conc.")
    fig.savefig(OUT / "concentration_dependence.png")
    plt.close(fig)
    results["temperature_sensitivity"] = sens


def main() -> None:
    ps.apply()
    OUT.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    for fn in (fig_oxidation, fig_validation, fig_convergence, fig_concentration_dependence):
        fn(results)
        print(f"wrote {fn.__name__}")
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if not isinstance(v, list)}, indent=2))


if __name__ == "__main__":
    main()
