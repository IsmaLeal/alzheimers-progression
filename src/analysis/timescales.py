"""
Tau Timescales by Seeding Region — FKPP Biomarker Curves

This script simulates and plots the evolution of total brain concentration
when tau misfolding starts in different anatomical regions (Braak stages). It
uses the simple FKPP model without accounting for clearance or its evolution.
It highlights how initial seeding in Braak I leads to slower global dynamics,
suggesting it's not a worst-case scenario.

Author: Ismael Leal
Date: 2024-04
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src.data.preprocessing import *
from src.models.tau_concentration_models import c_curves, dt


def run_timescale_analysis(tmax: int = 40, alpha: float = 2.1, rho: float = 0.01, save: bool=False) -> None:
    """
    Compute biomarker dynamics of FKPP model for initial tau seeding
    in each Braak stage.

    Parameters
    ----------
    tmax : int, optional
        Maximum simulation time in years. Default is 40.
    alpha : float, optional
        Reaction rate. Default is 40.
    rho : float, optional
        Effective diffusion coefficient. Default is 0.01.
    save : bool, optional
        Whether to save the figure as png. Default is False.
    """
    # Time resolution
    n_steps = round(tmax / dt)

    # Output dictionaries
    cs = {}
    mins_maxs = {}

    for idx, zone in enumerate(braak):
        # Initialise average and range trackers
        c_accum = np.zeros(n_steps)
        c_min = np.ones(n_steps) * np.inf
        c_max = np.ones(n_steps) * -np.inf

        for node in range(nv):
            if node + 1 in zone:  # node indices start at 1 in braak
                t, _, ctot = c_curves(alpha=alpha, rho=rho, firstnodes=node, tmax=tmax, clearance=False, c_tot=True)
                c_accum += ctot
                c_min = np.minimum(c_min, ctot)
                c_max = np.maximum(c_max, ctot)

        cs[idx] = c_accum / len(zone)
        mins_maxs[idx] = (c_min, c_max)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rcParams.update({"font.size": 19})

    colors = iter(["#dc3e04", "#451ddc", "#01dc04", "#dc01d9", "#000000", "#ffa11b", "#1a7c25"])

    for key, avg_curve in cs.items():
        color = next(colors)
        plt.plot(t, avg_curve, label=braaknames[key], color=color, linewidth=3)
        plt.fill_between(t, mins_maxs[key][0], mins_maxs[key][1], color=color, alpha=0.2)

    ax = plt.gca()
    ax.legend(loc="center left", bbox_to_anchor=(0.6, 0.3), fancybox=False, shadow=False, fontsize=16)
    ax.set_facecolor("#d9e9f9")
    ax.grid(color="white", linestyle="-", linewidth=1)
    ax.set_title(r"$\tau$ concentration vs time for seedings in different brain regions")
    ax.set_xlabel("Time (yrs)")
    ax.set_ylabel(r"$\tau$ concentration $p$")

    plt.tight_layout()
    if save:
        output_path = Path("results") / "timescales_by_seeding_region.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"✅ Plot saved to: {output_path.resolve()}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="FKPP biomarker dynamics by seeding region.")
    parser.add_argument("--tmax", type=float, default=40.0, help="Max simulation time (years)")
    parser.add_argument("--alpha", type=float, default=2.1, help="Reaction rate (default: 2.1)")
    parser.add_argument("--rho", type=float, default=0.01, help="Diffusion coefficient (default: 0.01)")
    parser.add_argument("--save", action="store_true", help="Save figure as png")
    args = parser.parse_args()

    run_timescale_analysis(tmax=args.tmax, alpha=args.alpha, rho=args.rho, save=args.save)


if __name__ == "__main__":
    main()
