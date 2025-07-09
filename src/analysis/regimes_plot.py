"""
Regime Comparison Plot - FKPP Tau Model

This script compares the dynamics of the FKPP reaction-diffusion model under
distinct parameter regimes: diffusion-dominated (rho/alpha >> 1) and growth-dominated
(rho/alpha << 1). The average concentration vs. time per Braak stage is plotted for both.

Author: Ismael Leal
Date: 2024-04
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from src.data.preprocessing import *
from src.models.tau_concentration_models import  c_curves


def plot_regimes_comparison(
        tmax: int = 40,
        save: bool = False,
        output_path: Path = None
) -> None:
    """
    Plot average tau concentration per Braak stage under two parameter regimes.

    Parameters
    ----------
    tmax : int, optional
        Total simulation time in years. Default is 40.
    save : bool, optional
        If True, saves the figure to `output_path`. Default is False.
    output_path : Path, optional
        Where to save the figure if `save=True`. If None, uses "results/regimes_plot.png".
    :return:
    """
    # Run both simulations
    t, c_diff = c_curves(tmax=40, alpha=0.6, rho=3, mass_conservation=False, clearance=False)
    _, c_growth = c_curves(tmax=40, alpha=2.1, rho=0.01, mass_conservation=False, clearance=False)

    # Plotting settings and figure
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 19})
    colorcitos1 = iter(['#dc3e04', '#451ddc', '#01dc04', '#dc01d9', '#000000', '#ffa11b', '#1a7c25'])
    colorcitos2 = iter(['#dc3e04', '#451ddc', '#01dc04', '#dc01d9', '#000000', '#ffa11b', '#1a7c25'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot diffusion-dominated regime
    for idx, zone in enumerate(braak):
        ax1.plot(t, np.mean(c_diff[:, np.array(zone) - 1], axis=1),
                 color=braakcolors[idx], linewidth=2)
    ax1.set_facecolor("#d9e9f9")
    ax1.grid(color="white", linestyle='-', linewidth=1)
    ax1.set_title(r'Diffusion-dominated regime $\frac{\rho}{\alpha} \gg 1$', pad=15)
    ax1.set_ylabel(r'$\tau$ concentration $p$')
    ax1.set_xlabel('Time (yrs)')

    # Plot growth-dominated regime
    for idx, zone in enumerate(braak):
        label = f"Braak stage {braaknames[idx]}"
        ax2.plot(t, np.mean(c_growth[:, np.array(zone) - 1], axis=1),
                 color=braakcolors[idx], linewidth=2, label=label)
    ax2.legend(loc='center left', bbox_to_anchor=(0.3, 0.4), fancybox=False, shadow=False, fontsize=18)
    ax2.set_facecolor("#d9e9f9")
    ax2.grid(color="white", linestyle='-', linewidth=1)
    ax2.set_title(r'Growth-dominated regime $\frac{\rho}{\alpha} \ll 1$', pad=15)
    #ax2.set_ylabel('Concentration $p$')
    ax2.set_xlabel('Time (yrs)')

    plt.tight_layout()

    if save:
        if output_path is None:
            output_path = Path("results") / "regimes_plot.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"✅ Saved plot to: {output_path.resolve()}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare tau concentration under diffusion- and growth-dominated regimes.")
    parser.add_argument("--tmax", type=int, default=40,
                        help="Simulation time (years). Default is 40.")
    parser.add_argument("--save", action="store_true",
                        help="Save the plot to a file instead of displaying.")
    parser.add_argument("--out", type=str, default=None,
                        help="Custom path to save the plot. Default: results/regimes_plot.png")

    args = parser.parse_args()

    output_path = Path(args.out) if args.out else None
    plot_regimes_comparison(tmax=args.tmax, save=args.save, output_path=output_path)


if __name__ == "__main__":
    main()