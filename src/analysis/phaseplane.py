"""
Phase Plane Analysis - Single-Node Tau Concentration vs. Clearance

This script computes and visualises the phase space of the coupled tau concentration
(p) and clearance (λ) model (no diffusion). It overlays streamlines of the vector
field and, optionally, the separatrix trajectory defined by the critical initial
concentration p_crit(λ0) along with nearby trajectories to illustrate basin boundaries.

Author: Ismael Leal
Date: 2024-04
"""

# === Imports ===
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Import FKPP simulation function
from src.models.tau_concentration_models import c_curves

# === Plotting settings ===
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 19})
colorcitos1 = iter(['#dc3e04', '#451ddc', '#01dc04', '#dc01d9', '#000000', '#ffa11b', '#1a7c25'])
colorcitos2 = iter(['#dc3e04', '#451ddc', '#01dc04', '#dc01d9', '#000000', '#ffa11b', '#1a7c25'])


def phaseplot(alpha=2.1, rho=0.01, l_crit=0.72, l_inf=0.01, l0_crit=1.4, beta=1, show_pcrit=True, save=False):
    """
    Plot the phase plane with streamlines for misfolded tau concentration and clearance.

    This function plots the phase plane with streamlines. Furthermore, the value of p_crit
    is calculated for the initial clearance l0_crit.

    Parameters
    ----------
    alpha : float, optional
        Nonlinear reaction rate. Default is 2.1.
    rho : float, optional
        Effective diffusion coefficient. Default is 0.01.
    l_crit : float, optional
        Critical clearance value. Default is 0.72.
    l_inf : float, optional
        Global minimum clearance value. Default is 0.01.
    l0_crit : float, optional
        Initial clearance critical value for the bifurcation. Used to compute the
        initial critical concentration value `c0_crit` that follows the trajectory
        separating both steady states in the phase plane. Default is 1.4.
    beta : float, optional
        Global kinetic constant representing node vulnerability. Default is 1.
    show_pcrit : bool, optional
        If True, the trajectory of the critical initial values for the given initial
        clearance will be shown, together with the trajectories of two points with
        slightly higher and slightly lower initial concentration values, to show that
        each converges to a different steady state. Default is True.
    save : bool, optional
        If True, saves the figure to `output_path`. Default is False.
    """
    # Set up a meshgrid and the derivatives for the streamplot
    l_vals, c_vals = np.meshgrid(np.linspace(0, 1.5, 200), np.linspace(0, 1, 200))

    def dcdt(c, l):
        return (l_crit - l_vals) * c_vals - alpha * c_vals**2
    def dldt(c, l):
     return beta * c_vals * (l_inf - l_vals)

    U = dcdt(c_vals, l_vals)
    V = dldt(c_vals, l_vals)

    # Initialise plot
    fig, ax = plt.subplots(1, 1, figsize=(11, 7))

    ax.streamplot(l_vals, c_vals, V, U, color="#451ddc")
    if show_pcrit:
        # Seed with p_crit
        c0_crit = ((alpha * (l_crit - l0_crit) +
                   beta * (l_inf - l_crit) +
                   beta * (l0_crit - l_inf) ** (alpha / beta) * (l_crit - l_inf) ** (1 - alpha / beta)) /
                  (alpha * (alpha - beta)))
        t, c_crit, l_crit = c_curves(c0=c0_crit, rho=rho, alpha=alpha, l0=l0_crit, beta=beta, firstnodes=[i for i in range(83)])
        ax.plot(l_crit[:, 26], c_crit[:, 26], linewidth=2, label=r'$p_0 = p_{crit}$')
        ax.scatter(l0_crit, c0_crit, color='red')
        ax.text(l0_crit, c0_crit, r'($\lambda_0$, p_$crit$($\lambda_0$))', color='black', verticalalignment='center', horizontalalignment='left', fontsize=14)

        p2 = c0_crit + 0.01
        _, c_2, l_2 = c_curves(c0=p2, rho=rho, alpha=alpha, l0=l0_crit, beta=beta, firstnodes=[i for i in range(83)])
        ax.plot(l_2[:, 26], c_2[:, 26], linewidth=2, label=r'$p_0 > p_{crit}$')
        ax.scatter(l0_crit, p2, color='red')
        ax.text(l0_crit, p2, r'($\lambda_0$, p_$crit$($\lambda_0$)+0.01)', color='black', verticalalignment='bottom',
                horizontalalignment='left', fontsize=14)

        p3 = c0_crit - 0.01
        _, c_3, l_3 = c_curves(c0=p3, rho=rho, alpha=alpha, l0=l0_crit, beta=beta, firstnodes=[i for i in range(83)])
        ax.plot(l_3[:, 26], c_3[:, 26], linewidth=2, label=r'$p_0 < p_{crit}$')
        ax.scatter(l0_crit, p3, color='red', label='Initial conditions')
        ax.text(l0_crit, p3, r'($\lambda_0$, p_$crit$($\lambda_0$)-0.01)', color='black', verticalalignment='top',
                horizontalalignment='left', fontsize=14)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True, fontsize=18)

    # Adjust background
    ax.set_facecolor("#d9e9f9")
    ax.grid(color="white", linestyle='-', linewidth=1)

    # Set plot title and labels
    ax.set_title('Trajectories in the phase space')
    ax.set_xlabel(r'Clearance $\lambda$')
    ax.set_ylabel(r'$\tau$ concentration $p$')

    ax.hlines(0, 0, 1.4, color='gray', linestyles='--', label=r'$p=0$')

    # Save or plot
    if save:
        output_path = Path("results") / "phase_plane.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"✅ Saved plot to: {output_path.resolve()}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot tau vs clearance phase plane")
    parser.add_argument("--l0_crit", type=float, default=1.4, help=r"Initial clearance value ($\lambda_0$) for bifurcation")
    parser.add_argument("--alpha", type=float, default=1.4, help=r"Nonlinear reaction/misfolding rate $\alpha$")
    parser.add_argument("--beta", type=float, default=1, help=r"Global kinetic constant representing node vulnerability $\beta$")
    parser.add_argument("--rho", type=float, default=0.01, help=r"Effective diffusion coefficient")
    parser.add_argument("--save", action="store_true", help=r"Save plot as png")
    
    args = parser.parse_args()
    phaseplot(l0_crit=args.l0_crit, alpha=args.alpha, beta=args.beta, rho=args.rho, save=args.save)


if __name__ == "__main__":
    main()
