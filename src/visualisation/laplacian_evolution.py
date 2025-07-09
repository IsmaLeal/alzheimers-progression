"""
Visualisation Script — Laplacian Evolution Heatmaps

This script computes the time-varying Laplacians under the exponential
damage model and displays heatmaps at three time snapshots: initial,
midpoint, and final. Used to analyse how structural connectivity is altered.

Author: Ismael Leal  
Date: 2024-04
"""

# === Imports ===
import argparse
import matplotlib.pyplot as plt
from src.models.tau_concentration_models import c_curves


def plot_laplacian_heatmaps(c0: float = 0.2, l0: float = 0.1, tmax: int = 100,
                             kappa: float = 0.9, cmap: str = "viridis") -> None:
    """
    Generates Laplacian heatmaps at start, mid, and end of simulation.

    Parameters
    ----------
    c0 : float, optional
        Initial tau concentration. Default is 0.2.
    l0 : float, optional
        Initial clearance. Default is 0.1.
    tmax : int, optional
        Maximum simulation time (years). Default is 100.
    kappa : float, optional
        Exponential decay parameter. Default is 0.9.
    cmap : str, optional
        Colormap for heatmaps. Default is "viridis".
    """
    # Compute evolving Laplacians
    t, _, _, laps = c_curves(
        c0=c0,
        l0=l0,
        tmax=tmax,
        mod_lap="exp",
        lap_ani=True,
        kappa=kappa,
        plot_c=False
    )

    # Extract snapshots
    lap1 = laps[0]
    lap2 = laps[len(laps) // 2]
    lap3 = laps[-1]

    vmin = min(lap1.min().min(), lap2.min().min(), lap3.min().min())
    vmax = max(lap1.max().max(), lap2.max().max(), lap3.max().max())

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(12, 6), constrained_layout=True)
    im = axs[0].imshow(lap1, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[0].set_title("Initial Laplacian")
    axs[1].imshow(lap2, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[1].set_title("Midpoint Laplacian")
    axs[2].imshow(lap3, vmin=vmin, vmax=vmax, cmap=cmap)
    axs[2].set_title("Final Laplacian")

    for ax in axs:
        ax.axis("off")

    fig.suptitle(r"Modified Graph Laplacian at different timepoints")
    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot Laplacian heatmaps over time under exponential damage.")
    parser.add_argument("--c0", type=float, default=0.2, help="Initial tau concentration (default: 0.2)")
    parser.add_argument("--l0", type=float, default=0.1, help="Initial clearance (default: 0.1)")
    parser.add_argument("--tmax", type=int, default=100, help="Maximum simulation time (default: 100)")
    parser.add_argument("--kappa", type=float, default=1.0, help="Exponential decay parameter (default: 1.0)")
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap for heatmaps (default: viridis)")
    args = parser.parse_args()

    plot_laplacian_heatmaps(
        c0=args.c0,
        l0=args.l0,
        tmax=args.tmax,
        kappa=args.kappa,
        cmap=args.cmap
    )


if __name__ == "__main__":
    main()
