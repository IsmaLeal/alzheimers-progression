"""
Seeding in Star Graphs — FKPP Tau Clearance Dynamics

This script simulates the evolution of clearance over time in multiple
star-shaped graphs with varying numbers of peripheral nodes. Each graph
is seeded at its central node with a range of initial concentrations.

The final clearance value at the central node is plotted vs. initial seeding
for each network size.

Author: Ismael Leal
Date: 2024-04
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def simulate_star_graph_seeding(
        ns=(10, 20, 30, 40, 50, 60),
        l0: float = 2,
        alpha: float = 2.1,
        rho: float = 0.01,
        beta: float = 1,
        l_crit: float = 0.72,
        l_inf: float = 0.01,
        tmax: float = 50,
        dt: float = 0.1,
        seed_resolution: int = 30,
        save: bool = False,
        output_path: Path = None
) -> None:
    """
    Run FKPP simulations on star graphs with increasing node count.

    Parameters
    ----------
    ns : tuple[int], optional
        List of node counts to simulate.
    l0, alpha, rho, beta, l_crit, l_inf : float, optional
        Model parameters (see FKPP-clearance coupled model equations).
    tmax : float, optional
        Max simulation time in years. Default is 50.
    dt : float, optional
        Timestep for numerical integration. Default is 0.1.
    seed_resolution : int, optional
        Number of different seed values to simulate. Default is 30.
    save : bool, optional
        Whether to save the plot. Default is False
    output_path : Path, optional
        Path to save figure if `save=True`.
    """
    t_vals = np.arange(0, tmax, step=dt)
    seeds = np.linspace(0, 1, seed_resolution)
    l_final = np.zeros((len(ns), len(seeds)))  # rows: graphs, cols: seeds

    for k, N in enumerate(ns):
        # Build star graph Laplacian
        w = np.zeros((N, N))
        w[0, 1:] = 1
        w[1:, 0] = 1
        d = np.diag(w @ np.ones(N))
        lap = d - w

        for j, c0 in enumerate(seeds):
            c = np.zeros((len(t_vals), N))
            l = np.zeros((len(t_vals), N))

            c[0, 0] = c0
            l[0, :] = l0

            for i in range(1, len(t_vals)):
                c_prev, l_prev = c[i - 1], l[i - 1]
                c[i] = c_prev + dt * (
                    -rho * (lap @ c_prev.T)
                    + (l_crit - l_prev) * c_prev
                    - alpha * c_prev**2
                )
                l[i] = l_prev + dt * beta * c_prev * (l_inf - l_prev)

            l_final[k, j] = l[-1, 0]

    # Plotting
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rcParams.update({"font.size": 20})

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = ["#dc3e04", "#451ddc", "#01dc04", "#dc01d9", "#583419", "#ffa11b", "#d1dc00"]
    for idx, N in enumerate(ns):
        ax.plot(seeds, l_final[idx], color=colors[idx % len(colors)], label=f"N = {N}")

    ax.legend(loc="upper right")
    ax.set_facecolor("#d9e9f9")
    ax.grid(color="white", linestyle="-", linewidth=1)
    ax.set_ylabel("Final clearance $\lambda$")
    ax.set_xlabel("Initial seeding $p_0$")
    ax.set_title("Effect of Initial Seeding on Final Clearance (Star Networks)")
    plt.tight_layout()

    if save:
        if output_path is None:
            output_path = Path("results") / "star_graph_seeding.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"✅ Plot saved to: {output_path.resolve()}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Simulate tau clearance in seeded star networks.")
    parser.add_argument("--tmax", type=float, default=50.0, help="Total simulation time (default: 50)")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step for simulation (default: 0.1)")
    parser.add_argument("--l0", type=float, default=2.0, help="Initial clearance (default: 2)")
    parser.add_argument("--save", action="store_true", help="Save the plot to disk.")
    parser.add_argument("--out", type=str, default=None, help="Custom output path (default: results/star_graph_seeding.png)")

    args = parser.parse_args()
    output_path = Path(args.out) if args.out else None

    simulate_star_graph_seeding(
        tmax=args.tmax,
        dt=args.dt,
        l0=args.l0,
        save=args.save,
        output_path=output_path
    )


if __name__ == "__main__":
    main()
