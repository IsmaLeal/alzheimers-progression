"""
Analysis Script - Braak V Activation Timing

This script computes the time at which Braak stage V nodes reach a specific
concentration threshold and relates it to the overall brain-wide concentration.
This is meant to be compared with clinically observed Jack curves, where activation
of Braak stage V typically occurs at ~60-80% of total misfolded tau load.

Author: Ismael Leal
Date: 2024-04
"""

# === Imports ===
import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.models.tau_concentration_models import c_curves
from src.data.preprocessing import braak


# === Main Function Returning Activation Time for Braak V and the Global Concentration at that Time ===
def analyse_braak5_activation(
        activation_threshold: float = 0.15,
        c0: float = 0.1,
        l0: float = 0.5,
        tmax: int = 200,
        plot: bool = False
) -> tuple[float, float]:
    """
    Analyse the time and global tau load at which Braak stage V reaches a given threshold.

    Parameters
    ----------
    activation_threshold : float, optional
        Local concentration threshold for Braak V to be considered "activated". Default is 0.15.
    c0 : float, optional
        Initial tau concentration at seeded nodes. Default is 0.1.
    l0 : float, optional
        Initial clearance value. Default is 0.5.
    tmax : float, optional
        Maximum simulation time (years). Default is 200.
    plot : bool, optional
        Whether to show a plot of the concentration dynamics. Default is False.

    Returns
    -------
    arrival_time : float
        Time (years) at which Braak V exceeds the threshold.
    total_c_at_activation : float
        Normalised total tau concentration at that time.
    """
    # Simulate the tau propagation across the brain network under a growth-dominated regime with clearance enabled
    t, c, c_total, l = c_curves(
        c0=c0,
        tmax=tmax,
        l0=l0,
        c_tot=True,
        plot_c=False
    )

    # Rescale total concentration to [0, 1] due to clearance-limited steady state
    scaled_c_total = c_total / np.max(c_total)

    # Compute mean concentration across Braak V nodes
    braak5_nodes = np.array(braak[4]) - 1
    c_braak5 = c[:, braak5_nodes]
    c_braak5_mean = np.mean(c_braak5, axis=1)

    # Compute activation time and global load at activation
    arrival_idx = np.argmax(c_braak5_mean > activation_threshold)
    arrival_time = t[arrival_idx]
    total_c_at_activation = scaled_c_total[arrival_idx]

    # Plot concentration curve with activation threshold
    if plot:
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
        plt.rcParams.update({"font.size": 19})
        fig, ax = plt.subplots(1, 1)
        ax.set_facecolor("#d9e9f9")
        ax.grid(color="white", linestyle='-', linewidth=1)
        ax.plot(t, c_braak5_mean, label="Braak V mean concentration")
        ax.axhline(y=activation_threshold, color="red", linestyle="--", label="Threshold")
        ax.axvline(x=arrival_time, color="gray", linestyle=":", label="Arrival time")
        ax.legend()
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Concentration")
        ax.set_title("Braak V Activation Analysis")
        ax.grid(True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True, fontsize=18)
        plt.show()

    return arrival_time, total_c_at_activation


def main():
    parser = argparse.ArgumentParser(description="Analyse Braak V activation timing and global tau load.")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Concentration threshold for Braak V activation (default: 0.15)")
    parser.add_argument("--c0", type=float, default=0.1,
                        help="Initial tau concentration at seeded nodes (default: 0.1)")
    parser.add_argument("--l0", type=float, default=0.5,
                        help="Initial clearance value (default: 0.5)")
    parser.add_argument("--tmax", type=int, default=200,
                        help="Maximum simulation time in years (default: 200)")
    parser.add_argument("--plot", action="store_true",
                        help="Show plot of concentration and activation threshold")
    args = parser.parse_args()

    arrival_time, total_c = analyse_braak5_activation(
        activation_threshold=args.threshold,
        c0=args.c0,
        l0=args.l0,
        tmax=args.tmax,
        plot=args.plot
    )

    # Report
    if arrival_time == 0:
        print("Braak V activation time is larger than `tmax`. Please increase its value or use the default")
    else:
        print(f"🧠 Arrival time for Braak V activation: {arrival_time:.2f} years")
    print(f"📊 Global tau load at that time: {total_c * 100:.1f}%")
    print("📚 Expected: Braak V activates at 60–80% total concentration (Jack curves)")


if __name__ == "__main__":
    main()