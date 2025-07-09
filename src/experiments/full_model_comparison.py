"""
Experiment Script — Full Model Comparison

This script plots tau concentration over time for each Braak region under four
different modelling regimes:
- FKPP baseline
- Coupled tau–clearance dynamics
- Linear damage
- Exponential damage

Each model is plotted with a different linestyle, and Braak region colors are consistent.

Author: Ismael Leal
Date: 2024-04
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src.models.run_models import *
from src.data.preprocessing import *


def plot_model_comparison(
        c0: float = 0.2,
        l0: float = 0.1,
        tmax: int = 100,
        kappa: float = 0.9,
        save: bool = False
):
    """
    Plots comparison of all models shown in the project report.

    Parameters
    ----------
    c0 : float, optional
        Initial tau concentration at seed nodes. Default is 0.2.
    l0 : float, optional
        Initial clearance value. Default is 0.1.
    tmax : int, optional
        Maximum simulation time (years). Default is 100.
    kappa : float, optional
        Exponential damage parameter (used in mod_lap="exp"). Default is 1.0.
    save: bool, optional
        Whether to save the figure. Default is False.
    """
    # Run all models
    results = run_all_models(c0=c0, l0=l0, tmax=tmax, kappa=kappa)
    t = results["t"]
    model_outputs = [results["fkpp"], results["coupled"], results["linear"], results["exp"]]
    model_names = ["FKPP", r"Coupled $p$ and $\lambda$", "Linear damage", "Exponential damage"]
    line_styles = ["dotted", "solid", "dashed", "dashdot"]

    # Plotting configuration
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rcParams.update({"font.size": 20})
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_facecolor("#d9e9f9")
    ax.grid(color="white", linestyle="-", linewidth=1)

    # Plot each model's mean Braak zone curve
    legend_model_handles = []
    for i, (model, name, style) in enumerate(zip(model_outputs, model_names, line_styles)):
        color_cycle = iter(["#dc3e04", "#451ddc", "#01dc04", "#dc01d9",
                            "#583419", "#ffa11b", "#d1dc00"])
        for zone in braak:
            ax.plot(t, np.mean(model[:, np.array(zone) - 1], axis=1),
                    color=next(color_cycle), linewidth=2, linestyle=style)
        legend_model_handles.append(
            plt.Line2D([], [], linestyle=style, color="black", label=name)
        )

    # Axis labels
    ax.set_xlabel(r"Time (yrs)")
    ax.set_ylabel(r"$\tau$ concentration $p$")

    # Adjust model legend
    ax.legend(handles=legend_model_handles, loc="center left", bbox_to_anchor=(0.5, 0.6), fontsize=13)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height * 0.9])

    # Global Braak legend
    braak_colors2 = iter(["#dc3e04", "#451ddc", "#01dc04", "#dc01d9",
                          "#583419", "#ffa11b", "#d1dc00"])
    legend_braak_handles = [
        plt.Line2D([], [], color=next(braak_colors2), label=braaknames[i])
        for i in range(len(braak))
    ]
    fig.legend(handles=legend_braak_handles, labels=list(braaknames.values()), loc="center left", fontsize=18, bbox_to_anchor=(0.67, 0.5))

    if save:
        output_path = Path("results") / "all_models_comparison.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"✅ Plot saved to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare different tau propagation models.")
    parser.add_argument("--save", action="store_true", help="Save the figure to png.")
    args = parser.parse_args()
    plot_model_comparison(save=args.save)


if __name__ == "__main__":
    main()
