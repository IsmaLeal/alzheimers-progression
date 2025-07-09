"""
Model Execution Script - Tau Propagation

This script runs the FKPP model and its three biologically motivated extensions:
    - Pure FKPP (no clearance or damage)
    - Coupled concentration–clearance dynamics
    - Clearance with linearly modulated damage in connectivity
    - Clearance with exponentially modulated damage in connectivity

These outputs are used for downstream analysis and visualisation of dynamic regimes
in tau pathology propagation models.

Author: Ismael Leal
Date: 2024-04
"""

# === Imports ===
import matplotlib.pyplot as plt
from src.models.tau_concentration_models import c_curves


# === Plotting Settings ===
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams.update({"font.size": 20})

def run_all_models(c0: float = 0.2, l0: float = 0.1, tmax: int = 100, kappa: float = 1.0) -> dict:
    """
    Run the FKPP model and its extensions under a shared set of parameters.

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

    Returns
    -------
    results : dict
        Dictionary with time vector `t` and concentration matrices for each model variant.
        Keys: "t", "fkpp", "coupled", "linear", "exp"
    """
    # FKPP (no clearance or damage)
    t, c_fkpp = c_curves(c0=c0, clearance=False, tmax=tmax)

    # Coupled concentration–clearance
    _, c_coupled, _ = c_curves(c0=c0, l0=l0, tmax=tmax)

    # Linearly modulated damage in connectivity
    _, c_linear, _, _ = c_curves(
        c0=c0, l0=l0, tmax=tmax,
        mod_lap="linear", lap_ani=True, plot_c=False
    )

    # Exponentially modulated damage in connectivity
    _, c_exp, _, _ = c_curves(
        c0=c0, l0=l0, tmax=tmax,
        mod_lap="exp", kappa=kappa, lap_ani=True, plot_c=False
    )

    return {
        "t": t,
        "fkpp": c_fkpp,
        "coupled": c_coupled,
        "linear": c_linear,
        "exp": c_exp
    }


__all__ = ["run_all_models"]
