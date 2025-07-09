import numpy as np
import matplotlib.pyplot as plt
from typing import List
from src.data.preprocessing import *

"""
This script plots the concentrations according to the
clearance-free FKPP model, incorporating the volume
of each node, hence enforcing mass conservation.

The partial differential equation (PDE) modelling the spread of misfolded
tau proteins is simplified to a system of ordinary differential equations (ODEs)
in time. These time derivatives are approximated by finite differences, resulting
in an explicit Euler scheme.
"""

dt = 0.1

lap_history = []
lap_history.append(lap)


def c_curves(
        c0: float = 1/20, l0: float = 0.5,
        alpha: float = 2.1, rho: float = 0.01,
        l_crit: float = 0.72, l_inf: float = 0.01,
        beta: float = 1, kappa: float = 1,
        tmax: int = 80,
        firstnodes: list = [27-1, 68-1],
        mass_conservation: bool = True, clearance: bool = True,
        all_nodes: bool = False, mod_lap: bool = False,
        lap_ani: bool =False, plot_c: bool = False,
        c_tot: bool = False
) -> List:
    """
    Plots the time evolution of the relative concentration of misfolded tau proteins.

    This function solves the explicit Euler scheme for the system of ODEs. The user can
    choose whether to plot the evolution of all nodes or the average grouped by Braak stage,
    whether to account for mass conservation or not, include the clearance model or not,
    and whether to include damage or not.

    Parameters
    ----------
    c0 : float, optional
        Relative initial concentration of the nodes with a nonzero initial seeding. Default is 1/20.
    l0 : float, optional
        Initial clearance value for all nodes. Default is 0.5.
    alpha : float, optional
        Nonlinear reaction rate. Default is 2.1.
    rho : float, optional
        Effective diffusion coefficient. Default is 0.01.
    l_crit : float, optional
        Critical clearance value. Default is 0.72.
    l_inf : float, optional
        Global minimum clearance level. Default is 0.01.
    beta : float, optional
    	Global kinetic constant representing node vulnerability. Default is 1.
    kappa : float, optional
    	Relative connectivity deterioration. Value between 0 and 1 that determines how fast axonal
    	pathways are damaged by the increasing concentration. Default is 1.
    tmax : int, optional
    	Maximum time of the simulation in years. Default is 80.
    firstnodes : list, optional
    	List displaying the indices of the nodes with a nonzero initial misfolded tau seeding.
    mass_conservation : bool, optional
    	If True, the diffusion term is scaled by the volume of each node to ensure mass conservation.
    	Default is True.
    clearance : bool, optional
    	If True, the system is solved taking clearance into account. Else, the system is the simple
    	FKPP reaction-diffusion model. Default is True.
    all_nodes : bool, optional
    	If True, all nodes' evolution is shown. Otherwise, an average concentration per Braak stage
    	is shown. Default is False.
    mod_lap : bool, optional
    	If True, the Laplacian matrix is modified as the concentration of misfolded tau proteins
    	increases, modelling neural damage. Default is False.
    lap_ani : bool, optional
    	If True, plots an animation of the evolution of the Laplacian. Default is False.
    plot_c : bool, optional
    	If True, the concentration of the different Braak stages averaged OR all nodes (depending on `all_nodes`)
    	is plotted. Default is False.
    c_tot : bool, optional
    	If True, a biomarker curve is plotted showing the total concentration accross all nodes.
    	Defaultis False.

    Returns
    -------
    vals : list
        List of arrays with at least two elements: an array with time values and an array
        with the corresponding concentration values for each node. If `clearance` is True,
        then an array with the clearance values for each node is added to `vals`. If
        `c_total` is also True, then a 1-d array with the normalised total concentration
        is added to `vals`, and if `mod_lap` is also True, then a list with the history of
        the Laplacian values will also be returned.
    """
    
    t_vals = np.arange(0, tmax, step=dt)    # Discretised time values
    global lap_history, lap, volumes

    if not mass_conservation:
        volumes = 1

    if clearance:
        # Initialise arrays to store concentration and clearance values
        c = np.zeros((len(t_vals), nv))
        l = np.zeros((len(t_vals), nv))
        q = np.zeros((len(t_vals), nv))

        # Enforce the initial conditions (homogeneous for clearance l)
        c[0, firstnodes] += c0
        l[0, :] += l0

        i = 1
        while i * dt < tmax:
            # Explicit finite difference scheme
            c[i, :] = c[i-1, :] + dt * (-rho * (lap @ c[i-1, :].T) / volumes + (l_crit - l[i-1, :]) * c[i-1, :] - alpha * c[i-1, :]**2)
            l[i, :] = l[i-1, :] + dt * (beta * c[i-1, :] * (l_inf - l[i-1, :]))
            q[i, :] = q[i-1, :] + dt * (beta * c[i-1, :] * (1 - q[i-1, :]))

            if mod_lap == 'exp':
                qi = q[i, :].reshape((-1, 1))
                qj = q[i, :].reshape((1, -1))
                a = (np.exp(2*kappa - qi - qj) - 1) / (np.exp(2*kappa) - 1)
                a[a < 0 ]= 0
                lap = lap * a
                lap[lap < 0] = 0
            elif mod_lap == 'linear':
                qi = q[i, :].reshape((-1, 1))
                qj = q[i, :].reshape((1, -1))
                a = (2 - qi - qj) / 2
                a[a < 0] = 0
                lap = lap * a
            elif mod_lap == 'nonlinear':
                ci = c[i, :].reshape((-1, 1))
                cj = c[i, :].reshape((1, -1))
                lap = lap * (1 - ci * cj)
            else:
                pass

            lap_history.append(lap)
            i += 1

    # If clearance is not accounted for
    else:
        # Initial condition for each node's concentration
        c = np.zeros((len(t_vals), nv))
        c[0, firstnodes] += c0

        i = 1
        while i * dt < tmax:
            c[i, :] = c[i-1, :] + dt * (-rho * (lap @ c[i-1, :].T) / volumes + alpha * c[i-1, :] * (1 - c[i-1, :]))

            if mod_lap == 'exp':
                ci = c[i, :].reshape((-1, 1))
                cj = c[i, :].reshape((1, -1))
                lap = lap * (np.exp(2 - ci - cj) - 1) / (np.exp(2) - 1)
            elif mod_lap == 'linear':
                ci = c[i, :].reshape((-1, 1))
                cj = c[i, :].reshape((1, -1))
                lap = lap * (2 - ci - cj) / 2
            else:
                pass

            lap_history.append(lap)
            i += 1

    vals = [t_vals, c]

    # Biomarker curve
    if c_tot:
        #c_total = np.mean(c1, axis=1)
        c_total = np.sum(c, axis=1)
        c_total = c_total / max(c_total)
        vals.append(c_total)

    # Plot braak and total biomarker curves
    if plot_c:
        # Plotting settings
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rcParams.update({'font.size': 20})
        colorcitos = iter(['#dc3e04', '#451ddc', '#01dc04', '#dc01d9', '#583419', '#ffa11b', '#d1dc00'])

        # Initialise plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        plt.subplots_adjust(right=0.8)

        # Plot
        if all_nodes:
            plot = [ax.plot(t_vals, c[:, i], color=braakcolors[j]) for i in range(nv) for j in range(len(braakcolors)) if i+1 in braak[j]]

            # Include legend with as many entries as Braak zones
            labels, handles = [], []
            for idx, colour in enumerate(braakcolors):
                # Create dummy line and label for each legend entry
                line = plt.Line2D([], [], color=colour, label=f'Braak zone {idx + 1}')
                label = f'Braak zone {idx + 1 if idx != 6 else "Unlabelled"}'
                handles.append(line)
                labels.append(label)

            # Show legend
            box = ax.get_position()
            ax.set_position([box.x0, box.y0+0.05, box.width*0.7, box.height*0.9])
            fig.legend(handles=handles, labels=labels, loc='upper right')
        else:
            [ax.plot(t_vals,
                     np.mean(c[:, np.array(zone)-1], axis=1),
                     color=next(colorcitos), linewidth=2, label='Braak stage ' + str(idx+1) if idx != 6 else "Unlabelled nodes") for idx, zone in enumerate(braak)]
            #plt.legend()
        if c_tot:
            plt.plot(t_vals, c_total, color='black', linewidth=2, linestyle='--', label='Biomarker curve')
        ax.legend(loc='center left', bbox_to_anchor=(0.45,0.4), fancybox=True, shadow=True, fontsize=18)
        ax.set_facecolor("#d9e9f9")
        ax.grid(color="white", linestyle='-', linewidth=1)
        ax.set_title(r'Growth-dominated regime $\frac{\rho}{\alpha} \ll 1$', pad=15)
        ax.set_ylabel('Concentration $p$')
        ax.set_xlabel('Time')

        plt.show()

    if clearance:
        vals.append(l)

    if lap_ani:
        vals.append(lap_history)

    return vals

__all__ = ["c_curves"]