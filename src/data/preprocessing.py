"""
Data Preprocessing Module

This module loads and defines constants and structures used across the project:
- Scaled node volumes
- Node positions
- Brain connectivity matrices (A2 as default)
- Brain parcellation into Braak stages
- Node Laplacian computation
- Plotting color palettes by Braak stage
- Label by Braak stage

All constants defined here are imported throughout the simulation, modelling, and
analysis pipelines.

Author: Ismael Leal
Date: 2024-04
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


# === Load Volumes and Connectivity Matrix ===

ROOT = Path(__file__).resolve().parents[2]
DB_DIR = ROOT / "databases"

# File paths
volumes_file = DB_DIR / "VolumeOfNodes.csv"
positions_file = DB_DIR / "NamesAndPosition.csv"
A2_file = DB_DIR / "A2.csv"
# A1_file = DB_DIR / "A1.csv"
# A0_file = DB_DIR / "A0.csv"

# Load data
volumes = pd.read_csv(volumes_file, header=None).iloc[:, -1].values         # Volumes
positions = pd.read_csv(positions_file, header=None).iloc[:, -3:].values    # Positions
A2 = pd.read_csv(A2_file, header=None)     # Connectivity matrix: A2_{ij} = n_{ij} / l^2_{ij}
# A1 = pd.read_csv(A1_file, header=None)     # Connectivity matrix: A1_{ij} = n_{ij} / l_{ij}
# A0 = pd.read_csv(A0_file, header=None)     # Connectivity matrix: A0_{ij} = n_{ij}

# Default adjacency matrix
A = A2


# === Network Size ===

nv = A.shape[0]


# === Braak Stage Groupings ===

braak1 = np.array([27, 68])
braak2 = np.array([40, 81])
braak3 = np.array([24, 25, 26, 41, 65, 66, 67, 82])
braak4 = np.array([12, 13, 14, 15, 28, 29, 30, 34, 53, 54, 55, 56, 69, 70, 71, 75])
braak6 = np.array([10, 11, 16, 21, 22, 51, 52, 57, 62, 63])
non_braak = np.array([35, 36, 37, 38, 39, 76, 77, 78, 79, 80])

# Braak V is the leftover set
braaks_set = set(braak1).union(braak2, braak3, braak4, braak6, non_braak)
nodes_set = set(range(1, nv+1))
braak5 = list(nodes_set - braaks_set)

braak = [braak1, braak2, braak3, braak4, braak5, braak6, non_braak]


# === Braak Colours and Names ===

# Use consistent matplotlib colormap
colors = plt.cm.get_cmap('tab10', 10)
braakcolors = [
    "#dc3e04",
    colors(2),
    colors(8),
    colors(4),
    colors(5),
    colors(10),
    colors(7)

]
braaknames = {
    0: 'Braak stage I',
    1: 'Braak stage II',
    2: 'Braak stage III',
    3: 'Braak stage IV',
    4: 'Braak stage V',
    5: 'Braak stage VI',
    6: 'Rest of brain'
}


# === Laplacian ===

D = np.diag(A @ np.ones(nv))    # Degree matrix (sum of weights connected to each node in the diagonal)
lap = D - A                     # Laplacian

__all__ = ["A", "lap", "volumes", "positions", "nv", "braak", "braakcolors", "braaknames"]
