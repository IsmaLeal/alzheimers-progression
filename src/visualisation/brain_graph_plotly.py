"""
Visualisation Script — 3D Brain Network Plot

This script generates a 3D network plot of the brain using Plotly,
where nodes are colour-coded by Braak region and sized by their volume.

Author: Ismael Leal
Date: 2024-04
"""
import argparse
import networkx as nx
import plotly.graph_objects as go

from src.data.preprocessing import *


def plot_brain_graph(volume_scale: float = 200.0):
    """
    Creates a 3D brain graph with Braak-colored nodes and volume-based sizing.

    Parameters
    ----------
    volume_scale : float, optional
        Factor to scale down node volumes for visual clarity. Default is 400.
    """
    scaled_volumes = volumes / volume_scale
    G = nx.from_numpy_array(A.values)

    # Assign coordinates and sizes
    node_pos = {i+1: (positions[i, 0], positions[i, 1], positions[i, 2]) for i in range(nv)}
    node_vol = {i+1: scaled_volumes[i] for i in range(nv)}

    braakcolors = ['#dc3e04', '#451ddc', '#01dc04', '#dc01d9', '#583419', '#ffa11b', '#d1dc00']

    # Create 3D plot
    fig = go.Figure()
    for idx, zone in enumerate(braak):
        x, y, z = zip(*[node_pos[node] for node in zone])
        node_sizes = [node_vol[node] for node in zone]
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(size=node_sizes, color=braakcolors[idx]),
            name=braaknames[idx]
        ))

    # Add edges
    for edge in G.edges():
        x_coords, y_coords, z_coords = zip(*[node_pos[node + 1] for node in edge])
        fig.add_trace(go.Scatter3d(
            x=[x_coords[0], x_coords[1]],
            y=[y_coords[0], y_coords[1]],
            z=[z_coords[0], z_coords[1]],
            mode="lines",
            line=dict(color="grey", width=2),
            showlegend=False
        ))

    # Layout settings
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
        legend=dict(title="Braak zones", itemsizing="constant")
    )

    fig.show()


def main():
    parser = argparse.ArgumentParser(description="3D brain graph visualisation by Braak stage and node volume.")
    parser.add_argument("--scale", type=float, default=400.0,
                        help="Volume scaling factor for node size (default: 400)")
    args = parser.parse_args()
    plot_brain_graph(volume_scale=args.scale)


if __name__ == "__main__":
    main()
