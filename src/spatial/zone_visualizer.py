"""
Vision-Based Customer Trajectory Analytics — Zone & Trajectory Visualizer Module.

This module renders 2D top-down floor maps showing defined retail store zones,
spatial trajectories, and zone-assigned trajectory points.
"""

import os
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.cm as cm

from src.analytics.zone import Zone
from src.trajectory.trajectory import Trajectory


def visualize_zones_and_trajectories(
    zones: List[Zone],
    trajectories: List[Trajectory],
    output_path: str,
    title: str = "Phase 6 — Spatial Zone Trajectory Overlay",
    grid_bounds: Optional[Tuple[float, float, float, float]] = None
) -> str:
    """
    Render a 2D floor-plane visualization with zone overlays and trajectories.

    Args:
        zones (List[Zone]): List of defined spatial retail zones.
        trajectories (List[Trajectory]): List of Trajectory objects with floor coordinates.
        output_path (str): File destination path for saving the PNG plot.
        title (str): Title string for the plot figure.
        grid_bounds (Optional[Tuple[float, float, float, float]]): (min_x, max_x, min_y, max_y).

    Returns:
        str: Output path where figure was saved.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # Color map mapping for zones
    zone_colors: Dict[str, str] = {}
    default_colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6", "#1abc9c", "#e67e22"]
    
    for idx, z in enumerate(zones):
        color = z.color if z.color else default_colors[idx % len(default_colors)]
        zone_colors[z.zone_id] = color

        # Draw Zone Polygon
        poly_arr = np.array(z.polygon)
        mpl_poly = MplPolygon(
            poly_arr,
            closed=True,
            facecolor=color,
            alpha=0.25,
            edgecolor=color,
            linewidth=2.5,
            linestyle="--",
            label=z.name
        )
        ax.add_patch(mpl_poly)

        # Draw Zone Name Label at Centroid
        centroid_x = float(np.mean(poly_arr[:, 0]))
        centroid_y = float(np.mean(poly_arr[:, 1]))
        ax.text(
            centroid_x, centroid_y,
            z.name,
            fontsize=10,
            fontweight="bold",
            color="#2c3e50",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=color)
        )

    # Track color palette for connecting trajectory lines
    track_colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(trajectories))))

    # Draw Trajectories
    for trk_idx, trk in enumerate(trajectories):
        valid_pts = [p for p in trk.points if p.floor_x is not None and p.floor_y is not None]
        if not valid_pts:
            continue

        fx = [p.floor_x for p in valid_pts]
        fy = [p.floor_y for p in valid_pts]

        # Draw continuous path line
        line_color = track_colors[trk_idx % len(track_colors)]
        ax.plot(
            fx, fy,
            linestyle="-",
            linewidth=1.8,
            color=line_color,
            alpha=0.7,
            label=f"Track ID {trk.track_id}"
        )

        # Draw points colored by zone assignment
        for p in valid_pts:
            p_color = zone_colors.get(p.zone_id, "#7f8c8d") if p.zone_id else "#95a5a6"
            ax.scatter(
                p.floor_x, p.floor_y,
                c=p_color,
                s=35,
                edgecolors="black",
                linewidths=0.5,
                zorder=4
            )

        # Mark Start (S) and End (E)
        ax.text(fx[0], fy[0], f"S-{trk.track_id}", fontsize=8, fontweight="bold", color="green", zorder=5)
        ax.text(fx[-1], fy[-1], f"E-{trk.track_id}", fontsize=8, fontweight="bold", color="red", zorder=5)

    # Configure axes limits and grid
    if grid_bounds:
        ax.set_xlim(grid_bounds[0], grid_bounds[1])
        ax.set_ylim(grid_bounds[2], grid_bounds[3])
    else:
        ax.autoscale_view()

    ax.set_xlabel("Floor X Coordinates (Metres / Units)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Floor Y Coordinates (Metres / Units)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path
