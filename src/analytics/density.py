"""
Vision-Based Customer Trajectory Analytics — Trajectory Density Heatmap Foundation.
===================================================================================
Phase 10.1: Deterministic 2D spatial binning of floor-plane trajectory observations
into a discrete grid, with technical rendering as a floor-plane density heatmap.

KEY ARCHITECTURAL & METHODOLOGICAL PRINCIPLES:
1. Pure Spatial Binning Layer:
   - Consumes pre-assigned floor coordinates (TrajectoryPoint.floor_x, floor_y)
     derived in Phase 5 from homography transformation.
   - Never reruns detection, tracking, homography, zone evaluation, or dwell analysis.
2. Sampled Point Density Semantics:
   - Grid cell values represent discrete counts of sampled trajectory observations
     falling within the spatial boundaries of each cell.
   - Each accepted observation contributes strictly +1 to its corresponding cell.
   - Raw counts measure spatial observation frequency at the video sampling rate
     (e.g., 5 FPS); they do NOT represent unique visitors, continuous physical dwell,
     human occupancy, customer foot traffic, or retail store performance.
3. Explicit Bounds & Resolution:
   - Grid spatial bounds (min_x, max_x, min_y, max_y) and cell dimensions must be
     explicitly provided by configuration or caller parameters.
   - The module NEVER dynamically infers bounds from observed point extrema,
     ensuring that spatial grid geometry remains invariant and comparable across runs.
4. Technical Faithfulness (No Smoothing or Synthetic Hotspots):
   - Cell counts are rendered directly using discrete nearest-neighbor cells
     (interpolation='nearest').
   - No Gaussian blurring, artificial smoothing, kernel density estimation (KDE),
     or arbitrary hotspot thresholds are applied.
5. Clip-Local Identity & Camera Independence:
   - Trajectory observations belong to session-local tracking IDs within a single
     camera calibration frame. No cross-camera stitching or Re-ID is performed.
"""

from dataclasses import dataclass, asdict
import math
import os
from typing import List, Dict, Optional, Tuple, Any, Sequence, Union
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Polygon as MplPolygon

from src.trajectory.trajectory import Trajectory, TrajectoryPoint
from src.analytics.zone import Zone


def _is_finite_number(val: Any) -> bool:
    """Check if value is a finite int or float (excluding booleans)."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    return math.isfinite(val)


@dataclass(frozen=True)
class GridBounds:
    """
    Explicit spatial boundary extent for a 2D floor-plane grid.

    Attributes:
        min_x: Minimum X coordinate in configured floor units (physical scale unverified).
        max_x: Maximum X coordinate in configured floor units (must be > min_x).
        min_y: Minimum Y coordinate in configured floor units (physical scale unverified).
        max_y: Maximum Y coordinate in configured floor units (must be > min_y).
    """
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def __post_init__(self) -> None:
        for name, val in [("min_x", self.min_x), ("max_x", self.max_x),
                          ("min_y", self.min_y), ("max_y", self.max_y)]:
            if not _is_finite_number(val):
                raise ValueError(f"GridBounds {name} must be a finite number, got {val!r}")
        if self.max_x <= self.min_x:
            raise ValueError(f"GridBounds max_x ({self.max_x}) must be strictly greater than min_x ({self.min_x})")
        if self.max_y <= self.min_y:
            raise ValueError(f"GridBounds max_y ({self.max_y}) must be strictly greater than min_y ({self.min_y})")

    @property
    def width(self) -> float:
        """Total spatial width along the X axis."""
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        """Total spatial height along the Y axis."""
        return self.max_y - self.min_y

    def contains(self, x: float, y: float) -> bool:
        """Check if point (x, y) lies within [min_x, max_x] and [min_y, max_y]."""
        return (self.min_x <= x <= self.max_x) and (self.min_y <= y <= self.max_y)

    def to_dict(self) -> Dict[str, float]:
        """Convert bounds to dictionary representation."""
        return {
            "min_x": self.min_x,
            "max_x": self.max_x,
            "min_y": self.min_y,
            "max_y": self.max_y,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_tuple(cls, bounds: Tuple[float, float, float, float]) -> "GridBounds":
        """Construct GridBounds from a (min_x, max_x, min_y, max_y) 4-tuple."""
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 4:
            raise ValueError(f"Expected a 4-element sequence (min_x, max_x, min_y, max_y), got {bounds!r}")
        return cls(min_x=float(bounds[0]), max_x=float(bounds[1]),
                   min_y=float(bounds[2]), max_y=float(bounds[3]))


@dataclass
class TrajectoryDensityGrid:
    """
    Deterministic 2D spatial binning container for floor-plane trajectory observations.

    Grid Matrix Orientation:
        - counts shape: (num_bins_y, num_bins_x)
        - Row index r in [0, num_bins_y - 1] corresponds to Y-axis bins from min_y up to max_y.
        - Column index c in [0, num_bins_x - 1] corresponds to X-axis bins from min_x up to max_x.
        - In matplotlib imshow(..., origin='lower'), row 0 is displayed at the bottom,
          matching Cartesian floor coordinates (X right, Y up).

    Attributes:
        bounds: Explicit GridBounds defining the spatial coordinate extent.
        counts: 2D numpy integer array (int32) containing observation counts per cell.
        num_bins_x: Number of discrete bins along the X axis.
        num_bins_y: Number of discrete bins along the Y axis.
        cell_size_x: Width of each cell along the X axis in floor units.
        cell_size_y: Height of each cell along the Y axis in floor units.
        total_evaluated_points: Total trajectory points inspected during evaluation.
        total_binned_points: Total points successfully mapped into grid cells.
        out_of_bounds_count: Points whose coordinates fell outside grid bounds.
        untransformed_count: Points lacking valid numeric floor coordinates (None or non-finite).
    """
    bounds: GridBounds
    counts: np.ndarray
    num_bins_x: int
    num_bins_y: int
    cell_size_x: float
    cell_size_y: float
    total_evaluated_points: int
    total_binned_points: int
    out_of_bounds_count: int
    untransformed_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.bounds, GridBounds):
            raise TypeError(f"bounds must be a GridBounds instance, got {type(self.bounds).__name__}")
        if not isinstance(self.counts, np.ndarray) or self.counts.ndim != 2:
            raise ValueError("counts must be a 2D numpy array")
        if self.counts.shape != (self.num_bins_y, self.num_bins_x):
            raise ValueError(
                f"counts shape {self.counts.shape} does not match (num_bins_y={self.num_bins_y}, num_bins_x={self.num_bins_x})"
            )
        if self.num_bins_x < 1 or self.num_bins_y < 1:
            raise ValueError("num_bins_x and num_bins_y must be at least 1")
        if self.cell_size_x <= 0.0 or self.cell_size_y <= 0.0:
            raise ValueError("cell_size_x and cell_size_y must be positive")

    @property
    def max_cell_count(self) -> int:
        """Maximum observation count observed across any single grid cell."""
        return int(np.max(self.counts)) if self.counts.size > 0 else 0

    def get_cell_bounds(self, row: int, col: int) -> Tuple[float, float, float, float]:
        """
        Get the spatial bounding box (min_x, max_x, min_y, max_y) of a specific cell.

        Args:
            row: Y-axis bin index (0 <= row < num_bins_y).
            col: X-axis bin index (0 <= col < num_bins_x).

        Returns:
            Tuple of (cell_min_x, cell_max_x, cell_min_y, cell_max_y).
        """
        if not (0 <= row < self.num_bins_y):
            raise IndexError(f"row index {row} out of range [0, {self.num_bins_y - 1}]")
        if not (0 <= col < self.num_bins_x):
            raise IndexError(f"col index {col} out of range [0, {self.num_bins_x - 1}]")

        c_min_x = round(self.bounds.min_x + col * self.cell_size_x, 4)
        c_max_x = round(c_min_x + self.cell_size_x, 4)
        c_min_y = round(self.bounds.min_y + row * self.cell_size_y, 4)
        c_max_y = round(c_min_y + self.cell_size_y, 4)
        return (c_min_x, c_max_x, c_min_y, c_max_y)

    def to_dict(self) -> Dict[str, Any]:
        """Convert density grid metadata and statistics to serializable dictionary."""
        return {
            "bounds": self.bounds.to_dict(),
            "grid_shape": [int(self.num_bins_y), int(self.num_bins_x)],
            "cell_size": [round(self.cell_size_x, 4), round(self.cell_size_y, 4)],
            "num_bins_x": self.num_bins_x,
            "num_bins_y": self.num_bins_y,
            "total_evaluated_points": self.total_evaluated_points,
            "total_binned_points": self.total_binned_points,
            "out_of_bounds_count": self.out_of_bounds_count,
            "untransformed_count": self.untransformed_count,
            "max_cell_count": self.max_cell_count,
            "counts": self.counts.tolist(),
        }


def compute_density_grid(
    trajectories: Sequence[Trajectory],
    bounds: Union[GridBounds, Tuple[float, float, float, float]],
    cell_size: Optional[Union[float, Tuple[float, float]]] = 0.5,
    grid_shape: Optional[Tuple[int, int]] = None,
) -> TrajectoryDensityGrid:
    """
    Bin floor-plane trajectory observations into a deterministic 2D spatial density grid.

    SEMANTICS & BOUNDARY HANDLING:
    - Half-Open Intervals: Points are mapped into cells defined by [x_c, x_c + dx) and [y_r, y_r + dy).
    - Upper Boundary Inclusion: Points lying exactly on max_x or max_y are clamped into
      the final bin (col = num_bins_x - 1, row = num_bins_y - 1).
    - Out-of-Bounds Points: Observations where x < min_x, x > max_x, y < min_y, or y > max_y
      are excluded from the grid and recorded in `out_of_bounds_count`.
    - Untransformed Points: Points where floor_x or floor_y is None or non-finite are recorded
      in `untransformed_count`.
    - Empty Input: If trajectories is empty or contains no in-bounds observations, a zero-filled
      grid of the requested dimensions is returned with total_binned_points = 0.

    Args:
        trajectories: Sequence of Trajectory objects with populated floor_x, floor_y coordinates.
        bounds: Spatial bounds as a GridBounds instance or (min_x, max_x, min_y, max_y) tuple.
        cell_size: Desired spatial cell dimension (float for square cells, or (size_x, size_y)).
                   Ignored if grid_shape is provided. Defaults to 0.5 floor units.
        grid_shape: Optional explicit bin counts (num_bins_x, num_bins_y). If provided, overrides cell_size.

    Returns:
        TrajectoryDensityGrid: Evaluated spatial density grid and diagnostic statistics.
    """
    if not isinstance(trajectories, (list, tuple)):
        raise TypeError(f"trajectories must be a sequence, got {type(trajectories).__name__}")

    # Validate and resolve bounds
    if isinstance(bounds, GridBounds):
        grid_bounds = bounds
    elif isinstance(bounds, (tuple, list)):
        grid_bounds = GridBounds.from_tuple(tuple(bounds))  # type: ignore
    else:
        raise TypeError(f"bounds must be GridBounds or 4-tuple, got {type(bounds).__name__}")

    # Resolve grid dimensions
    if grid_shape is not None:
        if not isinstance(grid_shape, (tuple, list)) or len(grid_shape) != 2:
            raise ValueError(f"grid_shape must be a 2-tuple (num_bins_x, num_bins_y), got {grid_shape!r}")
        nb_x, nb_y = grid_shape
        if not isinstance(nb_x, int) or isinstance(nb_x, bool) or nb_x < 1:
            raise ValueError(f"num_bins_x must be a positive integer >= 1, got {nb_x!r}")
        if not isinstance(nb_y, int) or isinstance(nb_y, bool) or nb_y < 1:
            raise ValueError(f"num_bins_y must be a positive integer >= 1, got {nb_y!r}")
        num_bins_x = nb_x
        num_bins_y = nb_y
        cell_size_x = grid_bounds.width / num_bins_x
        cell_size_y = grid_bounds.height / num_bins_y
    elif cell_size is not None:
        if isinstance(cell_size, (int, float)) and not isinstance(cell_size, bool):
            cs_x = float(cell_size)
            cs_y = float(cell_size)
        elif isinstance(cell_size, (tuple, list)) and len(cell_size) == 2:
            cs_x = float(cell_size[0])
            cs_y = float(cell_size[1])
        else:
            raise ValueError(f"cell_size must be a positive float or (size_x, size_y) tuple, got {cell_size!r}")

        if not _is_finite_number(cs_x) or cs_x <= 0.0:
            raise ValueError(f"cell_size_x must be a finite positive number, got {cs_x!r}")
        if not _is_finite_number(cs_y) or cs_y <= 0.0:
            raise ValueError(f"cell_size_y must be a finite positive number, got {cs_y!r}")

        # Compute bin counts from cell sizes
        num_bins_x = max(1, int(round(grid_bounds.width / cs_x)))
        num_bins_y = max(1, int(round(grid_bounds.height / cs_y)))
        cell_size_x = grid_bounds.width / num_bins_x
        cell_size_y = grid_bounds.height / num_bins_y
    else:
        raise ValueError("Either cell_size or grid_shape must be specified")

    counts = np.zeros((num_bins_y, num_bins_x), dtype=np.int32)
    total_eval = 0
    total_binned = 0
    out_of_bounds = 0
    untransformed = 0

    min_x = grid_bounds.min_x
    max_x = grid_bounds.max_x
    min_y = grid_bounds.min_y
    max_y = grid_bounds.max_y

    for traj in trajectories:
        if not isinstance(traj, Trajectory):
            raise TypeError(f"Expected Trajectory instance in sequence, got {type(traj).__name__}")
        for pt in traj.points:
            total_eval += 1
            fx = pt.floor_x
            fy = pt.floor_y

            # Check if coordinates exist and are valid numbers
            if fx is None or fy is None or not _is_finite_number(fx) or not _is_finite_number(fy):
                untransformed += 1
                continue

            # Check if point falls within spatial bounds
            if fx < min_x or fx > max_x or fy < min_y or fy > max_y:
                out_of_bounds += 1
                continue

            # Compute column (X) index
            if fx >= max_x:
                col = num_bins_x - 1
            else:
                col = int(math.floor((fx - min_x) / cell_size_x))
                col = min(num_bins_x - 1, max(0, col))

            # Compute row (Y) index
            if fy >= max_y:
                row = num_bins_y - 1
            else:
                row = int(math.floor((fy - min_y) / cell_size_y))
                row = min(num_bins_y - 1, max(0, row))

            counts[row, col] += 1
            total_binned += 1

    return TrajectoryDensityGrid(
        bounds=grid_bounds,
        counts=counts,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        total_evaluated_points=total_eval,
        total_binned_points=total_binned,
        out_of_bounds_count=out_of_bounds,
        untransformed_count=untransformed,
    )


def _create_figure_and_axes(figsize: Tuple[float, float], dpi: int) -> Tuple[Figure, Any]:
    """Create an isolated, non-interactive Figure and Axes without mutating process-wide backend."""
    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    return fig, ax


def render_density_heatmap(
    grid: TrajectoryDensityGrid,
    output_path: Optional[str] = None,
    title: str = "Floor-Plane Trajectory Density Heatmap",
    cmap: str = "viridis",
    zones: Optional[Sequence[Zone]] = None,
    figsize: Tuple[float, float] = (10.0, 8.0),
    dpi: int = 150,
    cbar_label: str = "Sampled Trajectory Points (Count)",
    unit_label: str = "configured floor units",
    ax: Optional[Any] = None,
) -> Optional[str]:
    """
    Render a discrete, technically faithful heatmap of the trajectory density grid.

    Rendering Invariants:
    - Nearest-neighbor interpolation: displays raw cell bin counts without artificial smoothing.
    - Cartesian orientation: origin='lower' aligns row 0 with min_y (bottom) and max row with max_y (top).
    - Equal aspect ratio: preserves physical floor-plane spatial proportions.

    Args:
        grid: TrajectoryDensityGrid containing the computed observation counts.
        output_path: Optional file path where the PNG image will be saved.
        title: Plot title string.
        cmap: Matplotlib colormap name (default 'viridis').
        zones: Optional sequence of Zone objects to overlay as polygon outlines.
        figsize: Figure dimensions (width, height) in inches.
        dpi: Resolution of the output rendering.
        cbar_label: Descriptive label for the colorbar legend.
        unit_label: Unit label for axis titles (default 'configured floor units').
        ax: Optional pre-existing matplotlib Axes instance to render onto.

    Returns:
        Optional[str]: Saved file path if output_path was provided, else None.
    """
    if not isinstance(grid, TrajectoryDensityGrid):
        raise TypeError(f"grid must be a TrajectoryDensityGrid, got {type(grid).__name__}")

    b = grid.bounds
    extent = [b.min_x, b.max_x, b.min_y, b.max_y]

    if ax is None:
        fig, ax = _create_figure_and_axes(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    # 1. Plot Heatmap Matrix
    # Use origin='lower' so row 0 is at min_y, matching standard floor coordinates
    # Use interpolation='nearest' to present discrete cell counts without smoothing
    im = ax.imshow(
        grid.counts,
        extent=extent,
        origin="lower",
        cmap=cmap,
        aspect="equal",
        interpolation="nearest",
    )

    # 2. Add Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10, fontweight="bold")

    # 3. Optional Zone Polygon Overlays
    if zones:
        for z in zones:
            if not isinstance(z, Zone):
                continue
            poly_pts = np.array(z.polygon)
            color = z.color if z.color else "#ffffff"
            mpl_poly = MplPolygon(
                poly_pts,
                closed=True,
                facecolor="none",
                edgecolor=color,
                linewidth=2.0,
                linestyle="--",
                label=z.name,
                zorder=3,
            )
            ax.add_patch(mpl_poly)

            # Zone label at centroid
            cx = float(np.mean(poly_pts[:, 0]))
            cy = float(np.mean(poly_pts[:, 1]))
            ax.text(
                cx, cy,
                z.name,
                fontsize=8,
                fontweight="bold",
                color="white",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6, edgecolor=color),
                zorder=4,
            )

    # 4. Axis Labels, Grid, and Title
    ax.set_xlim(b.min_x, b.max_x)
    ax.set_ylim(b.min_y, b.max_y)
    ax.set_xlabel(f"Floor X Coordinates ({unit_label})", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Floor Y Coordinates ({unit_label})", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.5, color="white" if cmap in ["viridis", "magma", "inferno"] else "gray")

    fig.tight_layout()

    if output_path:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        return output_path

    return None
