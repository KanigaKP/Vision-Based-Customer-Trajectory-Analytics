"""
Phase 10 — Unit & Integration Tests for Trajectory Density Heatmaps.
====================================================================
Validates 2D spatial binning of floor-plane trajectories into a discrete density grid:
1. GridBounds validation and edge cases.
2. Empty trajectories and zero-density grid initialization.
3. Known cell counts, spatial orientation, and Cartesian coordinate mapping.
4. Cell-boundary coordinates and interval inclusion semantics.
5. Out-of-bounds and untransformed point handling with diagnostic invariant checks.
6. Accumulation across multiple clip-local tracks.
7. Explicit grid_shape vs cell_size resolution.
8. Serialization to dictionary via to_dict().
9. Faithful heatmap rendering to file (non-blank verification, zone overlay, coordinate labels).
10. End-to-end integration boundary from Trajectory objects through density grid, zone overlay, and JSON export.
11. Verification that rendering does not mutate the process-wide Matplotlib backend.
12. Handling of sparse and boundary-truncated trajectories with diagnostic preservation.
"""

import os
import pytest
import numpy as np

from src.trajectory.trajectory import Trajectory, TrajectoryPoint
from src.analytics.zone import Zone
from src.analytics.density import (
    GridBounds,
    TrajectoryDensityGrid,
    compute_density_grid,
    render_density_heatmap,
)


def _make_traj(track_id: int, floor_coords: list, start_frame: int = 0) -> Trajectory:
    """Helper to construct Trajectory with given floor coordinates."""
    traj = Trajectory(track_id=track_id)
    for idx, (fx, fy) in enumerate(floor_coords):
        traj.add_point(
            TrajectoryPoint(
                track_id=track_id,
                frame_idx=start_frame + idx,
                timestamp_sec=round(idx * 0.2, 3),
                image_x=100.0,
                image_y=200.0,
                floor_x=fx,
                floor_y=fy,
            )
        )
    return traj


# ==============================================================================
# 1. GridBounds Validation
# ==============================================================================
def test_grid_bounds_validation():
    """Test 1: GridBounds validates coordinate order, finite numbers, and dimensions."""
    # Valid bounds
    b = GridBounds(min_x=0.0, max_x=10.0, min_y=0.0, max_y=6.0)
    assert b.width == 10.0
    assert b.height == 6.0
    assert b.contains(5.0, 3.0)
    assert b.contains(0.0, 0.0)
    assert b.contains(10.0, 6.0)
    assert not b.contains(-0.1, 3.0)
    assert not b.contains(5.0, 6.1)

    # Rejection of invalid coordinate ordering
    with pytest.raises(ValueError, match="strictly greater"):
        GridBounds(min_x=10.0, max_x=5.0, min_y=0.0, max_y=6.0)

    with pytest.raises(ValueError, match="strictly greater"):
        GridBounds(min_x=0.0, max_x=10.0, min_y=6.0, max_y=6.0)

    # Rejection of non-finite numbers
    with pytest.raises(ValueError, match="finite number"):
        GridBounds(min_x=float("nan"), max_x=10.0, min_y=0.0, max_y=6.0)

    # From tuple constructor
    b2 = GridBounds.from_tuple((0.0, 10.0, 0.0, 8.0))
    assert b2.max_y == 8.0


# ==============================================================================
# 2. Empty Input Behavior
# ==============================================================================
def test_empty_trajectories_input():
    """Test 2: Empty inputs produce a valid zero-filled grid without exceptions."""
    bounds = GridBounds(0.0, 10.0, 0.0, 6.0)

    # Empty trajectory list
    grid_empty = compute_density_grid([], bounds=bounds, cell_size=0.5)
    assert grid_empty.total_evaluated_points == 0
    assert grid_empty.total_binned_points == 0
    assert grid_empty.out_of_bounds_count == 0
    assert grid_empty.untransformed_count == 0
    assert grid_empty.max_cell_count == 0
    assert np.all(grid_empty.counts == 0)
    assert grid_empty.counts.shape == (12, 20)  # 6.0/0.5=12 rows (Y), 10.0/0.5=20 cols (X)

    # Trajectory with no points
    empty_traj = Trajectory(track_id=1, points=[])
    grid_no_pts = compute_density_grid([empty_traj], bounds=bounds, cell_size=0.5)
    assert grid_no_pts.total_evaluated_points == 0
    assert grid_no_pts.total_binned_points == 0
    assert np.all(grid_no_pts.counts == 0)


# ==============================================================================
# 3. Known Cell Counts & Spatial Orientation
# ==============================================================================
def test_known_cell_counts_and_orientation():
    """Test 3: Floor points map deterministically into expected grid cells and axes."""
    # Bounds: [0, 10] x [0, 6], cell_size = 1.0 => 6 rows (Y), 10 cols (X)
    bounds = GridBounds(0.0, 10.0, 0.0, 6.0)

    # Points designed to hit specific cells:
    # Point 1: (0.5, 0.5) -> col 0, row 0 (bottom-left)
    # Point 2: (9.5, 5.5) -> col 9, row 5 (top-right)
    # Point 3: (4.2, 2.8) -> col 4, row 2 (center-ish)
    # Point 4: (4.8, 2.1) -> col 4, row 2 (same cell as Point 3!)
    traj = _make_traj(1, [
        (0.5, 0.5),
        (9.5, 5.5),
        (4.2, 2.8),
        (4.8, 2.1),
    ])

    grid = compute_density_grid([traj], bounds=bounds, cell_size=1.0)

    assert grid.counts.shape == (6, 10)
    assert grid.total_evaluated_points == 4
    assert grid.total_binned_points == 4
    assert grid.out_of_bounds_count == 0

    # Verify counts at expected coordinates
    assert grid.counts[0, 0] == 1  # Bottom-left: row 0, col 0
    assert grid.counts[5, 9] == 1  # Top-right: row 5, col 9
    assert grid.counts[2, 4] == 2  # Accumulated in cell (row 2, col 4)
    assert grid.max_cell_count == 2

    # Verify other cells are strictly 0
    mask = np.ones((6, 10), dtype=bool)
    mask[0, 0] = False
    mask[5, 9] = False
    mask[2, 4] = False
    assert np.all(grid.counts[mask] == 0)


# ==============================================================================
# 4. Cell-Boundary Coordinates Semantics
# ==============================================================================
def test_boundary_coordinates_semantics():
    """Test 4: Half-open interval rules and upper-boundary clamping."""
    # Bounds: [0, 4] x [0, 4], cell_size = 2.0 => 2 rows, 2 cols
    # Cell (0,0): [0.0, 2.0) x [0.0, 2.0)
    # Cell (0,1): [2.0, 4.0] x [0.0, 2.0)
    # Cell (1,0): [0.0, 2.0) x [2.0, 4.0]
    # Cell (1,1): [2.0, 4.0] x [2.0, 4.0]
    bounds = GridBounds(0.0, 4.0, 0.0, 4.0)

    traj = _make_traj(1, [
        (0.0, 0.0),  # Exact min bounds -> (0, 0)
        (1.99, 1.99), # Interior of (0, 0) -> (0, 0)
        (2.0, 0.0),  # Exactly on boundary x=2.0 -> enters col 1 -> (0, 1)
        (0.0, 2.0),  # Exactly on boundary y=2.0 -> enters row 1 -> (1, 0)
        (4.0, 4.0),  # Exact max bounds -> clamped into final cell (1, 1)
    ])

    grid = compute_density_grid([traj], bounds=bounds, cell_size=2.0)
    assert grid.counts.shape == (2, 2)
    assert grid.counts[0, 0] == 2  # (0.0, 0.0) and (1.99, 1.99)
    assert grid.counts[0, 1] == 1  # (2.0, 0.0)
    assert grid.counts[1, 0] == 1  # (0.0, 2.0)
    assert grid.counts[1, 1] == 1  # (4.0, 4.0)
    assert grid.total_binned_points == 5


# ==============================================================================
# 5. Out-of-Bounds & Untransformed Point Diagnostics
# ==============================================================================
def test_out_of_bounds_and_untransformed_points():
    """Test 5: Points outside bounds or lacking floor coordinates are counted in diagnostics."""
    bounds = GridBounds(0.0, 10.0, 0.0, 6.0)

    traj = _make_traj(1, [
        (5.0, 3.0),   # In bounds (binned)
        (-1.0, 3.0),  # Out of bounds X < min_x
        (11.0, 3.0),  # Out of bounds X > max_x
        (5.0, -0.5),  # Out of bounds Y < min_y
        (5.0, 6.5),   # Out of bounds Y > max_y
        (None, 3.0),  # Untransformed None
        (5.0, None),  # Untransformed None
        (float("nan"), 3.0), # Untransformed NaN
    ])

    grid = compute_density_grid([traj], bounds=bounds, cell_size=1.0)

    assert grid.total_evaluated_points == 8
    assert grid.total_binned_points == 1
    assert grid.out_of_bounds_count == 4
    assert grid.untransformed_count == 3
    # Verify strict invariant
    assert grid.total_evaluated_points == (
        grid.total_binned_points + grid.out_of_bounds_count + grid.untransformed_count
    )


# ==============================================================================
# 6. Multi-Track Accumulation
# ==============================================================================
def test_multi_track_accumulation():
    """Test 6: Multiple independent tracks accumulate into density grid without ID merging."""
    bounds = GridBounds(0.0, 10.0, 0.0, 6.0)

    # Track 10 passes through cell (row 2, col 3)
    t1 = _make_traj(10, [(3.2, 2.2), (3.4, 2.4)])
    # Track 20 passes through same cell (row 2, col 3)
    t2 = _make_traj(20, [(3.6, 2.6)])
    # Track 30 passes through cell (row 0, col 1)
    t3 = _make_traj(30, [(1.5, 0.5)])

    grid = compute_density_grid([t1, t2, t3], bounds=bounds, cell_size=1.0)

    assert grid.total_binned_points == 4
    assert grid.counts[2, 3] == 3  # 2 points from t1 + 1 point from t2
    assert grid.counts[0, 1] == 1  # 1 point from t3
    assert grid.max_cell_count == 3


# ==============================================================================
# 7. Explicit grid_shape Parameter
# ==============================================================================
def test_explicit_grid_shape_parameter():
    """Test 7: Specifying grid_shape sets exact matrix dimensions and bin widths."""
    bounds = GridBounds(0.0, 10.0, 0.0, 5.0)

    # Request exactly 50 bins in X, 25 bins in Y
    grid = compute_density_grid([], bounds=bounds, grid_shape=(50, 25))

    assert grid.counts.shape == (25, 50)
    assert grid.num_bins_x == 50
    assert grid.num_bins_y == 25
    assert round(grid.cell_size_x, 4) == 0.2
    assert round(grid.cell_size_y, 4) == 0.2

    # Verify cell bounds helper
    cb = grid.get_cell_bounds(row=0, col=0)
    assert cb == (0.0, 0.2, 0.0, 0.2)

    cb_last = grid.get_cell_bounds(row=24, col=49)
    assert cb_last == (9.8, 10.0, 4.8, 5.0)


# ==============================================================================
# 8. Serialization Round-Trip
# ==============================================================================
def test_to_dict_serialization():
    """Test 8: TrajectoryDensityGrid serializes completely to dictionary."""
    bounds = GridBounds(0.0, 4.0, 0.0, 2.0)
    traj = _make_traj(1, [(1.0, 1.0)])
    grid = compute_density_grid([traj], bounds=bounds, cell_size=1.0)

    d = grid.to_dict()
    assert d["num_bins_x"] == 4
    assert d["num_bins_y"] == 2
    assert d["total_binned_points"] == 1
    assert d["max_cell_count"] == 1
    assert isinstance(d["counts"], list)
    assert len(d["counts"]) == 2  # 2 rows
    assert len(d["counts"][0]) == 4  # 4 cols


# ==============================================================================
# 9. Heatmap Rendering, Labeling & Non-Blank Output
# ==============================================================================
def test_rendering_produces_nonblank_heatmap(tmp_path, monkeypatch):
    """Test 9: render_density_heatmap generates non-empty PNG and verifies coordinate labels."""
    import matplotlib.pyplot as plt

    bounds = GridBounds(0.0, 10.0, 0.0, 6.0)
    t = _make_traj(1, [(2.0, 2.0), (2.0, 2.0), (8.0, 4.0)])
    grid = compute_density_grid([t], bounds=bounds, cell_size=0.5)

    # Optional zone overlay
    zone_a = Zone("zone_a", "Shelf Area", [(1.0, 1.0), (4.0, 1.0), (4.0, 4.0), (1.0, 4.0)], color="#3498db")

    # Spy on _create_figure_and_axes to inspect configured Axes labels
    import src.analytics.density as density_mod

    captured_axes = []
    real_create = density_mod._create_figure_and_axes

    def spy_create(*args, **kwargs):
        fig, ax = real_create(*args, **kwargs)
        captured_axes.append(ax)
        return fig, ax

    monkeypatch.setattr(density_mod, "_create_figure_and_axes", spy_create)

    # 1. Default unit_label execution
    out_file = str(tmp_path / "test_heatmap.png")
    result = render_density_heatmap(
        grid=grid,
        output_path=out_file,
        title="Test Density Heatmap",
        zones=[zone_a],
    )

    assert result == out_file
    assert os.path.exists(out_file)
    file_size = os.path.getsize(out_file)
    assert file_size > 5000, f"Heatmap file size too small ({file_size} bytes), possibly blank"

    # Verify default X and Y labels use 'configured floor units' and do not claim metres
    ax_default = captured_axes[0]
    assert ax_default.get_xlabel() == "Floor X Coordinates (configured floor units)"
    assert ax_default.get_ylabel() == "Floor Y Coordinates (configured floor units)"
    assert "metres" not in ax_default.get_xlabel().lower()
    assert "metres" not in ax_default.get_ylabel().lower()
    assert "meter" not in ax_default.get_xlabel().lower()
    assert "meter" not in ax_default.get_ylabel().lower()

    # 2. Supported custom unit_label execution
    out_custom = str(tmp_path / "test_heatmap_custom.png")
    result_custom = render_density_heatmap(
        grid=grid,
        output_path=out_custom,
        title="Custom Unit Heatmap",
        unit_label="normalized grid units",
    )

    assert result_custom == out_custom
    assert os.path.exists(out_custom)
    ax_custom = captured_axes[1]
    assert ax_custom.get_xlabel() == "Floor X Coordinates (normalized grid units)"
    assert ax_custom.get_ylabel() == "Floor Y Coordinates (normalized grid units)"


# ==============================================================================
# 10. Density Pipeline Integration Boundary
# ==============================================================================
def test_density_pipeline_integration_boundary(tmp_path):
    """Test 10: Validates end-to-end integration boundary from Trajectory objects
    to density grid computation, diagnostic tracking, zone-overlay rendering,
    and structured JSON-serializable export without external model/video dependencies.
    """
    # 1. Setup floor bounds and zone definitions
    bounds = GridBounds(min_x=0.0, max_x=10.0, min_y=0.0, max_y=8.0)
    corridor_zone = Zone(
        zone_id="south_corridor",
        name="South Corridor",
        polygon=[(0.0, 4.0), (10.0, 4.0), (10.0, 8.0), (0.0, 8.0)],
        color="#2ecc71",
    )

    # 2. Construct multi-track trajectory collection mimicking upstream output
    # Track 101: 5 points within corridor zone
    t1 = _make_traj(101, [(2.0, 5.0), (2.5, 5.5), (3.0, 6.0), (3.5, 6.5), (4.0, 7.0)])
    # Track 102: 3 points in north area (in bounds, out of corridor)
    t2 = _make_traj(102, [(1.0, 1.0), (1.5, 1.5), (2.0, 2.0)])
    # Track 103: 4 points, with 2 out of bounds and 1 untransformed None
    t3 = _make_traj(103, [(5.0, 3.0), (11.0, 3.0), (-1.0, 2.0), (None, 2.0)])

    trajectories = [t1, t2, t3]

    # 3. Compute density grid
    grid = compute_density_grid(
        trajectories=trajectories,
        bounds=bounds,
        cell_size=0.5,
    )

    # 4. Verify structural grid invariants
    assert grid.num_bins_x == 20
    assert grid.num_bins_y == 16
    assert grid.total_evaluated_points == 12
    assert grid.total_binned_points == 9
    assert grid.out_of_bounds_count == 2
    assert grid.untransformed_count == 1
    assert grid.total_evaluated_points == (
        grid.total_binned_points + grid.out_of_bounds_count + grid.untransformed_count
    )

    # 5. Verify render execution with zone overlay to file
    out_png = str(tmp_path / "integration_heatmap.png")
    rendered_path = render_density_heatmap(
        grid=grid,
        output_path=out_png,
        title="Integration Boundary Density Heatmap",
        zones=[corridor_zone],
        unit_label="configured floor units",
    )
    assert rendered_path == out_png
    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 5000

    # 6. Verify serialization contract
    summary = grid.to_dict()
    assert summary["bounds"]["min_x"] == 0.0
    assert summary["bounds"]["max_x"] == 10.0
    assert summary["bounds"]["min_y"] == 0.0
    assert summary["bounds"]["max_y"] == 8.0
    assert summary["total_binned_points"] == 9
    assert summary["out_of_bounds_count"] == 2
    assert summary["untransformed_count"] == 1
    assert len(summary["counts"]) == 16
    assert len(summary["counts"][0]) == 20


# ==============================================================================
# 11. Backend Side-Effect Isolation Verification
# ==============================================================================
def test_rendering_does_not_mutate_matplotlib_backend(tmp_path):
    """Test 11: render_density_heatmap uses isolated rendering and does not mutate process backend."""
    import matplotlib
    initial_backend = matplotlib.get_backend()

    bounds = GridBounds(0.0, 10.0, 0.0, 6.0)
    t = _make_traj(1, [(2.0, 2.0)])
    grid = compute_density_grid([t], bounds=bounds, cell_size=0.5)

    out_file = str(tmp_path / "backend_check.png")
    render_density_heatmap(grid=grid, output_path=out_file)

    current_backend = matplotlib.get_backend()
    assert current_backend == initial_backend, (
        f"Matplotlib backend unexpectedly mutated from {initial_backend} to {current_backend}"
    )
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 5000


# ==============================================================================
# 12. Sparse & Boundary-Truncated Trajectory Handling
# ==============================================================================
def test_sparse_and_boundary_truncated_trajectories_handling(tmp_path):
    """Test 12: Validates density grid and rendering with sparse, boundary-truncated
    trajectories (matching MERL recorded video pattern) without external video or weights.
    """
    bounds = GridBounds(min_x=0.0, max_x=10.0, min_y=0.0, max_y=6.0)

    # Construct synthetic trajectory matching MERL boundary-truncated distribution:
    # 3 points just inside bounds (e.g., near top right border),
    # 71 points outside bounds (y > 6.0 or x > 10.0).
    in_bounds_pts = [(9.8, 5.8), (9.8, 5.8), (9.7, 5.2)]
    out_bounds_pts = [(10.5, 6.2 + 0.05 * i) for i in range(71)]
    all_pts = in_bounds_pts + out_bounds_pts

    traj = _make_traj(1, all_pts)

    grid = compute_density_grid([traj], bounds=bounds, cell_size=0.5)

    assert grid.total_evaluated_points == 74
    assert grid.total_binned_points == 3
    assert grid.out_of_bounds_count == 71
    assert grid.untransformed_count == 0
    # Strict conservation invariant
    assert grid.total_evaluated_points == (
        grid.total_binned_points + grid.out_of_bounds_count + grid.untransformed_count
    )
    assert grid.max_cell_count == 2  # cell at (9.8, 5.8) has 2 points

    # Verify rendering of sparse grid with zone overlays succeeds
    zone = Zone(
        zone_id="shelf_b",
        name="Shelf B",
        polygon=[(6.0, 1.0), (9.5, 1.0), (9.5, 5.0), (6.0, 5.0)],
        color="#2ecc71",
    )
    out_file = str(tmp_path / "sparse_heatmap.png")
    result = render_density_heatmap(
        grid=grid,
        output_path=out_file,
        title="Sparse / Boundary-Truncated Test Heatmap",
        zones=[zone],
        unit_label="configured floor units",
    )

    assert result == out_file
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 5000

    # Verify serialization
    d = grid.to_dict()
    assert d["total_evaluated_points"] == 74
    assert d["total_binned_points"] == 3
    assert d["out_of_bounds_count"] == 71
    assert d["untransformed_count"] == 0
    assert d["max_cell_count"] == 2



