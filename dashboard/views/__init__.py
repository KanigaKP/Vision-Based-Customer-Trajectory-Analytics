"""
Direction A Presentation Views Package.
"""
from dashboard.views.overview import render_overview_view
from dashboard.views.new_analysis import render_new_analysis_view
from dashboard.views.results_workspace import render_results_workspace_view
from dashboard.views.placement_candidate import render_placement_candidate_view
from dashboard.views.evidence_details import render_evidence_details_view
from dashboard.views.sample_library import render_sample_library_view
from dashboard.views.spatial_studio import render_spatial_studio_view
from dashboard.views.main_workflow import render_main_workflow

__all__ = [
    "render_main_workflow",
    "render_overview_view",
    "render_new_analysis_view",
    "render_results_workspace_view",
    "render_placement_candidate_view",
    "render_evidence_details_view",
    "render_sample_library_view",
    "render_spatial_studio_view",
]

