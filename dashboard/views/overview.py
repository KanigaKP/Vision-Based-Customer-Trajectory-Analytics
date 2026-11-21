"""
Overview / Sample Shop Library View.
====================================
Preserves the workspace overview as the shop-selection screen for both sample actions:
- View Sample Shops (open saved results immediately)
- Test Sample Shops (rerun analysis with matching camera setup and zones)
"""

from dashboard.views.sample_library import render_sample_library_view, render_overview_view

__all__ = ["render_overview_view", "render_sample_library_view"]
