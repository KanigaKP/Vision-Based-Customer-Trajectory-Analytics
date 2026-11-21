"""
Direction A Visual Theme and UI Component Primitives.
=====================================================
Strictly offline: uses system fonts, local SVG rendering, and zero remote CDN dependencies.
Design Tokens adhere to docs/UI_UX_DESIGN_FOUNDATION.md.
"""

from typing import Dict, Any, List, Optional, Tuple
import html


# Design Tokens
COLOR_CANVAS_BG = "#FBFBF9"       # Warm White / Light Cream
COLOR_TEXT_HEADING = "#0F172A"    # Deep Charcoal
COLOR_TEXT_BODY = "#334155"       # Slate 700
COLOR_TEXT_MUTED = "#64748B"      # Slate 500
COLOR_BORDER = "#E2E8F0"          # Light Gray Line
COLOR_CARD_BG = "#FFFFFF"         # Pure White
COLOR_TEAL_ACCENT = "#0D9488"     # Teal Primary Accent
COLOR_TEAL_LIGHT = "#F0FDFA"      # Teal 50
COLOR_TEAL_BORDER = "#99F6E4"     # Teal 200
COLOR_WITHHELD_BG = "#F8FAFC"      # Clean Slate 50 (replaces harsh red)
COLOR_WITHHELD_TEXT = "#0F172A"    # Deep Navy / Slate 900
COLOR_WITHHELD_BORDER = "#CBD5E1"  # Architectural Slate 300

# Strict Zone Colors across retail datasets
ZONE_COLORS = {
    # MERL Supermarket
    "shelf_a": "#3498DB",               # Soft Blue
    "shelf_b": "#2ECC71",               # Soft Green
    "main_aisle": "#F39C12",             # Amber/Orange
    "checkout": "#E74C3C",               # Red/Coral
    # PRCV Site 03 Boutique
    "side_display_left": "#3498DB",      # Soft Blue
    "display_table_center": "#2ECC71",   # Soft Green
    "center_aisle": "#F39C12",           # Amber/Orange
    "checkout_counter": "#E74C3C",       # Red/Coral
    # PRCV Site 02 Luxury Jewelry
    "showcase_display_front": "#3498DB", # Soft Blue
    "showcase_display_back": "#2ECC71",  # Soft Green
    "browsing_aisle": "#F39C12",         # Amber/Orange
    "entrance_foyer": "#9B59B6",         # Amethyst Purple
}

# Sequential Teal Heatmap Palette (Offline)
DENSITY_PALETTE = [
    (0, "#F0FDFA"),   # 0 counts (Teal 50)
    (1, "#CCFBF1"),   # 1-2 counts (Teal 100)
    (3, "#5EEAD4"),   # 3-5 counts (Teal 300)
    (6, "#14B8A6"),   # 6-10 counts (Teal 500)
    (11, "#0F766E"),  # 11-20 counts (Teal 700)
    (21, "#134E4A"),  # 21+ counts (Teal 900)
]


def inject_direction_a_theme() -> str:
    """Return CSS string for Direction A design system to inject into Streamlit."""
    return f"""
    <style>
    /* Direction A Global Typography & Palette */
    :root {{
        --canvas-bg: {COLOR_CANVAS_BG};
        --ink-heading: {COLOR_TEXT_HEADING};
        --ink-body: {COLOR_TEXT_BODY};
        --ink-muted: {COLOR_TEXT_MUTED};
        --card-bg: {COLOR_CARD_BG};
        --line: {COLOR_BORDER};
        --teal: {COLOR_TEAL_ACCENT};
        --teal-light: {COLOR_TEAL_LIGHT};
    }}
    
    [data-testid="stAppViewContainer"] {{
        background-color: var(--canvas-bg);
        background-image: radial-gradient(rgba(148, 163, 184, 0.22) 1px, transparent 1px);
        background-size: 20px 20px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: var(--ink-body);
    }}
    
    [data-testid="stHeader"] {{
        background-color: rgba(251, 251, 249, 0.94);
        backdrop-filter: blur(8px);
    }}
    
    [data-testid="stMainBlockContainer"] {{
        max-width: 1440px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }}
    
    /* Navigation Sidebar */
    [data-testid="stSidebar"] {{
        background: #0B132B;
        border-right: 1px solid #1E293B;
    }}
    
    [data-testid="stSidebar"] * {{
        color: #F8FAFC !important;
    }}
    
    [data-testid="stSidebar"] [data-testid="stRadio"] label {{
        border-radius: 8px;
        padding: 0.45rem 0.75rem;
        transition: background 0.15s ease, color 0.15s ease;
        font-weight: 500;
    }}
    
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
        background: rgba(255, 255, 255, 0.09) !important;
    }}

    /* Card styling */
    .dir-a-card {{
        background: var(--card-bg);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04), 0 1px 2px -1px rgba(15, 23, 42, 0.03);
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }}
    .dir-a-card:hover {{
        border-color: #CBD5E1;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }}

    .dir-a-card-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.85rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #F1F5F9;
    }}

    .dir-a-card-title {{
        font-size: 1.05rem;
        font-weight: 750;
        color: var(--ink-heading);
        margin: 0;
        letter-spacing: -0.01em;
    }}

    /* Badges */
    .dir-a-badge {{
        display: inline-flex;
        align-items: center;
        padding: 0.22rem 0.65rem;
        border-radius: 6px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-family: "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
    }}

    .dir-a-badge-withheld {{
        background-color: #F0FDFA;
        color: #0F766E;
        border: 1px solid #99F6E4;
    }}

    .dir-a-badge-synthetic {{
        background-color: #FFFBEB;
        color: #B45309;
        border: 1px solid #FDE68A;
    }}

    .dir-a-badge-verified {{
        background-color: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
    }}

    .dir-a-badge-neutral {{
        background-color: #F1F5F9;
        color: #475569;
        border: 1px solid #CBD5E1;
    }}

    .dir-a-badge-alert {{
        background-color: #F8FAFC;
        color: #475569;
        border: 1px solid #CBD5E1;
    }}

    /* Primary Streamlit Action Buttons */
    div.stButton > button[kind="primary"], [data-testid="baseButton-primary"] {{
        background-color: var(--teal) !important;
        border: 1px solid var(--teal) !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 8px rgba(13, 148, 136, 0.25) !important;
        transition: all 0.15s ease-in-out !important;
    }}

    div.stButton > button[kind="primary"]:hover, [data-testid="baseButton-primary"]:hover {{
        background-color: #0F766E !important;
        border-color: #0F766E !important;
        box-shadow: 0 6px 18px rgba(13, 148, 136, 0.35) !important;
        color: #FFFFFF !important;
    }}

    div.stButton > button[kind="secondary"], [data-testid="baseButton-secondary"] {{
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        color: #0F172A !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.02) !important;
        transition: all 0.15s ease-in-out !important;
    }}

    div.stButton > button[kind="secondary"]:hover, [data-testid="baseButton-secondary"]:hover {{
        background-color: #F8FAFC !important;
        border-color: #94A3B8 !important;
        color: #0F172A !important;
    }}

    /* Radio button active states & selection indicators */
    [data-testid="stRadio"] [aria-checked="true"] {{
        border-color: var(--teal) !important;
    }}
    [data-testid="stRadio"] [aria-checked="true"] > div {{
        background-color: var(--teal) !important;
    }}
    [data-testid="stRadio"] input[type="radio"]:checked + div {{
        border-color: var(--teal) !important;
    }}

    /* Tab active highlight */
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
        background-color: var(--teal) !important;
    }}
    [data-testid="stTabs"] [aria-selected="true"] {{
        color: var(--teal) !important;
        font-weight: 700 !important;
    }}

    /* Metric Tiles & Value Formatting (prevent awkward hyphenation/splitting) */
    .dir-a-metric-tile {{
        background: var(--card-bg);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }}

    .dir-a-metric-tile:hover {{
        border-color: #CBD5E1;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        transform: translateY(-1px);
    }}

    .dir-a-metric-label {{
        font-size: 0.76rem;
        font-weight: 700;
        color: var(--ink-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.35rem;
    }}

    .dir-a-metric-val {{
        font-family: "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
        font-size: 1.45rem;
        font-weight: 780;
        color: var(--ink-heading);
        line-height: 1.2;
        letter-spacing: -0.02em;
        word-break: normal;
        overflow-wrap: break-word;
        hyphens: none;
    }}

    .dir-a-metric-subtext {{
        font-size: 0.78rem;
        color: var(--ink-muted);
        margin-top: 0.35rem;
        word-break: normal;
        overflow-wrap: break-word;
    }}

    /* Modern Table & DataFrame Overrides */
    [data-testid="stTable"], .stDataFrame {{
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }}
    table {{
        font-size: 0.86rem !important;
    }}
    th {{
        background-color: #F8FAFC !important;
        color: #475569 !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        font-size: 0.75rem !important;
        padding: 0.65rem 0.9rem !important;
        border-bottom: 2px solid #E2E8F0 !important;
    }}
    td {{
        padding: 0.6rem 0.9rem !important;
        border-bottom: 1px solid #F1F5F9 !important;
    }}
    tr:hover td {{
        background-color: #F0FDFA !important;
    }}

    /* Evidence Governance & Withheld Policy Card */
    .dir-a-withheld-box {{
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border: 1px solid #CBD5E1;
        border-left: 4px solid var(--teal);
        border-radius: 12px;
        padding: 1.15rem 1.4rem;
        margin: 1rem 0;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
    }}

    .dir-a-withheld-box h4 {{
        color: #0F172A;
        margin: 0 0 0.55rem 0;
        font-size: 0.95rem;
        font-weight: 750;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }}

    .dir-a-withheld-box ul {{
        margin: 0;
        padding-left: 1.25rem;
        color: #334155;
        font-size: 0.88rem;
        line-height: 1.55;
    }}

    .dir-a-withheld-box li {{
        margin-bottom: 0.35rem;
    }}

    /* Hero Banner */
    .dir-a-hero {{
        background: #0F172A;
        border-radius: 16px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.5rem;
        color: #F8FAFC;
    }}

    .dir-a-hero-eyebrow {{
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #2DD4BF;
        margin-bottom: 0.4rem;
    }}

    .dir-a-hero h1 {{
        color: #FFFFFF !important;
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
    }}

    .dir-a-hero p {{
        color: #CBD5E1;
        font-size: 0.98rem;
        line-height: 1.55;
        margin: 0;
        max-width: 920px;
    }}

    /* Accessible Focus States & Touch-Friendly Targets */
    div.stButton > button, [data-testid="baseButton-secondary"], [data-testid="baseButton-primary"] {{
        min-height: 42px;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.15s ease-in-out;
    }}

    button:focus-visible, input:focus-visible, select:focus-visible, [data-testid="stRadio"] label:focus-within {{
        outline: 2px solid var(--teal) !important;
        outline-offset: 2px !important;
    }}

    /* Mobile Responsive Optimizations (~390px viewport) */
    @media (max-width: 640px) {{
        [data-testid="stMainBlockContainer"] {{
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
            padding-top: 0.5rem !important;
        }}
        .dir-a-hero {{
            padding: 1.25rem 1.1rem !important;
            border-radius: 12px !important;
        }}
        .dir-a-hero h1 {{
            font-size: 1.45rem !important;
        }}
        .dir-a-hero p {{
            font-size: 0.88rem !important;
        }}
        div.stButton > button, [data-testid="baseButton-secondary"], [data-testid="baseButton-primary"] {{
            min-height: 44px !important;
            width: 100% !important;
        }}
        /* Ensure combobox/selectbox chevrons never get forced to 100% width */
        div[data-testid="stSelectbox"] button, [data-baseweb="select"] button, [role="group"] button {{
            width: auto !important;
            min-width: unset !important;
            min-height: unset !important;
            flex: 0 0 auto !important;
        }}
        div[data-testid="stSelectbox"] input, [role="group"] input {{
            width: 100% !important;
            flex: 1 1 auto !important;
        }}
        [data-testid="column"] {{
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }}
        .dir-a-card {{
            padding: 1rem 1rem !important;
        }}
    }}

    /* Executive Alert Styling: Replaces harsh pink/red error boxes with modern slate/teal cards */
    div[data-testid="stAlert"] {{
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%) !important;
        border: 1px solid #CBD5E1 !important;
        border-left: 4px solid var(--teal) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04) !important;
        padding: 1rem 1.25rem !important;
    }}
    div[data-testid="stAlert"] * {{
        color: #0F172A !important;
    }}
    div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {{
        color: #334155 !important;
        font-size: 0.88rem !important;
        line-height: 1.55 !important;
        margin: 0 !important;
    }}
    div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] strong {{
        color: var(--teal) !important;
        font-weight: 750 !important;
    }}
    div[data-testid="stAlert"] svg {{
        color: var(--teal) !important;
        fill: var(--teal) !important;
    }}
    </style>
    """


def render_density_legend_html() -> str:
    """Return an accessible HTML snippet for the sequential teal density palette legend."""
    items = [
        ("#F0FDFA", "#134E4A", "0 counts"),
        ("#CCFBF1", "#134E4A", "1–2"),
        ("#5EEAD4", "#0F766E", "3–5"),
        ("#14B8A6", "#FFFFFF", "6–10"),
        ("#0F766E", "#FFFFFF", "11–20"),
        ("#134E4A", "#FFFFFF", "21+"),
    ]
    swatches = "".join(
        f'<div style="display:inline-flex; align-items:center; margin-right:8px; margin-bottom:4px;">'
        f'<span style="display:inline-block; width:16px; height:16px; border-radius:3px; background:{bg}; '
        f'border:1px solid #CBD5E1; margin-right:4px;"></span>'
        f'<span style="font-size:0.75rem; color:#475569; font-family:monospace;">{label}</span>'
        f'</div>'
        for bg, fg, label in items
    )
    return (
        f'<div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:0.6rem 0.8rem; margin:0.6rem 0 1rem 0;">'
        f'<div style="font-size:0.75rem; font-weight:700; color:#334155; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.35rem;">'
        f'Sequential Density Scale (Discrete Binned Counts):</div>'
        f'<div style="display:flex; flex-wrap:wrap; align-items:center;">{swatches}</div>'
        f'</div>'
    )


def render_metric_card_html(label: str, value: str, subtext: Optional[str] = None, border_color: Optional[str] = None) -> str:
    """Return HTML snippet for a Direction A metric card."""
    border_style = f"border-left: 4px solid {border_color};" if border_color else ""
    subtext_html = f'<div class="dir-a-metric-subtext">{html.escape(subtext)}</div>' if subtext else ""
    return (
        f'<div class="dir-a-metric-tile" style="{border_style}">'
        f'<div class="dir-a-metric-label">{html.escape(label)}</div>'
        f'<div class="dir-a-metric-val">{html.escape("Unavailable" if value is None else str(value))}</div>'
        f'{subtext_html}'
        f'</div>'
    )


def render_withheld_banner_html(reasons: List[str], title: str = "STATUS: RECOMMENDATION WITHHELD") -> str:
    """Return HTML snippet for the prominent evidence withholding banner."""
    li_items = "".join([f"<li>{html.escape(r)}</li>" for r in reasons])
    return (
        f'<div class="dir-a-withheld-box">'
        f'<h4>🛡️ {html.escape(title)}</h4>'
        f'<ul>{li_items}</ul>'
        f'</div>'
    )


def render_store_owner_tip(tip_title: str, tip_text: str) -> str:
    """Return an executive, friendly Store Owner Guide tip box."""
    return (
        f'<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-left: 4px solid #0D9488; '
        f'border-radius: 8px; padding: 0.85rem 1.15rem; margin: 0.85rem 0 1.15rem 0;">'
        f'<div style="font-weight: 750; font-size: 0.9rem; color: #0F172A; margin-bottom: 0.25rem;">'
        f'🏪 Store Owner Guide: {html.escape(tip_title)}</div>'
        f'<div style="font-size: 0.84rem; color: #475569; line-height: 1.45;">{html.escape(tip_text)}</div>'
        f'</div>'
    )


def render_store_example_card(
    title: str,
    store_icon: str,
    store_badge: str,
    subtitle: str,
    description: str,
    highlights: List[str],
    run_id: str,
    is_selected: bool = False,
) -> str:
    """Return an elegant, store-owner friendly benchmark store card with zero markdown code-block leaking."""
    border_color = "#0D9488" if is_selected else "#CBD5E1"
    bg_color = "#F0FDFA" if is_selected else "#FFFFFF"
    selected_badge = (
        '<span style="background: #0D9488; color: #FFFFFF; font-size: 0.65rem; font-weight: 800; '
        'padding: 3px 8px; border-radius: 9999px; letter-spacing: 0.05em;">ACTIVE STORE</span>'
        if is_selected else ''
    )
    
    pills_html = "".join([
        f'<span style="background: {"rgba(13, 148, 136, 0.12)" if is_selected else "#F1F5F9"}; '
        f'color: {"#0F766E" if is_selected else "#334155"}; font-size: 0.72rem; font-weight: 600; '
        f'padding: 3px 7px; border-radius: 6px; margin-right: 5px; display: inline-block; margin-bottom: 4px;">'
        f'{html.escape(h)}</span>'
        for h in highlights
    ])
    
    return (
        f'<div style="border: 2px solid {border_color}; background: {bg_color}; border-radius: 12px; '
        f'padding: 1.15rem; min-height: 260px; display: flex; flex-direction: column; justify-content: space-between; '
        f'box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04); margin-bottom: 0.6rem;">'
        f'<div>'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">'
        f'<span style="font-size: 0.72rem; font-weight: 800; color: #0F766E; text-transform: uppercase; font-family: monospace;">'
        f'{store_icon} {html.escape(store_badge)}</span>'
        f'{selected_badge}'
        f'</div>'
        f'<h4 style="margin: 0.2rem 0 0.15rem 0; color: #0F172A; font-size: 1.1rem; font-weight: 750;">{html.escape(title)}</h4>'
        f'<div style="font-size: 0.8rem; font-weight: 600; color: #0D9488; margin-bottom: 0.45rem;">{html.escape(subtitle)}</div>'
        f'<p style="font-size: 0.82rem; color: #475569; margin: 0 0 0.6rem 0; line-height: 1.45;">{html.escape(description)}</p>'
        f'<div style="margin-bottom: 0.4rem;">{pills_html}</div>'
        f'</div>'
        f'<div style="font-size: 0.72rem; color: #64748B; font-family: monospace; border-top: 1px solid {"#99F6E4" if is_selected else "#E2E8F0"}; padding-top: 0.45rem;">'
        f'Dataset File: <code>{html.escape(run_id)}</code>'
        f'</div>'
        f'</div>'
    )


def generate_floor_plan_svg(
    bounds: Optional[Tuple[float, float, float, float]] = None,
    zones: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    zone_colors: Optional[Dict[str, str]] = None,
    zone_names: Optional[Dict[str, str]] = None,
    candidate_hotspot: Optional[Dict[str, Any]] = None,
    candidate_zone_id: Optional[str] = None,
    selected_zone_id: Optional[str] = None,
    width_px: int = 700,
    height_px: int = 420,
) -> Optional[str]:
    """
    Generate an offline SVG floor map representing run-specific configured floor bounds.
    Returns None if bounds or zones are not provided (does NOT silently attach MERL geometry).
    """
    if bounds is None or zones is None or len(zones) == 0:
        return None

    try:
        x_min, x_max, y_min, y_max = (float(v) for v in bounds)
    except (ValueError, TypeError):
        return None

    if x_max <= x_min or y_max <= y_min:
        return None

    pad = 40
    plot_w = width_px - 2 * pad
    plot_h = height_px - 2 * pad

    def to_svg_x(x: float) -> float:
        return pad + ((x - x_min) / (x_max - x_min)) * plot_w

    def to_svg_y(y: float) -> float:
        # SVG origin is top-left, flip Y so y_min is at bottom
        return pad + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h

    palette = ["#3498DB", "#2ECC71", "#F39C12", "#E74C3C", "#9B59B6", "#1ABC9C", "#E67E22"]
    if zone_colors is None:
        zone_colors = {}
    if zone_names is None:
        zone_names = {}

    svg_elements = []
    # Background
    svg_elements.append(
        f'<rect x="0" y="0" width="{width_px}" height="{height_px}" fill="#FFFFFF" rx="8" stroke="#E2E8F0" stroke-width="1"/>'
    )
    # Floor boundary
    fl_x = to_svg_x(x_min)
    fl_y = to_svg_y(y_max)
    fl_w = to_svg_x(x_max) - fl_x
    fl_h = to_svg_y(y_min) - fl_y
    svg_elements.append(
        f'<rect x="{fl_x}" y="{fl_y}" width="{fl_w}" height="{fl_h}" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.5" stroke-dasharray="4,4"/>'
    )

    # Dynamic Grid lines based on actual bounds
    span_x = x_max - x_min
    span_y = y_max - y_min
    num_grid_x = 4
    num_grid_y = 3
    for i in range(1, num_grid_x + 1):
        gx = x_min + i * (span_x / (num_grid_x + 1))
        sx = to_svg_x(gx)
        svg_elements.append(f'<line x1="{sx:.1f}" y1="{pad}" x2="{sx:.1f}" y2="{pad+plot_h}" stroke="#F1F5F9" stroke-width="1"/>')
        svg_elements.append(f'<text x="{sx:.1f}" y="{height_px - 15}" font-size="10" fill="#94A3B8" text-anchor="middle">X={gx:.1f}</text>')

    for j in range(1, num_grid_y + 1):
        gy = y_min + j * (span_y / (num_grid_y + 1))
        sy = to_svg_y(gy)
        svg_elements.append(f'<line x1="{pad}" y1="{sy:.1f}" x2="{pad+plot_w}" y2="{sy:.1f}" stroke="#F1F5F9" stroke-width="1"/>')
        svg_elements.append(f'<text x="{pad - 8}" y="{sy + 4:.1f}" font-size="10" fill="#94A3B8" text-anchor="end">Y={gy:.1f}</text>')

    # Zones
    for idx, (zid, poly) in enumerate(zones.items()):
        color = zone_colors.get(zid, palette[idx % len(palette)])
        pts_str = " ".join([f"{to_svg_x(px):.1f},{to_svg_y(py):.1f}" for px, py in poly])
        is_cand = (zid == candidate_zone_id)
        is_sel = (zid == selected_zone_id)
        if is_cand:
            stroke_w = "3.5"
            stroke_color = "#D97706"
            dash_attr = ' stroke-dasharray="6,3"'
            fill_alpha = "0.28"
        elif is_sel:
            stroke_w = "3.2"
            stroke_color = "#0D9488"
            dash_attr = ' stroke-dasharray="3,3"'
            fill_alpha = "0.26"
        else:
            stroke_w = "1.5"
            stroke_color = color
            dash_attr = ""
            fill_alpha = "0.20"

        svg_elements.append(
            f'<polygon points="{pts_str}" fill="{color}" fill-opacity="{fill_alpha}" stroke="{stroke_color}" stroke-width="{stroke_w}"{dash_attr}/>'
        )
        # Zone centroid for label
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        display_label = zone_names.get(zid, zid.replace("_", " ").title())
        svg_elements.append(
            f'<text x="{to_svg_x(cx):.1f}" y="{to_svg_y(cy):.1f}" font-size="12" font-weight="700" fill="{color}" text-anchor="middle">{display_label}</text>'
        )
        if is_cand:
            svg_elements.append(
                f'<text x="{to_svg_x(cx):.1f}" y="{to_svg_y(cy) + 16:.1f}" font-size="10" font-weight="800" fill="#B45309" text-anchor="middle">★ [SELECTED CANDIDATE]</text>'
            )
        elif is_sel:
            svg_elements.append(
                f'<text x="{to_svg_x(cx):.1f}" y="{to_svg_y(cy) + 16:.1f}" font-size="10" font-weight="800" fill="#0D9488" text-anchor="middle">★ [SELECTED GEOFENCE]</text>'
            )

    # Candidate hotspot cell highlight (ONLY when coordinates are present and valid)
    if candidate_hotspot and isinstance(candidate_hotspot, dict):
        hx_min = candidate_hotspot.get("x_min")
        hx_max = candidate_hotspot.get("x_max")
        hy_min = candidate_hotspot.get("y_min")
        hy_max = candidate_hotspot.get("y_max")
        if all(isinstance(v, (int, float)) for v in [hx_min, hx_max, hy_min, hy_max]):
            hx = to_svg_x(float(hx_min))
            hy = to_svg_y(float(hy_max))
            hw = to_svg_x(float(hx_max)) - hx
            hh = to_svg_y(float(hy_min)) - hy
            svg_elements.append(
                f'<rect x="{hx:.1f}" y="{hy:.1f}" width="{hw:.1f}" height="{hh:.1f}" fill="#F59E0B" fill-opacity="0.45" stroke="#B45309" stroke-width="2" stroke-dasharray="3,3"/>'
            )
            svg_elements.append(
                f'<text x="{hx + hw/2:.1f}" y="{hy + hh/2 + 4:.1f}" font-size="10" font-weight="800" fill="#78350F" text-anchor="middle">★ Hotspot</text>'
            )

    # Title & Axis Legend
    svg_elements.append(
        f'<text x="{pad}" y="{pad - 14}" font-size="11" font-weight="700" fill="#475569">Floor Geometry (Bounds: X [{x_min:.1f}..{x_max:.1f}] × Y [{y_min:.1f}..{y_max:.1f}])</text>'
    )

    svg_content = "\n  ".join(svg_elements)
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width_px} {height_px}" width="100%" style="max-width: {width_px}px; height: auto; display: block; margin: 0 auto;">\n  {svg_content}\n</svg>'


def generate_isometric_floor_plan_svg(
    bounds: Optional[Tuple[float, float, float, float]] = None,
    zones: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    zone_colors: Optional[Dict[str, str]] = None,
    zone_names: Optional[Dict[str, str]] = None,
    candidate_hotspot: Optional[Dict[str, Any]] = None,
    candidate_zone_id: Optional[str] = None,
    selected_zone_id: Optional[str] = None,
    width_px: int = 700,
    height_px: int = 420,
    zone_dwell: Optional[Dict[str, Any]] = None,
    dwell_scale: float = 0.05,
) -> Optional[str]:
    """
    Render an optional isometric representation derived strictly from the run's 2D floor geometry.
    Applies standard 30-degree isometric projection to 2D configured coordinates.
    When zone_dwell is provided, vertical extrusion height encodes observed cumulative dwell duration,
    with zero-dwell or unobserved zones remaining un-extruded at ground elevation (0.0 units).
    Labeled explicitly with its 2D floor-model basis; does NOT claim 3D sensor capture or physical height.
    """
    if bounds is None or zones is None or len(bounds) != 4 or not zones:
        return None

    x_min, x_max, y_min, y_max = [float(v) for v in bounds]
    if x_max <= x_min or y_max <= y_min:
        return None

    zone_colors = zone_colors or ZONE_COLORS
    zone_names = zone_names or {}

    # Center origin of 2D floor
    cx_floor = (x_min + x_max) / 2.0
    cy_floor = (y_min + y_max) / 2.0

    # Isometric angles: cos(30 deg) = sqrt(3)/2, sin(30 deg) = 0.5
    cos30 = 0.86602540378
    sin30 = 0.5

    # Scale calculation to fit viewport
    span_x = (x_max - x_min)
    span_y = (y_max - y_min)
    max_iso_w = (span_x + span_y) * cos30
    max_iso_h = (span_x + span_y) * sin30

    pad = 40.0
    avail_w = width_px - pad * 2
    avail_h = height_px - pad * 2

    scale = min(avail_w / max_iso_w, avail_h / max_iso_h) * 0.85
    orig_screen_x = width_px / 2.0
    orig_screen_y = height_px / 2.0 + 15.0

    def to_iso(fx: float, fy: float, fz: float = 0.0) -> Tuple[float, float]:
        dx = (fx - cx_floor)
        dy = (fy - cy_floor)
        sx = (dx - dy) * cos30 * scale + orig_screen_x
        sy = (dx + dy) * sin30 * scale - fz * scale + orig_screen_y
        return sx, sy

    svg_elements = []

    # Floor boundary outline (bounding quad on ground plane)
    b_pts = [
        to_iso(x_min, y_min),
        to_iso(x_max, y_min),
        to_iso(x_max, y_max),
        to_iso(x_min, y_max),
    ]
    b_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in b_pts)
    svg_elements.append(
        f'<polygon points="{b_str}" fill="#F8FAFC" stroke="#94A3B8" stroke-width="1.5" stroke-dasharray="4,4"/>'
    )

    # Render zone polygons on isometric ground plane with analytical dwell extrusion
    for zid, poly in zones.items():
        if not poly or len(poly) < 3:
            continue
        color = zone_colors.get(zid, ZONE_COLORS.get(zid, "#64748B"))
        is_cand = (zid == candidate_zone_id)
        is_sel = (zid == selected_zone_id)

        # Calculate wall height based on observed dwell span if zone_dwell is provided
        dwell_val = None
        if zone_dwell is not None:
            z_info = zone_dwell.get(zid, {})
            if isinstance(z_info, dict):
                dwell_val = z_info.get("observed_dwell_span_sec", z_info.get("total_dwell_sec"))
            elif isinstance(z_info, (int, float)):
                dwell_val = float(z_info)

            if dwell_val is not None and dwell_val > 0.0:
                wall_h = max(0.12, min(1.2, float(dwell_val) * dwell_scale))
            else:
                wall_h = 0.0
        else:
            wall_h = 0.35 if is_cand else 0.20

        # Base vertices
        base_pts = [to_iso(px, py, 0.0) for px, py in poly]
        top_pts = [to_iso(px, py, wall_h) for px, py in poly]

        base_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in base_pts)
        top_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in top_pts)

        # Floor footprint
        svg_elements.append(
            f'<polygon points="{base_str}" fill="{color}" fill-opacity="0.15" stroke="{color}" stroke-width="1" stroke-dasharray="2,2"/>'
        )

        # Subtle extruded walls when wall_h > 0
        if wall_h > 0.0:
            for i in range(len(poly)):
                j = (i + 1) % len(poly)
                wall_pts = [base_pts[i], base_pts[j], top_pts[j], top_pts[i]]
                w_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in wall_pts)
                fill_alpha = "0.32" if is_cand else ("0.26" if is_sel else "0.18")
                svg_elements.append(f'<polygon points="{w_str}" fill="{color}" fill-opacity="{fill_alpha}" stroke="{color}" stroke-width="1"/>')

        # Top face
        if is_cand:
            stroke_color = "#D97706"
            stroke_w = "3.2"
            dash_attr = ' stroke-dasharray="6,3"'
            fill_op = "0.38"
        elif is_sel:
            stroke_color = "#0D9488"
            stroke_w = "3.0"
            dash_attr = ' stroke-dasharray="3,3"'
            fill_op = "0.34"
        else:
            stroke_color = color
            stroke_w = "1.2"
            dash_attr = ""
            fill_op = "0.22" if wall_h > 0 else "0.08"

        svg_elements.append(
            f'<polygon points="{top_str}" fill="{color}" fill-opacity="{fill_op}" stroke="{stroke_color}" stroke-width="{stroke_w}"{dash_attr}/>'
        )

        # Label at top face centroid
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        lx, ly = to_iso(cx, cy, wall_h + 0.1)
        display_label = zone_names.get(zid, zid.replace("_", " ").title())
        if dwell_val is not None:
            dwell_suffix = f" ({dwell_val:.1f}s)" if dwell_val > 0 else " (0.0s)"
        else:
            dwell_suffix = ""
        
        if is_cand:
            badge_text = f"★ CANDIDATE: {display_label}{dwell_suffix}"
            badge_color = "#B45309"
        elif is_sel:
            badge_text = f"★ SELECTED: {display_label}{dwell_suffix}"
            badge_color = "#0D9488"
        else:
            badge_text = f"{display_label}{dwell_suffix}"
            badge_color = color

        svg_elements.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="11" font-weight="700" fill="{badge_color}" text-anchor="middle">{badge_text}</text>'
        )

    # Highlight hotspot on top if present
    if candidate_hotspot and isinstance(candidate_hotspot, dict):
        hx_min = candidate_hotspot.get("x_min")
        hx_max = candidate_hotspot.get("x_max")
        hy_min = candidate_hotspot.get("y_min")
        hy_max = candidate_hotspot.get("y_max")
        if all(isinstance(v, (int, float)) for v in [hx_min, hx_max, hy_min, hy_max]):
            cand_h = 0.0
            if zone_dwell is not None and candidate_zone_id:
                cand_info = zone_dwell.get(candidate_zone_id, {})
                if isinstance(cand_info, dict):
                    c_dwell = cand_info.get("observed_dwell_span_sec", cand_info.get("total_dwell_sec", 0.0))
                    if c_dwell and float(c_dwell) > 0:
                        cand_h = max(0.12, min(1.2, float(c_dwell) * dwell_scale))
            elif zone_dwell is None:
                cand_h = 0.35
            h_pts = [
                to_iso(float(hx_min), float(hy_min), cand_h + 0.02),
                to_iso(float(hx_max), float(hy_min), cand_h + 0.02),
                to_iso(float(hx_max), float(hy_max), cand_h + 0.02),
                to_iso(float(hx_min), float(hy_max), cand_h + 0.02),
            ]
            h_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in h_pts)
            svg_elements.append(
                f'<polygon points="{h_str}" fill="#F59E0B" fill-opacity="0.6" stroke="#B45309" stroke-width="2" stroke-dasharray="3,3"/>'
            )
            hc_x = (float(hx_min) + float(hx_max)) / 2.0
            hc_y = (float(hy_min) + float(hy_max)) / 2.0
            hlx, hly = to_iso(hc_x, hc_y, cand_h + 0.12)
            svg_elements.append(
                f'<text x="{hlx:.1f}" y="{hly:.1f}" font-size="10" font-weight="800" fill="#78350F" text-anchor="middle">★ Hotspot Cell</text>'
            )

    # Isometric Header & Extrusion Scale Legend Card
    svg_elements.append(
        f'<text x="{pad}" y="{pad - 14}" font-size="11" font-weight="700" fill="#475569">Analytical 3D Isometric View (Observed Dwell Extrusion)</text>'
    )
    svg_elements.append(
        f'<g transform="translate({width_px - 165}, {pad - 20})">'
        f'<rect x="0" y="0" width="155" height="66" fill="#FFFFFF" fill-opacity="0.94" rx="6" stroke="#CBD5E1" stroke-width="1"/>'
        f'<text x="77" y="14" font-size="8.5" font-weight="800" fill="#334155" text-anchor="middle" letter-spacing="0.05em">DWELL EXTRUSION SCALE</text>'
        f'<line x1="16" y1="46" x2="16" y2="24" stroke="#0D9488" stroke-width="2.5" stroke-linecap="round"/>'
        f'<circle cx="16" cy="46" r="2.5" fill="#64748B"/>'
        f'<circle cx="16" cy="24" r="2.5" fill="#0D9488"/>'
        f'<text x="26" y="27" font-size="8.5" font-family="monospace" font-weight="600" fill="#0F766E">Max Peak Dwell</text>'
        f'<text x="26" y="49" font-size="8.5" font-family="monospace" font-weight="600" fill="#64748B">0.0s Flat Ground</text>'
        f'<text x="77" y="60" font-size="7.5" fill="#94A3B8" text-anchor="middle">1.0s dwell ≈ {dwell_scale:.2f} floor units</text>'
        f'</g>'
    )
    svg_elements.append(
        f'<text x="{width_px / 2.0:.1f}" y="{height_px - 10}" font-size="9.5" font-weight="500" fill="#64748B" text-anchor="middle">Derived strictly from 2D floor geometry. Heights encode observed dwell duration; not physical 3D sensor capture or ceiling clearance.</text>'
    )

    svg_content = "\n  ".join(svg_elements)
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width_px} {height_px}" width="100%" style="max-width: {width_px}px; height: auto; display: block; margin: 0 auto;">\n  {svg_content}\n</svg>'

