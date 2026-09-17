import itertools
import os
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import load_data, build_name_indexes, judge_mask, advocate_mask, top_names, unique_names, filter_by_judge

# Must be the very first Streamlit command in the script. A real favicon
# and browser-tab title (instead of Streamlit's generic default) is one of
# the simplest things that makes this read as a real product rather than
# a script someone is running locally.
st.set_page_config(
    page_title="PHC Case Management",
    page_icon="\u2696\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- PALETTE -----------------------------
ACCENT = "#4F46E5"        # primary indigo - headline accents
ACCENT_LIGHT = "#818CF8"  # lighter indigo for gradients
BG = "#F6F7FB"            # soft neutral app background (subtle, not stark white)
CARD_BG = "#FFFFFF"       # card / chart surface
MUTED_BG = "#F8F9FF"      # very light tint used behind KPI icon chips
BORDER = "#E7E9F5"        # soft border color
TEXT_DARK = "#1E1B4B"     # near-black with a hint of indigo, for headings
TEXT_MUTED = "#6B7280"    # secondary text

KPI_ACCENTS = {
    "indigo": "#4F46E5",
    "teal": "#0EA5A4",
    "orange": "#F59E0B",
    "green": "#10B981",
    "red": "#EF4444",
    "purple": "#9333EA",
}

COLOR_LEGEND = [
    ("indigo", "Case Volume (totals & filtered counts)"),
    ("purple", "Judges / Justices"),
    ("teal", "Courts & time-based metrics"),
    ("orange", "Lawyers / Advocates"),
    ("green", "Clean / positive trend"),
    ("red", "Flagged for review / declining trend"),
]
ACCENT_MEANING = dict(COLOR_LEGEND)

_ICON_PATHS = {
    "home": '<path d="M4 11.5 12 4l8 7.5"/><path d="M6 10v9a1 1 0 0 0 1 1h3v-6h4v6h3a1 1 0 0 0 1-1v-9"/>',
    "overview": '<path d="M4 20V10M10 20V4M16 20v-7"/><path d="M3 20h18"/>',
    "court": '<path d="M3 21h18M4 21V10l8-5 8 5v11"/><path d="M8 21V13M12 21V13M16 21V13"/>',
    "distribution": '<circle cx="12" cy="12" r="8.5"/><path d="M12 3.5V12l6.5 3.2"/>',
    "judge": '<circle cx="12" cy="7.5" r="3"/><path d="M6 20c0-3.3 2.7-6 6-6s6 2.7 6 6"/>',
    "cases": '<path d="M3.5 6.5a1.5 1.5 0 0 1 1.5-1.5h4l1.6 1.8H19a1.5 1.5 0 0 1 1.5 1.5v7.7a1.5 1.5 0 0 1-1.5 1.5H5a1.5 1.5 0 0 1-1.5-1.5z"/>',
    "justice": '<path d="M12 3v16"/><path d="M6.5 6h11"/><path d="M6.5 6l-3 5.2a3 3 0 0 0 6 0z"/><path d="M17.5 6l-3 5.2a3 3 0 0 0 6 0z"/><path d="M8 20h8"/>',
    "lawyer": '<rect x="3.5" y="7.5" width="17" height="12" rx="2"/><path d="M8.5 7.5v-2a2 2 0 0 1 2-2h3a2 2 0 0 1 2 2v2"/><path d="M3.5 12.5h17"/>',
    "clipboard": '<rect x="5.5" y="4" width="13" height="17" rx="2"/><rect x="9" y="2.3" width="6" height="3.4" rx="1"/><path d="M9 11h6M9 15h6"/>',
    "flag": '<path d="M5.5 21V4"/><path d="M5.5 4.5h11l-2 4 2 4h-11"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.7 2.7L16 9.5"/>',
    "trend": '<path d="M4 16.5 9.5 11l3.5 3.5L20 7"/><path d="M14.5 7H20v5.5"/>',
    "bar": '<path d="M4 20V11M10 20V4M16 20v-6"/><path d="M3 20h18"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 8h.01"/><path d="M11 11.5h1.3v5"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5.3l3.5 2"/>',
}


def svg_icon(name, size=22):
    path = _ICON_PATHS.get(name, _ICON_PATHS["info"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="{size}" height="{size}" '
        f'fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round">{path}</svg>'
    )

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

#MainMenu, footer, div[data-testid="stToolbar"],
div[data-testid="stDecoration"], div[data-testid="stStatusWidget"] {{
    visibility: hidden !important;
    height: 0 !important;
}}
header[data-testid="stHeader"] {{
    background: {BG} !important;
    height: 2.6rem !important;
}}
header[data-testid="stHeader"] [data-testid="stMainMenu"],
header[data-testid="stHeader"] [data-testid="stToolbarActions"] {{
    visibility: hidden !important;
}}
button[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {{
    visibility: visible !important;
    height: auto !important;
}}
.stApp {{
    background: {BG} !important;
}}
.block-container {{
    padding-top: 3.2rem;
    padding-bottom: 2.5rem;
    padding-left: 3rem;
    padding-right: 3rem;
    max-width: 100% !important;
    width: 100% !important;
}}

h1, h2, h3, h4 {{
    font-family: 'Poppins', 'Inter', sans-serif;
    color: {TEXT_DARK} !important;
}}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] ol,
[data-testid="stMarkdownContainer"] ul {{
    color: {TEXT_DARK} !important;
}}
h3, h4 {{
    margin-top: 1.7rem;
    margin-bottom: 0.9rem;
    font-weight: 600;
}}
.phc-title {{
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.9rem;
    color: #000000;
    margin-bottom: 0.2rem;
}}
.phc-subtitle {{
    color: {TEXT_MUTED};
    font-size: 0.95rem;
    margin-bottom: 1.2rem;
}}

div[data-baseweb="tab-list"] {{
    display: flex;
    width: 100%;
    gap: 4px;
    background: {MUTED_BG};
    border: 1px solid {BORDER};
    padding: 6px;
    border-radius: 10px;
    margin-bottom: 1.4rem;
}}
button[data-baseweb="tab"] {{
    flex: 1 1 auto;
    border-radius: 8px !important;
    padding: 11px 18px !important;
    border: none !important;
    background: transparent !important;
    color: {TEXT_MUTED} !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    transition: background .15s ease, color .15s ease !important;
}}
button[data-baseweb="tab"]:hover {{
    background: rgba(79,70,229,0.06) !important;
    color: {ACCENT} !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    background: {ACCENT} !important;
    color: #fff !important;
}}
div[data-baseweb="tab-highlight"] {{ display: none; }}
div[data-baseweb="tab-border"] {{ display: none; }}

.kpi-card {{
    display: flex;
    align-items: center;
    gap: 14px;
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(16,24,40,0.06);
    transition: box-shadow .2s ease, border-color .2s ease;
    margin-bottom: 12px;
    min-height: 88px;
    height: 100%;
    box-sizing: border-box;
    overflow: visible;
    cursor: default;
}}
div[data-testid="stHorizontalBlock"]:has(.kpi-card) {{
    align-items: stretch;
}}
div[data-testid="stHorizontalBlock"]:has(.kpi-card) > div[data-testid="stColumn"] {{
    display: flex;
    flex-direction: column;
}}
div[data-testid="stHorizontalBlock"]:has(.kpi-card) > div[data-testid="stColumn"] div[data-testid="stVerticalBlock"] {{
    height: 100%;
}}
.kpi-card:hover {{
    box-shadow: 0 2px 8px rgba(16,24,40,0.10);
    border-color: #D7DAE8;
}}
.kpi-icon {{
    width: 46px;
    height: 46px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--accent-bg);
    color: var(--accent);
    flex-shrink: 0;
}}
.kpi-label {{
    font-size: 12px;
    color: {TEXT_MUTED};
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .04em;
    line-height: 1.2;
}}
.kpi-value {{
    font-size: 21px;
    font-weight: 700;
    color: {TEXT_DARK};
    margin-top: 2px;
    line-height: 1.2;
}}
.kpi-delta {{
    font-size: 12px;
    font-weight: 700;
    margin-top: 3px;
}}
@keyframes phcFadeIn {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

.phc-section-chip {{
    display: inline-block;
    background: {MUTED_BG};
    color: {ACCENT};
    border: 1px solid {BORDER};
    font-weight: 600;
    font-size: 12.5px;
    text-transform: uppercase;
    letter-spacing: .05em;
    padding: 5px 14px;
    border-radius: 999px;
    margin-bottom: 10px;
}}

.phc-filter-chip-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    padding-top: 2px;
}}
.phc-filter-chip {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: {MUTED_BG};
    border: 1px solid {BORDER};
    color: {TEXT_MUTED};
    font-size: 12.5px;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 999px;
}}
.phc-filter-chip strong {{
    color: {TEXT_DARK};
    font-weight: 700;
}}

.phc-insight-box {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-left: 4px solid {ACCENT};
    border-radius: 10px;
    padding: 14px 20px;
    margin: 4px 0 20px 0;
    box-shadow: 0 1px 3px rgba(16,24,40,0.06);
    font-size: 0.93rem;
    line-height: 1.6;
    color: {TEXT_DARK};
}}

@media (max-width: 680px) {{
    div[data-baseweb="tab-list"] {{
        flex-wrap: wrap;
        border-radius: 18px;
    }}
    button[data-baseweb="tab"] {{
        flex: 1 1 45%;
    }}
    .kpi-card {{
        min-height: unset;
    }}
    .kpi-value {{
        font-size: 19px;
    }}
    .phc-title {{
        font-size: 1.4rem;
    }}
}}

div[data-baseweb="select"] > div {{
    border-radius: 12px !important;
    border: 1px solid {BORDER} !important;
    background: #fff !important;
    box-shadow: 0 1px 2px rgba(31,41,55,0.03);
    transition: border-color .2s ease, box-shadow .2s ease;
}}
div[data-baseweb="select"] > div:hover {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.12) !important;
}}
div[data-testid="stTextInput"] input {{
    border-radius: 12px !important;
    border: 1px solid {BORDER} !important;
}}
div[data-testid="stTextInput"] input:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.12) !important;
}}

.stButton > button, .stDownloadButton > button {{
    border-radius: 8px !important;
    border: 1px solid transparent !important;
    background: {ACCENT} !important;
    color: #fff !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: background .2s ease, box-shadow .2s ease !important;
    box-shadow: 0 1px 3px rgba(16,24,40,0.10);
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    background: #4338CA !important;
    box-shadow: 0 2px 6px rgba(16,24,40,0.16);
}}
section[data-testid="stFileUploaderDropzone"] {{
    border-radius: 14px !important;
    border: 1.5px dashed {BORDER} !important;
    background: {MUTED_BG} !important;
}}
section[data-testid="stFileUploaderDropzone"] button {{
    border-radius: 999px !important;
    font-weight: 600 !important;
}}

div[data-testid="stPlotlyChart"] {{
    background-color: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 18px;
    margin-top: 6px;
    margin-bottom: 28px;
    box-shadow: 0 1px 3px rgba(16,24,40,0.06);
    transition: box-shadow .2s ease, border-color .2s ease;
    width: 100%;
    box-sizing: border-box;
}}
div[data-testid="stPlotlyChart"]:hover {{
    box-shadow: 0 2px 8px rgba(16,24,40,0.10);
    border-color: #D7DAE8;
}}
div[data-testid="stPlotlyChart"] iframe {{
    width: 100% !important;
    resize: none !important;
    overflow: visible !important;
    display: block;
}}
div[data-testid="stPlotlyChart"], div[data-testid="stPlotlyChart"] * {{
    overflow: visible;
    resize: none !important;
}}
div[data-testid="stPlotlyChart"] *::-webkit-scrollbar {{
    display: none !important;
    width: 0 !important;
    height: 0 !important;
}}
div[data-testid="stPlotlyChart"] .modebar {{
    display: none !important;
}}

div[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    overflow: hidden;
    margin-top: 10px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(16,24,40,0.06);
}}
div[data-testid="stDataFrame"] [data-testid="stElementToolbar"] {{
    background: {CARD_BG};
}}
div[data-testid="stDataFrame"] ::-webkit-scrollbar:horizontal {{
    height: 0px;
}}
div[data-testid="stDataFrame"] ::-webkit-scrollbar-corner {{
    background: transparent;
}}

div[data-testid="stAlert"] {{
    border-radius: 14px !important;
    border: 1px solid {BORDER} !important;
}}

section[data-testid="stSidebar"] {{
    display: none !important;
}}
div[data-testid="collapsedControl"] {{
    display: none !important;
}}

button[data-baseweb="tab"] {{ font-weight: 600; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------- KPI CARD HELPERS -----------------------------
def kpi_card(icon, label, value, accent="indigo", delta=None, delta_positive=True, delay=0.0):
    color = KPI_ACCENTS.get(accent, KPI_ACCENTS["indigo"])
    meaning = ACCENT_MEANING.get(accent, "")
    delta_html = ""
    if delta not in (None, ""):
        arrow = "\u2191" if delta_positive else "\u2193"
        delta_color = "#0F9D58" if delta_positive else "#EF4444"
        delta_html = f'<div class="kpi-delta" style="color:{delta_color}">{arrow} {delta}</div>'
    html = (
        f'<div class="kpi-card" style="animation-delay:{delay}s;" title="{meaning}">'
        f'<div><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{delta_html}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def kpi_row(items):
    cols = st.columns(len(items), gap="large")
    for i, (col, item) in enumerate(zip(cols, items)):
        with col:
            kpi_card(delay=round(i * 0.07, 2), **item)


def section_chip(text):
    st.markdown(f'<span class="phc-section-chip">{text}</span>', unsafe_allow_html=True)


def filter_summary_bar(active: dict, keys_to_reset: list, reset_button_key: str):
    chips = {k: v for k, v in active.items() if v not in (None, "All", "")}
    left, right = st.columns([5, 1.1])
    with left:
        if chips:
            chip_html = "".join(
                f'<span class="phc-filter-chip">{k}: <strong>{v}</strong></span>' for k, v in chips.items()
            )
            st.markdown(f'<div class="phc-filter-chip-row">{chip_html}</div>', unsafe_allow_html=True)
        else:
            st.caption("No filters applied \u2014 showing all data.")
    with right:
        if chips and st.button("\u21bb Reset filters", key=reset_button_key, **_full_width_kwargs(st.button)):
            for k in keys_to_reset:
                st.session_state.pop(k, None)
            st.rerun()


def insight_box(html_text):
    st.markdown(f'<div class="phc-insight-box">{html_text}</div>', unsafe_allow_html=True)


# ----------------------------- COLOR HELPERS -----------------------------
MASTER_PALETTE = (
    px.colors.qualitative.Bold + px.colors.qualitative.Set3
    + px.colors.qualitative.Dark24 + px.colors.qualitative.Pastel
)
_category_color_map = {}


def color_map_for(values):
    out = {}
    for v in pd.unique(pd.Series(list(values)).astype(str)):
        if v not in _category_color_map:
            _category_color_map[v] = MASTER_PALETTE[len(_category_color_map) % len(MASTER_PALETTE)]
        out[v] = _category_color_map[v]
    return out


SOLID_COLORS = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E", "#BC4749",
    "#1D3557", "#E76F51", "#457B9D", "#7209B7", "#06A77D", "#E9C46A",
    "#3D5A80", "#9D4EDD", "#DA627D",
]
SEQUENTIAL_SCALES = [
    px.colors.sequential.Viridis, px.colors.sequential.Plasma, px.colors.sequential.Tealgrn,
    px.colors.sequential.Sunset, px.colors.sequential.Mint, px.colors.sequential.Burg,
    px.colors.sequential.Blues, px.colors.sequential.Purples, px.colors.sequential.Oranges,
]

_solid_cycle = itertools.cycle(SOLID_COLORS)
_seq_cycle = itertools.cycle(SEQUENTIAL_SCALES)


def next_solid():
    return next(_solid_cycle)


def next_sequential():
    return next(_seq_cycle)


PLOTLY_CONFIG = {
    "displayModeBar": False,
    "scrollZoom": False,
    "doubleClick": False,
    "responsive": True,
    "staticPlot": False,
}

import inspect as _inspect
_PLOTLY_CHART_SUPPORTS_HEIGHT = "height" in _inspect.signature(st.plotly_chart).parameters


def _full_width_kwargs(func) -> dict:
    """Version-proof 'make this widget full width' kwargs.

    Newer Streamlit (roughly 1.4x+) uses width='stretch' / width='content'.
    Older Streamlit uses use_container_width=True / False. Rather than
    assume which one is installed, inspect the ACTUAL installed function's
    signature at runtime and hand back whichever kwarg it really supports.
    This means the app keeps working across Streamlit upgrades/downgrades
    without needing requirements.txt to be pinned exactly right - it's a
    functional fallback, not just a version guess."""
    params = _inspect.signature(func).parameters
    if "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


def render_chart(fig):
    """Single choke point every chart on the dashboard renders through.
    Wrapped in try/except: a single malformed figure should show an inline
    warning in place of that one chart, not take down the whole tab/app."""
    try:
        kwargs = dict(config=PLOTLY_CONFIG)
        kwargs.update(_full_width_kwargs(st.plotly_chart))
        if _PLOTLY_CHART_SUPPORTS_HEIGHT:
            kwargs["height"] = int(fig.layout.height or 440)
        st.plotly_chart(fig, **kwargs)
    except Exception as e:
        st.warning(f"This chart couldn't render right now ({type(e).__name__}).")


def _wrap_title(text: str, max_len: int = 26) -> str:
    if len(text) <= max_len or "<br>" in text:
        return text
    mid = len(text) // 2
    left = text.rfind(" ", 0, mid)
    right = text.find(" ", mid)
    if left == -1 and right == -1:
        return text
    if right == -1 or (left != -1 and (mid - left) <= (right - mid)):
        split_at = left
    else:
        split_at = right
    return text[:split_at] + "<br>" + text[split_at + 1:]


def style_fig(fig, height=460, legend_title=None, show_legend=None):
    has_legend = show_legend if show_legend is not None else (legend_title is not None)
    original_title = fig.layout.title.text or ""
    wrapped_title = _wrap_title(original_title)
    title_wraps = wrapped_title != original_title
    fig.update_layout(
        height=height,
        margin=dict(l=50, r=60, t=65, b=85),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title_text=legend_title if legend_title is not None else "",
        font=dict(family="Inter, sans-serif", color=TEXT_DARK, size=12),
        title=dict(text=wrapped_title, font=dict(family="Poppins, sans-serif", size=15, color=TEXT_DARK), pad=dict(b=16)),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter, sans-serif"),
        xaxis_title_standoff=16,
        yaxis_title_standoff=16,
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(rangemode="tozero", automargin=True)
    fig.update_yaxes(rangemode="tozero", automargin=True)
    is_percent_chart = "%" in (fig.layout.title.text or "")
    if is_percent_chart:
        is_h_bar = any(
            getattr(t, "type", "") == "bar" and getattr(t, "orientation", None) == "h" for t in fig.data
        )
        pct_axis_kwargs = dict(range=[0, 100], dtick=25, ticksuffix="%", tickangle=0,
                                tickfont=dict(size=10))
        if is_h_bar:
            fig.update_xaxes(**pct_axis_kwargs)
        else:
            fig.update_yaxes(**pct_axis_kwargs)
    is_polar = any(getattr(t, "type", "") == "scatterpolar" for t in fig.data)
    if is_polar:
        fig.update_layout(margin=dict(l=70, r=70, t=60, b=50), height=max(height, 420))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    angle=45, tickangle=45, showline=False,
                    gridcolor=BORDER, tickfont=dict(size=9, color=TEXT_MUTED),
                ),
                angularaxis=dict(
                    gridcolor=BORDER, tickfont=dict(size=11, color=TEXT_DARK),
                ),
            )
        )
    has_colorbar = fig.layout.coloraxis.colorscale is not None
    if has_colorbar:
        fig.update_layout(margin=dict(l=50, r=110, t=65, b=65))
    if show_legend is not None:
        fig.update_layout(showlegend=show_legend)
    if has_legend:
        trace_names = [getattr(t, "name", None) for t in fig.data if getattr(t, "name", None)]
        n_items = len(set(trace_names)) or len(fig.data)
        max_label_len = max([len(n) for n in trace_names], default=8)
        if max_label_len > 18:
            items_per_row = 2
        elif max_label_len > 12:
            items_per_row = 3
        elif max_label_len > 8:
            items_per_row = 4
        else:
            items_per_row = 5
        legend_rows = max(1, -(-n_items // items_per_row))
        legend_block_px = 34 + legend_rows * 20
        bottom_margin = 60 + legend_block_px
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-(legend_block_px / max(height, 1)),
                xanchor="center",
                x=0.5,
                font=dict(size=10),
                title_font=dict(size=11),
                tracegroupgap=4,
                itemwidth=30,
            ),
            margin=dict(l=50, r=60, t=65, b=bottom_margin),
            height=max(height, 280 + bottom_margin),
        )
    bar_traces = [t for t in fig.data if getattr(t, "type", "") == "bar"]
    category_positions = []
    for t in bar_traces:
        cats = t.y if (getattr(t, "orientation", None) or "v") == "h" else t.x
        category_positions.extend(list(cats) if cats is not None else [])
    is_stacked_or_grouped = len(bar_traces) > 1 and len(category_positions) != len(set(category_positions))
    too_dense = len(category_positions) > 12
    skip_labels = is_stacked_or_grouped or too_dense
    for trace in fig.data:
        if getattr(trace, "type", "") != "bar":
            continue
        if skip_labels:
            continue
        orientation = getattr(trace, "orientation", None) or "v"
        is_pct = "%" in (fig.layout.title.text or "")
        if orientation == "h":
            trace.texttemplate = "%{x:,.2f}%" if is_pct else "%{x:,.0f}"
        else:
            trace.texttemplate = "%{y:,.2f}%" if is_pct else "%{y:,.0f}"
        trace.textposition = "auto"
        trace.textfont = dict(size=10, color=TEXT_MUTED)
        trace.insidetextfont = dict(size=10, color="#FFFFFF")
        trace.cliponaxis = False

    is_pct_hover = "%" in (fig.layout.title.text or "")
    for trace in fig.data:
        ttype = getattr(trace, "type", "")
        if ttype == "pie":
            trace.hovertemplate = "<b>%{label}</b><br>%{value:,.2f}%<extra></extra>" if is_pct_hover \
                else "<b>%{label}</b><br>%{value:,.0f}<extra></extra>"
        elif ttype == "bar":
            orientation = getattr(trace, "orientation", None) or "v"
            if orientation == "h":
                fmt = "%{x:,.2f}%" if is_pct_hover else "%{x:,.0f}"
                trace.hovertemplate = "<b>%{y}</b><br>" + fmt + "<extra></extra>"
            else:
                fmt = "%{y:,.2f}%" if is_pct_hover else "%{y:,.0f}"
                trace.hovertemplate = "<b>%{x}</b><br>" + fmt + "<extra></extra>"

    if title_wraps:
        fig.update_layout(
            height=(fig.layout.height or height) + 22,
            margin=dict(t=(fig.layout.margin.t or 65) + 22),
        )
    return fig


DAY_ORDER = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]


# ----------------------------- TREND / ANOMALY HELPERS -----------------------------
def forecast_monthly(monthly_series: pd.Series, periods_ahead: int = 3):
    clean = monthly_series.dropna()
    if len(clean) < 4:
        return None, None, None
    x = np.arange(len(clean))
    y = clean.values.astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    future_x = np.arange(len(clean), len(clean) + periods_ahead)
    future_y = slope * future_x + intercept
    future_y = np.clip(future_y, 0, None)
    last_period = clean.index[-1]
    future_index = pd.period_range(last_period + 1, periods=periods_ahead, freq="M")
    return future_index, future_y, slope


def flag_anomalous_months(monthly_series: pd.Series, z_thresh: float = 2.0):
    clean = monthly_series.dropna()
    if len(clean) < 4 or clean.std() == 0:
        return pd.Series(dtype=float)
    z_scores = (clean - clean.mean()) / clean.std()
    return clean[z_scores.abs() > z_thresh]

# ----------------------------- DATA PATH -----------------------------
DEFAULT_DATA_PATH = "cause_lists_combined_2017_Jan_to_2026_July_MASTER.xlsx"
DEFAULT_PARQUET_PATH = os.path.splitext(DEFAULT_DATA_PATH)[0] + ".cache.parquet"

if os.path.exists(DEFAULT_DATA_PATH) or os.path.exists(DEFAULT_PARQUET_PATH):
    try:
        df = load_data(DEFAULT_DATA_PATH)
        name_idx = build_name_indexes(df)
    except Exception as e:
        st.error(f"Couldn't load the case data ({type(e).__name__}: {e}). Try reloading in a moment.")
        st.stop()
else:
    st.error(f"Couldn't find '{DEFAULT_DATA_PATH}' or its cached parquet. Put the combined cause-list file (or its .cache.parquet) next to this script.")
    st.stop()

_max_date = df["Hearing_Date"].max()
_min_date = df["Hearing_Date"].min()
LAST_UPDATED = _max_date.strftime("%d %B, %Y") if pd.notna(_max_date) else "N/A"
DATA_START = _min_date.strftime("%b %Y") if pd.notna(_min_date) else "N/A"
DATA_END = _max_date.strftime("%b %Y") if pd.notna(_max_date) else "N/A"

# ----------------------------- SIDEBAR (global, presentational) -----------------------------
ALL_FILTER_KEYS = ["ov_year", "ov_day", "infra_judge", "infra_year", "infra_cat", "dist_court", "judge_pick"]
with st.sidebar:
    st.markdown(f'<div style="font-weight:700;color:{TEXT_DARK};font-size:1.05rem;margin-bottom:4px;">'
                f'\u2696\ufe0f PHC Dashboard</div>', unsafe_allow_html=True)
    st.caption("Dataset overview")
    st.markdown(
        f'<div style="font-size:0.85rem;line-height:1.9;color:{TEXT_MUTED};">'
        f'Total cases: <strong style="color:{TEXT_DARK};">{len(df):,}</strong><br>'
        f'Courts on record: <strong style="color:{TEXT_DARK};">{df.loc[df["Court_No"] != "", "Court_No"].nunique():,}</strong><br>'
        f'Coverage: <strong style="color:{TEXT_DARK};">{DATA_START} \u2013 {DATA_END}</strong><br>'
        f'Last hearing on record: <strong style="color:{TEXT_DARK};">{LAST_UPDATED}</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("Filters")
    if st.button("\u21bb Reset all filters (every tab)", **_full_width_kwargs(st.button)):
        for k in ALL_FILTER_KEYS:
            st.session_state.pop(k, None)
        st.rerun()
    st.caption("Each tab keeps its own filters above its charts \u2014 this clears all of them at once.")

# ----------------------------- TOP NAV -----------------------------
st.markdown(
    f'<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">'
    f'<div class="phc-title" style="display:flex;align-items:center;gap:12px;">'
    f'<span>\u2696\ufe0f Peshawar High Court<br>Case Management</span></div>'
    f'<span style="display:inline-flex;align-items:center;gap:6px;background:{MUTED_BG};'
    f'border:1px solid {BORDER};border-radius:999px;padding:6px 14px;font-size:0.8rem;'
    f'color:{TEXT_MUTED};font-weight:500;white-space:nowrap;">'
    f'<span style="display:inline-flex;color:{ACCENT};">{svg_icon("clock", size=14)}</span>'
    f'Data last updated: <strong style="color:{TEXT_DARK};">{LAST_UPDATED}</strong></span>'
    f'</div>',
    unsafe_allow_html=True,
)

tab_home, tab_overview, tab_infra, tab_judge_court, tab_judge = st.tabs(
    ["\U0001F3E0 Home", "\U0001F4CA Overview", "\U0001F3DB\ufe0f Court Infrastructure Analysis",
     "\U0001F4C8 Distribution Analysis by Court", "\u2696\ufe0f Case Distribution by Judge"]
)

# ============================================================
# HOME
# ============================================================
with tab_home:
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.markdown("### The primary objective of this dashboard is to:")
        st.markdown("""
1. Provide clear visibility into **hearing schedules, bench compositions, and courtroom activity** across the Peshawar High Court.
2. Analyze case distribution across different legal categories, helping surface litigation trends and judicial workload patterns.
3. Monitor the balance between **regular hearings and flagged-for-review cases**, highlighting data-quality and backlog trends.
4. Facilitate evidence-based analysis of judicial processes through modern data visualization.
        """)
        st.info(
            "**Disclaimer:** The data presented in this dashboard is sourced from publicly available "
            "cause lists published by the Peshawar High Court. The information has been compiled and "
            "parsed for improved understanding and accessibility.",
        )
    with right:
        image_path = os.path.join(os.path.dirname(__file__), "assets", "phc_building.jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(os.path.dirname(__file__), "phc_building.jpg")
        if os.path.exists(image_path):
            st.image(image_path, caption="Peshawar High Court", **_full_width_kwargs(st.image))
        kpi_card(svg_icon("clock"), "Last Updated (latest hearing date in data)", LAST_UPDATED, accent="teal")

# ============================================================
# OVERVIEW
# ============================================================
with tab_overview:
    st.markdown("### Overview of Peshawar High Court Case Management")

    total_judges = len(name_idx["judge_names"])
    total_courts = df.loc[df["Court_No"] != "", "Court_No"].nunique()
    total_cases = len(df)
    total_lawyers = len(name_idx["advocate_names"])

    section_chip("Headline Totals")
    kpi_row([
        dict(icon=svg_icon("cases"), label="\U0001F4C1 Total Cases", value=f"{total_cases:,}", accent="indigo"),
        dict(icon=svg_icon("justice"), label="\u2696\ufe0f Total Justices", value=f"{total_judges:,}", accent="purple"),
        dict(icon=svg_icon("court"), label="\U0001F3DB\ufe0f Total Courts", value=f"{total_courts:,}", accent="teal"),
        dict(icon=svg_icon("lawyer"), label="\U0001F4BC Total Lawyers", value=f"{total_lawyers:,}", accent="orange"),
    ])

    by_year_all_kpi = df.loc[df["Year"] != "", "Year"].value_counts()
    years_sorted_kpi = sorted([y for y in by_year_all_kpi.index if str(y).strip()])
    growth_pct = None
    growth_positive = True
    if len(years_sorted_kpi) >= 2:
        prev_y, last_y = years_sorted_kpi[-2], years_sorted_kpi[-1]
        prev_val, last_val = by_year_all_kpi.get(prev_y, 0), by_year_all_kpi.get(last_y, 0)
        if prev_val:
            growth_pct = ((last_val - prev_val) / prev_val) * 100
            growth_positive = growth_pct >= 0

    dated = df.dropna(subset=["Hearing_Date"])
    busiest_month, busiest_month_n = "N/A", None
    if len(dated):
        month_counts = dated.groupby(dated["Hearing_Date"].dt.to_period("M")).size()
        if len(month_counts):
            busiest_month = month_counts.idxmax().strftime("%b %Y")
            busiest_month_n = int(month_counts.max())

    court_counts = df.loc[df["Court_No"] != "", "Court_No"].value_counts()
    busiest_court = f"Court No. {court_counts.idxmax()}" if len(court_counts) else "N/A"
    busiest_court_n = int(court_counts.max()) if len(court_counts) else None

    judge_counts = name_idx["judge_counts"]
    top_judge = judge_counts.index[0] if len(judge_counts) else "N/A"
    top_judge_n = int(judge_counts.iloc[0]) if len(judge_counts) else None

    eligible = court_counts[court_counts >= 50].index
    court_flag_pct = (
        df.loc[df["Court_No"].isin(eligible)]
        .groupby("Court_No", observed=True)["Needs_Review"]
        .apply(lambda s: (s == "Yes").mean() * 100)
    )
    top_flag_court = f"Court No. {court_flag_pct.idxmax()}" if len(court_flag_pct) else "N/A"
    top_flag_court_pct = float(court_flag_pct.max()) if len(court_flag_pct) else None

    advocate_counts = name_idx["advocate_counts"]
    top_advocate = advocate_counts.index[0] if len(advocate_counts) else "N/A"
    top_advocate_n = int(advocate_counts.iloc[0]) if len(advocate_counts) else None

    flagged_pct_alltime = (df["Needs_Review"] == "Yes").mean() * 100 if len(df) else 0.0
    flagged_last_year_pct = flagged_prev_year_pct = None
    if len(years_sorted_kpi) >= 2:
        flagged_last_year_pct = (df.loc[df["Year"] == last_y, "Needs_Review"] == "Yes").mean() * 100
        flagged_prev_year_pct = (df.loc[df["Year"] == prev_y, "Needs_Review"] == "Yes").mean() * 100

    summary_bits = [f"<strong>{total_cases:,} total cases</strong> on record across <strong>{total_courts} courts</strong>."]
    if growth_pct is not None:
        direction = "grew" if growth_positive else "declined"
        summary_bits.append(
            f"Case volume <strong>{direction} {abs(growth_pct):.1f}%</strong> year-over-year "
            f"({prev_y}\u2192{last_y})."
        )
    if busiest_court_n is not None:
        summary_bits.append(f"<strong>{busiest_court}</strong> carries the heaviest caseload ({busiest_court_n:,} cases).")
    if flagged_last_year_pct is not None and flagged_prev_year_pct is not None:
        trend_word = "down" if flagged_last_year_pct < flagged_prev_year_pct else "up"
        summary_bits.append(
            f"<strong>{flagged_last_year_pct:.1f}%</strong> of {last_y} cases are flagged for review "
            f"\u2014 {trend_word} from {flagged_prev_year_pct:.1f}% in {prev_y}."
        )
    else:
        summary_bits.append(f"<strong>{flagged_pct_alltime:.1f}%</strong> of all cases on record are flagged for review.")

    st.markdown(
        f'<div style="background:{CARD_BG};border:1px solid {BORDER};border-left:4px solid {ACCENT};'
        f'border-radius:10px;padding:16px 20px;margin:4px 0 22px 0;box-shadow:0 1px 3px rgba(16,24,40,0.06);">'
        f'<div style="font-size:0.72rem;font-weight:700;letter-spacing:0.06em;color:{ACCENT};'
        f'text-transform:uppercase;margin-bottom:6px;">\U0001F4CB Executive Summary</div>'
        f'<div style="font-size:0.95rem;line-height:1.6;color:{TEXT_DARK};">{" ".join(summary_bits)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    section_chip("Filters")
    years = sorted([y for y in df["Year"].unique() if str(y).strip()])
    fc1, fc2 = st.columns(2, gap="large")
    with fc1:
        year_filter = st.selectbox("Year", ["All"] + years, key="ov_year")
    with fc2:
        day_filter = st.selectbox("Hearing Day", ["All"] + sorted([d for d in df["Day"].unique() if d]), key="ov_day")

    fdf = df.copy()
    if year_filter != "All":
        fdf = fdf[fdf["Year"] == year_filter]
    if day_filter != "All":
        fdf = fdf[fdf["Day"] == day_filter]

    filter_summary_bar(
        {"Year": year_filter, "Hearing Day": day_filter},
        keys_to_reset=["ov_year", "ov_day"],
        reset_button_key="ov_reset",
    )

    if len(fdf) == 0:
        st.info("No cases match the selected filters. Try a different Year or Hearing Day above.")

    flagged_pct = (fdf["Needs_Review"] == "Yes").mean() * 100 if len(fdf) else 0.0
    clean_pct = 100 - flagged_pct if len(fdf) else 0.0
    FLAGGED_TARGET_PCT = 5.0
    flagged_within_target = flagged_pct <= FLAGGED_TARGET_PCT

    section_chip("Filtered Snapshot")
    kpi_row([
        dict(icon=svg_icon("clipboard"), label="\U0001F4CB Cases (filtered)", value=f"{len(fdf):,}", accent="indigo"),
        dict(
            icon=svg_icon("flag"), label="\U0001F6A9 Flagged for Review", value=f"{flagged_pct:.1f}%",
            accent="green" if flagged_within_target else "red",
            delta=f"Target: <{FLAGGED_TARGET_PCT:.0f}%", delta_positive=flagged_within_target,
        ),
        dict(icon=svg_icon("check"), label="\u2705 Clean / Final", value=f"{clean_pct:.1f}%", accent="green"),
        dict(
            icon=svg_icon("trend"), label=f"\U0001F4C8 YoY Growth ({years_sorted_kpi[-2]}\u2192{years_sorted_kpi[-1]})" if growth_pct is not None else "\U0001F4C8 YoY Growth",
            value=f"{growth_pct:+.1f}%" if growth_pct is not None else "N/A",
            accent="red" if growth_pct is not None and not growth_positive else "green",
            delta="vs prior year" if growth_pct is not None else None,
            delta_positive=growth_positive,
        ),
    ])

    st.download_button(
        "Download filtered data (CSV)",
        data=fdf.to_csv(index=False).encode("utf-8"),
        file_name=f"phc_cases_{year_filter}_{day_filter}.csv".replace(" ", "_"),
        mime="text/csv",
        key="ov_download_filtered",
    )

    section_chip("Key Insights (All-Time)")
    kpi_row([
        dict(icon=svg_icon("clock"), label="\u23F0 Busiest Month",
             value=busiest_month + (f" ({busiest_month_n:,})" if busiest_month_n is not None else ""),
             accent="indigo"),
        dict(icon=svg_icon("court"), label="\U0001F3DB\ufe0f Busiest Court",
             value=busiest_court + (f" ({busiest_court_n:,})" if busiest_court_n is not None else ""),
             accent="teal"),
        dict(icon=svg_icon("justice"), label="\u2696\ufe0f Most Active Justice",
             value=top_judge + (f" ({top_judge_n:,})" if top_judge_n is not None else ""),
             accent="purple"),
        dict(icon=svg_icon("flag"), label="\U0001F6A9 Highest Flagged-% Court (min. 50 cases)",
             value=top_flag_court + (f" ({top_flag_court_pct:.1f}%)" if top_flag_court_pct is not None else ""),
             accent="red"),
        dict(icon=svg_icon("lawyer"), label="\U0001F4BC Most Frequent Advocate",
             value=top_advocate + (f" ({top_advocate_n:,})" if top_advocate_n is not None else ""),
             accent="orange"),
    ])

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        remarks = fdf["Needs_Review"].value_counts(normalize=True).mul(100).round(2)
        remarks.index = remarks.index.map({"Yes": "Flagged for Review", "No": "Clean / Final"})
        fig = px.bar(remarks, orientation="h", title="Remarks Distribution (%)",
                     labels={"value": "% of Cases", "index": "Status"}, color=remarks.index,
                     color_discrete_map=color_map_for(remarks.index))
        render_chart(style_fig(fig, legend_title="Status"))

    with col2:
        section = fdf.loc[fdf["Section"] != "", "Section"].value_counts(normalize=True).mul(100).round(2)
        section = section[section >= 0.1].head(8)
        fig = px.bar(section, orientation="h", title="Case List Section (%)",
                     labels={"value": "% of Cases", "index": "Section"}, color=section.index,
                     color_discrete_map=color_map_for(section.index))
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        render_chart(style_fig(fig, legend_title="Section"))

    with col3:
        cat = fdf.loc[fdf["Case_Category"] != "", "Case_Category"].value_counts(normalize=True).mul(100).round(2)
        fig = px.pie(values=cat.values, names=cat.index, title="Categories (%)", hole=0.4,
                     color=cat.index, color_discrete_map=color_map_for(cat.index))
        render_chart(style_fig(fig, legend_title="Category"))

    col4, col5, col6 = st.columns(3, gap="large")
    with col4:
        by_court = fdf.loc[fdf["Court_No"] != "", "Court_No"].value_counts().sort_values(ascending=False).head(15)
        fig = px.bar(x=by_court.index.astype(str), y=by_court.values, title="No of Cases by Court No.",
                     labels={"x": "Court No.", "y": "Cases"}, color=by_court.index.astype(str),
                     color_discrete_map=color_map_for(by_court.index.astype(str)))
        render_chart(style_fig(fig, show_legend=False))

    with col5:
        top_titles = fdf.loc[fdf["Case_Title"] != "", "Case_Title"].value_counts().head(8)
        fig = px.bar(x=top_titles.values, y=top_titles.index, orientation="h",
                     title="Most-Repeated Case Titles (Single Hearing)",
                     labels={"x": "Cases", "y": "Case Title"}, color_discrete_sequence=[next_solid()])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        render_chart(style_fig(fig, show_legend=False))

    with col6:
        by_day = fdf.loc[fdf["Day_Norm"] != "", "Day_Norm"].value_counts().reindex(DAY_ORDER).fillna(0)
        fig = px.line_polar(r=by_day.values, theta=[d.title() for d in by_day.index], line_close=True,
                             title="Cases by Hearing Day (Weekly Pattern)", color_discrete_sequence=[next_solid()])
        fig.update_traces(fill="toself")
        render_chart(style_fig(fig, show_legend=False))

    col7, col8, col9 = st.columns(3, gap="large")
    with col7:
        trend = fdf.dropna(subset=["Hearing_Date"]).groupby(fdf["Hearing_Date"].dt.to_period("M")).size()
        trend.index = trend.index.astype(str)
        fig = px.area(x=trend.index, y=trend.values, title="Case Volume by Month",
                      labels={"x": "Month", "y": "Cases"}, color_discrete_sequence=[next_solid()])
        render_chart(style_fig(fig, show_legend=False))

    with col8:
        top_judges = top_names(fdf["Judges"], sep_pattern=r'&', top_n=10)
        fig = px.bar(x=top_judges.values, y=top_judges.index, orientation="h", title="Top 10 Justices by Caseload",
                     labels={"x": "Cases", "y": "Justice"}, color_discrete_sequence=[next_solid()])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        render_chart(style_fig(fig, show_legend=False))

    with col9:
        _all_advocates_raw = pd.concat([fdf["Petitioner_Advocate"], fdf["Respondent_Advocates"]])
        top_lawyers = top_names(_all_advocates_raw, top_n=10)
        fig = px.bar(x=top_lawyers.values, y=top_lawyers.index, orientation="h", title="Top 10 Advocates by Appearances",
                     labels={"x": "Appearances", "y": "Advocate"}, color_discrete_sequence=[next_solid()])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        render_chart(style_fig(fig, show_legend=False))

    st.markdown("#### Case Volume by Court and Category")
    court_cat = fdf.loc[(fdf["Court_No"] != "") & (fdf["Case_Category"] != "")].groupby(
        ["Court_No", "Case_Category"], observed=True).size().reset_index(name="Cases")
    fig = px.treemap(court_cat, path=["Case_Category", "Court_No"], values="Cases",
                      title="Cases by Category and Court (Treemap)", color="Cases",
                      color_continuous_scale=next_sequential())
    render_chart(style_fig(fig))

    st.markdown("#### Long-Term Trend (All-Time, Unfiltered)")
    col10, col11 = st.columns(2, gap="large")
    with col10:
        years_all = sorted([y for y in df["Year"].unique() if str(y).strip()])
        by_year_all = df.loc[df["Year"] != "", "Year"].value_counts().reindex(years_all).fillna(0)
        fig = px.bar(x=by_year_all.index, y=by_year_all.values, title="Total Cases by Year (All-Time)",
                     labels={"x": "Year", "y": "Cases"}, color=by_year_all.index,
                     color_discrete_map=color_map_for(by_year_all.index))
        render_chart(style_fig(fig, legend_title="Year"))
    with col11:
        yr_cat = df.loc[(df["Year"] != "") & (df["Case_Category"] != "")].groupby(
            ["Year", "Case_Category"], observed=True).size().reset_index(name="Cases")
        fig = px.bar(yr_cat, x="Year", y="Cases", color="Case_Category", title="Case Category Mix by Year (All-Time)",
                     barmode="stack", color_discrete_map=color_map_for(yr_cat["Case_Category"]),
                     category_orders={"Year": years_all})
        render_chart(style_fig(fig, legend_title="Category"))

    st.markdown("#### Trend & Outlook (All-Time, Unfiltered)")
    monthly_all = df.dropna(subset=["Hearing_Date"]).groupby(
        df["Hearing_Date"].dt.to_period("M")).size().sort_index()
    fc_index, fc_values, fc_slope = forecast_monthly(monthly_all, periods_ahead=3)
    anomalies = flag_anomalous_months(monthly_all)

    if fc_index is not None:
        actual_x = [str(p) for p in monthly_all.index]
        forecast_x = [str(p) for p in fc_index]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual_x, y=monthly_all.values, mode="lines+markers",
                                  name="Actual", line=dict(color=next_solid(), width=2)))
        bridge_x = [actual_x[-1]] + forecast_x
        bridge_y = [monthly_all.values[-1]] + list(fc_values)
        fig.add_trace(go.Scatter(x=bridge_x, y=bridge_y, mode="lines+markers",
                                  name="Forecast (linear trend)",
                                  line=dict(color="#F59E0B", width=2, dash="dash")))
        fig.update_layout(title="Monthly Case Volume: Actual vs 3-Month Forecast")
        render_chart(style_fig(fig, legend_title="Series"))

        trend_word = "rising" if fc_slope > 0 else "falling" if fc_slope < 0 else "flat"
        outlook_bits = [
            f"Based on a simple linear trend over the full history, monthly case volume is currently "
            f"<strong>{trend_word}</strong> by about <strong>{abs(fc_slope):,.0f} cases/month</strong>, "
            f"projecting to roughly <strong>{fc_values[-1]:,.0f} cases</strong> by {fc_index[-1]}."
        ]
        if len(anomalies):
            worst = anomalies.abs().sort_values(ascending=False).head(3)
            months_txt = ", ".join(f"{p} ({int(monthly_all[p]):,})" for p in worst.index)
            outlook_bits.append(f"Unusually high/low-volume months on record: <strong>{months_txt}</strong>.")
        else:
            outlook_bits.append("No months deviate sharply from the historical average.")
        insight_box(" ".join(outlook_bits))
        st.caption(
            "Forecast is a simple linear projection (no seasonality) for directional context only, "
            "not an operational planning figure. Anomalies flag months more than 2 standard deviations "
            "from the historical monthly average."
        )
    else:
        st.info("Not enough dated history yet to compute a trend forecast.")



# ============================================================
# COURT INFRASTRUCTURE ANALYSIS
# ============================================================
with tab_infra:
    st.markdown("### Court Infrastructure Analysis")

    section_chip("Filters")
    fc1, fc2, fc3 = st.columns(3, gap="large")
    with fc1:
        judge_names = name_idx["judge_names"]
        judge_pick = st.selectbox("Justice Name", ["All"] + judge_names, key="infra_judge")
    with fc2:
        years2 = sorted([y for y in df["Year"].unique() if str(y).strip()])
        year_pick = st.selectbox("Year", ["All"] + years2, key="infra_year")
    with fc3:
        cat_pick = st.selectbox("Case Category", ["All"] + sorted([c for c in df["Case_Category"].unique() if c]),
                                 key="infra_cat")

    fdf = df
    if judge_pick != "All":
        fdf = fdf[judge_mask(fdf["Judges"], judge_pick)]
    if year_pick != "All":
        fdf = fdf[fdf["Year"] == year_pick]
    if cat_pick != "All":
        fdf = fdf[fdf["Case_Category"] == cat_pick]

    filter_summary_bar(
        {"Justice": judge_pick, "Year": year_pick, "Category": cat_pick},
        keys_to_reset=["infra_judge", "infra_year", "infra_cat"],
        reset_button_key="infra_reset",
    )

    if len(fdf) == 0:
        st.info("No cases match the selected filters. Try adjusting the Justice, Year, or Category above.")

    section_chip("Filtered Snapshot")
    lawyers_f = len(unique_names(fdf["Petitioner_Advocate"]) | unique_names(fdf["Respondent_Advocates"]))
    kpi_row([
        dict(icon=svg_icon("cases"), label="\U0001F4C1 Total Cases (filtered)", value=f"{len(fdf):,}", accent="indigo"),
        dict(icon=svg_icon("court"), label="\U0001F3DB\ufe0f Total Courts (filtered)", value=f"{fdf.loc[fdf['Court_No'] != '', 'Court_No'].nunique():,}", accent="teal"),
        dict(icon=svg_icon("lawyer"), label="\U0001F4BC Total Lawyers (filtered)", value=f"{lawyers_f:,}", accent="orange"),
    ])

    if len(fdf):
        _by_court_insight = fdf.loc[fdf["Court_No"] != "", "Court_No"].value_counts()
        _bits = [f"<strong>{len(fdf):,} cases</strong> in this view, across <strong>{fdf.loc[fdf['Court_No'] != '', 'Court_No'].nunique()} courts</strong>."]
        if len(_by_court_insight):
            _bits.append(
                f"Court No. <strong>{_by_court_insight.idxmax()}</strong> handles the most in this selection "
                f"({int(_by_court_insight.max()):,} cases)."
            )
            _avg_court_load = _by_court_insight.mean()
            if _avg_court_load > 0:
                _imbalance_x = _by_court_insight.max() / _avg_court_load
                if _imbalance_x >= 1.3:
                    _bits.append(
                        f"That's <strong>{_imbalance_x:.1f}x</strong> the average court's caseload in this view "
                        f"({_avg_court_load:,.0f} cases/court)."
                    )
        _flagged_infra_pct = (fdf["Needs_Review"] == "Yes").mean() * 100
        _bits.append(f"<strong>{_flagged_infra_pct:.1f}%</strong> of these cases are flagged for review.")
        insight_box(" ".join(_bits))

    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        by_court = fdf.loc[fdf["Court_No"] != "", "Court_No"].value_counts().sort_values(ascending=False).head(15)
        fig = px.bar(x=by_court.values, y=by_court.index.astype(str), orientation="h",
                     title="No of Cases Handled in Each Court", labels={"x": "Cases", "y": "Court No."},
                     color=by_court.index.astype(str), color_discrete_map=color_map_for(by_court.index.astype(str)))
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        render_chart(style_fig(fig, show_legend=False))

    with col2:
        court_lawyer_map = {}
        court_judge_map = {}
        for court, grp in fdf.groupby("Court_No", observed=True):
            if court == "":
                continue
            lset = unique_names(grp["Petitioner_Advocate"]) | unique_names(grp["Respondent_Advocates"])
            court_lawyer_map[court] = len(lset)
            court_judge_map[court] = len(unique_names(grp["Judges"], sep_pattern=r'&'))
        s = pd.Series(court_lawyer_map).sort_values(ascending=False).head(15)
        fig = px.bar(x=s.values, y=s.index.astype(str), orientation="h",
                     title="No of Lawyers Connected to Each Court", labels={"x": "Lawyers", "y": "Court No."},
                     color=s.index.astype(str), color_discrete_map=color_map_for(s.index.astype(str)))
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        render_chart(style_fig(fig, show_legend=False))

    with col3:
        j = pd.Series(court_judge_map).sort_values(ascending=False).head(15)
        fig = px.bar(x=j.values, y=j.index.astype(str), orientation="h",
                     title="No of Justices Sitting in Each Court", labels={"x": "Justices", "y": "Court No."},
                     color_discrete_sequence=[next_solid()])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        render_chart(style_fig(fig, show_legend=False))

    col4, col5, col6 = st.columns(3, gap="large")
    with col4:
        cat_infra = fdf.loc[fdf["Case_Category"] != "", "Case_Category"].value_counts()
        fig = px.pie(values=cat_infra.values, names=cat_infra.index, title="Cases by Category", hole=0.4,
                     color=cat_infra.index, color_discrete_map=color_map_for(cat_infra.index))
        render_chart(style_fig(fig, legend_title="Category"))

    with col5:
        sect_infra = fdf.loc[fdf["Section"] != "", "Section"].value_counts().head(12).reset_index()
        sect_infra.columns = ["Section", "Cases"]
        fig = px.treemap(sect_infra, path=["Section"], values="Cases", title="Top Sections (Treemap)",
                          color="Cases", color_continuous_scale=next_sequential())
        render_chart(style_fig(fig))

    with col6:
        trend_infra = fdf.dropna(subset=["Hearing_Date"]).groupby(fdf["Hearing_Date"].dt.to_period("M")).size()
        trend_infra.index = trend_infra.index.astype(str)
        fig = px.area(x=trend_infra.index, y=trend_infra.values, title="Monthly Case Volume",
                      labels={"x": "Month", "y": "Cases"}, color_discrete_sequence=[next_solid()])
        render_chart(style_fig(fig, show_legend=False))

    col7, col8, col9 = st.columns(3, gap="large")
    with col7:
        review_court = fdf.loc[fdf["Court_No"] != ""].groupby(["Court_No", "Needs_Review"], observed=True).size().reset_index(name="Cases")
        review_court["Needs_Review"] = review_court["Needs_Review"].map({"Yes": "Flagged", "No": "Clean"})
        top_courts_list = fdf.loc[fdf["Court_No"] != "", "Court_No"].value_counts().head(10).index
        review_court = review_court[review_court["Court_No"].isin(top_courts_list)]
        fig = px.bar(review_court, x="Court_No", y="Cases", color="Needs_Review", barmode="stack",
                     title="Clean vs Flagged Cases by Court (Top 10)",
                     labels={"Court_No": "Court No.", "Cases": "Cases"},
                     color_discrete_map=color_map_for(review_court["Needs_Review"]))
        render_chart(style_fig(fig, legend_title="Status"))

    with col8:
        by_day_infra = fdf.loc[fdf["Day_Norm"] != "", "Day_Norm"].value_counts().reindex(DAY_ORDER).fillna(0)
        fig = px.line_polar(r=by_day_infra.values, theta=[d.title() for d in by_day_infra.index], line_close=True,
                             title="Hearings by Day of Week", color_discrete_sequence=[next_solid()])
        fig.update_traces(fill="toself")
        render_chart(style_fig(fig, show_legend=False))

    with col9:
        top_titles_infra = fdf.loc[fdf["Case_Title"] != "", "Case_Title"].value_counts().head(8)
        fig = px.bar(x=top_titles_infra.values, y=top_titles_infra.index, orientation="h",
                     title="Most Repeatedly Heard Case Titles", labels={"x": "Cases", "y": "Case Title"},
                     color_discrete_sequence=[next_solid()])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        render_chart(style_fig(fig, show_legend=False))

    col10, col11 = st.columns(2, gap="large")
    with col10:
        top_lawyers_infra = top_names(pd.concat([fdf["Petitioner_Advocate"], fdf["Respondent_Advocates"]]), top_n=10)
        fig = px.bar(x=top_lawyers_infra.values, y=top_lawyers_infra.index, orientation="h",
                     title="Top 10 Advocates by Appearances (Filtered)", labels={"x": "Appearances", "y": "Advocate"},
                     color_discrete_sequence=[next_solid()])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        render_chart(style_fig(fig, show_legend=False))
    with col11:
        by_year_infra = fdf.loc[fdf["Year"] != "", "Year"].value_counts().sort_index()
        fig = px.area(x=by_year_infra.index, y=by_year_infra.values, title="Case Volume by Year (Filtered)",
                      labels={"x": "Year", "y": "Cases"}, color_discrete_sequence=[next_solid()])
        render_chart(style_fig(fig, show_legend=False))

    st.markdown("#### Court Workload Profile (bubble size = Justices sitting)")
    workload = pd.DataFrame({
        "Court_No": [str(c) for c in court_lawyer_map.keys()],
        "Lawyers": list(court_lawyer_map.values()),
    })
    workload["Cases"] = workload["Court_No"].map(
        {str(k): v for k, v in fdf.loc[fdf["Court_No"] != "", "Court_No"].value_counts().items()})
    workload["Justices"] = workload["Court_No"].map({str(k): v for k, v in court_judge_map.items()})
    fig = px.scatter(workload, x="Cases", y="Lawyers", size="Justices", color="Court_No",
                      title="Cases vs Lawyers per Court (Bubble = No. of Justices)",
                      labels={"Cases": "Total Cases", "Lawyers": "Distinct Lawyers", "Court_No": "Court No."},
                      color_discrete_map=color_map_for(workload["Court_No"]), size_max=40)
    fig_workload = style_fig(fig, legend_title="Court No.", height=520)
    fig_workload.update_xaxes(automargin=True)
    fig_workload.update_yaxes(automargin=True)
    render_chart(fig_workload)

    st.markdown("#### Court Leaderboard")
    st.caption("Ranked by total case volume - click any column header to re-sort.")
    flagged_by_court = fdf.loc[fdf["Court_No"] != ""].groupby("Court_No", observed=True)["Needs_Review"].apply(
        lambda s: round((s == "Yes").mean() * 100, 1))
    leaderboard = workload.rename(columns={
        "Court_No": "Court No.", "Cases": "Total Cases",
        "Lawyers": "Distinct Lawyers", "Justices": "Distinct Justices",
    }).copy()
    leaderboard["Flagged %"] = leaderboard["Court No."].map({str(k): v for k, v in flagged_by_court.items()})
    leaderboard = leaderboard[["Court No.", "Total Cases", "Flagged %", "Distinct Justices", "Distinct Lawyers"]]
    leaderboard = leaderboard.sort_values("Total Cases", ascending=False)
    leaderboard["Flagged %"] = leaderboard["Flagged %"].fillna(0.0)
    lb_height = min(480, 46 + 36 * len(leaderboard))
    try:
        raw_max = leaderboard["Flagged %"].max()
        safe_max = raw_max if pd.notna(raw_max) and raw_max > 0 else 1.0
        st.dataframe(
            leaderboard, hide_index=True, height=lb_height,
            column_config={
                "Flagged %": st.column_config.ProgressColumn(
                    "Flagged %", format="%.1f%%", min_value=0.0, max_value=float(safe_max),
                ),
            },
            **_full_width_kwargs(st.dataframe),
        )
    except Exception:
        st.dataframe(leaderboard, hide_index=True, height=lb_height, **_full_width_kwargs(st.dataframe))

    st.markdown("#### Detail table")
    show_cols = ["Court_No", "Judges", "Case_Category", "Section", "Sr_No", "Case_No", "Case_Title", "Hearing_Date"]
    detail = fdf[show_cols].sort_values("Hearing_Date")

    search_term = st.text_input(
        "Search this table (Case No., Case Title, or Judges)",
        key="infra_detail_search",
        placeholder="e.g. W.P.No.1234/2023, or a party/judge name",
    )
    if search_term.strip():
        mask = (
            detail["Case_No"].str.contains(search_term, case=False, na=False)
            | detail["Case_Title"].str.contains(search_term, case=False, na=False)
            | detail["Judges"].str.contains(search_term, case=False, na=False)
        )
        detail = detail[mask]

    st.caption(f"Showing {len(detail):,} of {len(fdf):,} filtered rows - scroll within the table to browse (header stays fixed).")
    detail_height = min(480, 46 + 36 * max(len(detail), 1))
    st.dataframe(
        detail,
        height=detail_height,
        hide_index=True,
        **_full_width_kwargs(st.dataframe),
    )
    st.download_button(
        "Download this table (CSV)",
        data=detail.to_csv(index=False).encode("utf-8"),
        file_name="phc_detail_table.csv",
        mime="text/csv",
        key="infra_download_detail",
    )

# ============================================================
# DISTRIBUTION ANALYSIS BY COURT
# ============================================================
with tab_judge_court:
    st.markdown("### Distribution Analysis by Court")

    section_chip("Filter")
    court_options = sorted([c for c in df["Court_No"].unique() if c], key=str)
    court_pick = st.selectbox("Court No.", ["All"] + court_options, key="dist_court")
    fdf = df if court_pick == "All" else df[df["Court_No"] == court_pick]

    filter_summary_bar(
        {"Court No.": court_pick},
        keys_to_reset=["dist_court"],
        reset_button_key="dist_reset",
    )

    if len(fdf) == 0:
        st.info("No cases match the selected Court No. Try a different selection above.")

    section_chip("Snapshot")
    lawyers_f = len(unique_names(fdf["Petitioner_Advocate"]) | unique_names(fdf["Respondent_Advocates"]))
    judges_f = len(unique_names(fdf["Judges"], sep_pattern=r'&'))
    reviewed = (fdf["Needs_Review"] == "Yes").sum()
    kpi_row([
        dict(icon=svg_icon("cases"), label="\U0001F4C1 Total Cases", value=f"{len(fdf):,}", accent="indigo"),
        dict(icon=svg_icon("lawyer"), label="\U0001F4BC Total Lawyers Interacted", value=f"{lawyers_f:,}", accent="orange"),
        dict(icon=svg_icon("justice"), label="\u2696\ufe0f Judges Sitting Here", value=f"{judges_f:,}", accent="purple"),
        dict(icon=svg_icon("flag"), label="\U0001F6A9 Flagged for Review", value=f"{reviewed:,}", accent="red"),
    ])

    if len(fdf):
        _bits = [f"<strong>{len(fdf):,} cases</strong> in this view, involving <strong>{lawyers_f:,} lawyers</strong> and <strong>{judges_f} judges</strong>."]
        _bits.append(f"<strong>{(reviewed / len(fdf) * 100):.1f}%</strong> are flagged for review.")
        _sect_counts = fdf.loc[fdf["Section"] != "", "Section"].value_counts()
        if len(_sect_counts):
            _top_sect_pct = _sect_counts.iloc[0] / _sect_counts.sum() * 100
            _bits.append(
                f"Most common section: <strong>{_sect_counts.index[0]}</strong> "
                f"(<strong>{_top_sect_pct:.1f}%</strong> of cases here)."
            )
        _monthly_court = fdf.dropna(subset=["Hearing_Date"]).groupby(
            fdf["Hearing_Date"].dt.to_period("M")).size().sort_index()
        _court_anomalies = flag_anomalous_months(_monthly_court)
        if len(_court_anomalies):
            _worst = _court_anomalies.abs().sort_values(ascending=False).head(2)
            _months_txt = ", ".join(f"{p} ({int(_monthly_court[p]):,})" for p in _worst.index)
            _bits.append(f"Unusually high/low-volume months for this selection: <strong>{_months_txt}</strong>.")
        insight_box(" ".join(_bits))

    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        by_cat = fdf.loc[fdf["Case_Category"] != "", "Case_Category"].value_counts()
        fig = px.bar(x=by_cat.index, y=by_cat.values, title="No of Cases by Category",
                     labels={"x": "Category", "y": "Cases"}, color=by_cat.index,
                     color_discrete_map=color_map_for(by_cat.index))
        render_chart(style_fig(fig, legend_title="Category"))
    with col2:
        by_remark = fdf["Needs_Review"].value_counts(normalize=True).mul(100).round(2)
        by_remark.index = by_remark.index.map({"Yes": "Flagged", "No": "Clean"})
        fig = px.pie(values=by_remark.values, names=by_remark.index, title="% of Cases by Remarks", hole=0.4,
                     color=by_remark.index, color_discrete_map=color_map_for(by_remark.index))
        render_chart(style_fig(fig, legend_title="Status"))
    with col3:
        by_day = fdf.loc[fdf["Day_Norm"] != "", "Day_Norm"].value_counts().reindex(DAY_ORDER).fillna(0)
        fig = px.bar(x=[d.title() for d in by_day.index], y=by_day.values, title="No of Cases by Hearing Day",
                     labels={"x": "Day", "y": "Cases"}, color=by_day.index,
                     color_discrete_map=color_map_for(by_day.index))
        render_chart(style_fig(fig, show_legend=True))

    col4, col5, col6 = st.columns(3, gap="large")
    with col4:
        trend = fdf.dropna(subset=["Hearing_Date"]).groupby(fdf["Hearing_Date"].dt.to_period("M")).size()
        trend.index = trend.index.astype(str)
        fig = px.line(x=trend.index, y=trend.values, title="No of Cases by Month", markers=True,
                      labels={"x": "Month", "y": "Cases"}, color_discrete_sequence=[next_solid()])
        render_chart(style_fig(fig, show_legend=False))
    with col5:
        by_year = fdf.loc[fdf["Year"] != "", "Year"].value_counts().sort_index()
        fig = px.area(x=by_year.index, y=by_year.values, title="No of Cases by Year",
                      labels={"x": "Year", "y": "Cases"}, color_discrete_sequence=[next_solid()])
        render_chart(style_fig(fig, show_legend=False))
    with col6:
        sect_dist = fdf.loc[fdf["Section"] != "", "Section"].value_counts().head(10).reset_index()
        sect_dist.columns = ["Section", "Cases"]
        fig = px.treemap(sect_dist, path=["Section"], values="Cases", title="Top 10 Sections (Treemap)",
                          color="Cases", color_continuous_scale=next_sequential())
        render_chart(style_fig(fig))


    col7, col8, col9 = st.columns(3, gap="large")
    with col7:
        top_titles = fdf.loc[fdf["Case_Title"] != "", "Case_Title"].value_counts().head(8)
        fig = px.bar(x=top_titles.values, y=top_titles.index, orientation="h", title="Most-Repeated Case Titles",
                     labels={"x": "Cases", "y": "Case Title"}, color_discrete_sequence=[next_solid()])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        render_chart(style_fig(fig, show_legend=False))
    with col8:
        top_lawyers_c = top_names(pd.concat([fdf["Petitioner_Advocate"], fdf["Respondent_Advocates"]]), top_n=8)
        fig = px.bar(x=top_lawyers_c.values, y=top_lawyers_c.index, orientation="h",
                     title="Top Advocates Appearing in this Court", labels={"x": "Appearances", "y": "Advocate"},
                     color_discrete_sequence=[next_solid()])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        render_chart(style_fig(fig, show_legend=False))
    with col9:
        top_judges_c = top_names(fdf["Judges"], sep_pattern=r'&', top_n=8)
        fig = px.bar(x=top_judges_c.values, y=top_judges_c.index, orientation="h",
                     title="Justices Sitting in this Court", labels={"x": "Cases", "y": "Justice"},
                     color_discrete_sequence=[next_solid()])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        render_chart(style_fig(fig, show_legend=False))

    yr_cat_court = fdf.loc[(fdf["Year"] != "") & (fdf["Case_Category"] != "")].groupby(
        ["Year", "Case_Category"], observed=True).size().reset_index(name="Cases")
    years_court = sorted([y for y in fdf["Year"].unique() if str(y).strip()])
    fig = px.bar(yr_cat_court, x="Year", y="Cases", color="Case_Category", title="Case Category Mix by Year (This Court)",
                 barmode="stack", color_discrete_map=color_map_for(yr_cat_court["Case_Category"]), category_orders={"Year": years_court})
    render_chart(style_fig(fig, legend_title="Category"))

    st.markdown("#### Monthly Caseload Spread by Year (distribution shape)")
    box_df = fdf.dropna(subset=["Hearing_Date"]).copy()
    box_df["Year"] = box_df["Hearing_Date"].dt.year.astype(str)
    monthly_counts = box_df.groupby(["Year", box_df["Hearing_Date"].dt.month], observed=True).size().reset_index(name="Cases")
    fig = px.box(monthly_counts, x="Year", y="Cases", title="Distribution of Monthly Case Counts per Year",
                 labels={"Cases": "Cases in a Month"}, color="Year", color_discrete_map=color_map_for(monthly_counts["Year"]))
    render_chart(style_fig(fig, legend_title="Year"))

# ============================================================
# CASE DISTRIBUTION BY JUDGE
# ============================================================
with tab_judge:
    st.markdown("### Case Distribution by Judge")

    judge_names = name_idx["judge_names"]
    if not judge_names:
        st.warning("No individual judge names could be parsed from the Judges column.")
        st.stop()

    section_chip("Filter")
    judge_pick = st.selectbox("Justice Name", judge_names, key="judge_pick")
    fdf = filter_by_judge(df, judge_pick)

    section_chip("Snapshot")
    lawyers_f = len(unique_names(fdf["Petitioner_Advocate"]) | unique_names(fdf["Respondent_Advocates"]))
    reviewed = (fdf["Needs_Review"] == "Yes").sum()
    per_day = fdf.dropna(subset=["Hearing_Date"]).groupby("Hearing_Date").size()
    avg_per_day = f"{per_day.mean():.1f}" if len(per_day) else "N/A"
    kpi_row([
        dict(icon=svg_icon("cases"), label="\U0001F4C1 Total Cases", value=f"{len(fdf):,}", accent="indigo"),
        dict(icon=svg_icon("lawyer"), label="\U0001F4BC Total Lawyers Interacted", value=f"{lawyers_f:,}", accent="orange"),
        dict(icon=svg_icon("flag"), label="\U0001F6A9 Flagged for Review", value=f"{reviewed:,}", accent="red"),
        dict(icon=svg_icon("bar"), label="\U0001F4CA Avg Cases per Hearing Day", value=avg_per_day, accent="teal"),
    ])

    if len(fdf):
        _bits = [f"<strong>{judge_pick}</strong> has presided over <strong>{len(fdf):,} cases</strong> on record."]
        _cat_counts = fdf.loc[fdf["Case_Category"] != "", "Case_Category"].value_counts()
        if len(_cat_counts):
            _bits.append(f"Most common category: <strong>{_cat_counts.idxmax()}</strong> ({int(_cat_counts.max()):,} cases).")
        _bits.append(f"<strong>{(reviewed / len(fdf) * 100):.1f}%</strong> of these cases are flagged for review.")
        _all_judge_counts = name_idx["judge_counts"]
        if len(_all_judge_counts):
            _peer_avg = _all_judge_counts.mean()
            if _peer_avg > 0:
                _vs_peer_pct = (len(fdf) - _peer_avg) / _peer_avg * 100
                _direction = "more" if _vs_peer_pct >= 0 else "fewer"
                _bits.append(
                    f"That's <strong>{abs(_vs_peer_pct):.0f}% {_direction}</strong> cases than the average "
                    f"justice ({_peer_avg:,.0f} cases)."
                )
        _monthly_judge = fdf.dropna(subset=["Hearing_Date"]).groupby(
            fdf["Hearing_Date"].dt.to_period("M")).size().sort_index()
        _judge_anomalies = flag_anomalous_months(_monthly_judge)
        if len(_judge_anomalies):
            _worst = _judge_anomalies.abs().sort_values(ascending=False).head(2)
            _months_txt = ", ".join(f"{p} ({int(_monthly_judge[p]):,})" for p in _worst.index)
            _bits.append(f"Unusually high/low-volume sitting months: <strong>{_months_txt}</strong>.")
        insight_box(" ".join(_bits))

    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        by_remark = fdf["Needs_Review"].value_counts()
        by_remark.index = by_remark.index.map({"Yes": "Flagged", "No": "Clean"})
        fig = px.bar(x=by_remark.index, y=by_remark.values, title="No of Cases by Remarks",
                     labels={"x": "Status", "y": "Cases"}, color=by_remark.index,
                     color_discrete_map=color_map_for(by_remark.index))
        render_chart(style_fig(fig, legend_title="Status"))
    with col2:
        by_cat = fdf.loc[fdf["Case_Category"] != "", "Case_Category"].value_counts()
        fig = px.pie(values=by_cat.values, names=by_cat.index, title="% by Case Type", hole=0.4,
                     color=by_cat.index, color_discrete_map=color_map_for(by_cat.index))
        render_chart(style_fig(fig, legend_title="Category"))
    with col3:
        top_titles = fdf.loc[fdf["Case_Title"] != "", "Case_Title"].value_counts().head(7)
        fig = px.bar(x=top_titles.values, y=top_titles.index, orientation="h", title="Most Repeatedly Heard Cases",
                     labels={"x": "Cases", "y": "Case Title"}, color_discrete_sequence=[next_solid()])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        render_chart(style_fig(fig, show_legend=False))

    col4, col5, col6 = st.columns(3, gap="large")
    with col4:
        trend = fdf.dropna(subset=["Hearing_Date"]).groupby(fdf["Hearing_Date"].dt.date).size()
        fig = px.line(x=list(trend.index), y=trend.values, title="No of Cases by Date", markers=True,
                      labels={"x": "Date", "y": "Cases"}, color_discrete_sequence=[next_solid()])
        render_chart(style_fig(fig, show_legend=False))
    with col5:
        by_court_j = fdf.loc[fdf["Court_No"] != "", "Court_No"].value_counts().head(10)
        fig = px.bar(x=by_court_j.index.astype(str), y=by_court_j.values, title="Courts Presided Over",
                     labels={"x": "Court No.", "y": "Cases"}, color=by_court_j.index.astype(str),
                     color_discrete_map=color_map_for(by_court_j.index.astype(str)))
        render_chart(style_fig(fig, show_legend=False))
    with col6:
        by_year_j = fdf.loc[fdf["Year"] != "", "Year"].value_counts().sort_index()
        fig = px.area(x=by_year_j.index, y=by_year_j.values, title="Caseload by Year",
                      labels={"x": "Year", "y": "Cases"}, color_discrete_sequence=[next_solid()])
        render_chart(style_fig(fig, show_legend=False))


    col7, col8, col9 = st.columns(3, gap="large")
    with col7:
        sect_j = fdf.loc[fdf["Section"] != "", "Section"].value_counts().head(10).reset_index()
        sect_j.columns = ["Section", "Cases"]
        fig = px.treemap(sect_j, path=["Section"], values="Cases", title="Sections Handled (Treemap)",
                          color="Cases", color_continuous_scale=next_sequential())
        render_chart(style_fig(fig))
    with col8:
        top_lawyers_j = top_names(pd.concat([fdf["Petitioner_Advocate"], fdf["Respondent_Advocates"]]), top_n=8)
        fig = px.bar(x=top_lawyers_j.values, y=top_lawyers_j.index, orientation="h",
                     title="Top Advocates Appearing Before this Justice", labels={"x": "Appearances", "y": "Advocate"},
                     color_discrete_sequence=[next_solid()])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        render_chart(style_fig(fig, show_legend=False))
    with col9:
        by_day_j = fdf.loc[fdf["Day_Norm"] != "", "Day_Norm"].value_counts().reindex(DAY_ORDER).fillna(0)
        fig = px.line_polar(r=by_day_j.values, theta=[d.title() for d in by_day_j.index], line_close=True,
                             title="Sitting Pattern by Day of Week", color_discrete_sequence=[next_solid()])
        fig.update_traces(fill="toself")
        render_chart(style_fig(fig, show_legend=False))

    yr_cat_judge = fdf.loc[(fdf["Year"] != "") & (fdf["Case_Category"] != "")].groupby(
        ["Year", "Case_Category"], observed=True).size().reset_index(name="Cases")
    years_judge = sorted([y for y in fdf["Year"].unique() if str(y).strip()])
    fig = px.bar(yr_cat_judge, x="Year", y="Cases", color="Case_Category", title="Case Category Mix by Year (This Justice)",
                 barmode="stack", color_discrete_map=color_map_for(yr_cat_judge["Case_Category"]), category_orders={"Year": years_judge})
    render_chart(style_fig(fig, legend_title="Category"))

    st.markdown("#### Monthly Caseload Heatmap")
    heat_df = fdf.dropna(subset=["Hearing_Date"]).copy()
    if len(heat_df):
        heat_df["Year"] = heat_df["Hearing_Date"].dt.year
        heat_df["Month"] = heat_df["Hearing_Date"].dt.strftime("%b")
        pivot = heat_df.groupby(["Year", "Month"], observed=True).size().reset_index(name="Cases")
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot_table = pivot.pivot(index="Year", columns="Month", values="Cases").reindex(columns=month_order)
        fig = px.imshow(pivot_table, title="Cases Heard by Year and Month", labels=dict(x="Month", y="Year", color="Cases"),
                         color_continuous_scale=next_sequential(), aspect="auto")
        render_chart(style_fig(fig))
    else:
        st.info("No dated hearings available for this justice to build a heatmap.")
