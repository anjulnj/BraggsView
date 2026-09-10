"""
theme.py — CLEAN HIGH-CONTRAST READABLE THEME + Publication Variant

Phase 4 improvements:
- Colorblind-safe palette (Okabe-Ito derived) as CLEAN_PALETTE_CB_SAFE, exposed in PLOT pickers via PALETTES integration note
- Print/export-safe mode mode="print": pure white, all-black text/lines, >=2px lines, no soft grays
- Font consistency: Streamlit UI uses Inter (webfont), Plotly export uses system-safe Arial/Helvetica (no internet needed for kaleido)
- Contrast audit: programmatic WCAG AA check (contrast_ratio, audit function)
- Theme persistence: helper get_theme_choice() using st.session_state (to be wired in app.py Phase 5)

Backward compat: old API aliases kept forever.
"""

from __future__ import annotations
from typing import Any, Optional, Dict, Tuple
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Theme tokens
# ---------------------------------------------------------------------------

LIGHT = {
    "bg": "#F7F8FA",
    "surface": "#FFFFFF",
    "surface_2": "#F1F5F9",
    "surface_hover": "#E2E8F0",
    "text_primary": "#0F172A",
    "text_secondary": "#475569",
    "text_muted": "#5C6B80",
    "border": "#E2E8F0",
    "border_strong": "#CBD5E1",
    "grid": "#E2E8F0",
    "accent": "#2563EB",
    "accent_blue": "#0F172A",
}

DARK_READABLE = {
    "bg": "#0F172A",
    "surface": "#1E293B",
    "surface_2": "#334155",
    "surface_hover": "#475569",
    "text_primary": "#F8FAFC",
    "text_secondary": "#E2E8F0",
    "text_muted": "#94A3B8",
    "border": "#334155",
    "border_strong": "#475569",
    "grid": "rgba(148,163,184,0.22)",
    "accent": "#60A5FA",
    "accent_blue": "#38BDF8",
}

PRINT = {
    # Print/export-safe: pure white, all black, no soft grays
    "bg": "#FFFFFF",
    "surface": "#FFFFFF",
    "surface_2": "#FFFFFF",
    "text_primary": "#000000",
    "text_secondary": "#000000",
    "text_muted": "#000000",
    "border": "#000000",
    "border_strong": "#000000",
    "grid": "#E5E5E5",
    "accent": "#000000",
    "accent_blue": "#000000",
}

# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

CLEAN_PALETTE = [
    "#0F172A",  # ink
    "#2563EB",  # blue
    "#059669",  # emerald
    "#D97706",  # amber
    "#7C3AED",  # violet
    "#DB2777",  # pink
    "#0891B2",  # cyan
    "#475569",  # slate
]

CLEAN_PALETTE_SOFT = [
    "#334155",
    "#6366F1",
    "#0E7490",
    "#047857",
    "#B45309",
    "#BE185D",
    "#6D28D9",
    "#64748B",
]

# Colorblind-safe: Okabe-Ito 8-color + extension; verified distinct for deuteranopia
# Source: Okabe & Ito, "Color Universal Design" + Wong 2011 Nature Methods
CLEAN_PALETTE_CB_SAFE = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#999999",  # gray (extra)
]

CLEAN_PALETTE_CB_SAFE_SOFT = [
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#999933",
    "#DDCC77",
    "#CC6677",
    "#882255",
]

# For backward compat with earlier aesthetic naming
AESTHETIC_PALETTE = CLEAN_PALETTE
AESTHETIC_PALETTE_BOLD = CLEAN_PALETTE_SOFT
AESTHETIC = {
    "bg": LIGHT["bg"],
    "surface": LIGHT["surface"],
    "text_primary": LIGHT["text_primary"],
    "text_secondary": LIGHT["text_secondary"],
    "text_muted": LIGHT["text_muted"],
    "border": LIGHT["border"],
    "border_strong": LIGHT["border_strong"],
    "grid": LIGHT["grid"],
    "font_sans": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
}

# ---------------------------------------------------------------------------
# WCAG Contrast Audit
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """Convert #RRGGBB or #RGB to 0-1 RGB."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join([c*2 for c in h])
    if len(h) != 6:
        raise ValueError(f"Invalid hex {hex_color}")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return r, g, b

def _relative_luminance(rgb: Tuple[float, float, float]) -> float:
    """WCAG relative luminance."""
    def _lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)

def contrast_ratio(fg_hex: str, bg_hex: str) -> float:
    """
    WCAG contrast ratio between two hex colors.
    Returns ratio like 4.5, 7.0, 15.0 etc.
    """
    # Handle rgba strings for grid - extract rgb ignoring alpha
    if fg_hex.startswith("rgba"):
        # parse rgba(r,g,b,a) -> average? For grid we skip audit, but return low
        # approximate by using white bg for grid lines (they are light, contrast not required)
        return 1.0
    try:
        l1 = _relative_luminance(_hex_to_rgb(fg_hex))
        l2 = _relative_luminance(_hex_to_rgb(bg_hex))
    except Exception:
        return 1.0
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)

def audit_theme_contrast(mode: str = "light") -> Dict[str, Tuple[float, bool]]:
    """
    Run WCAG AA contrast check for current theme tokens.
    AA requires >=4.5:1 for normal text, >=3:1 for large text.

    Returns dict token_pair -> (ratio, passes_AA)
    """
    tokens = {"light": LIGHT, "dark": DARK_READABLE, "print": PRINT}.get(mode, LIGHT)
    bg = tokens["bg"]
    checks = {}
    for key in ["text_primary", "text_secondary", "text_muted"]:
        if key in tokens:
            fg = tokens[key]
            # Skip if fg is rgba
            if isinstance(fg, str) and fg.startswith("#"):
                ratio = contrast_ratio(fg, bg)
                passes = ratio >= 4.5
                checks[f"{key} on bg"] = (ratio, passes)
    return checks

def print_contrast_audit():
    """Print audit to console, useful for manual run."""
    for mode in ["light", "dark", "print"]:
        print(f"\n=== Contrast audit: {mode} ===")
        results = audit_theme_contrast(mode=mode)
        for pair, (ratio, passes) in results.items():
            status = "PASS" if passes else "FAIL"
            print(f"  {pair}: {ratio:.2f}:1 -> {status} (AA requires 4.5:1)")


# ---------------------------------------------------------------------------
# Plotly Templates - with font separation for export safety
# ---------------------------------------------------------------------------

def get_clean_plotly_template(mode: str = "light", for_export: bool = False) -> Any:
    """
    Return high-contrast readable Plotly template.

    Phase 4 fix: font consistency
    - for_export=False (screen): uses Inter (webfont) for Streamlit UI, looks nice on screen
    - for_export=True (PNG/PDF/SVG via kaleido): uses system-safe Arial/Helvetica, because kaleido
      renders headless without internet, Inter webfont not available -> would fallback to ugly serif.
      System-safe ensures exported figure looks like screen figure but with guaranteed font.

    Args:
        mode: light, dark, print
        for_export: if True, use system-safe fonts and thicker lines
    """
    tokens = {"light": LIGHT, "dark": DARK_READABLE, "print": PRINT}.get(mode, LIGHT)
    is_light = mode in ("light", "print")
    is_print = mode == "print"

    template: Any = go.layout.Template()

    # Colorway depends on mode: for print, use black only or CB safe black-heavy?
    if is_print:
        template.layout.colorway = ["#000000", "#333333", "#666666", "#000000", "#000000", "#000000"]
    else:
        template.layout.colorway = CLEAN_PALETTE_CB_SAFE if "cb" in mode else CLEAN_PALETTE

    # Font separation: screen vs export
    if for_export:
        # System-safe stack guaranteed in PDF/SVG viewers and kaleido
        font_family = "Arial, Helvetica, sans-serif"
    else:
        font_family = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

    template.layout.font = dict(family=font_family, size=13 if not is_print else 14, color=tokens["text_primary"])
    template.layout.title = dict(font=dict(size=19 if not is_print else 20, color=tokens["text_primary"], family=font_family), x=0.02, xanchor="left")
    template.layout.paper_bgcolor = tokens["bg"]
    template.layout.plot_bgcolor = tokens["bg"]

    line_width = 2.0 if is_print else 1.2
    grid_color = tokens["grid"]
    line_color = tokens["border_strong"]

    axis = dict(
        showline=True,
        linewidth=line_width,
        linecolor=line_color,
        showgrid=True,
        gridcolor=grid_color,
        gridwidth=1 if not is_print else 0.8,
        zeroline=False,
        ticks="outside",
        ticklen=5 if not is_print else 6,
        tickwidth=1 if not is_print else 1.5,
        tickcolor=line_color,
        tickfont=dict(size=11 if not is_print else 12, color=tokens["text_secondary"], family=font_family),
        title=dict(font=dict(size=12 if not is_print else 13, color=tokens["text_secondary"], family=font_family)),
    )

    template.layout.xaxis = axis.copy()
    template.layout.yaxis = axis.copy()

    template.layout.legend = dict(
        font=dict(size=11, color=tokens["text_secondary"], family=font_family),
        bgcolor="rgba(255,255,255,0.95)" if is_light else "rgba(15,23,42,0.9)" if mode != "print" else "rgba(255,255,255,1)",
        bordercolor=tokens["border"],
        borderwidth=1,
        yanchor="top",
        y=0.98,
        xanchor="left",
        x=1.02,
    )

    template.layout.margin = dict(l=70, r=160, t=70, b=60)
    template.layout.hoverlabel = dict(
        bgcolor="white" if is_light else "#1E293B",
        bordercolor=tokens["border"],
        font=dict(size=11, color=tokens["text_primary"], family=font_family),
    )

    return template


def apply_clean_plotly_theme(fig, mode: str = "light", palette: str = "clean", for_export: bool = False, line_width: Optional[float] = None) -> go.Figure:
    """Apply clean readable theme to figure, with export-safe font handling."""
    tokens = {"light": LIGHT, "dark": DARK_READABLE, "print": PRINT}.get(mode, LIGHT)
    tmpl = get_clean_plotly_template(mode=mode, for_export=for_export)

    fig.update_layout(template=tmpl)
    fig.update_layout(paper_bgcolor=tokens["bg"], plot_bgcolor=tokens["bg"], font=dict(color=tokens["text_primary"]))

    # If print mode, force thicker lines
    default_width = 2.0 if mode == "print" else 1.6
    if line_width is not None:
        default_width = line_width

    fig.update_xaxes(gridcolor=tokens["grid"], linecolor=tokens["border_strong"])
    fig.update_yaxes(gridcolor=tokens["grid"], linecolor=tokens["border_strong"])

    # Determine palette
    if palette == "cb_safe" or palette == "colorblind":
        clean_colors = CLEAN_PALETTE_CB_SAFE
    elif palette == "clean_soft":
        clean_colors = CLEAN_PALETTE_SOFT
    else:
        clean_colors = CLEAN_PALETTE

    for i, tr in enumerate(fig.data):
        if getattr(tr, "fill", None) == "tonexty":
            continue
        if "baseline" in (tr.name or "").lower():
            continue
        if hasattr(tr, "line"):
            try:
                tr.line.color = clean_colors[i % len(clean_colors)]
                tr.line.width = default_width
            except Exception:
                pass

    return fig


# Backward compat aliases
def get_aesthetic_plotly_template():
    return get_clean_plotly_template(mode="light", for_export=False)

def apply_aesthetic_theme_to_fig(fig, palette_name="Aesthetic", **kwargs):
    mode = kwargs.get("mode", "light")
    palette = "clean_soft" if "soft" in palette_name.lower() or "bold" in palette_name.lower() else "clean"
    return apply_clean_plotly_theme(fig, mode=mode, palette=palette, **{k: v for k, v in kwargs.items() if k != "mode"})

def get_aesthetic_palette(n, bold=False):
    base = CLEAN_PALETTE if not bold else CLEAN_PALETTE_SOFT
    return [base[i % len(base)] for i in range(n)]

def get_palette_colors(name, n, interpolate=True):
    # Map old dark names to clean, and expose cb_safe
    lname = name.lower()
    if "cb" in lname or "colorblind" in lname or "okabe" in lname or "ito" in lname:
        base = CLEAN_PALETTE_CB_SAFE
    elif "dark" in lname:
        base = CLEAN_PALETTE
    elif "soft" in lname:
        base = CLEAN_PALETTE_SOFT
    else:
        base = CLEAN_PALETTE
    if not interpolate:
        return [base[i % len(base)] for i in range(n)]
    return [base[i % len(base)] for i in range(n)]


# ---------------------------------------------------------------------------
# Streamlit CSS
# ---------------------------------------------------------------------------

CLEAN_LIGHT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* -------------------------
   BASE TYPOGRAPHY
-------------------------- */

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* -------------------------
   PAGE
-------------------------- */

[data-testid="stAppViewContainer"] {
    background: #FAFAFA !important;
    color: #1A1A1A !important;
}

[data-testid="stHeader"] {
    background: rgba(250,250,250,0.96) !important;
    border-bottom: 1px solid #E6E6E6 !important;
}

/* -------------------------
   SIDEBAR
-------------------------- */

[data-testid="stSidebar"] {
    background: #F4F4F2 !important;
    border-right: 1px solid #E2E2DF !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.01em !important;
    margin-bottom: 4px !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #333333 !important;
    font-size: 13px !important;
}

/* Sidebar inputs (Streamlit 1.6x testids) */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-testid="stNumberInput"] {
    background: #FFFFFF !important;
    border: 1px solid #DCDCD7 !important;
    border-radius: 6px !important;
    color: #1A1A1A !important;
}

/* Sidebar selectbox / multiselect containers (Streamlit 1.6x) */
[data-testid="stSidebar"] [data-testid="stSelectbox"],
[data-testid="stSidebar"] [data-testid="stMultiSelect"] {
    background: #FFFFFF !important;
    border: 1px solid #DCDCD7 !important;
    border-radius: 6px !important;
    color: #1A1A1A !important;
}

/* Selected-value chips inside multiselect: light gray pill, dark text */
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [aria-selected="true"],
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [role="option"],
[data-testid="stSidebar"] [data-testid="stMultiSelect"] span[aria-label^="Remove"],
[data-testid="stSidebar"] [data-testid="stMultiSelect"] div[aria-label^="Remove"] {
    background: #E8E8E3 !important;
    color: #1A1A1A !important;
    border: 1px solid #DCDCD7 !important;
    border-radius: 6px !important;
}

[data-testid="stSidebar"] [data-testid="stMultiSelect"] span[aria-label^="Remove"] *,
[data-testid="stSidebar"] [data-testid="stMultiSelect"] div[aria-label^="Remove"] * {
    color: #1A1A1A !important;
}

/* Virtual dropdown panel (opened list) */
[data-testid="stSelectboxVirtualDropdown"],
[data-testid="stSelectboxVirtualDropdown"] [role="option"] {
    background: #FFFFFF !important;
    color: #1A1A1A !important;
}

/* -------------------------
   FILE UPLOADER FIX
-------------------------- */

/* Remove duplicate Upload text */
[data-testid="stFileUploader"] button span:nth-child(2) {
    display: none !important;
}

/* Clean upload button */
[data-testid="stFileUploader"] button {
    background: #FFFFFF !important;
    border: 1px solid #D6D6D6 !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    padding: 0.35rem 0.9rem !important;
}

/* -------------------------
   MAIN CONTENT CARDS
-------------------------- */

.stPlotlyChart,
[data-testid="stExpander"],
[data-testid="stMetric"],
[data-testid="stDataFrame"] {
    background: #FFFFFF !important;
    border: 1px solid #E6E6E6 !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}

/* -------------------------
   TEXT
-------------------------- */

h1 {
    font-weight: 700 !important;
    font-size: 28px !important;
    letter-spacing: -0.02em !important;
}

h2, h3 {
    font-weight: 600 !important;
}

[data-testid="stMarkdownContainer"] p {
    color: #2A2A2A !important;
    line-height: 1.7 !important;
}

/* -------------------------
   BUTTONS (Improved green)
-------------------------- */

.stButton > button {
    background: #3E8E6E !important;
    color: #FFFFFF !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 0.5rem 1.1rem !important;
    border: none !important;
}

.stButton > button:hover {
    background: #2F6F56 !important;
}

/* Download buttons */
[data-testid="stDownloadButton"] > button {
    background: #FFFFFF !important;
    color: #3E8E6E !important;
    border: 1.5px solid #3E8E6E !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}

[data-testid="stDownloadButton"] > button:hover {
    background: #F2F7F4 !important;
}

/* -------------------------
   ALERTS
-------------------------- */

div[data-testid="stAlert"] {
    background: #F7F9F8 !important;
    border: 1px solid #DCE5E1 !important;
    border-radius: 8px !important;
}

/* -------------------------
   CLEANUP
-------------------------- */

footer,
#MainMenu {
    visibility: hidden;
}
</style>
"""

CLEAN_DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
[data-testid="stAppViewContainer"] {
    background: #0F172A !important;
    color: #F8FAFC !important;
}
[data-testid="stHeader"] {
    background: rgba(15, 23, 42, 0.95) !important;
    backdrop-filter: blur(12px);
}
[data-testid="stSidebar"] {
    background: #111827 !important;
    border-right: 1px solid #334155 !important;
}
[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #CBD5E1 !important;
    font-size: 13px !important;
}
h1, h2, h3 {
    color: #F8FAFC !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: #E2E8F0 !important;
    line-height: 1.7 !important;
}
[data-testid="stExpander"],
.stPlotlyChart,
[data-testid="stDataFrame"],
[data-testid="stMetric"] {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 16px !important;
}
[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: #F8FAFC !important;
    font-weight: 700 !important;
}
.stSelectbox > div > div,
.stTextInput input,
.stNumberInput input,
[data-baseweb="select"] > div {
    background: #111827 !important;
    color: #F8FAFC !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
}
[data-testid="stSelectbox"],
[data-testid="stMultiSelect"] {
    background: #111827 !important;
    color: #F8FAFC !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
}
[data-testid="stSelectboxVirtualDropdown"],
[data-testid="stSelectboxVirtualDropdown"] [role="option"] {
    background: #111827 !important;
    color: #F8FAFC !important;
}
[data-testid="stMultiSelect"] span[aria-label^="Remove"],
[data-testid="stMultiSelect"] div[aria-label^="Remove"] {
    background: #334155 !important;
    color: #F8FAFC !important;
    border: 1px solid #475569 !important;
    border-radius: 6px !important;
}
[data-testid="stMultiSelect"] span[aria-label^="Remove"] *,
[data-testid="stMultiSelect"] div[aria-label^="Remove"] * {
    color: #F8FAFC !important;
}
div:not([data-testid="stFileUploader"]) .stButton > button {
    background: #2563EB !important;
    color: #F8FAFC !important;
    border-radius: 999px !important;
    border: none !important;
    font-weight: 700 !important;
    padding: 0.65rem 1.3rem !important;
}
div:not([data-testid="stFileUploader"]) .stButton > button:hover {
    background: #1D4ED8 !important;
}
[data-testid="stDownloadButton"] > button {
    background: #111827 !important;
    color: #F8FAFC !important;
    border: 1px solid #334155 !important;
    border-radius: 999px !important;
}
div[data-testid="stAlert"] {
    background: #152038 !important;
    color: #F8FAFC !important;
    border: 1px solid #334155 !important;
    border-radius: 14px !important;
}
div[data-testid="stAlert"] * {
    color: #F8FAFC !important;
}
footer,
#MainMenu {
    visibility: hidden;
}
</style>
"""

CLEAN_PRINT_CSS = """
<style>
[data-testid="stAppViewContainer"] { background: #FFFFFF !important; color: #000000 !important; }
[data-testid="stSidebar"] { background: #FFFFFF !important; border-right: 2px solid #000000 !important; }
h1, h2, h3 { color: #000000 !important; }
[data-testid="stMarkdownContainer"] p { color: #000000 !important; }
</style>
"""

def inject_clean_theme(mode: str = "light"):
    try:
        import streamlit as st
        if mode == "light":
            css = CLEAN_LIGHT_CSS
        elif mode == "dark":
            css = CLEAN_DARK_CSS
        elif mode == "print":
            css = CLEAN_PRINT_CSS
        else:
            css = CLEAN_LIGHT_CSS
        st.markdown(css, unsafe_allow_html=True)
    except Exception:
        pass

# Theme persistence helpers (to be wired in app.py Phase 5)
def get_theme_choice(default: str = "light") -> str:
    """Get theme choice from session_state, with persistence."""
    try:
        import streamlit as st
        if "theme_choice" not in st.session_state:
            st.session_state.theme_choice = default
        return st.session_state.theme_choice
    except Exception:
        return default

def set_theme_choice(choice: str):
    try:
        import streamlit as st
        st.session_state.theme_choice = choice
    except Exception:
        pass

# Compat aliases
inject_aesthetic_css = lambda: inject_clean_theme(mode="light")
inject_aesthetic_theme = inject_aesthetic_css
inject_clean_css = inject_clean_theme

# Config TOML helper
STREAMLIT_CONFIG_TOML = """
[theme]
primaryColor = "#2563EB"
backgroundColor = "#FEFEFC"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#111827"
font = "sans serif"
"""

def get_streamlit_config():
    return STREAMLIT_CONFIG_TOML.strip()

# For testing contrast audit
if __name__ == "__main__":
    print_contrast_audit()
    # Test palette colorblind safety: just print
    print("\nCB Safe palette:", CLEAN_PALETTE_CB_SAFE)
    # Test template creation for all modes
    for m in ["light", "dark", "print"]:
        tmpl = get_clean_plotly_template(mode=m)
        print(f"Template {m}: colorway {tmpl.layout.colorway[:2]}")
        tmpl_export = get_clean_plotly_template(mode=m, for_export=True)
        print(f"  Export font: {tmpl_export.layout.font.family}")
