"""
pubfigure.py — Publication Figure Studio (self-contained, keeps app.py light)

Builds a fully-editable, journal-quality XRD figure with full control over:
- per-file display names (legend labels) without renaming files
- title, x/y axis labels, font sizes
- legend placement (inside/outside, corner, orientation), on/off
- per-file colors and line widths, grid on/off, x/y ranges
- size presets: single column (3.5"), 1.5 column (5"), double column (7")
- export: SVG + PDF (vector), PNG at 300/600 DPI (raster)

Export helpers delegate to plotting.figure_to_* which use kaleido.
EPS is NOT available (Kaleido v1 limitation) — we offer SVG/PDF/PNG.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from plotting import create_xrd_figure, figure_to_png_bytes, figure_to_svg_bytes, figure_to_pdf_bytes

# Journal size presets in inches (width, height). 72 px/inch at base.
SIZE_PRESETS: Dict[str, Dict[str, float]] = {
    "Single column (3.5 in)": {"width_in": 3.5, "height_in": 2.6},
    "1.5 column (5 in)": {"width_in": 5.0, "height_in": 3.5},
    "Double column (7 in)": {"width_in": 7.0, "height_in": 4.5},
    "Square (6 x 5 in)": {"width_in": 6.0, "height_in": 5.0},
}

LEGEND_POSITIONS: Dict[str, Dict] = {
    "Inside top-right": {"x": 0.99, "y": 0.99, "xanchor": "right", "yanchor": "top"},
    "Inside top-left": {"x": 0.01, "y": 0.99, "xanchor": "left", "yanchor": "top"},
    "Inside bottom-right": {"x": 0.99, "y": 0.01, "xanchor": "right", "yanchor": "bottom"},
    "Inside bottom-left": {"x": 0.01, "y": 0.01, "xanchor": "left", "yanchor": "bottom"},
    "Outside right (upper)": {"x": 1.02, "y": 0.99, "xanchor": "left", "yanchor": "top"},
    "Outside right (middle)": {"x": 1.02, "y": 0.5, "xanchor": "left", "yanchor": "middle"},
}

EXPORT_FONT = "Arial, Helvetica, sans-serif"


def _resolve_size_preset(name: str) -> Dict[str, float]:
    return SIZE_PRESETS.get(name, SIZE_PRESETS["Single column (3.5 in)"])


def create_publication_figure(
    data_dict: Dict[str, pd.DataFrame],
    display_names: Optional[Dict[str, str]] = None,
    palette_name: str = "Clean",
    title: str = "",
    x_label: str = "2\u03b8 (\u00b0)",
    y_label: str = "Intensity (a.u.)",
    legend_visible: bool = True,
    legend_position: str = "Inside top-right",
    legend_orientation: str = "v",
    legend_font_size: int = 10,
    tick_font_size: int = 11,
    label_font_size: int = 13,
    title_font_size: int = 16,
    line_width: float = 1.5,
    color_overrides: Optional[Dict[str, str]] = None,
    show_grid: bool = True,
    x_range: Optional[tuple] = None,
    y_range: Optional[tuple] = None,
    mode: str = "light",
    for_export: bool = True,
) -> Any:
    """
    Create a publication-quality XRD figure with full editorial control.

    Builds an overlay figure from the processed patterns (no stack/grid),
    then overrides trace names/colors/widths and layout per the settings.
    Uses system-safe Arial fonts and white background so the exported figure
    is identical to the preview and safe for journals.
    """
    fig = create_xrd_figure(
        data_dict=data_dict,
        palette_name=palette_name,
        stack_mode=False,
        offset_spacing=0.0,
        twotheta_label=x_label,
        intensity_label=y_label,
        title=title,
        template="plotly_white",
        normalization_method="none",
        show_peak_labels=False,
    )

    # Per-trace overrides: display name, color, line width
    display_names = display_names or {}
    names = sorted(data_dict.keys())
    from plotting import get_palette_colors
    colors = get_palette_colors(palette_name, len(names), interpolate=True)

    for i, tr in enumerate(fig.data):
        orig = tr.name
        label = display_names.get(orig, orig)
        tr.name = label
        override = (color_overrides or {}).get(orig) or (color_overrides or {}).get(label)
        if override:
            tr.line.color = override
        elif orig in names:
            tr.line.color = colors[names.index(orig) % len(colors)]
        tr.line.width = line_width

    # Layout overrides
    lp = LEGEND_POSITIONS.get(legend_position, LEGEND_POSITIONS["Inside top-right"])
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        title=dict(text=title or None, font=dict(size=title_font_size, family=EXPORT_FONT, color="#000000"), x=0, xanchor="left"),
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis=dict(tickfont=dict(size=tick_font_size, family=EXPORT_FONT), title_font=dict(size=label_font_size, family=EXPORT_FONT)),
        yaxis=dict(tickfont=dict(size=tick_font_size, family=EXPORT_FONT), title_font=dict(size=label_font_size, family=EXPORT_FONT)),
        legend=dict(
            visible=legend_visible,
            orientation="h" if legend_orientation == "h" else "v",
            x=lp["x"], y=lp["y"], xanchor=lp["xanchor"], yanchor=lp["yanchor"],
            font=dict(size=legend_font_size, family=EXPORT_FONT, color="#000000"),
        ),
        font=dict(family=EXPORT_FONT, size=11),
        margin=dict(l=70, r=20, t=60, b=60),
    )

    fig.update_xaxes(showgrid=show_grid, showline=True)
    fig.update_yaxes(showgrid=show_grid, showline=True)

    if x_range and x_range[0] is not None and x_range[1] is not None:
        fig.update_xaxes(range=list(x_range))
    if y_range and y_range[0] is not None and y_range[1] is not None:
        fig.update_yaxes(range=list(y_range))

    return fig


def figure_to_journal_bytes(fig, fmt: str, width_in: float, height_in: float, dpi: int = 300) -> bytes:
    """
    Export a figure at journal dimensions.

    - SVG / PDF: vector, base 72 px/inch (dpi ignored)
    - PNG: raster at width_in*dpi x height_in*dpi pixels (true hi-res)
    """
    if fmt == "png":
        width_px = int(round(width_in * dpi))
        height_px = int(round(height_in * dpi))
        return figure_to_png_bytes(fig, width=width_px, height=height_px, scale=1)
    width_px = int(round(width_in * 72))
    height_px = int(round(height_in * 72))
    if fmt == "svg":
        return figure_to_svg_bytes(fig, width=width_px, height=height_px, scale=1)
    if fmt == "pdf":
        return figure_to_pdf_bytes(fig, width=width_px, height=height_px, scale=1)
    raise ValueError(f"Unsupported format {fmt}")


def _render_gated_export_button(
    key: str,
    label: str,
    generate_fn,
    filename: str,
    mime: str,
    fig_sig: str,
    help_text: Optional[str] = None,
) -> None:
    """
    On-demand, cached export button (self-contained copy of the same pattern
    used in app.py — kept local here since this module is deliberately
    self-contained, per the module docstring).

    Kaleido-based exports cost multiple seconds each. Previously this module
    called figure_to_journal_bytes() unconditionally for all four formats
    (SVG, PDF, PNG@300, PNG@600) on every single Streamlit rerun — i.e. on
    every unrelated widget interaction anywhere in the Publication Studio,
    since expander bodies execute regardless of collapsed/expanded state.
    That was a large chunk of the app feeling slow. This renders a
    "Generate <label>" button that computes the export only when clicked,
    caches the bytes in session_state, and only shows a real download button
    while `fig_sig` (a cheap ~10ms hash of the current figure's content)
    still matches what was current at generation time — so a stale export
    is never silently offered after the user changes a setting.
    """
    import streamlit as st

    bytes_key = f"_pub_export_bytes_{key}"
    sig_key = f"_pub_export_sig_{key}"
    err_key = f"_pub_export_err_{key}"

    if st.button(f"⚙️ Generate {label}", key=f"pub_gen_{key}", help=help_text):
        try:
            st.session_state[bytes_key] = generate_fn()
            st.session_state[sig_key] = fig_sig
            st.session_state.pop(err_key, None)
        except Exception as e:
            st.session_state[err_key] = str(e)
            st.session_state.pop(bytes_key, None)

    cached_bytes = st.session_state.get(bytes_key)
    cached_sig = st.session_state.get(sig_key)
    if cached_bytes is not None and cached_sig == fig_sig:
        st.download_button(label, cached_bytes, filename, mime, key=f"pub_dl_{key}")
    elif err_key in st.session_state:
        st.caption(f"{label}: {st.session_state[err_key][:200]}")
    elif cached_bytes is not None:
        st.caption("Settings changed — click Generate again for an up-to-date file.")
    else:
        st.caption(f"Click 'Generate {label}' to prepare this file.")


def render_publication_studio(
    processed_dict: Dict[str, pd.DataFrame],
    theme_mode: str = "light",
) -> None:
    """
    Render the Publication Figure Studio inside a Streamlit expander.

    Reads all controls, builds the figure live, shows a preview with the
    Plotly toolbar, and offers SVG/PDF/PNG(300/600dpi) downloads.
    """
    import streamlit as st

    if not processed_dict:
        st.warning("Load at least one pattern to use the Publication Studio.")
        return

    filenames = sorted(processed_dict.keys())
    default_labels = {f: f for f in filenames}

    st.markdown("Customize a **journal-ready figure** independent of the quick-compare chart. Names below are *display labels only* — files are untouched.")

    # --- Display labels (editable per file) ---
    labels_df = pd.DataFrame(
        [{"Filename": f, "Legend label": default_labels[f]} for f in filenames]
    )
    edited = st.data_editor(
        labels_df,
        key="pub_labels",
        hide_index=True,
        width="stretch",
        num_rows="fixed",
    )
    display_names = {row["Filename"]: str(row["Legend label"]) or row["Filename"] for _, row in edited.iterrows()}

    c1, c2 = st.columns(2)
    with c1:
        title = st.text_input("Figure title", value="", key="pub_title", help="Leave empty for no title (journals often prefer none).")
        x_label = st.text_input("X-axis label", value="2\u03b8 (\u00b0)", key="pub_xlabel")
        y_label = st.text_input("Y-axis label", value="Intensity (a.u.)", key="pub_ylabel")
    with c2:
        size_preset = st.selectbox("Figure size (publication)", options=list(SIZE_PRESETS.keys()), key="pub_size")
        dpi_choice = st.selectbox("PNG resolution (DPI)", options=[300, 600], index=0, key="pub_dpi",
                                  help="Applies to PNG raster only. SVG/PDF are vector at any size.")
        palette_name = st.selectbox("Color palette", options=["Clean", "Clean Soft", "CB Safe (Okabe-Ito)", "Plotly", "Set2", "Pastel", "Vivid"], key="pub_palette",
                                    help="CB Safe is colorblind-safe Okabe-Ito.")

    c3, c4 = st.columns(2)
    with c3:
        legend_visible = st.checkbox("Show legend", value=True, key="pub_legend_show")
        legend_position = st.selectbox("Legend position", options=list(LEGEND_POSITIONS.keys()), key="pub_legend_pos") if legend_visible else "Inside top-right"
        legend_orientation = st.radio("Legend orientation", ["v", "h"], index=0, horizontal=True, key="pub_legend_orient") if legend_visible else "v"
    with c4:
        legend_font_size = st.slider("Legend font size", 7, 16, 10, key="pub_legend_font")
        tick_font_size = st.slider("Tick font size", 8, 18, 11, key="pub_tick_font")
        label_font_size = st.slider("Axis label font size", 9, 20, 13, key="pub_label_font")
        title_font_size = st.slider("Title font size", 10, 24, 16, key="pub_title_font")

    c5, c6 = st.columns(2)
    with c5:
        line_width = st.slider("Line width (px)", 0.5, 4.0, 1.5, step=0.25, key="pub_lw")
        show_grid = st.checkbox("Show gridlines", value=True, key="pub_grid")
    with c6:
        global_min = float(min((df["twotheta"].min() for df in processed_dict.values() if not df.empty), default=0))
        global_max = float(max((df["twotheta"].max() for df in processed_dict.values() if not df.empty), default=80))
        x_range_opt = st.selectbox("X range", ["Full data", "Custom"], key="pub_xrange_mode")
        x_range = None
        if x_range_opt == "Custom":
            lo, hi = st.slider("2\u03b8 range", float(global_min), float(global_max), (float(global_min), float(global_max)), step=0.5, key="pub_xrange")
            x_range = (lo, hi)
        y_range_opt = st.selectbox("Y range", ["Auto", "Custom"], key="pub_yrange_mode")
        y_range = None
        if y_range_opt == "Custom":
            lo, hi = st.slider("Y range", 0.0, 5.0, (0.0, 1.5), step=0.1, key="pub_yrange")
            y_range = (lo, hi)

    # Optional per-file color overrides
    with st.expander("Per-file colors (optional)", expanded=False):
        st.caption("Leave blank to use the palette. Enter a hex color like #2563EB.")
        color_rows = []
        for f in filenames:
            color_rows.append({"Filename": f, "Color (hex, optional)": ""})
        color_df = pd.DataFrame(color_rows)
        color_edited = st.data_editor(color_df, key="pub_colors", hide_index=True, width="stretch", num_rows="fixed")
        color_overrides = {}
        for _, row in color_edited.iterrows():
            val = str(row["Color (hex, optional)"] or "").strip()
            if val.startswith("#") and len(val) == 7:
                color_overrides[str(row["Filename"])] = val

    # Build live
    fig = create_publication_figure(
        data_dict=processed_dict,
        display_names=display_names,
        palette_name=palette_name,
        title=title,
        x_label=x_label,
        y_label=y_label,
        legend_visible=legend_visible,
        legend_position=legend_position,
        legend_orientation=legend_orientation,
        legend_font_size=legend_font_size,
        tick_font_size=tick_font_size,
        label_font_size=label_font_size,
        title_font_size=title_font_size,
        line_width=line_width,
        color_overrides=color_overrides,
        show_grid=show_grid,
        x_range=x_range,
        y_range=y_range,
        mode=theme_mode,
        for_export=True,
    )

    st.plotly_chart(fig, width="stretch", key="pub_chart", theme=None, config={"displayModeBar": True, "displaylogo": False, "scrollZoom": True})

    # --- Export ---
    size = _resolve_size_preset(size_preset)
    width_in, height_in = size["width_in"], size["height_in"]
    dpi = int(dpi_choice)

    st.markdown(f"**Export** — figure will be {width_in:.1f} × {height_in:.1f} in. SVG/PDF are vector; PNG is {width_in*dpi:.0f}×{height_in*dpi:.0f} px.")

    # Cheap (~10ms) signature covering both the figure content and the export
    # dimensions/DPI (which affect the exported bytes but aren't part of the
    # in-memory fig object itself) — used to detect stale cached exports.
    pub_fig_sig = hashlib.md5((fig.to_json() + f"|{width_in}|{height_in}|{dpi}").encode()).hexdigest()

    ec1, ec2, ec3, ec4 = st.columns(4)
    with ec1:
        _render_gated_export_button(
            key="svg", label="SVG (vector)",
            generate_fn=lambda: figure_to_journal_bytes(fig, "svg", width_in, height_in),
            filename="xrd_publication.svg", mime="image/svg+xml", fig_sig=pub_fig_sig,
            help_text="Renders via Kaleido (~3s).",
        )
    with ec2:
        _render_gated_export_button(
            key="pdf", label="PDF (vector)",
            generate_fn=lambda: figure_to_journal_bytes(fig, "pdf", width_in, height_in),
            filename="xrd_publication.pdf", mime="application/pdf", fig_sig=pub_fig_sig,
            help_text="Renders via Kaleido (~4s).",
        )
    with ec3:
        _render_gated_export_button(
            key="png300", label="PNG @ 300 DPI",
            generate_fn=lambda: figure_to_journal_bytes(fig, "png", width_in, height_in, dpi=300),
            filename="xrd_publication_300dpi.png", mime="image/png", fig_sig=pub_fig_sig,
            help_text="Renders via Kaleido.",
        )
    with ec4:
        if dpi == 600:
            _render_gated_export_button(
                key="png600", label="PNG @ 600 DPI",
                generate_fn=lambda: figure_to_journal_bytes(fig, "png", width_in, height_in, dpi=600),
                filename="xrd_publication_600dpi.png", mime="image/png", fig_sig=pub_fig_sig,
                help_text="Renders via Kaleido.",
            )

    st.caption("EPS is not available with Kaleido v1; journals accept SVG/PDF (vector) or 300–600 DPI PNG/TIFF.")
