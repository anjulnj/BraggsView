"""
plotting.py - Advanced Plotly figure building for XRD comparison.

Phase 3 enhanced:
- Consumes processing.two_theta_to_d as single source of truth for d-spacing axis
- Peak annotation overlay (peaks_dict param)
- Reference stick pattern subplot rows
- Vector export helpers (SVG/PDF via kaleido)
- Consistent axis metadata based on normalization method
- Multi-panel (small multiples) layout mode
- Legend truncation for long filenames
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union, Literal
import io
import zipfile
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import d-spacing util from processing as single source of truth (Phase 2 -> Phase 3)
# Use absolute import only to avoid relative import error when run as script (no parent package)
try:
    from processing import two_theta_to_d
except ImportError:
    # Fallback inline implementation if processing not available
    def two_theta_to_d(two_theta_deg, wavelength=1.5406):
        tt = np.asarray(two_theta_deg, dtype=float)
        theta = np.radians(tt / 2.0)
        sin_theta = np.sin(theta)
        with np.errstate(divide='ignore', invalid='ignore'):
            d = wavelength / (2.0 * np.where(sin_theta < 1e-6, np.nan, sin_theta))
        return d

# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

PALETTES: Dict[str, List[str]] = {
    "Plotly": px.colors.qualitative.Plotly,
    "D3": px.colors.qualitative.D3,
    "G10": px.colors.qualitative.G10,
    "T10": px.colors.qualitative.T10,
    "Alphabet": px.colors.qualitative.Alphabet,
    "Dark24": px.colors.qualitative.Dark24,
    "Light24": px.colors.qualitative.Light24,
    "Set1": px.colors.qualitative.Set1,
    "Set2": px.colors.qualitative.Set2,
    "Set3": px.colors.qualitative.Set3,
    "Pastel": px.colors.qualitative.Pastel,
    "Safe": px.colors.qualitative.Safe,
    "Vivid": px.colors.qualitative.Vivid,
    "Bold": px.colors.qualitative.Bold,
    "Viridis": px.colors.sequential.Viridis,
    "Plasma": px.colors.sequential.Plasma,
    "Turbo": px.colors.sequential.Turbo,
    "Cividis": px.colors.sequential.Cividis,
    "Inferno": px.colors.sequential.Inferno,
    "Magma": px.colors.sequential.Magma,
    "Blues": px.colors.sequential.Blues,
    "Reds": px.colors.sequential.Reds,
    "BuPu": px.colors.sequential.BuPu,
    "Emrld": px.colors.sequential.Emrld,
    "Spectral": px.colors.diverging.Spectral,
}

LARGE_DATA_THRESHOLD = 3000

def _is_sequential_palette(name: str) -> bool:
    return name in {"Viridis", "Plasma", "Turbo", "Cividis", "Inferno", "Magma", "Blues", "Reds", "BuPu", "Emrld", "Spectral"}

def get_palette_colors(name: str, n: int, interpolate: bool = True) -> List[str]:
    if n <= 0:
        return []
    if name not in PALETTES:
        name = "Plotly"
    base = PALETTES[name]
    if not base:
        base = PALETTES["Plotly"]
    if _is_sequential_palette(name) and interpolate and n > 1:
        try:
            from plotly.colors import sample_colorscale
            scale = [[i / (len(base) - 1), c] for i, c in enumerate(base)] if len(base) > 1 else [[0, base[0]], [1, base[0]]]
            sampled = sample_colorscale(scale, [i / (n - 1) if n > 1 else 0.5 for i in range(n)])
            return sampled
        except Exception:
            return [base[i % len(base)] for i in range(n)]
    else:
        if n <= len(base):
            return base[:n]
        return [base[i % len(base)] for i in range(n)]

def estimate_max_intensity(data_dict: Dict[str, pd.DataFrame]) -> float:
    max_int = 0.0
    for df in data_dict.values():
        if df.empty:
            continue
        try:
            m = float(np.nanmax(df["intensity"].values))
            if np.isfinite(m) and m > max_int:
                max_int = m
        except Exception:
            continue
    return max_int if max_int > 0 else 1.0

def _should_use_gl(n_points: int) -> bool:
    return n_points > LARGE_DATA_THRESHOLD

def _truncate_filename(name: str, max_len: int = 50) -> str:
    """Truncate long filenames for legend sanity, preserving extension."""
    if len(name) <= max_len:
        return name
    # Keep start and end with ellipsis
    # Try to preserve extension
    if "." in name:
        stem, ext = name.rsplit(".", 1)
        ext = "." + ext
        # Reserve len for ext + 3 dots
        keep = max_len - len(ext) - 3
        if keep <= 10:
            return name[:max_len-3] + "..."
        return stem[:keep] + "..." + ext
    else:
        return name[:max_len-3] + "..."

def _get_y_label(normalization_method: str = "none", intensity_label_override: Optional[str] = None) -> str:
    """Consistent axis metadata: normalized vs counts based on norm method."""
    if intensity_label_override is not None:
        return intensity_label_override
    norm = (normalization_method or "none").lower()
    if norm in ("none", "", "raw", "off", "no"):
        return "Intensity (counts)"
    elif "zscore" in norm or "standard" in norm:
        return "Intensity (z-score, a.u.)"
    else:
        return "Intensity (normalized, a.u.)"

# ---------------------------------------------------------------------------
# Vector export helpers (publishable)
# ---------------------------------------------------------------------------

def figure_to_svg_bytes(fig: go.Figure, width: int = 1000, height: int = 600, scale: int = 2) -> bytes:
    """
    Export figure to SVG bytes via kaleido (vector). Journals require vector.

    Args:
        fig: Plotly figure
        width, height: pixels
        scale: scale factor

    Returns:
        SVG bytes
    """
    try:
        # kaleido is already dependency for image export
        img_bytes = fig.to_image(format="svg", width=width, height=height, scale=scale)
        return img_bytes
    except Exception as e:
        # Fallback: try with engine param
        try:
            img_bytes = fig.to_image(format="svg", width=width, height=height, scale=scale, engine="kaleido")
            return img_bytes
        except Exception as e2:
            raise RuntimeError(f"SVG export failed (kaleido may be missing or misconfigured): {e} / {e2}")

def figure_to_pdf_bytes(fig: go.Figure, width: int = 1000, height: int = 600, scale: int = 2) -> bytes:
    """
    Export figure to PDF bytes via kaleido.
    """
    try:
        img_bytes = fig.to_image(format="pdf", width=width, height=height, scale=scale)
        return img_bytes
    except Exception as e:
        try:
            img_bytes = fig.to_image(format="pdf", width=width, height=height, scale=scale, engine="kaleido")
            return img_bytes
        except Exception as e2:
            raise RuntimeError(f"PDF export failed: {e} / {e2}")

def figure_to_png_bytes(fig: go.Figure, width: int = 1000, height: int = 600, scale: int = 3) -> bytes:
    """Export to high-res PNG bytes."""
    try:
        return fig.to_image(format="png", width=width, height=height, scale=scale)
    except Exception as e:
        raise RuntimeError(f"PNG export failed: {e}")

# ---------------------------------------------------------------------------
# Main figure creation - enhanced with peaks and reference
# ---------------------------------------------------------------------------

def create_xrd_figure(
    data_dict: Dict[str, pd.DataFrame],
    raw_data_dict: Optional[Dict[str, pd.DataFrame]] = None,
    palette_name: str = "Plotly",
    stack_mode: bool = False,
    offset_spacing: float = 0.0,
    log_y: bool = False,
    show_baseline: bool = False,
    baselines: Optional[Dict[str, np.ndarray]] = None,
    twotheta_label: str = "2θ (°)",
    intensity_label: Optional[str] = None,
    title: str = "XRD Pattern Comparison",
    template: str = "plotly_white",
    show_error_bands: bool = False,
    show_offset_labels: bool = True,
    wavelength: Optional[float] = None,
    d_spacing_label: str = "d (Å)",
    # Phase 3 new optional params (backward compatible)
    normalization_method: str = "none",
    peaks_dict: Optional[Dict[str, pd.DataFrame]] = None,
    reference_patterns: Optional[Dict[str, pd.DataFrame]] = None,
    layout_mode: str = "overlay",  # overlay, stack (via stack_mode bool compat), grid
    max_legend_length: int = 50,
    show_peak_labels: bool = True,
    peak_label_field: str = "twotheta",  # twotheta, d_spacing, hkl, intensity
) -> go.Figure:
    """
    Build advanced interactive Plotly figure for multiple XRD patterns.

    Phase 3 additions:
    - Uses processing.two_theta_to_d as single source of truth for secondary axis
    - Peak annotation overlay via peaks_dict
    - Reference stick patterns subplot rows via reference_patterns
    - Consistent y label based on normalization_method
    - Legend truncation via max_legend_length
    - layout_mode grid delegates to create_multi_panel_figure

    Backward compatibility: all new params have safe defaults, old calls unchanged.
    """
    # Handle layout_mode grid -> delegate
    if layout_mode == "grid":
        return create_multi_panel_figure(
            data_dict=data_dict,
            palette_name=palette_name,
            title=title,
            template=template,
            wavelength=wavelength,
            normalization_method=normalization_method,
            peaks_dict=peaks_dict,
            reference_patterns=reference_patterns,
            max_legend_length=max_legend_length,
        )

    # Resolve y label consistently
    y_label_resolved = _get_y_label(normalization_method, intensity_label)

    # If reference patterns provided, create subplot with 2 rows
    has_reference = reference_patterns is not None and len(reference_patterns) > 0

    if has_reference:
        # 2 rows: main (80%) and reference (20%)
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.82, 0.18],
            vertical_spacing=0.04,
            subplot_titles=(title, "Reference patterns (stick)"),
        )
    else:
        fig = go.Figure()

    if not data_dict:
        fig.update_layout(
            title="No data loaded",
            xaxis_title=twotheta_label,
            yaxis_title=y_label_resolved,
            template=template,
        )
        return fig

    filenames = sorted(data_dict.keys())
    n_files = len(filenames)
    colors = get_palette_colors(palette_name, n_files, interpolate=True)

    max_int = estimate_max_intensity(data_dict)
    if stack_mode and (offset_spacing == 0 or not np.isfinite(offset_spacing)):
        offset_spacing = max_int * 0.5

    all_x_min = min((float(df["twotheta"].min()) for df in data_dict.values() if not df.empty), default=0)
    all_x_max = max((float(df["twotheta"].max()) for df in data_dict.values() if not df.empty), default=80)

    annotations: List[Dict] = []

    for idx, fname in enumerate(filenames):
        df = data_dict[fname]
        if df.empty:
            continue
        color = colors[idx % len(colors)]
        x = df["twotheta"].values.astype(float)
        y = df["intensity"].values.astype(float)
        has_error = "error" in df.columns and show_error_bands
        y_plot = y + idx * offset_spacing if stack_mode else y

        use_gl = _should_use_gl(len(x))
        ScatterClass = go.Scattergl if use_gl else go.Scatter

        # Legend truncation
        display_name = _truncate_filename(fname, max_len=max_legend_length)

        # customdata for hover
        customdata = None
        if raw_data_dict and fname in raw_data_dict:
            raw_df = raw_data_dict[fname]
            if len(raw_df) == len(df):
                raw_intensity = raw_df["intensity"].values
                if has_error:
                    err = df["error"].values if "error" in df.columns else np.full_like(y, np.nan)
                    customdata = np.stack([err, raw_intensity], axis=-1)
                else:
                    customdata = raw_intensity
            else:
                try:
                    from scipy.interpolate import interp1d
                    f = interp1d(raw_df["twotheta"].values, raw_df["intensity"].values, kind="linear", bounds_error=False, fill_value=np.nan)
                    raw_interp = f(x)
                    if has_error:
                        err = df["error"].values if "error" in df.columns else np.full_like(y, np.nan)
                        customdata = np.stack([err, raw_interp], axis=-1)
                    else:
                        customdata = raw_interp
                except Exception:
                    pass

        hovertemplate = f"<b>{fname}</b><br>2θ: %{{x:.4f}}°<br>Intensity: %{{y:.2f}}"
        if has_error and not stack_mode and customdata is not None and hasattr(customdata, 'ndim') and customdata.ndim == 2:
            hovertemplate += "<br>±%{customdata[0]:.1f}"
        if raw_data_dict and fname in raw_data_dict and customdata is not None:
            if has_error and hasattr(customdata, 'ndim') and customdata.ndim == 2:
                hovertemplate += "<br>Raw: %{customdata[1]:.1f}"
            else:
                hovertemplate += "<br>Raw: %{customdata:.1f}"
        hovertemplate += "<extra></extra>"

        trace_args = dict(
            x=x,
            y=y_plot,
            mode="lines",
            name=display_name,
            legendgroup=fname,
            line=dict(color=color, width=1.6),
            hovertemplate=hovertemplate,
        )
        if customdata is not None:
            trace_args["customdata"] = customdata

        if has_reference:
            fig.add_trace(ScatterClass(**trace_args), row=1, col=1)
        else:
            fig.add_trace(ScatterClass(**trace_args))

        # Error bands
        if has_error and not stack_mode:
            try:
                err = df["error"].values.astype(float)
                y_upper = y_plot + err
                y_lower = y_plot - err
                upper = go.Scatter(x=x, y=y_upper, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
                lower = go.Scatter(x=x, y=y_lower, mode="lines", line=dict(width=0), fill="tonexty",
                                   fillcolor=color.replace("rgb", "rgba").replace(")", ",0.15)") if "rgb" in color else "rgba(100,100,100,0.15)",
                                   showlegend=False, hoverinfo="skip")
                if has_reference:
                    fig.add_trace(upper, row=1, col=1)
                    fig.add_trace(lower, row=1, col=1)
                else:
                    fig.add_trace(upper)
                    fig.add_trace(lower)
            except Exception:
                pass

        # Baseline
        if show_baseline and baselines and fname in baselines:
            baseline = baselines[fname]
            if baseline is not None and len(baseline) == len(df):
                y_base = baseline + (idx * offset_spacing if stack_mode else 0)
                base_trace = go.Scatter(x=x, y=y_base, mode="lines", name=f"{display_name} baseline",
                                        line=dict(color=color, width=1.2, dash="dashdot"), opacity=0.6, showlegend=False, hoverinfo="skip")
                if has_reference:
                    fig.add_trace(base_trace, row=1, col=1)
                else:
                    fig.add_trace(base_trace)

        # Peak annotation overlay - Phase 3 new
        if peaks_dict and fname in peaks_dict:
            peaks_df = peaks_dict[fname]
            if not peaks_df.empty and "twotheta" in peaks_df.columns:
                # Need to map peak positions to y_plot at those x
                # Interpolate y_plot at peak positions for marker height
                try:
                    # peaks_df may have intensity column from peak detection, but use interpolated y_plot for stacking consistency
                    peak_x = peaks_df["twotheta"].values.astype(float)
                    # Interpolate y_plot at peak_x
                    # Use np.interp (x must be sorted)
                    peak_y_interp = np.interp(peak_x, x, y_plot, left=np.nan, right=np.nan)
                    # Marker symbol
                    marker_trace = go.Scatter(
                        x=peak_x,
                        y=peak_y_interp,
                        mode="markers" + ("+text" if show_peak_labels else ""),
                        name=f"{display_name} peaks",
                        legendgroup=fname,
                        showlegend=False,
                        marker=dict(color=color, size=8, symbol="line-ns-open", line=dict(width=1.5)),
                        text=[f"{row[peak_label_field]:.2f}°" if peak_label_field in peaks_df.columns and np.isfinite(row.get(peak_label_field, np.nan)) else f"{row['twotheta']:.2f}°" for _, row in peaks_df.iterrows()] if show_peak_labels else None,
                        textposition="top center",
                        textfont=dict(size=9, color=color),
                        hovertemplate=f"<b>{fname} peak</b><br>2θ: %{{x:.3f}}°<br>Intensity: %{{y:.1f}}<br>FWHM: %{{customdata[0]:.3f}}°<br>d: %{{customdata[1]:.3f}} Å<extra></extra>",
                        customdata=np.stack([
                            peaks_df["fwhm"].values if "fwhm" in peaks_df.columns else np.full(len(peaks_df), np.nan),
                            peaks_df["d_spacing"].values if "d_spacing" in peaks_df.columns else np.full(len(peaks_df), np.nan),
                        ], axis=-1) if "fwhm" in peaks_df.columns else None,
                    )
                    if has_reference:
                        fig.add_trace(marker_trace, row=1, col=1)
                    else:
                        fig.add_trace(marker_trace)
                except Exception:
                    pass

        # Offset labels
        if stack_mode and show_offset_labels:
            x_label = all_x_max + (all_x_max - all_x_min) * 0.01
            y_label = float(np.nanmedian(y_plot)) if len(y_plot) > 0 else idx * offset_spacing
            try:
                mask_right = x >= (all_x_max - (all_x_max - all_x_min) * 0.2)
                if np.any(mask_right):
                    y_label = float(np.nanmean(y_plot[mask_right]))
            except Exception:
                pass
            annotations.append(dict(
                x=x_label, y=y_label, xref="x", yref="y",
                text=f"<b>{display_name}</b>", showarrow=False,
                font=dict(color=color, size=11), xanchor="left", yanchor="middle",
                bgcolor="rgba(255,255,255,0.6)", bordercolor=color, borderwidth=1, borderpad=2,
            ))

    # Reference stick patterns - bottom subplot
    if has_reference:
        ref_colors = get_palette_colors(palette_name, len(reference_patterns), interpolate=True)
        for r_idx, (ref_name, ref_df) in enumerate(reference_patterns.items()):
            if ref_df.empty:
                continue
            ref_color = ref_colors[r_idx % len(ref_colors)]
            ref_x = ref_df["twotheta"].values.astype(float)
            # Use intensity if present, else uniform tick height
            if "intensity" in ref_df.columns:
                ref_y = ref_df["intensity"].values.astype(float)
                # Normalize to 0-1 for tick display
                max_r = np.nanmax(ref_y) if len(ref_y) > 0 else 1.0
                if max_r > 0:
                    ref_y = ref_y / max_r
            else:
                ref_y = np.ones_like(ref_x) * 0.5

            # Stick: vertical lines from 0 to ref_y - we achieve with markers symbol line-ns-open
            # Also add text for hkl if present
            has_hkl = "hkl" in ref_df.columns
            text_labels = ref_df["hkl"].astype(str).values if has_hkl else None

            stick_trace = go.Scatter(
                x=ref_x,
                y=ref_y,
                mode="markers" + ("+text" if has_hkl else ""),
                name=f"Ref: {ref_name}",
                marker=dict(symbol="line-ns-open", size=14, color=ref_color, line=dict(width=2, color=ref_color)),
                text=text_labels,
                textposition="top center",
                textfont=dict(size=8, color=ref_color),
                hovertemplate=f"<b>Ref {ref_name}</b><br>2θ: %{{x:.3f}}°<br>Int: %{{y:.2f}}" + (f"<br>HKL: %{{text}}" if has_hkl else "") + "<extra></extra>",
            )
            fig.add_trace(stick_trace, row=2, col=1)

        # Update yaxis for reference row - small range, no grid
        fig.update_yaxes(title_text="Ref Int", row=2, col=1, showgrid=False, range=[0, 1.2])

    # Layout
    if has_reference:
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=18)),
            template=template,
            hovermode="closest",
            legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02,
                        bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0.15)", borderwidth=1, font=dict(size=11)),
            margin=dict(l=70, r=200, t=80, b=60),
            dragmode="zoom",
            annotations=annotations,
            font=dict(family="Inter, Arial, sans-serif"),
        )
        fig.update_xaxes(title_text=twotheta_label, row=2, col=1)
        fig.update_yaxes(title_text=y_label_resolved + (" (stacked)" if stack_mode else ""), row=1, col=1)
    else:
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=18)),
            xaxis_title=twotheta_label,
            yaxis_title=y_label_resolved + (" (stacked + offset)" if stack_mode else ""),
            template=template,
            hovermode="closest",
            legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02,
                        bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0.15)", borderwidth=1, font=dict(size=11)),
            margin=dict(l=70, r=260 if (stack_mode and show_offset_labels) else 180, t=70, b=65),
            dragmode="zoom",
            annotations=annotations,
            font=dict(family="Inter, Arial, sans-serif"),
        )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(180,180,180,0.25)", showline=True, linewidth=1.2, linecolor="black", mirror=True, ticks="outside", ticklen=5, title_font=dict(size=14), tickfont=dict(size=12))
    y_type = "log" if log_y else "linear"
    fig.update_yaxes(type=y_type, showgrid=True, gridwidth=1, gridcolor="rgba(180,180,180,0.25)", showline=True, linewidth=1.2, linecolor="black", mirror=True, ticks="outside", ticklen=5, title_font=dict(size=14), tickfont=dict(size=12))

    # Secondary d-spacing axis using processing.two_theta_to_d (single source of truth)
    # Only for single-panel (no reference) to avoid colliding with subplot xaxis2 (top row or top-right panel)
    if wavelength is not None and wavelength > 0 and not has_reference:
        try:
            tick_x = np.linspace(all_x_min, all_x_max, 8)
            tick_d = two_theta_to_d(tick_x, wavelength=wavelength)
            # Filter NaN
            valid = np.isfinite(tick_d)
            tick_x_valid = tick_x[valid]
            tick_d_valid = tick_d[valid]
            tick_text = [f"{d:.2f}" for d in tick_d_valid]

            fig.update_layout(
                xaxis2=dict(
                    title=dict(text=d_spacing_label, font=dict(size=13)),
                    overlaying="x",
                    side="top",
                    tickmode="array",
                    tickvals=tick_x_valid.tolist(),
                    ticktext=tick_text,
                    showgrid=False,
                    showline=True,
                    linewidth=1,
                    linecolor="black",
                )
            )
        except Exception:
            pass

    return fig


def create_multi_panel_figure(
    data_dict: Dict[str, pd.DataFrame],
    palette_name: str = "Plotly",
    title: str = "XRD Multi-Panel Comparison",
    template: str = "plotly_white",
    wavelength: Optional[float] = None,
    normalization_method: str = "none",
    peaks_dict: Optional[Dict[str, pd.DataFrame]] = None,
    reference_patterns: Optional[Dict[str, pd.DataFrame]] = None,
    max_legend_length: int = 40,
    ncols: int = 2,
) -> go.Figure:
    """
    Multi-panel (small multiples) comparison: each pattern in its own subplot.
    Fixes previous bug where right side / last panel didn't show:
    - shared_xaxes=False so all x-axes are independent and visible
    - subplot_titles padded to full grid size (nrows*ncols)
    - empty slots in last row hidden (xaxis/yaxis visible=False)
    - x title set on every bottom-row column, not just col=1
    """
    if not data_dict:
        fig = go.Figure()
        fig.update_layout(title="No data")
        return fig

    filenames = sorted(data_dict.keys())
    n = len(filenames)
    nrows = math.ceil(n / ncols)
    total_slots = nrows * ncols

    # Pad titles to full grid size - otherwise Plotly misaligns last row
    truncated_names = [_truncate_filename(fn, max_len=max_legend_length) for fn in filenames]
    subplot_titles = truncated_names + [""] * (total_slots - len(truncated_names))

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=subplot_titles,
        shared_xaxes=False,  # FIX: was True, hid right side tick labels and caused confusion
        shared_yaxes=False,
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    colors = get_palette_colors(palette_name, n, interpolate=True)
    y_label = _get_y_label(normalization_method)

    for idx, fname in enumerate(filenames):
        df = data_dict[fname]
        if df.empty:
            continue
        row = idx // ncols + 1
        col = idx % ncols + 1
        color = colors[idx % len(colors)]
        x = df["twotheta"].values
        y = df["intensity"].values

        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=_truncate_filename(fname, max_legend_length),
                                 line=dict(color=color, width=1.4), showlegend=False,
                                 hovertemplate=f"<b>{fname}</b><br>2θ: %{{x:.3f}}°<br>Int: %{{y:.2f}}<extra></extra>"),
                      row=row, col=col)

        # Peaks overlay
        if peaks_dict and fname in peaks_dict:
            pdf = peaks_dict[fname]
            if not pdf.empty and "twotheta" in pdf.columns:
                px_vals = pdf["twotheta"].values
                py = np.interp(px_vals, x, y, left=np.nan, right=np.nan)
                fig.add_trace(go.Scatter(x=px_vals, y=py, mode="markers",
                                         marker=dict(color=color, size=7, symbol="x"),
                                         showlegend=False, hoverinfo="skip"),
                              row=row, col=col)

        # Y label on first column only to avoid clutter, but show on all if ncols==1
        if col == 1 or ncols == 1:
            fig.update_yaxes(title_text=y_label, row=row, col=col, showgrid=True, gridcolor="rgba(180,180,180,0.2)")
        else:
            fig.update_yaxes(showgrid=True, gridcolor="rgba(180,180,180,0.2)", row=row, col=col)

        # X label will be set for all bottom row columns below

    # Hide unused empty subplots in last row
    for empty_idx in range(n, total_slots):
        r = empty_idx // ncols + 1
        c = empty_idx % ncols + 1
        # Make axes invisible for empty slots
        fig.update_xaxes(visible=False, showticklabels=False, row=r, col=c)
        fig.update_yaxes(visible=False, showticklabels=False, row=r, col=c)

    # Set X titles for all columns in bottom row (fixes right side missing title confusion)
    for c in range(1, ncols + 1):
        # Only if that bottom slot actually has data (or is within n)
        # For grid with incomplete last row, only set title if slot exists
        slot_idx = (nrows - 1) * ncols + (c - 1)
        if slot_idx < n:
            fig.update_xaxes(title_text="2θ (°)", row=nrows, col=c, showticklabels=True)

    # Also ensure all x axes show tick labels (since shared_xaxes=False)
    fig.update_xaxes(showticklabels=True, ticks="outside")

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        template=template,
        height=320 * nrows,
        showlegend=False,
        margin=dict(l=70, r=40, t=80, b=70),
    )

    # NOTE: We intentionally skip secondary d-spacing top axis for multi-panel grid
    # because Plotly's subplot xaxis2 is already used for row1 col2. Adding another
    # xaxis2 with overlaying="x" overwrites the top-right panel's x-axis, making it
    # disappear (bug reported in your screenshot where synthetic_cubic.xy title shows
    # but no plot). If d-spacing needed in grid mode, it should be added per-subplot
    # as annotations, not as global xaxis2.

    return fig


def create_difference_figure(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    name_a: str = "Pattern A",
    name_b: str = "Pattern B",
    step: float = 0.02,
    palette_name: str = "Plotly",
    title: str = "Difference Plot",
) -> go.Figure:
    try:
        from processing import interpolate_to_common_grid
    except ImportError:
        import processing as _proc
        interpolate_to_common_grid = _proc.interpolate_to_common_grid

    common_x, interp = interpolate_to_common_grid({name_a: df_a, name_b: df_b}, step=step)
    fig = go.Figure()
    if common_x.size == 0:
        fig.update_layout(title="No data for difference")
        return fig
    colors = get_palette_colors(palette_name, 3)
    for idx, (name, y) in enumerate(interp.items()):
        fig.add_trace(go.Scatter(x=common_x, y=y, mode="lines", name=name, line=dict(color=colors[idx], width=1.5)))
    if name_a in interp and name_b in interp:
        diff = interp[name_a] - interp[name_b]
        fig.add_trace(go.Scatter(x=common_x, y=diff, mode="lines", name=f"Difference ({name_a} - {name_b})", line=dict(color=colors[2], width=1.3, dash="dash")))
    fig.update_layout(title=title, xaxis_title="2θ (°)", yaxis_title="Intensity", template="plotly_white", hovermode="x unified")
    return fig


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def create_stats_table(data_dict: Dict[str, pd.DataFrame], include_noise: bool = True) -> pd.DataFrame:
    rows = []
    for fname, df in data_dict.items():
        if df.empty:
            rows.append({"Filename": fname, "N points": 0})
            continue
        try:
            max_idx = int(df["intensity"].idxmax())
            mean_v = float(df["intensity"].mean())
            std_v = float(df["intensity"].std())
            max_v = float(df["intensity"].max())
            try:
                from processing import estimate_noise_std
                noise = estimate_noise_std(df["intensity"].values)
            except Exception:
                noise = np.nan
            snr = max_v / noise if noise and noise > 0 else np.nan
            rows.append({
                "Filename": _truncate_filename(fname, max_len=60),
                "N points": len(df),
                "2θ min": f"{float(df['twotheta'].min()):.3f}",
                "2θ max": f"{float(df['twotheta'].max()):.3f}",
                "2θ range": f"{float(df['twotheta'].max() - df['twotheta'].min()):.3f}",
                "2θ at max": f"{float(df.loc[max_idx, 'twotheta']):.3f}",
                "Intensity max": f"{max_v:.2f}",
                "Intensity min": f"{float(df['intensity'].min()):.2f}",
                "Mean": f"{mean_v:.2f}",
                "Std": f"{std_v:.2f}",
                "Noise (MAD)": f"{noise:.2f}" if np.isfinite(noise) else "—",
                "SNR (est)": f"{snr:.1f}" if np.isfinite(snr) else "—",
            })
        except Exception as e:
            rows.append({"Filename": fname, "N points": len(df), "Error": str(e)[:80]})
    df_stats = pd.DataFrame(rows)
    desired_order = ["Filename", "N points", "2θ min", "2θ max", "2θ range", "2θ at max", "Intensity max", "Intensity min", "Mean", "Std", "Noise (MAD)", "SNR (est)"]
    cols = [c for c in desired_order if c in df_stats.columns] + [c for c in df_stats.columns if c not in desired_order]
    return df_stats[cols]


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_processed_csv_zip(data_dict: Dict[str, pd.DataFrame], include_metadata: bool = True) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, df in data_dict.items():
            if df.empty:
                continue
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            base = fname.rsplit(".", 1)[0] if "." in fname else fname
            out_name = f"{base}_processed.csv"
            zf.writestr(out_name, csv_buffer.getvalue())
        if include_metadata:
            meta = io.StringIO()
            meta.write("XRD Processed Data Export\n")
            meta.write(f"Files: {len(data_dict)}\n")
            for fname, df in data_dict.items():
                meta.write(f"- {fname}: {len(df)} points, 2θ {float(df['twotheta'].min()) if not df.empty else 'N/A'} - {float(df['twotheta'].max()) if not df.empty else 'N/A'}\n")
            zf.writestr("METADATA.txt", meta.getvalue())
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def export_combined_csv(data_dict: Dict[str, pd.DataFrame]) -> str:
    frames: List[pd.DataFrame] = []
    for fname, df in data_dict.items():
        if df.empty:
            continue
        tmp = df.copy()
        tmp["filename"] = fname
        frames.append(tmp)
    if not frames:
        return ""
    combined = pd.concat(frames, ignore_index=True)
    cols = ["filename", "twotheta", "intensity"]
    if "error" in combined.columns:
        cols.append("error")
    existing_cols = [c for c in cols if c in combined.columns]
    extra = [c for c in combined.columns if c not in existing_cols]
    combined = combined[existing_cols + extra]
    return combined.to_csv(index=False)

def export_common_grid_csv(data_dict: Dict[str, pd.DataFrame], step: Optional[float] = None, method: str = "linear") -> str:
    try:
        from processing import interpolate_to_common_grid
    except ImportError:
        import processing
        interpolate_to_common_grid = processing.interpolate_to_common_grid
    common_x, interp = interpolate_to_common_grid(data_dict, step=step, method=method)
    if common_x.size == 0:
        return ""
    out_df = pd.DataFrame({"twotheta": common_x})
    for name, y in interp.items():
        col = name.rsplit(".", 1)[0] if "." in name else name
        out_df[col] = y
    return out_df.to_csv(index=False)
