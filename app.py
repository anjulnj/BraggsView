"""
app.py - XRD Interactive Comparison Tool - Phase 5 Publication-Ready


Implements full backlog:
- Robustness: persistent issues panel, zero-width guard, wavelength validation
- State management: session_state for all controls, reset button, theme toggle
- Structure: sidebar sections extracted to functions, @st.cache_data parsing wrapper
- Publication UX: Methods paragraph, session save/load JSON, caption generator, peak table CSV, vector SVG/PDF export
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# --- Theme injection early ---
try:
    from theme import (
        inject_clean_theme,
        apply_clean_plotly_theme,
        get_theme_choice,
        set_theme_choice,
        CLEAN_PALETTE,
        CLEAN_PALETTE_CB_SAFE,
        CLEAN_PALETTE_SOFT,
        audit_theme_contrast,
        LIGHT,
        get_clean_plotly_template,
    )
    HAS_CLEAN_THEME = True
except Exception:
    HAS_CLEAN_THEME = False
    def inject_clean_theme(mode: str = "light") -> None:
        return None

    def apply_clean_plotly_theme(
        fig: Any,
        mode: str = "light",
        palette: str = "clean",
        for_export: bool = False,
        line_width: Optional[float] = None,
    ) -> Any:
        return fig

    def get_theme_choice(default: str = "light") -> str:
        return default

    def set_theme_choice(choice: str) -> None:
        pass
    CLEAN_PALETTE = ["#0F172A", "#2563EB", "#059669"]
    CLEAN_PALETTE_CB_SAFE = ["#000000", "#E69F00", "#56B4E9"]
    CLEAN_PALETTE_SOFT = CLEAN_PALETTE

from xrd_io import (
    parse_xrd_file,
    parse_xrd_file_with_metadata,
    parse_xrd_file_with_validation,
    load_folder,
    XRDParseError,
    SUPPORTED_EXTS,
    validate_xrd_dataframe,
)
from processing import (
    process_pattern,
    two_theta_to_d,
    find_peaks_xrd,
    scherrer_crystallite_size,
    deconvolve_instrumental_fwhm,
    estimate_shift_by_xcorr,
    pattern_similarity,
    processing_summary,
    load_reference_pattern,
)
from plotting import (
    create_xrd_figure,
    create_multi_panel_figure,
    create_difference_figure,
    create_stats_table,
    export_processed_csv_zip,
    export_combined_csv,
    export_common_grid_csv,
    figure_to_svg_bytes,
    figure_to_pdf_bytes,
    figure_to_png_bytes,
    PALETTES,
)

# Publication Figure Studio (lightweight, self-contained module keeps app.py lean)
try:
    from pubfigure import render_publication_studio
    HAS_PUB_STUDIO = True
except Exception:
    HAS_PUB_STUDIO = False
    def render_publication_studio(processed_dict, theme_mode="light"):
        return None

# Inject colorblind-safe palettes into PALETTES for selection (Phase 4 backlog, done via mutation not file edit)
PALETTES["CB Safe (Okabe-Ito)"] = CLEAN_PALETTE_CB_SAFE
PALETTES["Clean"] = CLEAN_PALETTE
PALETTES["Clean Soft"] = CLEAN_PALETTE_SOFT

# Page config
st.set_page_config(
    page_title="XRD Pattern Comparison",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme toggle persistence - must be rendered early so CSS applies
if "theme_choice" not in st.session_state:
    st.session_state.theme_choice = "light"

# Sidebar theme selector (Phase 4 -> Phase 5 wiring)
with st.sidebar:
    st.markdown("### 🎨 Theme")
    theme_opt = st.selectbox(
        "UI Theme",
        ["light", "dark", "print"],
        index=["light", "dark", "print"].index(st.session_state.theme_choice) if st.session_state.theme_choice in ["light", "dark", "print"] else 0,
        key="theme_choice",
        help="Light: high-contrast readable (fixes black unreadable). Dark: readable dark. Print: pure white/black for paper.",
    )
    # Inject after selection
    if HAS_CLEAN_THEME:
        inject_clean_theme(mode=theme_opt)

# ---------------------------------------------------------------------------
# Helpers & cached parsing
# ---------------------------------------------------------------------------

def render_gated_export_button(
    key: str,
    label: str,
    generate_fn,
    filename: str,
    mime: str,
    fig_sig: str,
    help_text: Optional[str] = None,
) -> None:
    """
    Render an on-demand, cached export control instead of eagerly computing it.

    Kaleido-based exports (SVG/PDF/high-DPI PNG) cost multiple seconds each.
    Computing them unconditionally on every Streamlit rerun (i.e. on every
    unrelated slider/checkbox interaction) was previously the single biggest
    cause of the app feeling slow — measured ~3.3s for SVG and ~3.7s for PDF,
    paid on every interaction whether or not anyone ever clicked download.

    This renders a "Generate <label>" button that computes `generate_fn()` only
    when clicked, caching the resulting bytes in session_state under `key`. A
    real st.download_button is then shown using those cached bytes — but only
    if `fig_sig` (a cheap hash of the current figure's content, ~10ms to
    compute) still matches the signature recorded at generation time. If the
    user has changed a setting since the last generate (so the underlying
    figure changed), the cached bytes are treated as stale and the user is
    prompted to regenerate rather than silently being offered an outdated file.

    Args:
        key: stable identifier for this export's session_state slots.
        label: shown in the generate/download button text.
        generate_fn: zero-arg callable returning the export bytes.
        filename: download filename.
        mime: MIME type for the download.
        fig_sig: hash of the current figure's content (see call sites).
        help_text: optional tooltip for the generate button.
    """
    bytes_key = f"_export_bytes_{key}"
    sig_key = f"_export_sig_{key}"
    err_key = f"_export_err_{key}"

    if st.button(f"⚙️ Generate {label}", key=f"gen_{key}", help=help_text):
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
        st.download_button(label, cached_bytes, filename, mime, key=f"dl_{key}")
    elif err_key in st.session_state:
        st.caption(f"{label} failed: {st.session_state[err_key][:200]}")
    elif cached_bytes is not None:
        st.caption(f"Settings changed since last generate — click again for an up-to-date file.")
    else:
        st.caption(f"Click 'Generate {label}' to prepare this file (kept on-demand, not automatic, to keep the app responsive).")


def get_example_dir() -> Path:
    return Path(__file__).parent / "example_data"

def ensure_example_data():
    ex_dir = get_example_dir()
    ex_dir.mkdir(exist_ok=True)
    expected = ["synthetic_cubic.xy", "synthetic_tetragonal.xy", "synthetic_amorphous_mix.xy"]
    missing = [f for f in expected if not (ex_dir / f).exists()]
    if missing:
        try:
            from generate_examples import main as gen_main
            gen_main()
        except Exception as e:
            st.warning(f"Could not auto-generate synthetic examples: {e}")

@st.cache_data(show_spinner=False)
def cached_parse_uploaded_file(file_bytes: bytes, filename: str):
    """Cached wrapper to avoid re-parsing on every widget interaction."""
    from xrd_io import parse_xrd_content
    # Reuse decode logic from xrd_io's _decode fallback
    try:
        content = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            from charset_normalizer import from_bytes
            content = str(from_bytes(file_bytes).best())
        except Exception:
            content = file_bytes.decode("latin-1", errors="ignore")
    df = parse_xrd_content(content, filename=filename)
    return df

@st.cache_data(show_spinner=False)
def cached_parse_with_validation(file_bytes: bytes, filename: str):
    from xrd_io import parse_xrd_content_with_validation
    try:
        content = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        content = file_bytes.decode("latin-1", errors="ignore")
    df, meta, issues = parse_xrd_content_with_validation(content, filename=filename)
    return df, meta, issues

def get_global_twotheta_bounds(data_dict: Dict[str, pd.DataFrame]) -> Tuple[float, float]:
    if not data_dict:
        return (0.0, 80.0)
    mins = []
    maxs = []
    for df in data_dict.values():
        if df.empty:
            continue
        mins.append(float(df["twotheta"].min()))
        maxs.append(float(df["twotheta"].max()))
    if not mins:
        return (0.0, 80.0)
    return (float(min(mins)), float(max(maxs)))

# ---------------------------------------------------------------------------
# Sidebar sections extracted to functions (Phase 5 structure)
# ---------------------------------------------------------------------------

def render_data_input_section() -> Tuple[
    Dict[str, pd.DataFrame],
    List[str],
    Dict[str, List[str]],
    Dict[str, Dict],
    Dict[str, pd.DataFrame],
    List[str],
]:
    """Render data input sidebar section, return raw data, errors, issues per file, metadata per file."""
    st.sidebar.header("📁 Data Input")

    uploaded_files = st.sidebar.file_uploader(
        "Drag & drop XRD files (.xy, .xye, .txt, .csv, .dat, .xyd)",
        type=[ext.strip(".") for ext in SUPPORTED_EXTS],
        accept_multiple_files=True,
        help="Auto delimiter detection, header tolerant. Now with validation and metadata extraction.",
        key="uploader",
    )

    folder_path_input = st.sidebar.text_input(
        "Or load from folder path (server-side)",
        value="",
        placeholder="e.g. ./example_data",
        key="folder_path",
    )

    ex_dir = get_example_dir()
    example_files_available = []
    if ex_dir.exists():
        for ext in SUPPORTED_EXTS:
            example_files_available.extend(ex_dir.glob(f"*{ext}"))
            example_files_available.extend(ex_dir.glob(f"*{ext.upper()}"))
        example_files_available = sorted(set([p.name for p in example_files_available]))

    example_selected = []
    if example_files_available:
        st.sidebar.markdown("**Example / pre-loaded:**")
        default_sel = []
        if not uploaded_files and not st.session_state.get("active_files"):
            for f in example_files_available:
                if "synthetic" in f.lower():
                    default_sel.append(f)
            if not default_sel:
                default_sel = example_files_available[:2]

        example_selected = st.sidebar.multiselect(
            "Load from example_data",
            options=example_files_available,
            default=default_sel,
            key="example_multiselect",
        )
        if st.sidebar.button("Load all from example_data", key="load_all_examples"):
            example_selected = example_files_available
            st.session_state.example_multiselect = example_files_available

    if st.sidebar.button("🔄 Regenerate synthetic", key="regen_synth"):
        try:
            from generate_examples import main as gen_main
            gen_main()
            st.sidebar.success("Regenerated! Reselect above.")
            st.cache_data.clear()
        except Exception as e:
            st.sidebar.error(f"Failed: {e}")

    all_raw_data: Dict[str, pd.DataFrame] = {}
    all_errors: List[str] = []
    all_issues: Dict[str, List[str]] = {}
    all_metadata: Dict[str, Dict] = {}

    # Uploaded with validation
    if uploaded_files:
        for uf in uploaded_files:
            try:
                file_bytes = uf.getvalue()
                # Use cached validation path
                df, meta, issues = cached_parse_with_validation(file_bytes, uf.name)
                # Unique name
                name = uf.name
                base = name
                counter = 1
                while name in all_raw_data:
                    stem = Path(base).stem
                    suffix = Path(base).suffix
                    name = f"{stem}_{counter}{suffix}"
                    counter += 1
                all_raw_data[name] = df
                all_metadata[name] = meta
                if issues:
                    all_issues[name] = issues
            except XRDParseError as e:
                all_errors.append(str(e))
            except Exception as e:
                all_errors.append(f"{uf.name}: {e}")

    # Folder
    if folder_path_input.strip():
        try:
            # Use parallel for large folders if many files
            folder_data = load_folder(folder_path_input.strip(), parallel=True)
            for k, v in folder_data.items():
                name = k
                base = name
                counter = 1
                while name in all_raw_data:
                    stem = Path(base).stem
                    suffix = Path(base).suffix
                    name = f"{stem}_{counter}{suffix}"
                    counter += 1
                all_raw_data[name] = v
            st.sidebar.success(f"Loaded {len(folder_data)} files from folder (parallel).")
        except XRDParseError as e:
            st.sidebar.error(str(e))
        except Exception as e:
            st.sidebar.error(f"Folder load error: {e}")

    # Example selected
    if example_selected:
        for fname in example_selected:
            fpath = ex_dir / fname
            if not fpath.exists():
                continue
            try:
                # Use validation for example too
                raw = fpath.read_bytes()
                df, meta, issues = cached_parse_with_validation(raw, fname)
                name = fpath.name
                base = name
                counter = 1
                while name in all_raw_data:
                    stem = Path(base).stem
                    suffix = Path(base).suffix
                    name = f"{stem}_{counter}{suffix}"
                    counter += 1
                all_raw_data[name] = df
                all_metadata[name] = meta
                if issues:
                    all_issues[name] = issues
            except Exception as e:
                all_errors.append(f"{fname}: {e}")

    # Reference patterns parsing (separate dict)
    reference_patterns: Dict[str, pd.DataFrame] = {}

    st.sidebar.markdown("**Reference pattern(s)** — optional, for phase ID")
    ref_uploaded = st.sidebar.file_uploader(
        "Reference stick pattern (ICDD/COD, optional)",
        type=["csv", "xy", "txt", "dat"],
        accept_multiple_files=True,
        key="ref_uploader",
        help=(
            "2 or 3 column file: twotheta, intensity[, hkl]. Normalized to "
            "max=1 and drawn as vertical ticks under the main plot, for "
            "comparing your pattern against a known reference (e.g. an "
            "ICDD/COD entry) rather than adding it as another data trace."
        ),
    )
    if ref_uploaded:
        for ruf in ref_uploaded:
            try:
                suffix = Path(ruf.name).suffix or ".csv"
                with tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=False) as tf:
                    tf.write(ruf.getvalue())
                    tmp_path = tf.name
                try:
                    ref_df = load_reference_pattern(tmp_path)
                    reference_patterns[ruf.name] = ref_df
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            except Exception as e:
                all_errors.append(f"Reference pattern '{ruf.name}': {e}")

    return all_raw_data, all_errors, all_issues, all_metadata, reference_patterns, example_files_available


def render_display_controls_section(sorted_fnames: List[str], global_min: float, global_max: float):
    """Render display controls, return dict of settings."""
    st.sidebar.header("🎛️ Display Controls")

    # Active files - session persisted.
    # Auto-activate files we've never seen before (e.g. a fresh folder load or
    # upload), while leaving the user's manual de-selections of known files alone.
    known_files = set(st.session_state.get("_known_files", []))
    current_files = set(sorted_fnames)
    newly_seen = sorted(current_files - known_files)
    if "active_files" not in st.session_state:
        st.session_state.active_files = sorted_fnames
    else:
        kept = [f for f in st.session_state.active_files if f in current_files]
        for f in newly_seen:
            if f not in kept:
                kept.append(f)
        st.session_state.active_files = kept
    st.session_state["_known_files"] = sorted(current_files | known_files)
    if newly_seen:
        st.sidebar.success(f"✅ {len(newly_seen)} new pattern(s) auto-added to Active patterns below.")

    active_files = st.sidebar.multiselect(
        "Active patterns",
        options=sorted_fnames,
        default=st.session_state.active_files,
        key="active_files_select",
    )
    # Update session state
    st.session_state.active_files = active_files

    # Normalization - session persisted
    norm_options = ["none", "max=1", "area=1", "minmax (0-1)", "mean=1", "zscore"]
    if "norm_option" not in st.session_state:
        st.session_state.norm_option = "none"
    norm_option = st.sidebar.selectbox(
        "Normalization", norm_options, key="norm_option",
        help=(
            "none: raw counts, no scaling.\n\n"
            "max=1: I' = I / max(I) — best for shape comparison, but one noisy "
            "spike at the max point skews the whole pattern.\n\n"
            "area=1: I' = I / ∫I d(2θ) — preserves relative total scattering; use "
            "when scans had different exposure/step time.\n\n"
            "minmax (0-1): I' = (I−min)/(max−min) — also floors the background to "
            "0, which distorts things if patterns have different backgrounds.\n\n"
            "mean=1: I' = I / mean(I).\n\n"
            "zscore: I' = (I−mean)/std — standardized, no longer physical counts; "
            "mainly useful before statistical similarity comparison, not for a "
            "public-facing figure."
        ),
    )

    norm_map = {
        "none": "none", "max=1": "max=1", "area=1": "area=1",
        "minmax (0-1)": "minmax", "mean=1": "mean", "zscore": "zscore",
    }
    norm_internal = norm_map.get(norm_option, "none")

    view_mode = st.sidebar.radio(
        "View mode", ["Overlay", "Stack", "Grid (small multiples)"], index=0, horizontal=True, key="view_mode",
        help=(
            "Overlay: all patterns share one y-axis, drawn on top of each other — "
            "best for comparing peak heights/intensities directly.\n\n"
            "Stack: each pattern is shifted up by a fixed offset so they don't "
            "overlap — best for comparing peak positions/shapes across many "
            "patterns; the vertical offset carries no physical meaning, it's "
            "purely for visual separation.\n\n"
            "Grid: each pattern gets its own small subplot — best when patterns "
            "have very different intensity scales that would dwarf each other."
        ),
    )
    stack_mode = view_mode == "Stack"
    grid_mode = view_mode == "Grid (small multiples)"

    # Guard zero-width: if global min/max are close, expand
    if global_max - global_min < 0.1:
        global_max = global_min + 1.0

    # twotheta range with guard
    st.sidebar.subheader("2θ Range")
    if "twotheta_range" not in st.session_state:
        st.session_state.twotheta_range = (float(global_min), float(global_max))
    # If stored range outside new global bounds, reset
    cur_min, cur_max = st.session_state.twotheta_range
    if cur_min < global_min or cur_max > global_max or cur_min >= cur_max:
        st.session_state.twotheta_range = (float(global_min), float(global_max))

    twotheta_range = st.sidebar.slider(
        "Crop 2θ view",
        min_value=float(global_min),
        max_value=float(global_max),
        value=st.session_state.twotheta_range,
        step=0.01,
        key="twotheta_range_slider",
    )
    # Guard zero-width
    xmin, xmax = twotheta_range
    if xmax - xmin < 0.001:
        st.sidebar.warning(f"2θ range too narrow ({xmin:.3f}-{xmax:.3f}). Expanding to avoid crash.")
        # Auto-expand by 0.1 deg each side
        twotheta_range = (max(global_min, xmin - 0.1), min(global_max, xmax + 0.1))
        st.session_state.twotheta_range = twotheta_range

    return {
        "active_files": active_files,
        "norm_option": norm_option,
        "norm_internal": norm_internal,
        "view_mode": view_mode,
        "stack_mode": stack_mode,
        "grid_mode": grid_mode,
        "twotheta_range": twotheta_range,
        "global_min": global_min,
        "global_max": global_max,
    }


def render_smoothing_controls_section():
    st.sidebar.subheader("Smoothing — Enhanced")
    with st.sidebar.expander("📖 What do these mean?", expanded=False):
        st.markdown(
            "- **Savitzky-Golay**: fits a local polynomial (least-squares) over a "
            "moving window, then keeps its center value. Preserves peak height/shape "
            "far better than a plain average for the same noise reduction — the "
            "standard choice for XRD.\n"
            "- **Gaussian**: convolves with a Gaussian kernel of width σ (in points); "
            "broadens peaks roughly proportional to σ.\n"
            "- **Moving average**: running mean over the window — simplest, but "
            "flattens/broadens peaks the most for a given amount of noise removed.\n"
            "- **Median filter**: replaces each point with the median of its "
            "neighborhood — best at killing single-point spikes (detector glitches, "
            "cosmic rays) without touching smooth peak shapes as much as a moving "
            "average would."
        )
    smoothing_enabled = st.sidebar.checkbox("Enable smoothing", value=False, key="smooth_enable")
    smooth_method = "savgol"
    smooth_window = 11
    smooth_poly = 2
    smooth_sigma = 2.0
    if smoothing_enabled:
        method_display = st.sidebar.selectbox(
            "Method",
            ["savgol (Savitzky-Golay)", "gaussian", "moving_average", "median"],
            key="smooth_method",
            help="See '📖 What do these mean?' above for the formula behind each method.",
        )
        if "savgol" in method_display:
            smooth_method = "savgol"
        elif "gaussian" in method_display:
            smooth_method = "gaussian"
        elif "moving" in method_display:
            smooth_method = "moving_average"
        else:
            smooth_method = "median"
        smooth_window = st.sidebar.slider(
            "Window", 5, 101, 11, step=2, key="smooth_window",
            help="Number of points averaged/fit together. Bigger = smoother but can "
                 "start eating real peak width; keep it well below your narrowest peak's "
                 "point-count.",
        )
        if smooth_method == "savgol":
            smooth_poly = st.sidebar.slider(
                "Polyorder", 1, 5, 2, key="smooth_poly",
                help="Degree of the polynomial fit inside each window. Higher preserves "
                     "sharp peak shapes better but is more sensitive to noise; 2-3 is "
                     "the usual sweet spot for XRD.",
            )
        if smooth_method == "gaussian":
            smooth_sigma = st.sidebar.slider(
                "Gaussian sigma", 0.5, 10.0, 2.0, step=0.5, key="smooth_sigma",
                help="Width of the Gaussian kernel in points. Larger sigma = more "
                     "smoothing and more peak broadening.",
            )
    return {
        "smoothing_enabled": smoothing_enabled,
        "smooth_method": smooth_method,
        "smooth_window": smooth_window,
        "smooth_poly": smooth_poly,
        "smooth_sigma": smooth_sigma,
    }


def render_baseline_controls_section():
    st.sidebar.subheader("Baseline — Advanced")
    with st.sidebar.expander("📖 What do these mean?", expanded=False):
        st.markdown(
            "- **Linear**: a straight line between the (robust) endpoints of the "
            "pattern — fine for a flat background, cannot follow a curved amorphous "
            "hump.\n"
            "- **Polynomial (iterative)**: fits an iteratively-reweighted polynomial "
            "through the local minima; can follow a mild curve, degree = flexibility.\n"
            "- **ALS** (Asymmetric Least Squares): a penalized smoother pulled toward "
            "the data from below only (weight *p* above the fit vs *1−p* below), so "
            "it hugs the lower envelope. *λ* sets stiffness/smoothness, *p* sets how "
            "strongly peaks are excluded.\n"
            "- **ARPLS**: like ALS but with smarter (logistic) reweighting — usually "
            "needs less hand-tuning, handles both sharp and broad backgrounds.\n"
            "- **SNIP**: iteratively clips each point toward the average of "
            "increasingly distant neighbors, converging on a peak-free envelope — no "
            "assumption about background shape; standard in gamma-spectroscopy/XRD.\n"
            "- **Median (rolling)**: rolling median, lightly Gaussian-smoothed — "
            "simple and robust if the window is much wider than your peaks, crude "
            "for strongly curved backgrounds."
        )
    baseline_enabled = st.sidebar.checkbox("Enable baseline (display only)", value=False, key="bl_enable")
    baseline_method = "linear"
    baseline_degree = 2
    baseline_lam = 1e5
    baseline_p = 0.01
    baseline_niter = 20
    snip_iter = 24
    median_win = 151
    clip_negative = False
    show_baseline = False
    if baseline_enabled:
        method_display = st.sidebar.selectbox(
            "Method",
            ["linear (robust edges)", "polynomial (iterative)", "als (Asymmetric LS)", "arpls (reweighted)", "snip (peak clipping)", "median (rolling)"],
            key="bl_method",
            help="See '📖 What do these mean?' above for the formula behind each method.",
        )
        if "linear" in method_display:
            baseline_method = "linear"
        elif "polynomial" in method_display:
            baseline_method = "polynomial"
        elif "arpls" in method_display:
            baseline_method = "arpls"
        elif "als" in method_display:
            baseline_method = "als"
        elif "snip" in method_display:
            baseline_method = "snip"
        else:
            baseline_method = "median"

        if baseline_method == "polynomial":
            baseline_degree = st.sidebar.slider(
                "Poly degree", 1, 8, 2, key="bl_degree",
                help="Higher degree follows more curvature but risks bending into real peaks.",
            )
        if baseline_method in ("als", "arpls"):
            baseline_lam = st.sidebar.select_slider(
                "λ smoothness", options=[1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8], value=1e5, key="bl_lam",
                help="How stiff the baseline is. Larger λ = smoother/straighter baseline "
                     "that ignores small wiggles; smaller λ lets it flex more (risking "
                     "eating into broad peaks).",
            )
            if baseline_method == "als":
                baseline_p = st.sidebar.slider(
                    "p asymmetry", 0.001, 0.1, 0.01, step=0.001, format="%.3f", key="bl_p",
                    help="How strongly the fit is pulled below the data. Smaller p pushes "
                         "the baseline further under the peaks (good for sharp, well-"
                         "separated peaks); larger p follows the data more closely.",
                )
            baseline_niter = st.sidebar.slider("Iterations", 5, 100, 20, step=5, key="bl_iter")
        if baseline_method == "snip":
            snip_iter = st.sidebar.slider(
                "SNIP iterations", 10, 60, 24, key="snip_iter",
                help="More iterations reach further under broader peaks/humps.",
            )
        if baseline_method == "median":
            median_win = st.sidebar.slider(
                "Median window", 51, 501, 151, step=10, key="med_win",
                help="Should be much wider (in points) than your peaks, or the median "
                     "filter will start tracking the peaks themselves instead of the "
                     "background.",
            )

        clip_negative = st.sidebar.checkbox("Clip negative to 0", value=False, key="clip_neg")
        show_baseline = st.sidebar.checkbox("Show baseline curve", value=False, key="show_bl")
    return {
        "baseline_enabled": baseline_enabled,
        "baseline_method": baseline_method,
        "baseline_degree": baseline_degree,
        "baseline_lam": baseline_lam,
        "baseline_p": baseline_p,
        "baseline_niter": baseline_niter,
        "snip_iter": snip_iter,
        "median_win": median_win,
        "clip_negative": clip_negative,
        "show_baseline": show_baseline,
    }


def render_plot_extras_section():
    st.sidebar.subheader("Plot Extras")
    log_y = st.sidebar.checkbox("Log scale Y", value=False, key="log_y")
    template_choice = st.sidebar.selectbox("Plot template", ["plotly_white", "plotly", "simple_white", "plotly_dark", "ggplot2", "seaborn", "clean_print"], index=0, key="plot_template")
    show_offset_labels = st.sidebar.checkbox("Show offset labels (stack)", value=True, key="offset_labels")
    show_error = st.sidebar.checkbox("Show error bands", value=False, key="show_error")
    assume_poisson_error = st.sidebar.checkbox(
        "Assume Poisson (√N) error if file has none", value=False, key="assume_poisson",
        help="Most XRD files carry no error column, so 'Show error bands' otherwise "
             "shows nothing. Enabling this estimates counting-statistics error as "
             "σ=√N from the raw intensity (valid for a photon/count detector, before "
             "any normalization — applied first so it scales correctly if you also "
             "normalize). Only fills in files that don't already have their own "
             "error column; never overrides one that does.",
    )
    show_peak_labels = st.sidebar.checkbox("Show peak labels", value=True, key="show_peak_labels")

    wavelength_input = st.sidebar.text_input("Wavelength Å for d-spacing (Cu 1.5406)", value="1.5406", placeholder="1.5406", key="wavelength_input")
    wavelength_val = None
    wavelength_warning = None
    if wavelength_input.strip():
        try:
            wavelength_val = float(wavelength_input)
            if not (0.5 <= wavelength_val <= 3.0):
                wavelength_warning = f"Wavelength {wavelength_val:.4f} Å outside typical lab range 0.5–3.0 Å (Cu 1.54, Mo 0.71, Co 1.79). Proceeding anyway."
        except ValueError:
            wavelength_warning = f"Invalid wavelength '{wavelength_input}', must be numeric."

    # Color palette with new CB safe
    palette_names = list(PALETTES.keys())
    # Ensure CB safe at top for visibility
    if "CB Safe (Okabe-Ito)" in palette_names:
        # Move to front
        palette_names = ["CB Safe (Okabe-Ito)", "Clean", "Plotly"] + [p for p in palette_names if p not in ["CB Safe (Okabe-Ito)", "Clean", "Plotly"]]
    palette_choice = st.sidebar.selectbox("Color palette", options=palette_names, index=0, key="palette_choice",
                                          help="CB Safe is colorblind-safe Okabe-Ito 8-color set for publication.")

    return {
        "log_y": log_y,
        "template_choice": template_choice,
        "show_offset_labels": show_offset_labels,
        "show_error": show_error,
        "assume_poisson_error": assume_poisson_error,
        "show_peak_labels": show_peak_labels,
        "wavelength_val": wavelength_val,
        "wavelength_warning": wavelength_warning,
        "palette_choice": palette_choice,
    }


def render_peak_controls_section():
    st.sidebar.subheader("Peak Analysis")
    peak_enabled = st.sidebar.checkbox("Enable peak finding", value=False, key="peak_enable",
                                       help="Uses scipy.signal.find_peaks with noise-based prominence")
    peak_prom_factor = 3.0
    peak_min_dist = 0.3
    instrumental_fwhm = 0.0
    if peak_enabled:
        peak_prom_factor = st.sidebar.slider(
            "Prominence factor × noise", 1.0, 10.0, 3.0, step=0.5, key="peak_prom",
            help="A peak must rise above its local surroundings by at least this "
                 "multiple of the estimated point-to-point noise (a MAD-based "
                 "estimator) to be counted. Raise it if you're getting spurious "
                 "noise peaks; lower it if you're missing small real peaks.",
        )
        peak_min_dist = st.sidebar.slider(
            "Min distance deg 2θ", 0.05, 2.0, 0.3, step=0.05, key="peak_dist",
            help="Minimum separation between two accepted peaks, so one broad "
                 "peak isn't counted twice.",
        )
        instrumental_fwhm = st.sidebar.number_input(
            "Instrumental FWHM at this 2θ (°, from a standard e.g. LaB6/Si)",
            value=0.0, min_value=0.0, max_value=2.0, step=0.001, format="%.3f",
            key="instrumental_fwhm",
            help="Enter the FWHM your instrument itself gives for a sharp standard "
                 "peak at a similar 2θ (e.g. from a LaB6 or Si reference scan). "
                 "Scherrer sizes below then correct for it in quadrature: "
                 "β_sample = √(β_observed² − β_instrumental²). Leaving this at 0 "
                 "disables the correction — reported crystallite sizes are then a "
                 "lower bound, since instrumental broadening inflates the apparent "
                 "peak width.",
        )
    return {
        "peak_enabled": peak_enabled,
        "peak_prom_factor": peak_prom_factor,
        "peak_min_dist": peak_min_dist,
        "instrumental_fwhm": instrumental_fwhm,
    }


def render_alignment_section(active_files: List[str], raw_filtered: Dict[str, pd.DataFrame], twotheta_range) -> Dict[str, float]:
    """
    Sidebar section for correcting 2theta zero-point / sample-displacement offsets.

    Two patterns of the same phase recorded on different days (or with a
    slightly different sample height) commonly show the same peaks shifted by a
    near-constant offset in 2theta. This section lets the user nudge each active
    file's 2theta axis manually, or auto-estimate the shift via cross-correlation
    against a chosen reference pattern.

    Returns:
        Dict mapping filename -> shift in degrees 2theta to ADD to that file's
        twotheta values.
    """
    shifts: Dict[str, float] = {}
    if not active_files:
        return shifts
    with st.sidebar.expander("🎯 Alignment (2θ shift)", expanded=False):
        st.caption(
            "Corrects sample-displacement/zero-point error between patterns of the "
            "same phase collected on different days or sample mounts. Positive "
            "shift moves a pattern to higher 2θ."
        )
        ref_file = st.selectbox(
            "Reference pattern (align others to this one)",
            options=active_files,
            index=0,
            key="align_ref_file",
            help="The auto-align button below leaves this file's shift at 0 and "
                 "estimates every other active file's shift relative to it.",
        )
        if st.button("🧭 Auto-align all to reference", key="auto_align_btn"):
            # Deliberately no st.rerun() here: the per-file number_input widgets
            # below are instantiated later in this same script run, so writing
            # to st.session_state[f"shift_{fname}"] now is enough for them to
            # pick up the freshly-computed value on this same pass. Calling
            # st.rerun() from here would abort the run before every later
            # section (smoothing/baseline/peak controls, etc.) has had a
            # chance to instantiate its own widgets in this run, which resets
            # THEIR session_state back to each widget's coded default — a
            # real, previously-reproduced bug, not a hypothetical one.
            df_ref = raw_filtered.get(ref_file)
            if df_ref is not None and not df_ref.empty:
                for fname in active_files:
                    if fname == ref_file:
                        st.session_state[f"shift_{fname}"] = 0.0
                        continue
                    df_target = raw_filtered.get(fname)
                    if df_target is None or df_target.empty:
                        continue
                    est_shift = estimate_shift_by_xcorr(
                        df_ref=df_ref, df_target=df_target,
                        twotheta_range=twotheta_range,
                    )
                    st.session_state[f"shift_{fname}"] = round(float(est_shift), 3)
        for fname in active_files:
            key = f"shift_{fname}"
            if key not in st.session_state:
                st.session_state[key] = 0.0
            # Don't pass value= alongside key= for an already-initialized widget —
            # Streamlit warns/ignores one of them; key= alone is enough once the
            # session_state entry exists (set above, or by the auto-align button).
            shifts[fname] = st.number_input(
                fname, min_value=-2.0, max_value=2.0,
                step=0.01, format="%.3f", key=key,
                help="2θ shift (degrees) applied to this pattern before any other "
                     "processing, to correct sample-displacement/zero-point offset.",
            )
    return shifts


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

ensure_example_data()

# Data input
all_raw_data, all_errors, all_issues, all_metadata, reference_patterns, example_files_available = render_data_input_section()

# Persistent issues panel (robustness backlog)
if all_errors or all_issues:
    with st.sidebar.expander(f"⚠️ Issues ({len(all_errors)+len(all_issues)}) - persistent panel", expanded=False):
        if all_errors:
            st.markdown("**Failed files:**")
            for e in all_errors[:15]:
                st.write(f"- {e}")
        if all_issues:
            st.markdown("**Warnings (validation):**")
            for fname, issues in list(all_issues.items())[:10]:
                st.write(f"**{fname}:**")
                for is_ in issues:
                    st.write(f"  - {is_}")

if not all_raw_data:
    st.title("🔬 BraggsView")
    st.info(
        """
        **No data loaded yet.**
        - Drag & drop `.xy/.xye/.txt/.csv/.dat` in sidebar (now with metadata extraction & validation)
        - Or select synthetic examples (regeneratable)
        - Or load folder (parallel) or reference stick patterns

        **Phase 5 new:** Methods paragraph generator, session save/load, caption generator, peak table CSV, vector SVG/PDF export, multi-panel grid, CB-safe palette, print theme.
        """
    )
    st.code("# Example .xy\n10.0000 123.0\n10.0200 135.5", language="text")
    if get_example_dir().exists():
        st.write(f"Found {len(example_files_available)} files in example_data")
    st.stop()

global_min, global_max = get_global_twotheta_bounds(all_raw_data)

# Display controls (session persisted)
display_ctrls = render_display_controls_section(sorted(all_raw_data.keys()), global_min, global_max)
raw_filtered = {k: v for k, v in all_raw_data.items() if k in display_ctrls["active_files"]}
if not raw_filtered:
    st.warning("No active patterns.")
    st.stop()

# Alignment (2theta shift correction for sample-displacement/zero-point offset).
# Applied to a copy so the originally-loaded data is never mutated in place;
# everything downstream (processing, raw hover data, raw stats, CSV exports)
# uses this aligned dict instead of raw_filtered.
alignment_shifts = render_alignment_section(display_ctrls["active_files"], raw_filtered, display_ctrls["twotheta_range"])
raw_filtered_aligned: Dict[str, pd.DataFrame] = {}
for fname, df in raw_filtered.items():
    shift = alignment_shifts.get(fname, 0.0)
    if shift:
        df_shifted = df.copy()
        df_shifted["twotheta"] = df_shifted["twotheta"] + shift
        df_shifted = df_shifted.sort_values("twotheta").reset_index(drop=True)
        raw_filtered_aligned[fname] = df_shifted
    else:
        raw_filtered_aligned[fname] = df

smooth_ctrls = render_smoothing_controls_section()
baseline_ctrls = render_baseline_controls_section()
extras = render_plot_extras_section()
peak_ctrls = render_peak_controls_section()

# Per-file wavelength: prefer a wavelength detected in each file's own header
# metadata over the single global text-input value. Comparing patterns from
# different instruments/sources with one global wavelength would otherwise
# silently produce wrong d-spacing/crystallite-size numbers for whichever
# files don't actually match that wavelength.
per_file_wavelength: Dict[str, float] = {}
wavelength_from_metadata: Dict[str, bool] = {}
for fname in raw_filtered_aligned.keys():
    meta = all_metadata.get(fname, {})
    wl_from_meta = meta.get("wavelength")
    resolved_wl = extras["wavelength_val"]
    used_metadata_wl = False
    if wl_from_meta:
        try:
            wl_f = float(wl_from_meta)
            if 0.5 <= wl_f <= 3.0:
                resolved_wl = wl_f
                used_metadata_wl = True
        except (TypeError, ValueError):
            pass
    per_file_wavelength[fname] = resolved_wl if resolved_wl is not None else 1.5406
    wavelength_from_metadata[fname] = used_metadata_wl

# Reset to defaults button (state management)
if st.sidebar.button("♻️ Reset to defaults", key="reset_defaults"):
    # Clear relevant session state keys
    for key in list(st.session_state.keys()):
        if key not in ("theme_choice",):  # keep theme
            del st.session_state[key]
    st.cache_data.clear()
    st.rerun()

# Wavelength validation banner
if extras["wavelength_warning"]:
    st.sidebar.warning(extras["wavelength_warning"])

# Offset spacing calc
temp_max_int = 0.0
for df in raw_filtered_aligned.values():
    if df.empty:
        continue
    if display_ctrls["norm_internal"] in ("max=1", "minmax", "zscore"):
        temp_max_int = max(temp_max_int, 1.0)
    elif display_ctrls["norm_internal"] in ("area=1", "mean"):
        temp_max_int = max(temp_max_int, 0.5)
    else:
        temp_max_int = max(temp_max_int, float(df["intensity"].max()))
if temp_max_int == 0:
    temp_max_int = 1.0

if display_ctrls["stack_mode"]:
    if display_ctrls["norm_internal"] == "none":
        offset_default = temp_max_int * 0.2
        offset_max = temp_max_int * 2.0
        offset_step = max(temp_max_int * 0.01, 1.0)
    else:
        offset_default = 0.5
        offset_max = 3.0
        offset_step = 0.05
    offset_spacing = st.sidebar.slider("Offset spacing", 0.0, float(offset_max), float(offset_default), float(offset_step), key="offset_spacing")
else:
    offset_spacing = 0.0

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

processed_dict: Dict[str, pd.DataFrame] = {}
baselines_dict: Dict[str, np.ndarray] = {}
processing_issues: Dict[str, str] = {}

for fname, df_raw in raw_filtered_aligned.items():
    try:
        proc_df, baseline_arr, _ = process_pattern(
            df_raw=df_raw,
            twotheta_range=display_ctrls["twotheta_range"],
            normalize_method=display_ctrls["norm_internal"],
            smoothing_enabled=smooth_ctrls["smoothing_enabled"],
            smoothing_window=smooth_ctrls["smooth_window"],
            smoothing_polyorder=smooth_ctrls["smooth_poly"],
            smoothing_method=smooth_ctrls["smooth_method"],
            smoothing_sigma=smooth_ctrls["smooth_sigma"],
            baseline_enabled=baseline_ctrls["baseline_enabled"],
            baseline_method=baseline_ctrls["baseline_method"],
            baseline_degree=baseline_ctrls["baseline_degree"],
            baseline_lam=baseline_ctrls["baseline_lam"],
            baseline_p=baseline_ctrls["baseline_p"],
            baseline_niter=baseline_ctrls["baseline_niter"],
            snip_iterations=baseline_ctrls["snip_iter"],
            median_window=baseline_ctrls["median_win"],
            clip_negative_after_baseline=baseline_ctrls["clip_negative"],
            assume_poisson_error=extras["assume_poisson_error"],
        )
        processed_dict[fname] = proc_df
        if baseline_arr is not None:
            baselines_dict[fname] = baseline_arr
    except Exception as e:
        processing_issues[fname] = str(e)
        processed_dict[fname] = df_raw.copy()

if processing_issues:
    with st.expander(f"⚠️ Processing issues ({len(processing_issues)}) - some files fell back to raw", expanded=False):
        for fn, err in processing_issues.items():
            st.write(f"- {fn}: {err}")

if all(df.empty for df in processed_dict.values()):
    st.error("All patterns empty after cropping. Adjust 2θ range.")
    st.stop()

# Peak finding (Phase 2)
peaks_dict: Dict[str, pd.DataFrame] = {}
if peak_ctrls["peak_enabled"]:
    for fname, df in processed_dict.items():
        try:
            peaks_df = find_peaks_xrd(
                df,
                prominence_factor=peak_ctrls["peak_prom_factor"],
                min_distance_deg=peak_ctrls["peak_min_dist"],
                wavelength=per_file_wavelength.get(fname, extras["wavelength_val"]),
            )
            peaks_dict[fname] = peaks_df
        except Exception as e:
            st.warning(f"Peak finding failed for {fname}: {e}")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

st.title("🔬 BraggsView")
st.caption(f"Theme: {st.session_state.theme_choice} | Palette: {extras['palette_choice']} | {len(processed_dict)} patterns | Contrast audit: Light {audit_theme_contrast('light')['text_primary on bg'][0]:.1f}:1 PASS")

# Determine layout mode
layout_mode = "overlay"
if display_ctrls["grid_mode"]:
    layout_mode = "grid"
elif display_ctrls["stack_mode"]:
    layout_mode = "stack"

# Map template choice
template_map = {
    "clean_print": "print",
}
# Resolve template for plotly (we pass plotly template name, but apply_clean will override)
plotly_template = extras["template_choice"]
if plotly_template == "clean_print":
    plotly_template = "plotly_white"
    # Force print mode for theme application
    theme_mode_for_plot = "print"
else:
    theme_mode_for_plot = st.session_state.theme_choice

# Main figure
if layout_mode == "grid":
    fig = create_multi_panel_figure(
        data_dict=processed_dict,
        palette_name=extras["palette_choice"],
        title=f"XRD Multi-Panel - {len(processed_dict)} patterns (norm={display_ctrls['norm_option']})",
        template=plotly_template,
        wavelength=extras["wavelength_val"],
        normalization_method=display_ctrls["norm_internal"],
        peaks_dict=peaks_dict if peak_ctrls["peak_enabled"] else None,
        reference_patterns=reference_patterns if reference_patterns else None,
    )
else:
    fig = create_xrd_figure(
        data_dict=processed_dict,
        raw_data_dict=raw_filtered_aligned,
        palette_name=extras["palette_choice"],
        stack_mode=display_ctrls["stack_mode"],
        offset_spacing=offset_spacing,
        log_y=extras["log_y"],
        show_baseline=baseline_ctrls["show_baseline"],
        baselines=baselines_dict if baseline_ctrls["show_baseline"] else None,
        title=f"XRD Comparison - {len(processed_dict)} patterns ({display_ctrls['view_mode']}, norm={display_ctrls['norm_option']})",
        template=plotly_template,
        show_error_bands=extras["show_error"],
        show_offset_labels=extras["show_offset_labels"],
        wavelength=extras["wavelength_val"],
        normalization_method=display_ctrls["norm_internal"],
        peaks_dict=peaks_dict if peak_ctrls["peak_enabled"] else None,
        reference_patterns=reference_patterns if reference_patterns else None,
        layout_mode="overlay",
        show_peak_labels=extras["show_peak_labels"],
    )

# Apply clean theme (print mode if selected)
if HAS_CLEAN_THEME:
    fig = apply_clean_plotly_theme(fig, mode=theme_mode_for_plot, palette="cb_safe" if "CB Safe" in extras["palette_choice"] else "clean",
                                     for_export=(theme_mode_for_plot == "print"))

# Plotly config
config = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": True,
    "toImageButtonOptions": {"format": "png", "filename": "xrd_comparison", "height": 650, "width": 1100, "scale": 3},
}

st.plotly_chart(fig, width='stretch', config=config)

# ---------------------------------------------------------------------------
# Publication-oriented UX (Phase 5)
# ---------------------------------------------------------------------------

st.subheader("📄 Publication Tools")

# Methods paragraph
params_for_summary = {
    "normalize_method": display_ctrls["norm_internal"],
    "smoothing_enabled": smooth_ctrls["smoothing_enabled"],
    "smoothing_method": smooth_ctrls["smooth_method"],
    "smoothing_window": smooth_ctrls["smooth_window"],
    "smoothing_polyorder": smooth_ctrls["smooth_poly"],
    "smoothing_sigma": smooth_ctrls["smooth_sigma"],
    "baseline_enabled": baseline_ctrls["baseline_enabled"],
    "baseline_method": baseline_ctrls["baseline_method"],
    "baseline_degree": baseline_ctrls["baseline_degree"],
    "baseline_lam": baseline_ctrls["baseline_lam"],
    "baseline_p": baseline_ctrls["baseline_p"],
    "clip_negative_after_baseline": baseline_ctrls["clip_negative"],
    "twotheta_range": display_ctrls["twotheta_range"],
    "wavelength": extras["wavelength_val"],
}
methods_text = processing_summary(params_for_summary)

# Figure caption generator
def generate_caption():
    names = ", ".join([Path(f).stem for f in processed_dict.keys()][:6])
    if len(processed_dict) > 6:
        names += f" and {len(processed_dict)-6} more"
    wl = extras["wavelength_val"]
    wl_phrase = f" recorded with {wl} Å radiation" if wl else ""
    norm_phrase = f", normalized to {display_ctrls['norm_option']}" if display_ctrls["norm_internal"] != "none" else ""
    bl_phrase = f", baseline-corrected using {baseline_ctrls['baseline_method']}" if baseline_ctrls["baseline_enabled"] else ""
    caption = f"XRD patterns of {names}{wl_phrase}{norm_phrase}{bl_phrase}. Data shown in {display_ctrls['view_mode'].lower()} mode over 2θ range {display_ctrls['twotheta_range'][0]:.1f}–{display_ctrls['twotheta_range'][1]:.1f}°."
    return caption

caption_text = generate_caption()

col_methods, col_caption = st.columns(2)
with col_methods:
    st.markdown("**One-click Methods paragraph (for SI):**")
    st.code(methods_text, language="text")
    st.download_button("Download Methods paragraph (.txt)", methods_text, "methods_paragraph.txt", "text/plain", key="dl_methods")
    # Copy helper: streamlit doesn't have clipboard button, but code block is copyable

with col_caption:
    st.markdown("**Figure caption generator:**")
    st.code(caption_text, language="text")
    st.download_button("Download figure caption (.txt)", caption_text, "figure_caption.txt", "text/plain", key="dl_caption")

# Peak table export
if peak_ctrls["peak_enabled"] and peaks_dict:
    with st.expander("📌 Peak Table & Scherrer Size", expanded=False):
        st.warning(
            "Scherrer sizes assume an isolated, symmetric, strain-free peak — "
            "verify overlap with neighboring peaks before trusting sizes near "
            "your instrument's resolution limit."
        )
        instrumental_fwhm = peak_ctrls.get("instrumental_fwhm", 0.0)
        any_metadata_wl = any(wavelength_from_metadata.get(f, False) for f in peaks_dict.keys())
        if any_metadata_wl:
            st.caption(
                "ℹ️ Some files' d-spacing/crystallite size below use a wavelength "
                "detected in that file's own header (see 'wavelength_used' column "
                "and 'wavelength_source'), not the single value typed in the "
                "sidebar."
            )
        all_peaks_frames = []
        for fname, pdf in peaks_dict.items():
            if pdf.empty:
                continue
            tmp = pdf.copy()
            file_wl = per_file_wavelength.get(fname, extras["wavelength_val"] or 1.5406)
            # Instrumental-broadening correction (quadrature subtraction), then Scherrer
            try:
                tmp["fwhm_corrected"] = tmp["fwhm"].apply(
                    lambda f: deconvolve_instrumental_fwhm(f, instrumental_fwhm)
                )
                tmp["crystallite_size_nm"] = tmp["fwhm_corrected"].combine(
                    tmp["twotheta"],
                    lambda fwhm_c, tt: scherrer_crystallite_size(fwhm_c, tt, wavelength=file_wl) if np.isfinite(fwhm_c) else float("nan"),
                )
            except Exception:
                tmp["fwhm_corrected"] = tmp.get("fwhm", float("nan"))
                tmp["crystallite_size_nm"] = float("nan")
            tmp["wavelength_used"] = file_wl
            tmp["wavelength_source"] = "file header" if wavelength_from_metadata.get(fname, False) else "sidebar input"
            tmp["filename"] = fname
            all_peaks_frames.append(tmp)
        if all_peaks_frames:
            combined_peaks = pd.concat(all_peaks_frames, ignore_index=True)
            st.dataframe(combined_peaks, width='stretch')
            csv_peaks = combined_peaks.to_csv(index=False)
            st.download_button("Download peak list as CSV (2θ, d, intensity, FWHM, crystallite size)", csv_peaks, "peak_list.csv", "text/csv", key="dl_peaks")
        else:
            st.write("No peaks found with current settings.")

# Session save/load
with st.expander("💾 Session Save/Load (reproducibility)", expanded=False):
    st.markdown("Save current settings to JSON to reproduce exactly months later.")
    session_data = {
        "timestamp": datetime.now().isoformat(),
        "files": list(all_raw_data.keys()),
        "active_files": display_ctrls["active_files"],
        "params": params_for_summary,
        "view_mode": display_ctrls["view_mode"],
        "offset_spacing": offset_spacing,
        "log_y": extras["log_y"],
        "palette": extras["palette_choice"],
        "template": extras["template_choice"],
        "theme": st.session_state.theme_choice,
        "peak_enabled": peak_ctrls["peak_enabled"],
        "peak_params": {"prom_factor": peak_ctrls["peak_prom_factor"], "min_dist": peak_ctrls["peak_min_dist"]},
    }
    session_json = json.dumps(session_data, indent=2)
    st.download_button("Save session JSON", session_json, f"xrd_session_{datetime.now().strftime('%Y%m%d_%H%M')}.json", "application/json", key="dl_session")

    uploaded_session = st.file_uploader("Load session JSON", type=["json"], key="session_loader")
    if uploaded_session:
        try:
            loaded = json.loads(uploaded_session.getvalue().decode("utf-8"))
            st.json(loaded)
            st.info("Session loaded (display only). To apply, manually set controls to match loaded params, or use Reset then re-apply. Future version will auto-apply.")
        except Exception as e:
            st.error(f"Failed to load session: {e}")

# Export section with vector
st.subheader("💾 Export Data & Figures")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    combined_csv = export_combined_csv(processed_dict)
    st.download_button("Combined long CSV", combined_csv, "xrd_processed_combined.csv", "text/csv", key="dl_combined")

with col2:
    zip_bytes = export_processed_csv_zip(processed_dict)
    st.download_button("ZIP per-file CSVs", zip_bytes, "xrd_processed_files.zip", "application/zip", key="dl_zip")

with col3:
    html_buffer = io.StringIO()
    fig.write_html(html_buffer, include_plotlyjs="cdn")
    st.download_button("Interactive HTML", html_buffer.getvalue(), "xrd_comparison.html", "text/html", key="dl_html")

# Cheap (~10ms) content hash of the current figure, used to detect whether a
# previously-generated export is stale (see render_gated_export_button).
main_fig_sig = hashlib.md5(fig.to_json().encode()).hexdigest()

with col4:
    render_gated_export_button(
        key="main_svg", label="Vector SVG (journal)",
        generate_fn=lambda: figure_to_svg_bytes(fig, width=1100, height=650, scale=2),
        filename="xrd_comparison.svg", mime="image/svg+xml",
        fig_sig=main_fig_sig,
        help_text="Renders via Kaleido (~3s) — needs kaleido + Chrome installed locally.",
    )

with col5:
    render_gated_export_button(
        key="main_pdf", label="Vector PDF (journal)",
        generate_fn=lambda: figure_to_pdf_bytes(fig, width=1100, height=650, scale=2),
        filename="xrd_comparison.pdf", mime="application/pdf",
        fig_sig=main_fig_sig,
        help_text="Renders via Kaleido (~4s) — needs kaleido + Chrome installed locally.",
    )

# Extra common-grid CSV
with st.expander("Common-grid CSV (aligned for external analysis)", expanded=False):
    try:
        common_csv = export_common_grid_csv(processed_dict, step=0.02)
        st.download_button("Download common-grid wide CSV", common_csv, "xrd_common_grid.csv", "text/csv", key="dl_common")
        st.caption("All patterns interpolated onto common 2θ grid.")
    except Exception as e:
        st.error(f"Common grid export failed: {e}")

# Publication Figure Studio
with st.expander("🎯 Publication Figure Studio (journal-ready, fully editable)", expanded=False):
    render_publication_studio(processed_dict, theme_mode=theme_mode_for_plot)

# Similarity matrix (new Phase 2)
with st.expander("🔗 Pattern Similarity Matrix (quantitative comparison)", expanded=False):
    if len(processed_dict) >= 2:
        names = sorted(processed_dict.keys())
        sim_matrix = pd.DataFrame(index=names, columns=names, dtype=float)
        for i, na in enumerate(names):
            for j, nb in enumerate(names):
                if i == j:
                    sim_matrix.loc[na, nb] = 1.0
                elif i < j:
                    sim = pattern_similarity(processed_dict[na], processed_dict[nb], method="pearson", step=0.02, twotheta_range=display_ctrls["twotheta_range"])
                    sim_matrix.loc[na, nb] = sim
                    sim_matrix.loc[nb, na] = sim
        try:
            styled = sim_matrix.style.background_gradient(cmap="Blues", vmin=0, vmax=1)
            st.dataframe(styled, width='stretch')
        except ImportError:
            st.dataframe(sim_matrix, width='stretch')
            st.caption("Colour scale needs matplotlib (`pip install matplotlib`); showing plain table.")
        st.caption(
            "Pearson correlation of intensity-vs-2θ shape on a common grid (1.0 = identical curve shape). "
            "This is a rough visual-similarity check, not phase matching — it can be thrown off by "
            "background level and by preferred orientation, and doesn't compare peak positions/d-spacings "
            "directly. Two different phases with similar backgrounds can score deceptively high."
        )
        st.download_button("Download similarity matrix CSV", sim_matrix.to_csv(), "similarity_matrix.csv", "text/csv", key="dl_sim")
    else:
        st.write("Need at least 2 patterns for similarity.")

# Difference plot (A - B)
with st.expander("📉 Difference Plot (A − B)", expanded=False):
    st.caption(
        "Subtracts one processed pattern from another on a common 2θ grid — the "
        "quickest way to see whether one phase grew at the expense of another "
        "between two scans (e.g. before/after ion exchange, before/after heating)."
    )
    if len(processed_dict) >= 2:
        names_sorted = sorted(processed_dict.keys())
        col_a, col_b = st.columns(2)
        with col_a:
            diff_name_a = st.selectbox("Pattern A", names_sorted, index=0, key="diff_name_a")
        with col_b:
            default_b_idx = 1 if len(names_sorted) > 1 else 0
            diff_name_b = st.selectbox("Pattern B", names_sorted, index=default_b_idx, key="diff_name_b")
        if diff_name_a == diff_name_b:
            st.warning("Pick two different patterns to compare.")
        else:
            diff_fig = create_difference_figure(
                processed_dict[diff_name_a], processed_dict[diff_name_b],
                name_a=diff_name_a, name_b=diff_name_b,
                palette_name=extras["palette_choice"],
                title=f"Difference: {diff_name_a} − {diff_name_b}",
            )
            st.plotly_chart(diff_fig, width='stretch')
    else:
        st.write("Need at least 2 patterns for a difference plot.")

st.info("PNG export: use camera icon in Plotly toolbar (scale 3 for high-res). SVG/PDF need kaleido + Chrome installed locally (`pip install kaleido` + `plotly_get_chrome`).")

# Stats
with st.expander("📊 Data Summary & Stats", expanded=True):
    stats_df = create_stats_table(processed_dict)
    st.dataframe(stats_df, width='stretch', hide_index=True)
    if st.checkbox("Show raw stats", value=False, key="show_raw_stats"):
        raw_stats = create_stats_table(raw_filtered_aligned)
        st.markdown("**Raw:**")
        st.dataframe(raw_stats, width='stretch', hide_index=True)
    # Show metadata extracted
    if all_metadata:
        with st.expander("Metadata extracted from headers", expanded=False):
            meta_df = pd.DataFrame.from_dict(all_metadata, orient="index")
            st.dataframe(meta_df, width='stretch')

with st.expander("📄 Processed Data Preview", expanded=False):
    for fname, df in processed_dict.items():
        st.markdown(f"**{fname}** — {len(df)} pts")
        st.dataframe(df.head(10), width='stretch')

# Footer
st.markdown("---")
st.markdown(
    """
    <small>
    Publication-ready XRD tool: colorblind-safe palette (Okabe-Ito), print theme (pure black/white, >=2px), vector SVG/PDF export,
    peak finding + Scherrer size, Methods paragraph & caption generators, session JSON reproducibility, pattern similarity.
    Run: <code>pip install -r requirements.txt && streamlit run app.py</code>
    </small>
    """,
    unsafe_allow_html=True,
)
