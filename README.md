# BraggsView 🔬

[![CI](https://github.com/anjulnj/BraggsView/actions/workflows/ci.yml/badge.svg)](https://github.com/anjulnj/BraggsView/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An interactive tool for visually comparing multiple XRD (X-ray diffraction) pattern files, with peak finding, Scherrer size estimation, and publication-ready export on top of the core comparison view.

Built with **Streamlit + Plotly** for browser-based interactivity, with robust file parsing, normalization, stacking, smoothing, and baseline subtraction.

## Why this instead of Profex / OriginLab / pymatgen?

These solve adjacent but different problems:
- **Profex/BGMN** and **TOPAS/GSAS** are Rietveld refinement suites — the right tool when you need a fitted structural model, but heavyweight for "does sample B look like sample A."
- **OriginLab** is general-purpose commercial plotting software with no XRD-specific parsing, metadata extraction, or domain logic (d-spacing, Scherrer size, baseline methods) built in.
- **pymatgen** has XRD utilities, but as a Python library — no drag-and-drop, no browser UI; you write code per comparison.

BraggsView's niche is specifically **fast, zero-code, browser-based visual comparison of many patterns at once** — drag in files, get an interactive multi-pattern figure, quick peak/Scherrer numbers, and a publication-ready export, without writing code or opening refinement software. It complements those tools rather than replacing them: reach for Profex/TOPAS once you actually need a refined structure.

## Limitations (honest, not exhaustive)

- **No Kα1/Kα2 stripping.** FWHM and Scherrer sizes from unstripped lab Cu data are systematically inflated at high 2θ, where the doublet visibly separates.
- **2θ is the only plotting axis; d/Q are not yet selectable as the primary axis.** Patterns collected at different wavelengths (e.g. synchrotron vs. lab Cu) cannot be directly overlaid today — d-spacing is only a secondary display axis.
- **The pattern similarity matrix compares intensity-vs-2θ shape (Pearson correlation), not peak positions.** It is a rough visual-similarity check, not phase matching — two different phases with similar backgrounds can score deceptively high, and the same phase with different preferred orientation can score low.
- **Preferred orientation is not corrected for or flagged.** Relative peak intensities in a textured/packed powder are not necessarily reliable, but nothing in the UI warns about this.
- **Peak positions are the nearest data point, not a fitted centroid** — sub-step-size shifts between patterns are not resolved by the current peak finder.

## Citation

If you use BraggsView in published work, please cite it — see [`CITATION.cff`](CITATION.cff).

## Features

### File Support
- Reads common XRD formats: `.xy`, `.xye`, `.txt`, `.csv`, `.dat`
- 2-column: `2theta, intensity`, optional 3rd error column
- Auto-detect delimiter (comma, tab, whitespace, semicolon)
- Skips header/comment lines automatically (lines starting with `#`, `%`, `;`, `!`, `//`, or non-numeric)
- Drag-and-drop multi-file upload (Streamlit)
- Folder-load option (load all matching files in a directory)
- Graceful error handling with user-facing messages

### Interactive Plotting (Plotly)
- Zoom in/out, pan on both axes (scrollZoom, drag)
- Hover tooltips showing 2θ, intensity, filename
- Toggle patterns via legend click
- **Stack / Overlay modes**:
  - Stack: vertically offset each pattern by adjustable amount
  - Overlay: all patterns on same y-axis
- **Normalization**: none, max=1, area=1
- **Log-scale Y** toggle
- Color-coded by filename, palette selector (Plotly, D3, G10, Viridis, etc.)
- **Smoothing**: Savitzky-Golay toggle with window + polyorder sliders (not altering raw)
- **Background subtraction**: linear or polynomial baseline toggle (display only) + optional dashed baseline visualization
- Export current view as PNG/SVG via Plotly modebar + export processed data as CSV/ZIP + HTML

### UI (Streamlit)
- Sidebar controls: file upload/multi-select, normalize dropdown, stack/overlay toggle, offset slider, smoothing, baseline, log toggle, 2θ range slider, palette
- Main panel: interactive Plotly plot
- Data table with stats per file: max intensity, 2θ at max, number of points, range
- Expandable raw vs processed preview

## Project Structure

```
project/
├── app.py                 # Streamlit entry point
├── xrd_io.py              # file parsing/loading functions
├── processing.py          # normalize, smooth, baseline, offset
├── plotting.py            # plotly figure-building
├── generate_examples.py   # generate synthetic .xy files
├── requirements.txt
├── README.md
└── example_data/          # synthetic examples + your files (auto-created)
    ├── synthetic_cubic.xy
    ├── synthetic_tetragonal.xy
    ├── synthetic_amorphous_mix.xy
    └── CH020_ND_L.txt etc.
```

## Installation & Running

### For non-programmers (no VS Code / no typing commands)

Double-click the launcher for your computer:

- **macOS:** `Start_BraggsView.command` (first launch may ask you to right-click → Open)
- **Windows:** `Start_BraggsView.bat` *or* `Start_BraggsView.pyw` (double-click either; the
  `.pyw` opens a small status window with a Stop button and needs no terminal at all —
  handy on managed/university PCs)

The launcher finds Python 3, creates a private app environment **outside the OneDrive
folder** (so switching Mac ↔ Windows can't corrupt it), installs packages, and opens the
app in your browser. You only need Python 3.9+ installed once
(https://www.python.org/downloads/). Close the launcher window to stop the app.

### For developers

```bash
# Clone or unzip project
cd project

# Create venv (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt

# Generate synthetic examples (first run auto-generates, but you can manual)
python generate_examples.py

# Run the app
streamlit run app.py
```

Then open browser at `http://localhost:8501`

### CLI / Folder Load

You can also run folder load via UI text input, or use modules programmatically:

```python
from xrd_io import load_folder, parse_xrd_file
from processing import process_pattern
from plotting import create_xrd_figure

data = load_folder("./example_data")
# data is dict filename -> DataFrame
df = parse_xrd_file("path/to/pattern.xy")
```

## File Format Details

Example `.xy`:

```
# Sample header, automatically skipped
# Wavelength = 1.54059
10.0000  123.0
10.0200  135.5
10.0400  128.2
```

CSV example:

```
2theta,intensity
10.0,123
10.02,135.5
```

- Delimiter auto-detected via regex split `[,\s\t;]+`
- Lines that fail numeric parsing are skipped (header tolerant)
- Sorts by 2θ ascending, drops duplicates
- Requires ≥3 valid points

## Processing Details (Display Only)

- **Normalization**:
  - `max=1`: `I / max(I)`
  - `area=1`: `I / trapz(I, 2θ)`
  - Toggle does NOT alter underlying file
- **Smoothing**: Savitzky-Golay (`scipy.signal.savgol_filter`), window auto-adjusted to odd < n
- **Baseline**:
  - Linear: mean of first/last 5% as endpoints, linear interpolation
  - Polynomial: bin data into 20 bins, find minima per bin, fit polynomial to minima (robust to peaks)
  - Subtraction optional clip to 0
  - Can show baseline dashed
- **Offset/Stack**: `y_display = y_processed + i * offset_spacing`
- **Cropping**: simple 2θ range filter

## Export

- **PNG/SVG**: Click camera icon in Plotly modebar (top-right of plot)
- **HTML**: Download button below plot saves interactive HTML with cdn plotly.js
- **CSV**:
  - Combined long format: one CSV with columns `filename, twotheta, intensity`
  - ZIP: per-file processed CSVs (`*_processed.csv`)
  - Processed means after cropping, normalization, smoothing, baseline (as displayed, except offset for stack is only for plotting; CSVs contain no stack offset, but you can include if needed)

## Error Handling

- Bad format, mismatched columns, empty files -> `XRDParseError` with user-facing message in Streamlit sidebar expander, not crash
- Smoothing window too large -> auto-adjust
- Log scale with ≤0 values handled by Plotly (blank at those points)
- Folder not found -> message

## Example Synthetic Data

`generate_examples.py` creates 3 patterns:

- `synthetic_cubic.xy`: Peaks at 28.4°, 32.2°, 47.4°, 56.4°...
- `synthetic_tetragonal.xy`: Peaks at 25.3°, 30.1°, 48°, 50.5°...
- `synthetic_amorphous_mix.xy`: Broad peaks + amorphous hump at 22°

Each with Gaussian peaks, linear background, noise.

You can also test with the 4 uploaded real files `CH020_ND_L.txt` etc. Negative 2θ values are preserved (some lab files start at -16°).

## Dependencies

- streamlit
- pandas, numpy
- plotly
- scipy (Savitzky-Golay)
- kaleido (optional for static export, but modebar works without)

Type hints & docstrings included throughout.

## Future Extensions (not implemented)

- Peak annotation by database
- Difference plots
- Alignment / 2θ offset correction

## License

MIT-like - free for research use.

## Screenshot (Conceptual)

- Sidebar: file upload, normalize, stack slider, smoothing, baseline, log, 2θ slider, palette
- Main: Plotly figure with legend toggle, hover, zoom
- Expanders: stats table, data preview, export buttons
cd "/Users/anjul001/Library/CloudStorage/OneDrive-KULeuven/My python projects/BraggsView"
python -m streamlit run app.py

MAC
cd "/Users/anjul001/Library/CloudStorage/OneDrive-KULeuven/My python projects/BraggsView"
python3 -m streamlit run app.py