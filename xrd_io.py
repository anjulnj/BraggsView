"""
xrd_io.py - Robust file parsing for XRD pattern files.

Supports .xy, .xye, .txt, .csv, .dat, .xyd with auto-delimiter detection,
header/comment skipping, and folder/batch loading.

Phase 1 improvements:
- Format detection: recognizes Bruker .raw/.brml, PANalytical .xrdml, Rigaku .ras and raises clear XRDParseError
- Metadata capture via parse_xrd_file_with_metadata() (new, non-breaking)
- Encoding robustness: UTF-8 -> charset-normalizer/chardet detection -> latin-1
- Validation: validate_xrd_dataframe() and parse_xrd_file_with_validation() returning issues list
- Performance: optional parallel loading in load_folder / load_multiple_files via ThreadPoolExecutor
"""

from __future__ import annotations

import re
import io
import os
import warnings
from pathlib import Path
from typing import Union, List, Dict, Tuple, BinaryIO, TextIO, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

# Supported extensions (ASCII)
SUPPORTED_EXTS = {".xy", ".xye", ".txt", ".csv", ".dat", ".xyd", ".xrd", ".uxd", ".ras", ".xrdml", ".brml", ".raw"}

COMMENT_STARTS = ("#", "%", ";", "!", "'", "//", "*")

DELIM_RE = re.compile(r'[,\s\t;]+')

class XRDParseError(Exception):
    """Custom exception for XRD parsing issues with user-friendly message."""
    pass

# ---------------------------------------------------------------------------
# Phase 1 - Format detection
# ---------------------------------------------------------------------------

def detect_special_format(content_start: str, filename: str = "unknown") -> Optional[str]:
    """
    Detect non-ASCII XRD formats that need explicit conversion.

    Returns None if content looks like regular ASCII XY, otherwise returns
    a user-friendly error message to raise.

    Checks:
    - XRDML XML: <?xml and <xrdMeasurements or <xrdml
    - BRML: <RawData, Bruker, *BRML* etc
    - RAW binary: null bytes or .raw extension with non-numeric start
    - RAS: Rigaku .ras often starts with *RAS or *HEADER etc but may still be parseable;
          we only trigger if content looks structured AND numeric parsing will likely fail
    """
    fname_lower = filename.lower()
    start_lower = content_start.lower()[:4000]  # first 4k chars

    # Null bytes -> binary
    if "\x00" in content_start[:1024]:
        if fname_lower.endswith(".raw"):
            return (
                f"File '{filename}' appears to be a Bruker binary .RAW file (contains null bytes). "
                "Binary RAW files cannot be read directly. Please export to ASCII .xy / .csv from "
                "DIFFRAC.EVA, or use Bruker's conversion tool (e.g., 'Convert RAW to UXD/XY'). "
                "If you have a Bruker .brml, export via Topas or EVA."
            )
        # generic binary
        return (
            f"File '{filename}' appears to be binary (null bytes detected). "
            "Common XRD binary formats (.raw, .brml) need conversion to ASCII .xy/.csv first."
        )

    # XRDML detection
    if fname_lower.endswith(".xrdml") or ("<?xml" in start_lower and ("<xrdmeasurements" in start_lower or "<xrdml" in start_lower)):
        return (
            f"File '{filename}' looks like a PANalytical XRDML XML file (<?xml ... <xrdMeasurements>). "
            "XRDML is XML-based and not 2-column ASCII. Please open in HighScore/PANalytical Data Viewer "
            "and export as ASCII .xy, .xye, or .csv (File → Save As → ASCII). "
            "Alternatively, use the xrdtools python package to parse XRDML."
        )

    # BRML detection
    if fname_lower.endswith(".brml") or "<rawdata" in start_lower or "<brml" in start_lower or "bruker" in start_lower and "<" in start_lower:
        # BRML is XML but specific to Bruker
        if "<" in start_lower[:2000] and "bruker" in start_lower:
            return (
                f"File '{filename}' looks like a Bruker BRML XML file. "
                "BRML export needs conversion: open in DIFFRAC.EVA / DIFFRAC.TOPAS and export as .xy/.uxd/.csv."
            )

    # RAS - Rigaku .ras can be text but often with *HEADER sections. We allow parsing normally,
    # but if file extension is .ras and first lines are mostly * keywords with no numeric data in first 50 lines,
    # we can still try parsing; our generic numeric parser will skip headers. So detect only if parsing will fail.
    # Here we just provide a hint if extension is .ras and content starts with *RAS_HEADER or similar and no numeric
    # in first few k. The caller will attempt parse anyway; this function returns None to allow attempt, but if
    # later parsing fails, we will augment error.
    # For explicit .raw/.brml already handled above.

    # UXD (Bruker) - usually ASCII but with _DRIVE etc. Might be parseable; we allow.
    return None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _is_comment_or_empty(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    for prefix in COMMENT_STARTS:
        if stripped.startswith(prefix):
            return True
    return False


def _parse_line(line: str) -> Optional[Tuple[float, float, Optional[float]]]:
    line = line.strip()
    if not line or _is_comment_or_empty(line):
        return None
    if not re.search(r'\d', line):
        return None
    parts = DELIM_RE.split(line)
    parts = [p for p in parts if p != ""]
    if len(parts) < 2:
        return None
    try:
        x = float(parts[0])
        y = float(parts[1])
        err = None
        if len(parts) >= 3:
            try:
                err = float(parts[2])
            except ValueError:
                err = None
        return (x, y, err)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

# regexes for common metadata in headers
WAVELENGTH_RE = re.compile(r'wavelength\s*[:=]\s*([0-9]+\.[0-9]+)', re.IGNORECASE)
WAVELENGTH_RE2 = re.compile(r'_diffrn_radiation_wavelength\s+([0-9]+\.[0-9]+)', re.IGNORECASE)
TARGET_RE = re.compile(r'(Cu|Co|Fe|Cr|Mo)\s*K[αa]?', re.IGNORECASE)
SAMPLE_NAME_RE = re.compile(r'(sample|name|id)\s*[:=]\s*([A-Za-z0-9_\-\. ]+)', re.IGNORECASE)
STEP_RE = re.compile(r'(step|increment)\s*[:=]\s*([0-9]+\.[0-9]+)', re.IGNORECASE)

def _extract_metadata(content: str) -> Dict[str, str]:
    """
    Extract metadata from header/comment lines.

    Looks for wavelength, target, sample name etc. in first 100 lines.

    Returns dict with keys if found: wavelength, target, sample, step, etc.
    """
    metadata: Dict[str, str] = {}
    lines = content.splitlines()[:150]  # header is usually early

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        # Skip numeric data lines - metadata is non-numeric headers
        if _parse_line(line) is not None:
            continue

        # wavelength
        m = WAVELENGTH_RE.search(line)
        if m:
            metadata["wavelength"] = m.group(1)
            continue
        m2 = WAVELENGTH_RE2.search(line)
        if m2:
            metadata["wavelength"] = m2.group(1)
            continue
        # target
        m = TARGET_RE.search(line)
        if m:
            metadata["target"] = m.group(0)
            continue
        # step
        m = STEP_RE.search(line)
        if m:
            metadata["step"] = m.group(2)
            continue
        # sample name - first line that is not comment? Heuristic: if line is not starting with # but contains alphanumeric and is short (<80 chars) and before numeric data
        # We'll capture first meaningful header line as sample if no other metadata
        # e.g. "CH020_ND_Lrls 70 BB_L" from uploaded files
        if len(line) < 100 and len(line) > 3 and "=" not in line and "Wavelength" not in line:
            # if not already has sample and line looks like identifier
            if "sample" not in metadata:
                # avoid lines that are just comments symbols
                if not line.startswith("#") and not re.match(r'^-?\d+(\.\d+)?\s', line):
                    # check if contains letters and not too many spaces
                    if re.search(r'[A-Za-z]', line) and len(line.split()) <= 6:
                        # store as sample hint, but don't override if already present
                        metadata.setdefault("sample_hint", line)
                        # also store as sample_name for backward compat
                        metadata.setdefault("sample", line)

    # Also look for common Bruker/PANalytical header keywords anywhere
    # e.g. _diffrn_radiation_type
    full_lower = content.lower()[:5000]
    if "cuka" in full_lower or "cu k" in full_lower:
        metadata.setdefault("target", "Cu Ka")
    elif "coka" in full_lower or "co k" in full_lower:
        metadata.setdefault("target", "Co Ka")

    return metadata


def _skip_ratio_issue(content: str, n_kept: int, threshold: float = 0.20) -> Optional[str]:
    """
    Warn when parsing silently dropped a large fraction of candidate data lines.

    A "candidate" line is any non-empty, non-comment line that contains at least
    one digit (i.e. looks like it was meant to be a data row). If more than
    `threshold` fraction of those candidates failed to parse into a clean
    (2theta, intensity) pair, the file's delimiter/column layout is probably not
    what the parser assumed, and the resulting plot may be built from only a
    fraction of the real data with no other visible sign of trouble.

    Returns a warning string, or None if the skip ratio is acceptable.
    """
    n_candidates = 0
    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if not stripped or _is_comment_or_empty(stripped):
            continue
        if not re.search(r'\d', stripped):
            continue
        n_candidates += 1
    if n_candidates == 0:
        return None
    skip_ratio = 1.0 - (n_kept / n_candidates)
    if skip_ratio > threshold:
        return (
            f"{skip_ratio * 100:.0f}% of candidate data lines were dropped while parsing "
            f"({n_kept} kept of {n_candidates}). The plot may be missing most of your data — "
            f"check the file's delimiter and column layout (expected 2-column 2θ/intensity)."
        )
    return None


def validate_xrd_dataframe(df: pd.DataFrame, raw_xs: Optional[List[float]] = None) -> List[str]:
    """
    Validate a parsed XRD DataFrame and return list of issues/warnings.

    Checks:
    - All intensities negative
    - Twot heta range out of typical 0-180
    - Non-monotonic raw xs before sorting
    - Large gaps in x
    - Very few points

    Args:
        df: sorted DataFrame (twotheta, intensity)
        raw_xs: optional original x list before sorting to check monotonicity

    Returns:
        List of warning strings (empty if no issues)
    """
    issues: List[str] = []
    if df.empty:
        issues.append("DataFrame is empty after parsing.")
        return issues

    x = df["twotheta"].values
    y = df["intensity"].values

    if np.all(y <= 0):
        issues.append("All intensities are <= 0 (negative or zero). Possibly column order swapped (2θ and intensity columns reversed) or baseline over-subtracted file.")

    if len(x) < 10:
        issues.append(f"Very few data points ({len(x)}). Expected at least 50-100 for a typical XRD pattern.")

    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmin < -20:
        issues.append(f"2θ min {xmin:.2f}° is unusually low (< -20°). Negative 2θ can be valid for some instruments but check calibration.")
    if xmax > 180:
        issues.append(f"2θ max {xmax:.2f}° exceeds 180°. Uncommon for lab XRD (typical 0-90° or 5-90°).")
    if xmin > 30:
        issues.append(f"2θ starts at {xmin:.2f}° (>30°). Pattern may be truncated, missing low-angle peaks.")

    # Check monotonicity of raw_xs if provided
    if raw_xs is not None and len(raw_xs) > 2:
        raw_arr = np.array(raw_xs, dtype=float)
        diffs = np.diff(raw_arr)
        # Count sign changes
        non_mono = np.sum(diffs < 0)
        if non_mono > len(raw_arr) * 0.1:  # >10% decreasing steps
            issues.append(f"Original 2θ values were non-monotonic ({non_mono} decreasing steps). Data sorted ascending for display, but check file for out-of-order rows.")

    # Gap detection
    if len(x) > 1:
        diffs_sorted = np.diff(np.sort(x))
        median_step = float(np.median(diffs_sorted)) if len(diffs_sorted) > 0 else 0
        if median_step > 0:
            large_gaps = diffs_sorted[diffs_sorted > median_step * 10]
            if len(large_gaps) > 0:
                max_gap = float(np.max(diffs_sorted))
                issues.append(f"Large gap detected in 2θ (max step {max_gap:.3f}° vs median {median_step:.4f}°). Possible missing region or concatenated files.")

    return issues


# ---------------------------------------------------------------------------
# Core parsing (existing contract preserved)
# ---------------------------------------------------------------------------

def parse_xrd_content(content: str, filename: str = "unknown") -> pd.DataFrame:
    """
    Parse XRD data from a string content.

    Args:
        content: Text content of file.
        filename: For error messages.

    Returns:
        DataFrame with columns 'twotheta','intensity', and optional 'error'

    Raises:
        XRDParseError if no valid data found.
    """
    # Phase 1: format detection first
    fmt_msg = detect_special_format(content, filename)
    if fmt_msg:
        raise XRDParseError(fmt_msg)

    data = []
    errors = []
    has_error_col = False
    raw_xs: List[float] = []
    line_num = 0
    skipped = 0

    for raw_line in content.splitlines():
        line_num += 1
        parsed = _parse_line(raw_line)
        if parsed is None:
            skipped += 1
            continue
        x, y, e = parsed
        data.append((x, y))
        raw_xs.append(x)
        if e is not None:
            errors.append(e)
            has_error_col = True
        else:
            if has_error_col:
                errors.append(np.nan)

    if len(data) < 3:
        # Provide more helpful error including possible format hint for .ras/.uxd etc
        hint = ""
        fname_lower = filename.lower()
        if fname_lower.endswith(".ras") and len(data) < 3:
            hint = " The file has .ras extension (Rigaku). Some RAS files use a binary format or have a different column layout. Try exporting as ASCII .xy from Rigaku software, or ensure file is not binary."
        if fname_lower.endswith(".uxd"):
            hint = " .uxd (Bruker) is often readable, but some UXD variants have extra header blocks. If file is not parsing, try re-exporting as .xy."
        raise XRDParseError(
            f"File '{filename}': Could not find enough numeric data "
            f"(found {len(data)} points, skipped {skipped} header/comment lines). "
            f"Expected 2-column 2θ/intensity data.{hint}"
        )

    arr = np.array(data, dtype=float)
    df = pd.DataFrame(arr, columns=["twotheta", "intensity"])

    if has_error_col:
        if len(errors) == len(df):
            df["error"] = errors
        else:
            df["error"] = pd.Series(errors).reindex(df.index)

    df = df.sort_values("twotheta").reset_index(drop=True)
    df = df.drop_duplicates(subset="twotheta", keep="last")
    df = df.dropna(subset=["intensity"])
    df = df[np.isfinite(df["twotheta"]) & np.isfinite(df["intensity"])]

    if df.empty or len(df) < 3:
        raise XRDParseError(f"File '{filename}': No valid finite data after cleaning.")

    return df


def _decode_bytes_with_fallback(raw: bytes, filename: str = "unknown") -> str:
    """
    Decode bytes with robust fallback: utf-8 -> charset-normalizer/chardet -> latin-1
    """
    # Try utf-8 strict first
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass

    # Try charset-normalizer if available
    try:
        from charset_normalizer import from_bytes as cn_from_bytes
        results = cn_from_bytes(raw)
        best = results.best()
        if best is not None:
            enc = best.encoding
            try:
                return raw.decode(enc)
            except Exception:
                pass
    except ImportError:
        pass
    except Exception:
        pass

    # Try chardet if available
    try:
        import chardet
        detected = chardet.detect(raw)
        enc = detected.get("encoding")
        if enc:
            try:
                return raw.decode(enc)
            except Exception:
                pass
    except ImportError:
        pass
    except Exception:
        pass

    # Final fallback latin-1 (never fails)
    try:
        return raw.decode("latin-1")
    except Exception as e:
        raise XRDParseError(f"File '{filename}': Cannot decode file with utf-8, charset detection, or latin-1: {e}")


def parse_xrd_file(file_input: Union[str, os.PathLike, BinaryIO, TextIO, io.BytesIO],
                   filename_hint: Optional[str] = None) -> pd.DataFrame:
    """
    Parse a file path or file-like object.

    Args:
        file_input: Path string, Path object, or file-like (has read()).
        filename_hint: Optional filename for error messages when file_input is file-like.

    Returns:
        DataFrame.

    Raises:
        XRDParseError.
    """
    fname = filename_hint or "unknown"
    content: str = ""

    if isinstance(file_input, (str, os.PathLike)):
        p = Path(file_input)
        fname = p.name
        if not p.exists():
            raise XRDParseError(f"File not found: {file_input}")
        if p.stat().st_size == 0:
            raise XRDParseError(f"File '{fname}' is empty.")
        # Read bytes then decode with fallback
        try:
            raw = p.read_bytes()
        except Exception as e:
            raise XRDParseError(f"File '{fname}': Cannot read file: {e}")

        # Early binary check
        fmt_msg = detect_special_format(raw[:4096].decode('latin-1', errors='ignore'), fname)
        if fmt_msg:
            # For binary .raw, still raise immediately
            if "\x00" in raw[:1024].decode('latin-1', errors='ignore'):
                raise XRDParseError(fmt_msg)
            # For XML, also raise
            if "XRDML" in fmt_msg or "BRML" in fmt_msg:
                # double-check by decoding as utf-8 for XML signature
                try:
                    txt_start = raw[:8192].decode('utf-8', errors='ignore')
                    fmt2 = detect_special_format(txt_start, fname)
                    if fmt2:
                        raise XRDParseError(fmt2)
                except XRDParseError:
                    raise
                except Exception:
                    pass

        content = _decode_bytes_with_fallback(raw, fname)

    else:
        # file-like
        try:
            if hasattr(file_input, "name"):
                fname = getattr(file_input, "name") or fname
            if hasattr(file_input, "getvalue"):
                raw = file_input.getvalue()
                if isinstance(raw, bytes):
                    # binary check before decode
                    txt_check = raw[:4096].decode('latin-1', errors='ignore')
                    fmt_msg = detect_special_format(txt_check, fname)
                    if fmt_msg and ("\x00" in txt_check or "XRDML" in fmt_msg or "BRML" in fmt_msg):
                        raise XRDParseError(fmt_msg)
                    content = _decode_bytes_with_fallback(raw, fname)
                else:
                    content = str(raw)
            elif hasattr(file_input, "read"):
                data = file_input.read()
                if isinstance(data, bytes):
                    txt_check = data[:4096].decode('latin-1', errors='ignore')
                    fmt_msg = detect_special_format(txt_check, fname)
                    if fmt_msg and ("\x00" in txt_check or "XRDML" in fmt_msg or "BRML" in fmt_msg):
                        raise XRDParseError(fmt_msg)
                    content = _decode_bytes_with_fallback(data, fname)
                else:
                    content = str(data)
                try:
                    file_input.seek(0)
                except Exception:
                    pass
            else:
                raise XRDParseError(f"Unsupported file input type: {type(file_input)}")
        except XRDParseError:
            raise
        except Exception as e:
            raise XRDParseError(f"File '{fname}': Failed to read uploaded file: {e}")

        if not content.strip():
            raise XRDParseError(f"File '{fname}' is empty after reading.")

    return parse_xrd_content(content, filename=fname)


# ---------------------------------------------------------------------------
# Phase 1 new API - non-breaking additions
# ---------------------------------------------------------------------------

def parse_xrd_content_with_metadata(content: str, filename: str = "unknown") -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    New in Phase 1: parse content and return (DataFrame, metadata_dict)

    Metadata includes wavelength, target, sample if found in headers.

    This is the safe replacement for parse_xrd_content when you need metadata;
    original parse_xrd_content signature unchanged.
    """
    metadata = _extract_metadata(content)
    metadata["filename"] = filename
    df = parse_xrd_content(content, filename=filename)
    return df, metadata


def parse_xrd_file_with_metadata(file_input: Union[str, os.PathLike, BinaryIO, TextIO, io.BytesIO],
                                 filename_hint: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    New in Phase 1: parse file and return (DataFrame, metadata_dict)

    Backward compatible: old parse_xrd_file still returns only DataFrame.
    Use this when you need wavelength etc.
    """
    fname = filename_hint or "unknown"
    content_for_meta = ""

    # Need content for metadata extraction - duplicate logic but reuse _decode
    if isinstance(file_input, (str, os.PathLike)):
        p = Path(file_input)
        fname = p.name
        try:
            raw = p.read_bytes()
            content_for_meta = _decode_bytes_with_fallback(raw, fname)
        except Exception:
            content_for_meta = ""
    else:
        try:
            if hasattr(file_input, "getvalue"):
                raw = file_input.getvalue()
                if isinstance(raw, bytes):
                    content_for_meta = _decode_bytes_with_fallback(raw, fname)
                else:
                    content_for_meta = str(raw)
            elif hasattr(file_input, "read"):
                data = file_input.read()
                if isinstance(data, bytes):
                    content_for_meta = _decode_bytes_with_fallback(data, fname)
                else:
                    content_for_meta = str(data)
                try:
                    file_input.seek(0)
                except Exception:
                    pass
        except Exception:
            content_for_meta = ""

    df = parse_xrd_file(file_input, filename_hint=filename_hint)
    metadata = _extract_metadata(content_for_meta) if content_for_meta else {}
    metadata["filename"] = fname if isinstance(file_input, (str, os.PathLike)) else (getattr(file_input, "name", fname) or fname)
    return df, metadata


def parse_xrd_content_with_validation(content: str, filename: str = "unknown") -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    New in Phase 1: parse with metadata and validation issues.

    Returns (df, metadata, issues_list)
    """
    metadata = _extract_metadata(content)
    metadata["filename"] = filename

    # Need raw xs before sorting for validation
    raw_xs: List[float] = []
    for line in content.splitlines():
        parsed = _parse_line(line)
        if parsed:
            raw_xs.append(parsed[0])

    df = parse_xrd_content(content, filename=filename)
    issues = validate_xrd_dataframe(df, raw_xs=raw_xs)
    skip_issue = _skip_ratio_issue(content, n_kept=len(df))
    if skip_issue:
        issues.insert(0, skip_issue)
    return df, metadata, issues


def parse_xrd_file_with_validation(file_input: Union[str, os.PathLike, BinaryIO, TextIO, io.BytesIO],
                                   filename_hint: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    New in Phase 1: wrapper returning df, metadata, issues

    Issues is List[str] warnings like non-monotonic, all-negative etc.
    """
    # Get content for metadata extraction
    content_for_meta = ""
    fname = filename_hint or "unknown"
    if isinstance(file_input, (str, os.PathLike)):
        p = Path(file_input)
        fname = p.name
        try:
            raw = p.read_bytes()
            content_for_meta = _decode_bytes_with_fallback(raw, fname)
        except Exception:
            content_for_meta = ""
    else:
        try:
            if hasattr(file_input, "getvalue"):
                raw = file_input.getvalue()
                if isinstance(raw, bytes):
                    content_for_meta = _decode_bytes_with_fallback(raw, fname)
                else:
                    content_for_meta = str(raw)
            elif hasattr(file_input, "read"):
                data = file_input.read()
                if isinstance(data, bytes):
                    content_for_meta = _decode_bytes_with_fallback(data, fname)
                else:
                    content_for_meta = str(data)
                try:
                    file_input.seek(0)
                except Exception:
                    pass
        except Exception:
            content_for_meta = ""

    df = parse_xrd_file(file_input, filename_hint=filename_hint)

    raw_xs = []
    if content_for_meta:
        for line in content_for_meta.splitlines():
            parsed = _parse_line(line)
            if parsed:
                raw_xs.append(parsed[0])
        metadata = _extract_metadata(content_for_meta)
    else:
        metadata = {}

    metadata["filename"] = fname if isinstance(file_input, (str, os.PathLike)) else (getattr(file_input, "name", fname) or fname)
    issues = validate_xrd_dataframe(df, raw_xs=raw_xs if raw_xs else None)
    if content_for_meta:
        skip_issue = _skip_ratio_issue(content_for_meta, n_kept=len(df))
        if skip_issue:
            issues.insert(0, skip_issue)
    return df, metadata, issues


# ---------------------------------------------------------------------------
# Folder / batch loading - now with optional parallel
# ---------------------------------------------------------------------------

def load_folder(folder_path: Union[str, os.PathLike],
                extensions: Optional[List[str]] = None,
                recursive: bool = False,
                parallel: bool = False,
                max_workers: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """
    Load all matching XRD files from a folder.

    Phase 1 enhancement: optional parallel loading for large folders.

    Returns:
        Dict mapping filename -> DataFrame. Skips files that fail with warning.
        If all fail, raises.

    Args:
        folder_path: Directory path.
        extensions: List of extensions to include. Defaults to SUPPORTED_EXTS.
        recursive: If True, search subfolders.
        parallel: If True, use ThreadPoolExecutor for I/O bound parallel parsing.
        max_workers: Max workers for parallel (None = auto).
    """
    if extensions is None:
        extensions = list(SUPPORTED_EXTS)
    extensions = [e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions]

    p = Path(folder_path)
    if not p.exists() or not p.is_dir():
        raise XRDParseError(f"Folder not found or not a directory: {folder_path}")

    pattern = "**/*" if recursive else "*"
    files = []
    for ext in extensions:
        files.extend(p.glob(f"{pattern}{ext}"))
        files.extend(p.glob(f"{pattern}{ext.upper()}"))
    files = sorted(set(files))

    if not files:
        raise XRDParseError(f"No files with extensions {extensions} found in folder '{folder_path}'")

    results: Dict[str, pd.DataFrame] = {}
    errors: List[str] = []

    if parallel:
        # Parallel version
        def _load_one(f: Path):
            try:
                df = parse_xrd_file(f)
                return (f.name, df, None)
            except Exception as e:
                return (f.name, None, str(e))

        workers = max_workers or min(32, (os.cpu_count() or 4) + 4)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_load_one, f): f for f in files}
            for fut in as_completed(futures):
                name, df, err = fut.result()
                if err is None and df is not None:
                    # Ensure unique name
                    key = name
                    base = key
                    counter = 1
                    while key in results:
                        stem = Path(base).stem
                        suffix = Path(base).suffix
                        key = f"{stem}_{counter}{suffix}"
                        counter += 1
                    results[key] = df
                else:
                    errors.append(f"{name}: {err}")
    else:
        # Serial (original)
        for f in files:
            try:
                df = parse_xrd_file(f)
                results[f.name] = df
            except XRDParseError as e:
                errors.append(str(e))
            except Exception as e:
                errors.append(f"{f.name}: Unexpected error {e}")

    if not results:
        raise XRDParseError(f"No valid XRD files loaded from '{folder_path}'. Errors: {'; '.join(errors[:5])}")

    return results


def load_multiple_files(file_inputs: List[Union[str, os.PathLike, BinaryIO]],
                        filename_hints: Optional[List[str]] = None,
                        parallel: bool = False,
                        max_workers: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """
    Load multiple files from a list of paths or file-like objects.

    Phase 1 enhancement: optional parallel.

    Args:
        file_inputs: List of inputs.
        filename_hints: Optional list of filenames for file-like inputs.
        parallel: If True, parallelize I/O bound parsing.
        max_workers: Max workers.

    Returns:
        Dict filename -> DataFrame.
    """
    results: Dict[str, pd.DataFrame] = {}
    errors: List[str] = []

    if parallel:
        def _load_one_idx(idx_fin):
            idx, fin = idx_fin
            hint = None
            if filename_hints and idx < len(filename_hints):
                hint = filename_hints[idx]
            try:
                df = parse_xrd_file(fin, filename_hint=hint)
                if isinstance(fin, (str, os.PathLike)):
                    key = Path(fin).name
                else:
                    key = hint or getattr(fin, "name", f"file_{idx}")
                return (idx, key, df, None)
            except Exception as e:
                name = hint or f"file_{idx}"
                return (idx, name, None, str(e))

        workers = max_workers or min(32, (os.cpu_count() or 4) + 4)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_load_one_idx, (i, fin)) for i, fin in enumerate(file_inputs)]
            for fut in as_completed(futures):
                idx, key, df, err = fut.result()
                if err is None and df is not None:
                    base_key = key
                    counter = 1
                    while key in results:
                        stem = Path(base_key).stem
                        suffix = Path(base_key).suffix
                        key = f"{stem}_{counter}{suffix}"
                        counter += 1
                    results[key] = df
                else:
                    errors.append(f"{key}: {err}")
    else:
        for idx, fin in enumerate(file_inputs):
            hint = None
            if filename_hints and idx < len(filename_hints):
                hint = filename_hints[idx]
            try:
                df = parse_xrd_file(fin, filename_hint=hint)
                if isinstance(fin, (str, os.PathLike)):
                    key = Path(fin).name
                else:
                    key = hint or getattr(fin, "name", f"file_{idx}")
                base_key = key
                counter = 1
                while key in results:
                    stem = Path(base_key).stem
                    suffix = Path(base_key).suffix
                    key = f"{stem}_{counter}{suffix}"
                    counter += 1
                results[key] = df
            except XRDParseError as e:
                errors.append(str(e))
            except Exception as e:
                name = hint or f"file_{idx}"
                errors.append(f"{name}: {e}")

    if not results and errors:
        raise XRDParseError(f"Failed to load all files: {'; '.join(errors[:5])}")

    return results


def compute_basic_stats(df: pd.DataFrame) -> Dict[str, float]:
    """Compute basic statistics for a pattern DataFrame."""
    if df.empty:
        return {}
    max_idx = df["intensity"].idxmax()
    return {
        "n_points": len(df),
        "twotheta_min": float(df["twotheta"].min()),
        "twotheta_max": float(df["twotheta"].max()),
        "twotheta_range": float(df["twotheta"].max() - df["twotheta"].min()),
        "intensity_max": float(df["intensity"].max()),
        "twotheta_at_max": float(df.loc[max_idx, "twotheta"]),
        "intensity_min": float(df["intensity"].min()),
        "intensity_mean": float(df["intensity"].mean()),
    }
