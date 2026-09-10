"""
processing.py - Advanced processing utilities for XRD patterns.

Phase 2 extended version (publication-quality):
- Extended normalization, multi-method smoothing, robust baselines
- NEW: d-spacing conversion (centralized, single source of truth)
- NEW: Peak finding & FWHM, Scherrer crystallite size
- NEW: Reference pattern loader, similarity metrics
- NEW: Processing summary sentence for Methods section
- NEW: Error propagation (approx) through smoothing
- Common-grid interpolation, difference patterns, noise estimation

All functions non-destructive, backward-compatible.
"""

from __future__ import annotations

from typing import Tuple, Optional, Literal, Dict, List, Union
import warnings
import math

import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter, find_peaks, peak_widths
    from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    SCIPY_AVAILABLE = False
    savgol_filter = None  # type: ignore
    gaussian_filter1d = None  # type: ignore
    find_peaks = None  # type: ignore

NormMethod = Literal["none", "max", "minmax", "area", "mean", "zscore", "max=1", "area=1"]
BaselineMethod = Literal["linear", "polynomial", "als", "arpls", "snip", "median"]

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_intensity(df: pd.DataFrame, method: str = "none") -> pd.DataFrame:
    if df.empty:
        return df.copy()
    m = method.lower().strip().replace(" ", "").replace("_", "")
    df_out = df.copy()
    if m in ("", "none", "off", "no", "raw"):
        return df_out
    y = df_out["intensity"].values.astype(float)
    x = df_out["twotheta"].values.astype(float) if "twotheta" in df_out else np.arange(len(y))
    if y.size == 0:
        return df_out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if m in ("max", "max=1", "max1", "max=1.0", "maximum"):
            ymax = np.nanmax(y)
            if not np.isfinite(ymax) or ymax == 0:
                return df_out
            scale = ymax
            df_out["intensity"] = y / scale
            if "error" in df_out.columns:
                df_out["error"] = df_out["error"] / scale
        elif m in ("minmax", "min-max", "0-1", "01", "minmax01"):
            ymin = np.nanmin(y)
            ymax = np.nanmax(y)
            denom = ymax - ymin
            if not np.isfinite(denom) or denom == 0:
                return df_out
            df_out["intensity"] = (y - ymin) / denom
            if "error" in df_out.columns:
                df_out["error"] = df_out["error"] / denom
        elif m in ("area", "area=1", "area1", "trapz", "int", "integral"):
            try:
                # np.trapz was removed in NumPy 2.0 in favor of np.trapezoid;
                # try the modern name first so recent NumPy gets a real
                # trapezoidal integral instead of silently falling through to
                # the cruder sum-based approximation below.
                trapezoid_fn = getattr(np, "trapezoid", None) or np.trapz
                area = trapezoid_fn(y, x)
            except Exception:
                area = np.nansum(y) * np.mean(np.diff(x)) if len(x) > 1 else np.nansum(y)
            if not np.isfinite(area) or area == 0:
                return df_out
            area = abs(area)
            df_out["intensity"] = y / area
            if "error" in df_out.columns:
                df_out["error"] = df_out["error"] / area
        elif m in ("mean", "mean=1", "mean1", "avg"):
            mean = np.nanmean(y)
            if not np.isfinite(mean) or mean == 0:
                return df_out
            df_out["intensity"] = y / mean
            if "error" in df_out.columns:
                df_out["error"] = df_out["error"] / mean
        elif m in ("zscore", "standard", "standardize", "std"):
            mean = np.nanmean(y)
            std = np.nanstd(y)
            if not np.isfinite(std) or std == 0:
                return df_out
            df_out["intensity"] = (y - mean) / std
            if "error" in df_out.columns:
                df_out["error"] = df_out["error"] / std
        else:
            pass
    return df_out


def normalize_intensity_array(y: np.ndarray, x: Optional[np.ndarray] = None, method: str = "max") -> np.ndarray:
    df = pd.DataFrame({"twotheta": x if x is not None else np.arange(len(y)), "intensity": y})
    out = normalize_intensity(df, method=method)
    return out["intensity"].values


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def _ensure_odd_window(window: int, n: int, polyorder: int) -> int:
    min_win = polyorder + 2
    if min_win % 2 == 0:
        min_win += 1
    window = int(window)
    if window < min_win:
        window = min_win
    if window % 2 == 0:
        window += 1
    if window > n:
        window = n if n % 2 == 1 else n - 1
        if window < min_win:
            return -1
    return window


def smooth_intensity(
    y: np.ndarray,
    window_length: int = 11,
    polyorder: int = 2,
    method: str = "savgol",
    sigma: float = 2.0,
    mode: str = "interp",
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return y.copy()
    method = method.lower().strip()
    try:
        if method in ("savgol", "savitzky-golay", "sg"):
            if not SCIPY_AVAILABLE or savgol_filter is None:
                return y.copy()
            polyorder = max(1, min(int(polyorder), 6))
            w = _ensure_odd_window(window_length, n, polyorder)
            if w == -1:
                return y.copy()
            return savgol_filter(y, window_length=w, polyorder=polyorder, mode=mode)
        elif method in ("gaussian", "gauss", "whittaker"):
            if SCIPY_AVAILABLE and gaussian_filter1d is not None:
                sig = float(sigma)
                if sig <= 0:
                    sig = 2.0
                return gaussian_filter1d(y, sigma=sig, mode="nearest")
            else:
                w = max(3, int(window_length) | 1)
                kernel = np.ones(w) / w
                return np.convolve(y, kernel, mode="same")
        elif method in ("moving_average", "moving", "mean", "boxcar", "uniform"):
            w = int(window_length)
            if w < 3:
                w = 3
            if w > n:
                w = n
            if SCIPY_AVAILABLE:
                return uniform_filter1d(y, size=w, mode="nearest")
            else:
                kernel = np.ones(w) / w
                return np.convolve(y, kernel, mode="same")
        elif method in ("median", "medfilt"):
            w = int(window_length)
            if w < 3:
                w = 3
            if w % 2 == 0:
                w += 1
            if w > n:
                w = n if n % 2 == 1 else n - 1
            if SCIPY_AVAILABLE and median_filter is not None:
                return median_filter(y, size=w, mode="nearest")
            else:
                y_pad = np.pad(y, (w // 2, w // 2), mode="edge")
                out = np.empty_like(y)
                for i in range(n):
                    out[i] = np.median(y_pad[i:i + w])
                return out
        else:
            if SCIPY_AVAILABLE and savgol_filter is not None:
                polyorder = max(1, min(int(polyorder), 5))
                w = _ensure_odd_window(window_length, n, polyorder)
                if w == -1:
                    return y.copy()
                return savgol_filter(y, window_length=w, polyorder=polyorder, mode=mode)
            return y.copy()
    except Exception:
        return y.copy()


def smooth_dataframe(
    df: pd.DataFrame,
    window_length: int = 11,
    polyorder: int = 2,
    method: str = "savgol",
    sigma: float = 2.0,
) -> pd.DataFrame:
    """
    Smooth intensity, with approx error propagation if error column present.

    Error propagation: smoothing reduces random noise; for simple moving average
    error ~ error / sqrt(window). For savgol/gaussian/median we approximate by
    smoothing error with same filter (conservative).
    """
    if df.empty:
        return df.copy()
    df_out = df.copy()
    y = df_out["intensity"].values.astype(float)
    y_s = smooth_intensity(y, window_length=window_length, polyorder=polyorder, method=method, sigma=sigma)
    df_out["intensity"] = y_s

    if "error" in df_out.columns:
        try:
            err = df_out["error"].values.astype(float)
            # Propagate: smooth error with same method, then scale by sqrt window for averaging methods
            err_s = smooth_intensity(err, window_length=window_length, polyorder=polyorder, method=method, sigma=sigma)
            if method in ("moving_average", "mean", "boxcar", "uniform"):
                # For boxcar averaging, error reduces by sqrt(N)
                w = max(1, int(window_length))
                err_s = err_s / math.sqrt(w)
            df_out["error"] = err_s
        except Exception:
            pass

    return df_out


# ---------------------------------------------------------------------------
# Baseline estimation
# ---------------------------------------------------------------------------

def estimate_baseline_linear(x: np.ndarray, y: np.ndarray, edge_fraction: float = 0.05, edge_mode: str = "median_low", percentile: float = 20.0) -> np.ndarray:
    n = len(y)
    if n < 2:
        return np.zeros_like(y, dtype=float)
    n_edge = max(1, int(n * edge_fraction))
    y_start_seg = y[:n_edge]
    y_end_seg = y[-n_edge:]
    def _robust(v):
        v = np.asarray(v)
        if edge_mode == "mean":
            return float(np.mean(v))
        elif edge_mode == "median":
            return float(np.median(v))
        elif edge_mode == "percentile":
            return float(np.percentile(v, percentile))
        else:
            return float(np.percentile(v, 30))
    y0 = _robust(y_start_seg)
    y1 = _robust(y_end_seg)
    x0, x1 = float(x[0]), float(x[-1])
    if x1 == x0:
        return np.full_like(y, (y0 + y1) / 2.0, dtype=float)
    baseline = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return baseline.astype(float)


def estimate_baseline_polynomial(x: np.ndarray, y: np.ndarray, degree: int = 2, num_bins: int = 20, iterative: bool = True, max_iter: int = 10, tol: float = 1e-3, clip_factor: float = 1.0) -> np.ndarray:
    n = len(y)
    if n < degree + 1:
        return np.zeros_like(y, dtype=float)
    degree = max(1, min(int(degree), 8))
    num_bins = max(degree + 1, min(int(num_bins), max(1, n // 2)))
    indices = np.array_split(np.arange(n), num_bins)
    x_min = []
    y_min = []
    for bin_idx in indices:
        if len(bin_idx) == 0:
            continue
        bin_y = y[bin_idx]
        local_idx = bin_idx[np.argmin(bin_y)]
        x_min.append(x[local_idx])
        y_min.append(y[local_idx])
    x_min = np.array(x_min, dtype=float)
    y_min = np.array(y_min, dtype=float)
    if len(x_min) <= degree:
        thresh = np.percentile(y, 20)
        mask = y <= thresh
        if np.sum(mask) > degree:
            x_min = x[mask]
            y_min = y[mask]
        else:
            x_min = x
            y_min = y
    best_baseline = None
    cur_x = x_min
    cur_y = y_min
    for it in range(max_iter if iterative else 1):
        order = np.argsort(cur_x)
        cur_x_sorted = cur_x[order]
        cur_y_sorted = cur_y[order]
        try:
            coeffs = np.polyfit(cur_x_sorted, cur_y_sorted, deg=degree)
            baseline = np.polyval(coeffs, x)
        except Exception:
            break
        best_baseline = baseline
        if not iterative:
            break
        resid = y - baseline
        neg = resid[resid < 0]
        if len(neg) == 0:
            break
        std_neg = np.std(neg)
        mean_neg = np.mean(neg)
        keep_mask = resid < (mean_neg + clip_factor * std_neg)
        if np.sum(keep_mask) < degree + 5:
            break
        new_x = x[keep_mask]
        new_y = y[keep_mask]
        if len(new_x) == len(cur_x) and np.allclose(new_x, cur_x):
            break
        cur_x, cur_y = new_x, new_y
        if it > 0 and best_baseline is not None:
            prev = np.polyval(np.polyfit(cur_x_sorted, cur_y_sorted, degree), x) if len(cur_x_sorted) > degree else baseline
            if np.mean((baseline - prev) ** 2) < tol:
                break
    if best_baseline is None:
        return estimate_baseline_linear(x, y)
    return best_baseline


def baseline_als(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10, return_weights: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if not SCIPY_AVAILABLE:
        return np.zeros_like(y)
    y = np.asarray(y, dtype=float)
    L = len(y)
    if L < 3:
        return np.zeros_like(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), dtype=float)
    D = lam * D.dot(D.transpose())
    w = np.ones(L, dtype=float)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + D
        try:
            z = spsolve(Z.tocsc(), w * y)
        except Exception:
            try:
                Z_dense = Z.toarray()
                z = np.linalg.solve(Z_dense, w * y)
            except Exception:
                z = np.zeros_like(y)
                break
        w = p * (y > z) + (1 - p) * (y < z)
    if return_weights:
        return z, w
    return z


def baseline_arpls(y: np.ndarray, lam: float = 1e5, ratio: float = 1e-6, niter: int = 100) -> np.ndarray:
    if not SCIPY_AVAILABLE:
        return baseline_als(y, lam=lam, p=0.01, niter=10)
    y = np.asarray(y, dtype=float)
    L = len(y)
    if L < 3:
        return np.zeros_like(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), dtype=float)
    D = lam * D.dot(D.transpose())
    w = np.ones(L, dtype=float)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + D
        try:
            z = spsolve(Z.tocsc(), w * y)
        except Exception:
            try:
                z = np.linalg.solve(Z.toarray(), w * y)
            except Exception:
                return baseline_als(y, lam=lam, niter=10)
        d = y - z
        dn = d[d < 0]
        if len(dn) == 0:
            dn = d
        m = np.mean(dn)
        s = np.std(dn)
        if s == 0:
            s = 1.0
        arg = 2 * (d - (2 * s - m)) / s
        arg = np.clip(arg, -30, 30)
        w_new = 1.0 / (1.0 + np.exp(arg))
        if np.linalg.norm(w - w_new) / (np.linalg.norm(w) + 1e-12) < ratio:
            w = w_new
            break
        w = w_new
    return z


def baseline_snip(y: np.ndarray, iterations: int = 24, window: Optional[int] = None, order: int = 2) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return np.zeros_like(y)
    if window is None:
        window = max(10, min(100, n // 15))
    window = int(window)
    baseline = y.copy()
    for p in range(iterations, 0, -1):
        w = max(1, min(window, p))
        left = np.empty_like(baseline)
        right = np.empty_like(baseline)
        left[w:] = baseline[:-w]
        left[:w] = baseline[:w]
        right[:-w] = baseline[w:]
        right[-w:] = baseline[-w:]
        candidate = (left + right) * 0.5
        baseline = np.minimum(baseline, candidate)
    return baseline


def baseline_median(y: np.ndarray, window: int = 151, smooth_window: int = 51, sigma: float = 5.0) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return np.zeros_like(y)
    w = int(window)
    if w % 2 == 0:
        w += 1
    w = max(3, min(w, n if n % 2 == 1 else n - 1))
    if SCIPY_AVAILABLE and median_filter is not None:
        med = median_filter(y, size=w, mode="nearest")
        if gaussian_filter1d is not None:
            med = gaussian_filter1d(med, sigma=sigma, mode="nearest")
        return med
    else:
        if SCIPY_AVAILABLE:
            return uniform_filter1d(y, size=w, mode="nearest")
        else:
            kernel = np.ones(w) / w
            return np.convolve(y, kernel, mode="same")


def estimate_baseline(x: np.ndarray, y: np.ndarray, method: str = "linear", poly_degree: int = 2, lam: float = 1e5, p: float = 0.01, niter: int = 20, snip_iterations: int = 24, median_window: int = 151) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return np.zeros_like(y)
    method = method.lower().strip()
    if method in ("poly", "polynom", "polynomial"):
        method = "polynomial"
    elif method in ("als", "asymmetric", "asls"):
        method = "als"
    elif method in ("arpls", "arp", "reweighted"):
        method = "arpls"
    elif method in ("snip", "clip", "lls"):
        method = "snip"
    elif method in ("median", "med", "rolling_median", "rolling"):
        method = "median"
    else:
        method = "linear" if method not in ("linear", "polynomial", "als", "arpls", "snip", "median") else method
    try:
        if method == "linear":
            return estimate_baseline_linear(x, y, edge_fraction=0.05, edge_mode="median_low")
        elif method == "polynomial":
            return estimate_baseline_polynomial(x, y, degree=poly_degree, num_bins=25, iterative=True)
        elif method == "als":
            return baseline_als(y, lam=lam, p=p, niter=niter)
        elif method == "arpls":
            return baseline_arpls(y, lam=lam, niter=niter)
        elif method == "snip":
            return baseline_snip(y, iterations=snip_iterations, window=None)
        elif method == "median":
            return baseline_median(y, window=median_window, sigma=5.0)
        else:
            return estimate_baseline_linear(x, y)
    except Exception:
        return estimate_baseline_linear(x, y)


# ---------------------------------------------------------------------------
# Phase 2 NEW: d-spacing utilities (centralized source of truth)
# ---------------------------------------------------------------------------

def two_theta_to_d(two_theta_deg: Union[float, np.ndarray], wavelength: float = 1.5406) -> Union[float, np.ndarray]:
    """
    Convert 2θ (degrees) to d-spacing (Å) using Bragg's law: λ = 2d sinθ

    Args:
        two_theta_deg: 2θ value(s) in degrees
        wavelength: X-ray wavelength in Å (Cu Ka 1.5406, Co 1.79026, etc.)

    Returns:
        d-spacing in same units as wavelength (Å if wavelength in Å). NaN where sinθ <= 0 or invalid.
    """
    tt = np.asarray(two_theta_deg, dtype=float)
    theta_rad = np.radians(tt / 2.0)
    sin_theta = np.sin(theta_rad)
    # Avoid division by zero or sin > 1
    with np.errstate(divide='ignore', invalid='ignore'):
        d = wavelength / (2.0 * sin_theta)
        d = np.where(sin_theta < 1e-6, np.nan, d)
        # For |sin| > 1 due to numerical, nan
        d = np.where(np.abs(sin_theta) > 1.0, np.nan, d)
    if np.ndim(two_theta_deg) == 0:
        return float(d) if np.ndim(d) == 0 else float(d.flat[0]) if d.size == 1 else d
    return d


def d_to_two_theta(d_spacing: Union[float, np.ndarray], wavelength: float = 1.5406) -> Union[float, np.ndarray]:
    """
    Inverse: d-spacing (Å) to 2θ (degrees): sinθ = λ/(2d), 2θ = 2*arcsin(...)

    Returns NaN where λ/(2d) > 1 (physically impossible, d too small for wavelength).
    """
    d_arr = np.asarray(d_spacing, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        sin_theta = wavelength / (2.0 * d_arr)
        # Clip for arcsin domain
        valid = np.abs(sin_theta) <= 1.0
        theta_rad = np.where(valid, np.arcsin(sin_theta), np.nan)
        two_theta = np.degrees(theta_rad * 2.0)
    if np.ndim(d_spacing) == 0:
        return float(two_theta) if np.ndim(two_theta) == 0 else float(two_theta.flat[0]) if two_theta.size == 1 else two_theta
    return two_theta


def add_d_spacing_column(df: pd.DataFrame, wavelength: float = 1.5406) -> pd.DataFrame:
    """Return copy of df with added 'd_spacing' column (Å) computed from twotheta."""
    if df.empty or "twotheta" not in df.columns:
        return df.copy()
    out = df.copy()
    out["d_spacing"] = two_theta_to_d(out["twotheta"].values, wavelength=wavelength)
    return out


# ---------------------------------------------------------------------------
# Phase 2 NEW: Peak finding & Scherrer
# ---------------------------------------------------------------------------

def _estimate_fwhm_single_peak(x: np.ndarray, y: np.ndarray, peak_idx: int, baseline: float = 0.0) -> float:
    """
    Estimate FWHM for a single peak at peak_idx.

    Finds left/right crossing at half-max (y_half = baseline + (y_peak - baseline)/2).
    Linear interpolation between points for sub-step accuracy.

    Returns FWHM in same units as x (degrees if x is 2θ). NaN if cannot estimate.
    """
    n = len(y)
    if peak_idx <= 0 or peak_idx >= n - 1:
        return float("nan")
    y_peak = y[peak_idx]
    if y_peak <= baseline:
        return float("nan")
    half = baseline + (y_peak - baseline) * 0.5

    # Left side search
    left_idx = peak_idx
    while left_idx > 0 and y[left_idx] > half:
        left_idx -= 1
    # Interpolate between left_idx and left_idx+1
    if left_idx < 0 or left_idx + 1 >= n:
        return float("nan")
    x0, x1 = x[left_idx], x[left_idx + 1]
    y0, y1 = y[left_idx], y[left_idx + 1]
    if y1 == y0:
        x_left = x0
    else:
        # linear interpolation: y = y0 + (x - x0)*(y1-y0)/(x1-x0) = half => x = x0 + (half - y0)*(x1-x0)/(y1-y0)
        x_left = x0 + (half - y0) * (x1 - x0) / (y1 - y0) if (y1 - y0) != 0 else x0

    # Right side
    right_idx = peak_idx
    while right_idx < n - 1 and y[right_idx] > half:
        right_idx += 1
    if right_idx <= 0 or right_idx >= n:
        return float("nan")
    x0, x1 = x[right_idx - 1], x[right_idx]
    y0, y1 = y[right_idx - 1], y[right_idx]
    if y1 == y0:
        x_right = x1
    else:
        x_right = x0 + (half - y0) * (x1 - x0) / (y1 - y0) if (y1 - y0) != 0 else x1

    fwhm = x_right - x_left
    if fwhm <= 0 or not np.isfinite(fwhm):
        return float("nan")
    return float(fwhm)


def find_peaks_xrd(
    df: pd.DataFrame,
    prominence: Optional[float] = None,
    prominence_factor: float = 3.0,
    min_distance_deg: float = 0.3,
    height: Optional[float] = None,
    min_height_factor: float = 0.05,
    wavelength: Optional[float] = None,
    estimate_fwhm: bool = True,
) -> pd.DataFrame:
    """
    Find peaks in XRD pattern with sensible defaults.

    Uses noise estimate to set prominence if not provided:
      prominence = max(prominence_factor * noise_std, min_height)

    Args:
        df: DataFrame with twotheta, intensity (must be sorted ascending)
        prominence: minimum prominence in intensity units (if None, auto from noise)
        prominence_factor: factor times noise_std if prominence auto
        min_distance_deg: minimum separation between peaks in degrees 2θ (converted to points)
        height: minimum peak height absolute
        min_height_factor: relative factor of max intensity if height not given
        wavelength: if given, compute d-spacing column
        estimate_fwhm: whether to compute FWHM per peak

    Returns:
        DataFrame with columns: twotheta, intensity, fwhm (if estimate), d_spacing (if wavelength),
        prominence, peak_idx (original index)
    """
    if df.empty or len(df) < 5:
        return pd.DataFrame(columns=["twotheta", "intensity", "fwhm", "d_spacing", "prominence", "peak_idx"])

    if not SCIPY_AVAILABLE or find_peaks is None:
        # Fallback: no scipy, return empty with warning? return max only
        max_idx = int(df["intensity"].idxmax())
        row = {
            "twotheta": float(df.loc[max_idx, "twotheta"]),
            "intensity": float(df.loc[max_idx, "intensity"]),
            "fwhm": float("nan"),
            "d_spacing": float("nan"),
            "prominence": float("nan"),
            "peak_idx": max_idx,
        }
        return pd.DataFrame([row])

    x = df["twotheta"].values.astype(float)
    y = df["intensity"].values.astype(float)

    # Estimate noise and step
    try:
        noise = estimate_noise_std(y, method="mad")
    except Exception:
        noise = float(np.std(y) * 0.1)

    if prominence is None:
        # Use factor * noise, with scale-relative floor so normalized data (max=1) still finds peaks
        ymax = float(np.max(y)) if len(y) else 1.0
        if not np.isfinite(ymax) or ymax <= 0:
            ymax = 1.0
        prominence = max(
            prominence_factor * (noise if noise > 0 else np.std(y) * 0.1),
            ymax * 0.005,
        )

    if height is None:
        height = float(np.max(y) * min_height_factor)

    # Convert min_distance_deg to points
    median_step = float(np.median(np.diff(x))) if len(x) > 1 else 0.02
    if median_step <= 0:
        median_step = 0.02
    min_distance_pts = max(1, int(min_distance_deg / median_step))

    # Find peaks
    peaks_idx, props = find_peaks(y, prominence=prominence, distance=min_distance_pts, height=height)

    if len(peaks_idx) == 0:
        return pd.DataFrame(columns=["twotheta", "intensity", "fwhm", "d_spacing", "prominence", "peak_idx", "height"])

    prominences = props.get("prominences", np.full(len(peaks_idx), np.nan))
    heights = props.get("peak_heights", y[peaks_idx])

    rows = []
    for pi, pk_idx in enumerate(peaks_idx):
        tt = float(x[pk_idx])
        intens = float(y[pk_idx])
        prom = float(prominences[pi]) if pi < len(prominences) else float("nan")
        h = float(heights[pi]) if pi < len(heights) else intens

        fwhm = float("nan")
        if estimate_fwhm:
            try:
                # Use local baseline as min in window around peak
                window_pts = max(10, int(2.0 / median_step))  # 2 deg window
                lo = max(0, pk_idx - window_pts)
                hi = min(len(y), pk_idx + window_pts + 1)
                local_min = float(np.min(y[lo:hi]))
                # baseline ~ local_min, but not too high
                baseline = local_min
                fwhm = _estimate_fwhm_single_peak(x, y, pk_idx, baseline=baseline)
            except Exception:
                fwhm = float("nan")

        d_sp = float("nan")
        if wavelength is not None and wavelength > 0:
            try:
                d_sp = float(two_theta_to_d(tt, wavelength=wavelength))
            except Exception:
                pass

        rows.append({
            "twotheta": tt,
            "intensity": intens,
            "fwhm": fwhm,
            "d_spacing": d_sp,
            "prominence": prom,
            "height": h,
            "peak_idx": int(pk_idx),
        })

    peaks_df = pd.DataFrame(rows)
    # Sort by intensity descending (most intense first) but keep twotheta order optional? Provide both sorted by 2theta for display
    # Return sorted by twotheta ascending for natural order
    peaks_df = peaks_df.sort_values("twotheta").reset_index(drop=True)
    return peaks_df


def deconvolve_instrumental_fwhm(fwhm_obs_deg: float, instrumental_fwhm_deg: float = 0.0) -> float:
    """
    Remove instrumental broadening from an observed FWHM in quadrature:
        beta_sample = sqrt(beta_observed^2 - beta_instrumental^2)

    This is the standard (Gaussian-approximation) correction crystallographers apply
    before a Scherrer calculation: the measured peak width is a convolution of the
    true sample broadening with the instrument's own resolution function, so simply
    subtracting the instrumental FWHM directly would overcorrect.

    Args:
        fwhm_obs_deg: measured FWHM in degrees 2theta.
        instrumental_fwhm_deg: FWHM of an instrumental standard (e.g. LaB6, Si) at
            a comparable 2theta, in degrees. 0 (default) disables the correction
            and returns fwhm_obs_deg unchanged.

    Returns:
        Corrected FWHM in degrees, or NaN if the observed peak is at or narrower
        than the instrumental resolution (i.e. unresolved from the instrument
        limit — no meaningful sample broadening can be extracted).
    """
    if not np.isfinite(fwhm_obs_deg):
        return float("nan")
    if instrumental_fwhm_deg is None or instrumental_fwhm_deg <= 0:
        return fwhm_obs_deg
    diff_sq = fwhm_obs_deg ** 2 - instrumental_fwhm_deg ** 2
    if diff_sq <= 0:
        return float("nan")
    return float(math.sqrt(diff_sq))


def estimate_shift_by_xcorr(
    df_ref: pd.DataFrame,
    df_target: pd.DataFrame,
    step: float = 0.01,
    max_shift_deg: float = 2.0,
    twotheta_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> float:
    """
    Estimate the 2theta shift (in degrees) that best aligns df_target onto df_ref,
    via cross-correlation on a common interpolated grid.

    This corrects the classic sample-displacement / zero-point error: two patterns
    of the same phase recorded on different days (or slightly different sample
    height) show the same peaks shifted by a near-constant offset in 2theta. Rather
    than eyeballing it, we interpolate both patterns onto a shared fine grid, then
    slide df_target over a +/- max_shift_deg window and pick the offset that
    maximizes the cross-correlation with df_ref.

    Returns:
        Shift in degrees 2theta to ADD to df_target's twotheta values to best align
        it with df_ref (0.0 if it cannot be estimated, e.g. too little overlap).
    """
    if df_ref.empty or df_target.empty:
        return 0.0
    xmin_r, xmax_r = None, None
    if twotheta_range is not None:
        xmin_r, xmax_r = twotheta_range
    common_x, interp = interpolate_to_common_grid(
        {"ref": df_ref, "tgt": df_target}, step=step, xmin=xmin_r, xmax=xmax_r, method="linear"
    )
    if common_x.size < 10:
        return 0.0
    y_ref = interp["ref"]
    y_tgt = interp["tgt"]
    mask = np.isfinite(y_ref) & np.isfinite(y_tgt)
    if np.sum(mask) < 10:
        return 0.0
    # Zero-mean so correlation isn't dominated by DC offset / differing backgrounds
    yr = np.where(mask, y_ref, 0.0)
    yt = np.where(mask, y_tgt, 0.0)
    yr = yr - np.mean(yr[mask])
    yt = yt - np.mean(yt[mask])
    max_lag_pts = max(1, int(round(max_shift_deg / step)))
    try:
        corr = np.correlate(yr, yt, mode="full")
    except Exception:
        return 0.0
    n = len(yr)
    lags = np.arange(-(n - 1), n)
    center = len(corr) // 2
    lo = max(0, center - max_lag_pts)
    hi = min(len(corr), center + max_lag_pts + 1)
    window_corr = corr[lo:hi]
    window_lags = lags[lo:hi]
    if window_corr.size == 0:
        return 0.0
    best_lag = window_lags[np.argmax(window_corr)]
    # np.correlate(yr, yt) peak at lag L means yt shifted by +L (in points) matches yr
    shift_deg = float(best_lag) * step
    return shift_deg


def scherrer_crystallite_size(
    fwhm_deg: float,
    two_theta_deg: float,
    wavelength: float = 1.5406,
    K: float = 0.9,
) -> float:
    """
    Scherrer equation: crystallite size D = K λ / (β cosθ)

    Args:
        fwhm_deg: FWHM in degrees 2θ (must be in radians for formula, we convert)
        two_theta_deg: peak position 2θ in degrees
        wavelength: X-ray wavelength in Å (Cu Ka 1.5406 Å). If you pass nm, result will be in nm.
        K: shape factor (0.9 typical)

    Returns:
        Crystallite size in same units as wavelength scaled to nm if wavelength in Å.
        We convert Å → nm (/10) for convenience, so if wavelength=1.5406 Å, result in nm.
        If FWHM invalid, returns NaN.
    """
    if not np.isfinite(fwhm_deg) or not np.isfinite(two_theta_deg) or fwhm_deg <= 0:
        return float("nan")
    # Convert FWHM to radians (β is in radians)
    beta_rad = math.radians(fwhm_deg)
    theta_rad = math.radians(two_theta_deg / 2.0)
    cos_theta = math.cos(theta_rad)
    if cos_theta == 0 or not np.isfinite(cos_theta):
        return float("nan")
    # D in same units as wavelength (Å)
    D_angstrom = K * wavelength / (beta_rad * cos_theta)
    # Convert Å to nm
    D_nm = D_angstrom / 10.0
    return float(D_nm)


# ---------------------------------------------------------------------------
# Phase 2 NEW: Reference pattern support
# ---------------------------------------------------------------------------

def load_reference_pattern(file_path: Union[str, "os.PathLike"],
                           has_hkl: bool = True) -> pd.DataFrame:
    """
    Load reference stick pattern (ICDD/COD style) for overlay.

    Expected CSV with columns: two_theta, intensity, hkl (optional)
    or 2-column xy (stick positions). Supports same flexible parsing as xrd_io
    but also keeps hkl if string column.

    Args:
        file_path: path to csv/xy file
        has_hkl: if True, attempt to parse 3rd column as hkl string

    Returns:
        DataFrame with twotheta, intensity, hkl (if present), normalized intensity max=1
    """
    from pathlib import Path
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Reference file not found: {file_path}")

    # Try to parse with pandas first for hkl handling
    try:
        # Peek first lines to infer delimiter
        content = p.read_text(encoding="utf-8", errors="ignore")[:5000]
        # If contains hkl or non-numeric third column, use csv reading with handling
        lines = []
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("%"):
                continue
            lines.append(line)

        # Try to detect separator and load via pandas with up to 3 columns
        # Use python engine, allow string in 3rd column
        try:
            # Quick attempt: read as csv with flexible sep
            df_tmp = pd.read_csv(p, comment="#", header=None, sep=None, engine="python", dtype=str)
            # Try convert first two cols to float, keep third as object
            if df_tmp.shape[1] >= 2:
                # Coerce first two
                df_tmp.iloc[:, 0] = pd.to_numeric(df_tmp.iloc[:, 0], errors="coerce")
                df_tmp.iloc[:, 1] = pd.to_numeric(df_tmp.iloc[:, 1], errors="coerce")
                # Drop rows where first two are NaN
                df_tmp = df_tmp.dropna(subset=[0, 1])
                if not df_tmp.empty:
                    out = pd.DataFrame({
                        "twotheta": df_tmp.iloc[:, 0].astype(float),
                        "intensity": df_tmp.iloc[:, 1].astype(float),
                    })
                    if has_hkl and df_tmp.shape[1] >= 3:
                        out["hkl"] = df_tmp.iloc[:, 2].astype(str)
                    out = out.sort_values("twotheta").reset_index(drop=True)
                    # Normalize reference to max=1 for tick height consistency
                    max_i = out["intensity"].max()
                    if max_i > 0:
                        out["intensity"] = out["intensity"] / max_i
                    return out
        except Exception:
            pass
    except Exception:
        pass

    # Fallback: use xrd_io parser (2-col)
    try:
        from xrd_io import parse_xrd_file as xrd_parse
        df = xrd_parse(p)
        # Normalize
        df = df.sort_values("twotheta").reset_index(drop=True)
        max_i = df["intensity"].max()
        if max_i > 0:
            df["intensity"] = df["intensity"] / max_i
        return df
    except Exception as e:
        raise ValueError(f"Could not parse reference pattern {file_path}: {e}")


# ---------------------------------------------------------------------------
# Phase 2 NEW: Similarity metrics
# ---------------------------------------------------------------------------

def pattern_similarity(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    method: str = "pearson",
    step: float = 0.02,
    twotheta_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> float:
    """
    Quantitative similarity between two patterns on common grid.

    Args:
        df_a, df_b: DataFrames with twotheta, intensity
        method: "pearson" (correlation) or "cosine" (cosine similarity)
        step: common grid step for interpolation
        twotheta_range: optional (xmin, xmax) to restrict comparison

    Returns:
        similarity float: Pearson r between -1 and 1, or cosine 0-1. NaN if not computable.
    """
    if df_a.empty or df_b.empty:
        return float("nan")

    # Note: twotheta_range filtering is handled below after interpolation via range_mask.
    # (Removed broken relative import that caused ImportError when running as script)

    # Use interpolate_to_common_grid
    common_x, interp = interpolate_to_common_grid({"a": df_a, "b": df_b}, step=step, method="linear")

    if common_x.size == 0:
        return float("nan")

    ya = interp["a"]
    yb = interp["b"]

    # Mask NaNs
    mask = np.isfinite(ya) & np.isfinite(yb)
    if np.sum(mask) < 10:
        return float("nan")

    ya = ya[mask]
    yb = yb[mask]

    # If range specified, further restrict to range
    if twotheta_range is not None:
        xmin, xmax = twotheta_range
        range_mask = np.ones(len(common_x[mask]), dtype=bool)
        if xmin is not None:
            range_mask &= common_x[mask] >= xmin
        if xmax is not None:
            range_mask &= common_x[mask] <= xmax
        ya = ya[range_mask]
        yb = yb[range_mask]
        if len(ya) < 5:
            return float("nan")

    method = method.lower().strip()
    try:
        if method in ("pearson", "correlation", "r"):
            # Pearson correlation
            if np.std(ya) == 0 or np.std(yb) == 0:
                return float("nan")
            corr = np.corrcoef(ya, yb)[0, 1]
            return float(corr)
        elif method in ("cosine", "cos"):
            # Cosine similarity
            norm_a = np.linalg.norm(ya)
            norm_b = np.linalg.norm(yb)
            if norm_a == 0 or norm_b == 0:
                return float("nan")
            cos_sim = float(np.dot(ya, yb) / (norm_a * norm_b))
            return cos_sim
        elif method in ("euclidean", "l2", "rms"):
            # Return negative distance? For similarity we want high = similar, so invert
            # Return 1/(1+RMSE)
            rmse = float(np.sqrt(np.mean((ya - yb) ** 2)))
            return float(1.0 / (1.0 + rmse))
        else:
            # default pearson
            return float(np.corrcoef(ya, yb)[0, 1])
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Phase 2 NEW: Processing summary for Methods paragraph
# ---------------------------------------------------------------------------

def processing_summary(params: Dict) -> str:
    """
    Generate human-readable Methods sentence from processing params.

    Args:
        params: dict with keys like:
            normalize_method, smoothing_enabled, smoothing_method, smoothing_window,
            smoothing_polyorder, baseline_enabled, baseline_method, baseline_degree,
            baseline_lam, twotheta_range (tuple), wavelength, etc.
            Values as used in process_pattern / app.py.

    Returns:
        Single sentence / paragraph suitable for SI Methods.

    Example:
        {"normalize_method":"max=1","smoothing_enabled":True,"smoothing_method":"savgol",
         "smoothing_window":11,"smoothing_polyorder":2,"baseline_enabled":True,
         "baseline_method":"arpls","baseline_lam":1e5,"twotheta_range":(10,70),
         "wavelength":1.5406}

        -> "XRD patterns were baseline-corrected using ARPLS (λ=1e5), smoothed with
            Savitzky-Golay (window=11, polyorder=2), normalized to maximum intensity (max=1),
            and cropped to 2θ range 10.0–70.0°, measured with Cu Kα radiation (λ=1.5406 Å)."
    """
    parts = []

    # Wavelength
    wl = params.get("wavelength")
    wl_str = ""
    if wl is not None:
        try:
            wl_f = float(wl)
            # Guess target from typical wavelengths
            target_str = "X-ray"
            if 1.54 <= wl_f <= 1.55:
                target_str = "Cu Kα"
            elif 1.79 <= wl_f <= 1.80:
                target_str = "Co Kα"
            elif 0.7 <= wl_f <= 0.72:
                target_str = "Mo Kα"
            wl_str = f"measured with {target_str} radiation (λ={wl_f:.4f} Å)"
        except Exception:
            wl_str = f"measured with λ={wl} Å"

    # Baseline
    if params.get("baseline_enabled"):
        b_method = str(params.get("baseline_method", "linear")).lower()
        if b_method == "polynomial":
            deg = params.get("baseline_degree", 2)
            parts.append(f"baseline-corrected using polynomial (degree={deg})")
        elif b_method in ("als", "arpls"):
            lam = params.get("baseline_lam", params.get("baseline_lambda", "1e5"))
            # Format lam nicely
            try:
                lam_f = float(lam)
                lam_s = f"{lam_f:.0e}"
            except Exception:
                lam_s = str(lam)
            method_name = "ARPLS" if b_method == "arpls" else "ALS"
            if b_method == "als":
                p_val = params.get("baseline_p", 0.01)
                parts.append(f"baseline-corrected using {method_name} (λ={lam_s}, p={p_val})")
            else:
                parts.append(f"baseline-corrected using {method_name} (λ={lam_s})")
        elif b_method == "snip":
            it = params.get("snip_iterations", 24)
            parts.append(f"baseline-corrected using SNIP (iterations={it})")
        elif b_method == "median":
            win = params.get("median_window", 151)
            parts.append(f"baseline-corrected using rolling median (window={win})")
        else:
            parts.append(f"baseline-corrected using linear")

        if params.get("clip_negative_after_baseline"):
            # Append to last part
            if parts:
                parts[-1] += ", with negative intensities clipped to zero"

    # Smoothing
    if params.get("smoothing_enabled"):
        s_method = str(params.get("smoothing_method", params.get("smoothing_method", "savgol"))).lower()
        win = params.get("smoothing_window", 11)
        poly = params.get("smoothing_polyorder", 2)
        sigma = params.get("smoothing_sigma", 2.0)
        if "savgol" in s_method:
            parts.append(f"smoothed with Savitzky-Golay (window={win}, polyorder={poly})")
        elif "gauss" in s_method:
            parts.append(f"smoothed with Gaussian (σ={sigma})")
        elif "mov" in s_method:
            parts.append(f"smoothed with moving average (window={win})")
        elif "median" in s_method:
            parts.append(f"smoothed with median filter (window={win})")
        else:
            parts.append(f"smoothed (method={s_method}, window={win})")

    # Normalization
    norm = str(params.get("normalize_method", "none")).lower()
    if norm in ("max=1", "max", "max1"):
        parts.append("normalized to maximum intensity (max=1)")
    elif norm in ("area=1", "area", "area1"):
        parts.append("normalized to integrated area (area=1)")
    elif "minmax" in norm:
        parts.append("normalized to 0–1 range (min-max)")
    elif "mean" in norm:
        parts.append("normalized to mean intensity (mean=1)")
    elif "zscore" in norm:
        parts.append("standardized (z-score)")

    # Crop range
    trange = params.get("twotheta_range")
    if trange is not None:
        try:
            xmin, xmax = trange
            if xmin is not None and xmax is not None:
                parts.append(f"cropped to 2θ range {float(xmin):.1f}–{float(xmax):.1f}°")
            elif xmin is not None:
                parts.append(f"cropped to 2θ ≥ {float(xmin):.1f}°")
            elif xmax is not None:
                parts.append(f"cropped to 2θ ≤ {float(xmax):.1f}°")
        except Exception:
            pass

    # Compose sentence
    if not parts:
        base = "XRD patterns were plotted without additional processing"
    else:
        # Join with commas, last with "and"
        if len(parts) == 1:
            base = f"XRD patterns were {parts[0]}"
        else:
            base = f"XRD patterns were {', '.join(parts[:-1])}, and {parts[-1]}"

    if wl_str:
        base += f", {wl_str}"

    base += "."

    # Capitalize first letter already, but ensure
    return base[0].upper() + base[1:] if base else base


# ---------------------------------------------------------------------------
# Other utilities (remaining from Phase 1)
# ---------------------------------------------------------------------------

def subtract_baseline(df: pd.DataFrame, baseline: np.ndarray, clip_negative: bool = False) -> pd.DataFrame:
    if len(baseline) != len(df):
        return df.copy()
    df_out = df.copy()
    y = df_out["intensity"].values.astype(float) - baseline.astype(float)
    if clip_negative:
        y = np.maximum(y, 0.0)
    df_out["intensity"] = y
    # Error column stays same (baseline uncertainty not propagated - documented as approx)
    return df_out


def apply_offset(y: np.ndarray, offset: float) -> np.ndarray:
    return np.asarray(y, dtype=float) + float(offset)


def crop_twotheta_range(df: pd.DataFrame, xmin: Optional[float] = None, xmax: Optional[float] = None) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    mask = np.ones(len(df), dtype=bool)
    vals = df["twotheta"].values
    if xmin is not None:
        mask &= vals >= xmin
    if xmax is not None:
        mask &= vals <= xmax
    return df.loc[mask].reset_index(drop=True)


def interpolate_to_common_grid(data_dict: Dict[str, pd.DataFrame], step: Optional[float] = None, xmin: Optional[float] = None, xmax: Optional[float] = None, method: str = "linear") -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if not data_dict:
        return np.array([]), {}
    mins = []
    maxs = []
    steps = []
    for df in data_dict.values():
        if df.empty or len(df) < 2:
            continue
        mins.append(float(df["twotheta"].min()))
        maxs.append(float(df["twotheta"].max()))
        diffs = np.diff(df["twotheta"].values)
        if len(diffs) > 0:
            steps.append(float(np.median(diffs)))
    if not mins:
        return np.array([]), {}
    if xmin is None:
        xmin = min(mins)
    if xmax is None:
        xmax = max(maxs)
    if step is None:
        step = float(np.median(steps)) if steps else 0.02
        step = max(step, 0.001)
    common_x = np.arange(xmin, xmax + step / 2, step)
    interpolated: Dict[str, np.ndarray] = {}
    for name, df in data_dict.items():
        if df.empty:
            interpolated[name] = np.full_like(common_x, np.nan, dtype=float)
            continue
        x = df["twotheta"].values.astype(float)
        y = df["intensity"].values.astype(float)
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        try:
            f = interp1d(x, y, kind=method, bounds_error=False, fill_value=np.nan, assume_sorted=True) if SCIPY_AVAILABLE else None
            if f is not None:
                y_interp = f(common_x)
            else:
                y_interp = np.interp(common_x, x, y, left=np.nan, right=np.nan)
        except Exception:
            y_interp = np.interp(common_x, x, y, left=np.nan, right=np.nan)
        interpolated[name] = y_interp
    return common_x, interpolated


def compute_difference(df_a: pd.DataFrame, df_b: pd.DataFrame, step: float = 0.02) -> pd.DataFrame:
    common_x, interp = interpolate_to_common_grid({"a": df_a, "b": df_b}, step=step, method="linear")
    if common_x.size == 0:
        return pd.DataFrame(columns=["twotheta", "intensity"])
    diff = interp["a"] - interp["b"]
    return pd.DataFrame({"twotheta": common_x, "intensity": diff})


def estimate_noise_std(y: np.ndarray, method: str = "mad") -> float:
    y = np.asarray(y, dtype=float)
    if len(y) < 5:
        return 0.0
    if method == "mad":
        diff = np.diff(y)
        med = np.median(diff)
        mad = np.median(np.abs(diff - med))
        return float(mad * 1.4826)
    else:
        y_s = smooth_intensity(y, window_length=21, polyorder=2, method="savgol")
        resid = y - y_s
        return float(np.std(resid))


def estimate_poisson_error(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a Poisson counting-statistics error column (sigma = sqrt(N)) when the
    file didn't come with its own 'error' column.

    Only valid for raw photon/count-detector intensities — this must be
    applied BEFORE any normalization or baseline subtraction, since sqrt(N)
    is a property of the original counts, not of a rescaled value. Callers
    that then normalize will correctly propagate this error alongside the
    intensity (see normalize_intensity), because it is added first here.

    Returns a copy; does nothing (returns df unchanged) if an 'error' column
    already exists, so this never overwrites a file's own error column.
    """
    if "error" in df.columns or df.empty:
        return df
    out = df.copy()
    out["error"] = np.sqrt(np.maximum(out["intensity"].values.astype(float), 0.0))
    return out


def process_pattern(
    df_raw: pd.DataFrame,
    twotheta_range: Tuple[Optional[float], Optional[float]] = (None, None),
    normalize_method: str = "none",
    smoothing_enabled: bool = False,
    smoothing_window: int = 11,
    smoothing_polyorder: int = 2,
    smoothing_method: str = "savgol",
    smoothing_sigma: float = 2.0,
    baseline_enabled: bool = False,
    baseline_method: str = "linear",
    baseline_degree: int = 2,
    baseline_lam: float = 1e5,
    baseline_p: float = 0.01,
    clip_negative_after_baseline: bool = False,
    assume_poisson_error: bool = False,
    **kwargs,
) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
    xmin, xmax = twotheta_range
    df = crop_twotheta_range(df_raw, xmin, xmax)
    if df.empty:
        return df, None, None
    if assume_poisson_error:
        df = estimate_poisson_error(df)
    df = normalize_intensity(df, method=normalize_method)
    if smoothing_enabled:
        s_method = kwargs.get("smoothing_method", smoothing_method)
        s_sigma = kwargs.get("smoothing_sigma", smoothing_sigma)
        df = smooth_dataframe(df, window_length=smoothing_window, polyorder=smoothing_polyorder, method=s_method, sigma=s_sigma)
    baseline_arr = None
    if baseline_enabled:
        x = df["twotheta"].values
        y = df["intensity"].values
        baseline_arr = estimate_baseline(
            x, y,
            method=baseline_method,
            poly_degree=baseline_degree,
            lam=kwargs.get("baseline_lam", baseline_lam),
            p=kwargs.get("baseline_p", baseline_p),
            niter=kwargs.get("baseline_niter", 20),
            snip_iterations=kwargs.get("snip_iterations", 24),
            median_window=kwargs.get("median_window", 151),
        )
        df = subtract_baseline(df, baseline_arr, clip_negative=clip_negative_after_baseline)
    try:
        noise = estimate_noise_std(df["intensity"].values, method="mad")
    except Exception:
        noise = None
    return df, baseline_arr, noise
