"""
tests/test_processing.py - Unit tests for processing.py on synthetic data.

Covers: normalization methods, smoothing, baseline estimation, Bragg's law
(2theta <-> d), Scherrer crystallite size, instrumental-broadening
correction, and 2theta-shift estimation via cross-correlation.

Run with: pytest tests/
"""
import numpy as np
import pandas as pd
import pytest

from processing import (
    normalize_intensity,
    smooth_intensity,
    estimate_baseline,
    subtract_baseline,
    two_theta_to_d,
    d_to_two_theta,
    scherrer_crystallite_size,
    deconvolve_instrumental_fwhm,
    estimate_shift_by_xcorr,
    crop_twotheta_range,
)


def _synthetic_pattern(n=500, peak_center=30.0, peak_width=0.3, peak_height=1000.0,
                        background=50.0, noise_std=2.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(10, 70, n)
    y = background + peak_height * np.exp(-0.5 * ((x - peak_center) / peak_width) ** 2)
    y = y + rng.normal(0, noise_std, size=n)
    return pd.DataFrame({"twotheta": x, "intensity": y})


# --- Normalization ---

def test_normalize_max_scales_to_one():
    df = _synthetic_pattern()
    out = normalize_intensity(df, method="max=1")
    assert np.isclose(out["intensity"].max(), 1.0)


def test_normalize_none_is_unchanged():
    df = _synthetic_pattern()
    out = normalize_intensity(df, method="none")
    assert np.allclose(out["intensity"].values, df["intensity"].values)


def test_normalize_area_preserves_integral_scaling():
    df = _synthetic_pattern()
    out = normalize_intensity(df, method="area=1")
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # np.trapz removed in NumPy 2.0
    area = trapezoid(out["intensity"].values, out["twotheta"].values)
    assert np.isclose(abs(area), 1.0, atol=1e-6)


def test_normalize_zscore_has_zero_mean_unit_std():
    df = _synthetic_pattern()
    out = normalize_intensity(df, method="zscore")
    assert np.isclose(out["intensity"].mean(), 0.0, atol=1e-8)
    # normalize_intensity uses np.nanstd (population std, ddof=0); pandas'
    # .std() defaults to ddof=1 (sample std), so match ddof here rather than
    # use the pandas default, or this diverges by sqrt(n/(n-1)) for finite n.
    assert np.isclose(out["intensity"].std(ddof=0), 1.0, atol=1e-8)


def test_normalize_does_not_mutate_input():
    df = _synthetic_pattern()
    original = df["intensity"].copy()
    normalize_intensity(df, method="max=1")
    assert np.allclose(df["intensity"].values, original.values)


# --- Smoothing ---

def test_savgol_smoothing_reduces_noise_roughness():
    df = _synthetic_pattern(noise_std=5.0)
    smoothed = smooth_intensity(df["intensity"].values, window_length=11, polyorder=2, method="savgol")
    raw_roughness = np.std(np.diff(df["intensity"].values))
    smooth_roughness = np.std(np.diff(smoothed))
    assert smooth_roughness < raw_roughness


@pytest.mark.parametrize("method", ["savgol", "gaussian", "moving_average", "median"])
def test_smoothing_preserves_array_length(method):
    df = _synthetic_pattern()
    out = smooth_intensity(df["intensity"].values, window_length=11, method=method)
    assert len(out) == len(df)
    assert np.all(np.isfinite(out))


# --- Baseline ---

def test_linear_baseline_near_background_level_away_from_peak():
    df = _synthetic_pattern(background=50.0, peak_center=30.0)
    x, y = df["twotheta"].values, df["intensity"].values
    baseline = estimate_baseline(x, y, method="linear")
    edge_mask = (x < 15) | (x > 60)
    assert np.median(np.abs(baseline[edge_mask] - 50.0)) < 10.0


def test_baseline_subtraction_reduces_offpeak_background():
    df = _synthetic_pattern(background=50.0)
    x, y = df["twotheta"].values, df["intensity"].values
    baseline = estimate_baseline(x, y, method="arpls")
    corrected = subtract_baseline(df, baseline)
    edge_mask = (x < 15) | (x > 60)
    assert np.median(np.abs(corrected["intensity"].values[edge_mask])) < np.median(np.abs(y[edge_mask]))


@pytest.mark.parametrize("method", ["linear", "polynomial", "als", "arpls", "snip", "median"])
def test_baseline_methods_all_run_without_crashing(method):
    df = _synthetic_pattern()
    x, y = df["twotheta"].values, df["intensity"].values
    baseline = estimate_baseline(x, y, method=method)
    assert len(baseline) == len(y)
    assert np.all(np.isfinite(baseline))


# --- d-spacing / Bragg's law ---

def test_two_theta_d_round_trip():
    wavelength = 1.5406
    two_theta = 30.0
    d = two_theta_to_d(two_theta, wavelength=wavelength)
    back = d_to_two_theta(d, wavelength=wavelength)
    assert np.isclose(back, two_theta, atol=1e-6)


def test_two_theta_to_d_matches_braggs_law_by_hand():
    # Bragg's law: lambda = 2 d sin(theta)  =>  d = lambda / (2 sin(theta))
    wavelength = 1.5406
    two_theta = 28.44
    theta_rad = np.radians(two_theta / 2.0)
    expected_d = wavelength / (2.0 * np.sin(theta_rad))
    d = two_theta_to_d(two_theta, wavelength=wavelength)
    assert np.isclose(d, expected_d, atol=1e-8)


# --- Scherrer ---

def test_scherrer_larger_fwhm_gives_smaller_crystallite():
    size_narrow = scherrer_crystallite_size(fwhm_deg=0.1, two_theta_deg=30.0)
    size_broad = scherrer_crystallite_size(fwhm_deg=0.5, two_theta_deg=30.0)
    assert size_broad < size_narrow


def test_scherrer_nan_for_invalid_fwhm():
    assert np.isnan(scherrer_crystallite_size(fwhm_deg=0.0, two_theta_deg=30.0))
    assert np.isnan(scherrer_crystallite_size(fwhm_deg=-1.0, two_theta_deg=30.0))


# --- Instrumental broadening correction ---

def test_deconvolve_instrumental_fwhm_matches_quadrature_formula():
    corrected = deconvolve_instrumental_fwhm(fwhm_obs_deg=0.3, instrumental_fwhm_deg=0.1)
    expected = float(np.sqrt(0.3 ** 2 - 0.1 ** 2))
    assert np.isclose(corrected, expected)


def test_deconvolve_instrumental_fwhm_nan_when_unresolved():
    result = deconvolve_instrumental_fwhm(fwhm_obs_deg=0.1, instrumental_fwhm_deg=0.3)
    assert np.isnan(result)


def test_deconvolve_zero_instrumental_is_noop():
    assert deconvolve_instrumental_fwhm(0.3, 0.0) == 0.3


# --- 2theta alignment via cross-correlation ---

def test_estimate_shift_recovers_known_shift():
    df_ref = _synthetic_pattern(peak_center=30.0, noise_std=0.5, seed=1)
    df_shifted = df_ref.copy()
    true_shift = 0.5
    df_shifted["twotheta"] = df_shifted["twotheta"] + true_shift
    estimated = estimate_shift_by_xcorr(df_ref, df_shifted, step=0.02, max_shift_deg=2.0)
    # estimated is the shift to ADD to df_shifted to align it back onto df_ref,
    # i.e. it should recover -true_shift.
    assert abs(estimated - (-true_shift)) < 0.05


# --- Cropping ---

def test_crop_twotheta_range():
    df = _synthetic_pattern()
    cropped = crop_twotheta_range(df, xmin=20.0, xmax=40.0)
    assert cropped["twotheta"].min() >= 20.0
    assert cropped["twotheta"].max() <= 40.0
