"""
tests/test_xrd_io.py - Unit tests for xrd_io.py parsing and validation.

Covers: delimiter auto-detection, comment/header skipping, malformed-line
tolerance, duplicate-2theta handling, negative-2theta preservation, error
column detection, and the high-skip-ratio warning.

Run with: pytest tests/
"""
import numpy as np
import pandas as pd
import pytest

from xrd_io import (
    parse_xrd_content,
    parse_xrd_content_with_validation,
    XRDParseError,
    validate_xrd_dataframe,
)


def test_parse_basic_two_column():
    content = "10.0 100.0\n10.02 105.0\n10.04 110.0\n"
    df = parse_xrd_content(content, filename="test.xy")
    assert list(df["twotheta"]) == [10.0, 10.02, 10.04]
    assert list(df["intensity"]) == [100.0, 105.0, 110.0]


def test_parse_skips_comment_lines():
    content = "# header\n% also comment\n; also\n! also\n10.0 100.0\n10.02 105.0\n10.04 110.0\n"
    df = parse_xrd_content(content, filename="test.xy")
    assert len(df) == 3


@pytest.mark.parametrize("sep", [",", "\t", ";", "    "])
def test_parse_delimiter_variants(sep):
    content = f"10.0{sep}100.0\n10.02{sep}105.0\n10.04{sep}110.0\n"
    df = parse_xrd_content(content)
    assert list(df["twotheta"]) == [10.0, 10.02, 10.04]
    assert list(df["intensity"]) == [100.0, 105.0, 110.0]


def test_parse_error_column_detected():
    content = "10.0 100.0 5.0\n10.02 105.0 5.1\n10.04 110.0 5.2\n"
    df = parse_xrd_content(content)
    assert "error" in df.columns
    assert list(df["error"]) == [5.0, 5.1, 5.2]


def test_parse_duplicate_twotheta_keeps_last_occurrence():
    content = "10.0 100.0\n10.0 999.0\n10.02 105.0\n10.04 110.0\n"
    df = parse_xrd_content(content)
    assert len(df) == 3
    row = df[df["twotheta"] == 10.0]
    assert row["intensity"].iloc[0] == 999.0


def test_parse_sorts_ascending_even_if_input_unsorted():
    content = "10.04 110.0\n10.0 100.0\n10.02 105.0\n"
    df = parse_xrd_content(content)
    assert list(df["twotheta"]) == [10.0, 10.02, 10.04]


def test_parse_negative_twotheta_preserved():
    # Some lab instruments start scans below 0 deg; this must survive parsing.
    content = "-16.0 50.0\n-15.98 55.0\n-15.96 60.0\n"
    df = parse_xrd_content(content)
    assert df["twotheta"].min() == -16.0


def test_parse_malformed_lines_skipped_not_crashed():
    content = "this is garbage\n10.0 100.0\nalso garbage 12\n10.02 105.0\n10.04 110.0\n"
    df = parse_xrd_content(content)
    assert len(df) == 3


def test_parse_too_few_points_raises():
    content = "10.0 100.0\n10.02 105.0\n"  # only 2 valid points, minimum is 3
    with pytest.raises(XRDParseError):
        parse_xrd_content(content)


def test_parse_empty_content_raises():
    with pytest.raises(XRDParseError):
        parse_xrd_content("")


def test_validate_flags_all_nonpositive_intensity():
    df = pd.DataFrame({"twotheta": [10.0, 20.0, 30.0], "intensity": [-1.0, -2.0, -3.0]})
    issues = validate_xrd_dataframe(df)
    assert any("negative" in i.lower() or "<= 0" in i for i in issues)


def test_validate_flags_too_few_points():
    df = pd.DataFrame({"twotheta": [10.0, 20.0], "intensity": [1.0, 2.0]})
    issues = validate_xrd_dataframe(df)
    assert any("few" in i.lower() for i in issues)


def test_validate_clean_pattern_has_no_issues():
    x = np.linspace(10, 70, 200)
    y = 50.0 + np.abs(np.sin(x))
    df = pd.DataFrame({"twotheta": x, "intensity": y})
    issues = validate_xrd_dataframe(df)
    assert issues == []


def test_skip_ratio_warns_on_mostly_garbage_file():
    # 10 garbage lines (each with a stray digit) + 3 real data lines: most
    # candidate lines fail to parse, which should surface a clear warning
    # instead of silently plotting a nearly-empty pattern.
    lines = [f"garbage text {i} x" for i in range(10)]
    lines += ["10.0 100.0", "10.02 105.0", "10.04 110.0"]
    content = "\n".join(lines)
    df, meta, issues = parse_xrd_content_with_validation(content, filename="bad.xy")
    assert len(df) == 3
    assert any("dropped" in i.lower() for i in issues)


def test_skip_ratio_silent_on_clean_file():
    content = "# header\n10.0 100.0\n10.02 105.0\n10.04 110.0\n"
    df, meta, issues = parse_xrd_content_with_validation(content, filename="clean.xy")
    assert not any("dropped" in i.lower() for i in issues)
