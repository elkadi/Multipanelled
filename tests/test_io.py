import pandas as pd

from src.multipanneled_app.io import detect_numeric_columns, validate_columns


def test_validate_columns_reports_missing_and_type_errors():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    result = validate_columns(df, required=["a", "c"], numeric=["a", "b"])
    assert not result.ok
    assert any("Missing required column" in e for e in result.errors)
    assert any("must be numeric" in e for e in result.errors)


def test_detect_numeric_columns():
    df = pd.DataFrame({"x": [1, 2], "y": [1.5, 2.1], "g": ["a", "b"]})
    assert set(detect_numeric_columns(df)) == {"x", "y"}
