import pandas as pd

from src.multipanneled_app.export import build_export_zip


def test_build_export_zip_contains_data():
    data = build_export_zip({"a.txt": b"hello"}, {"stats.csv": pd.DataFrame({"x": [1]})})
    assert isinstance(data, bytes)
    assert len(data) > 20
