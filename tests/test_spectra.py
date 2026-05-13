import pandas as pd

from src.multipanneled_app.spectra import (
    detect_spectral_axis_columns,
    from_wide,
    preprocess_spectra,
    run_pca,
)


def test_detect_spectral_columns_wide():
    df = pd.DataFrame({"sample_id": ["s1"], "group": ["A"], "400": [1.0], "401": [2.0]})
    cols = detect_spectral_axis_columns(df, ["sample_id", "group"])
    assert cols == ["400", "401"]


def test_preprocess_and_pca():
    df = pd.DataFrame({"sample_id": ["s1", "s2"], "group": ["A", "B"], "400": [1.0, 2.0], "401": [1.5, 2.5], "402": [2.0, 3.0]})
    spec = from_wide(df, ["sample_id", "group"])
    proc = preprocess_spectra(spec, crop_min=400, crop_max=401, normalization="snv")
    assert proc.X.shape == (2, 2)
    scores, loadings, _ = run_pca(proc, n_components=2)
    assert "PC1" in scores.columns
    assert "PC1" in loadings.columns
