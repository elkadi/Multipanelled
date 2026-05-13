# Multipanelled

Streamlit app for publication-quality single-panel and multipanel figures, including first-class **Spectra Mode**.

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

## Spectra Mode (wide, long, or one-file-per-spectrum)
Supported spectral representations:
- **Wide matrix**: `sample_id, group, time, 400, 401, ...`
- **Long/tidy**: `sample_id, axis, intensity, group`
- **One file per spectrum**: upload many files and infer `sample_id` from filename, optional metadata merge.

Capabilities:
- Automatic spectral axis detection from numeric column names.
- Metadata column selection and group coloring.
- Optional x-axis reversal for wavenumber/Raman.
- Crop range, Savitzky-Golay smoothing, baseline correction, derivatives (1st/2nd), normalization (min-max, vector norm, AUC, SNV).
- Spectral plots: overlay, grouped means, mean±SEM/SD, stacked, heatmap, PCA scores/loadings, peak vs outcome, feature-importance-style loading vs axis.
- Processed spectra CSV export and multipanel SVG/PDF export.

## Example files
- `examples/synthetic_measured_samples.csv`
- `examples/synthetic_spectra_wide.csv`
- `examples/synthetic_spectra_long.csv`
- `examples/spectrum_sample_A.csv`, `examples/spectrum_sample_B.csv`, `examples/spectrum_metadata.csv`
