# Multipanelled

A Streamlit application for turning notebook-style figure workflows into publication-quality single-panel and multipanel exports.

## Features
- Multi-CSV upload and data preview.
- Single-panel plotting with manuscript presets.
- Plot types: scatter, regression, line, bar, box, violin, feature-importance, predicted-vs-measured.
- Optional stats tables: group summary, one-way ANOVA, Tukey HSD, Pearson, linear regression stats, repeated-measures correlation (if `pingouin` installed).
- Panel tray to compose multipanel layouts (including 2x2) with panel labels.
- Export as PNG, SVG, PDF, and ZIP bundle with figures + stats CSV tables.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

## Expected input format
Any CSV with user-selectable columns; typical fields:
- Numeric x/y columns (e.g., `time`, `modulus`, `measured`, `predicted`)
- Optional grouping column (e.g., `group`)
- Optional repeated-measures subject/replicate column (e.g., `sample_id`)

See `examples/synthetic_measured_samples.csv` for a runnable sample dataset.

## Example workflow
1. Upload one or more CSV files.
2. Select a dataset and choose columns + plot type.
3. Generate and download a single panel.
4. Add panels to tray.
5. Build multipanel grid (e.g., 2x2) and export SVG/PDF/PNG.
6. Optionally compute/export stats tables and download ZIP.

## Notebook conversion notes
The plotting and stats logic are refactored into modules under `src/multipanneled_app/` so no notebook-local paths are required.

## Screenshots
- Run the app and capture screenshots from:
  - Single panel preview state
  - Panel tray + multipanel builder state

