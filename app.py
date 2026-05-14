from __future__ import annotations

import io
import pandas as pd
import streamlit as st

from src.multipanneled_app.export import build_export_zip, fig_to_bytes
from src.multipanneled_app.io import load_csv_bytes, validate_columns
from src.multipanneled_app.multipanel import compose_multipanel
from src.multipanneled_app.plotting import PlotConfig, TEMPLATES, create_plot
from src.multipanneled_app import stats as stats_mod
from src.multipanneled_app import spectra as sp

st.set_page_config(page_title="Multipanelled", layout="wide")
st.title("Multi-panel Figure Builder")

for key, default in [("datasets", {}), ("panel_tray", []), ("stats_tables", {})]:
    if key not in st.session_state:
        st.session_state[key] = default

mode = st.radio("Mode", ["Standard Mode", "Spectra Mode"], horizontal=True)
uploads = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
if uploads:
    for up in uploads:
        st.session_state.datasets[up.name] = load_csv_bytes(up.name, up.getvalue())
if not st.session_state.datasets:
    st.stop()

if mode == "Standard Mode":
    selected_name = st.selectbox("Dataset", list(st.session_state.datasets.keys()))
    df = st.session_state.datasets[selected_name]
    st.dataframe(df.head())
    cols = list(df.columns)
    plot_type = st.selectbox("Plot type", ["scatter", "regression", "line", "bar", "box", "violin", "feature importance", "predicted-vs-measured"])
    template = st.selectbox("Template", ["None", *TEMPLATES.keys()])
    if template != "None":
        plot_type = TEMPLATES[template]
    x, y = st.selectbox("X", cols), st.selectbox("Y", cols)
    hue = st.selectbox("Hue", ["", *cols]) or None
    v = validate_columns(df, required=[x, y], numeric=[x, y])
    if v.ok:
        fig, _ = create_plot(df, PlotConfig(plot_type=plot_type, x=x, y=y, hue=hue))
        st.pyplot(fig)
        png = fig_to_bytes(fig, "png")
        st.download_button("PNG", png, file_name="panel.png")
        if st.button("Add panel"):
            st.session_state.panel_tray.append({"name": f"{selected_name}:{plot_type}", "png": png})

else:
    st.subheader("Spectra Mode")
    fmt = st.selectbox("Spectral data format", ["Wide matrix", "Long/tidy", "One file per spectrum"])
    names = list(st.session_state.datasets.keys())

    if fmt == "Wide matrix":
        name = st.selectbox("Wide spectra CSV", names)
        df = st.session_state.datasets[name]
        meta_cols = st.multiselect("Metadata columns", df.columns.tolist(), default=[c for c in ["sample_id", "group", "time"] if c in df.columns])
        axis_cols = sp.detect_spectral_axis_columns(df, meta_cols)
        st.write(f"Detected spectral axis columns: {len(axis_cols)}")
        spec = sp.from_wide(df, meta_cols)
    elif fmt == "Long/tidy":
        name = st.selectbox("Long spectra CSV", names)
        df = st.session_state.datasets[name]
        sid = st.selectbox("Sample ID", df.columns.tolist())
        axis_col = st.selectbox("Spectral axis column", df.columns.tolist())
        intensity = st.selectbox("Intensity column", df.columns.tolist())
        meta_cols = st.multiselect("Metadata columns", df.columns.tolist())
        spec = sp.from_long(df, sid, axis_col, intensity, meta_cols)
    else:
        selected = st.multiselect("Select per-spectrum files", names)
        axis_col = st.text_input("Axis column", "axis")
        intensity = st.text_input("Intensity column", "intensity")
        metadata_name = st.selectbox("Optional metadata CSV", ["", *names])
        metadata = st.session_state.datasets[metadata_name] if metadata_name else None
        file_map = {k: st.session_state.datasets[k] for k in selected}
        if not file_map:
            st.stop()
        spec = sp.from_files(file_map, axis_col, intensity, metadata=metadata)

    reverse_x = st.checkbox("Reverse X axis (Raman/wavenumber)", value=False)
    c1, c2 = st.columns(2)
    crop_min = c1.text_input("Crop min", "")
    crop_max = c2.text_input("Crop max", "")
    smooth = st.number_input("Savitzky-Golay window (odd, 0 off)", 0, 51, 0, step=2)
    smooth_poly = st.number_input("Savitzky-Golay polyorder", 1, 6, 2)
    baseline = st.checkbox("Baseline correction (subtract per-spectrum min)")
    derivative = st.selectbox("Derivative", [0, 1, 2])
    norm = st.selectbox("Normalization", ["none", "min-max", "vector norm", "auc", "snv"])

    processed = sp.preprocess_spectra(spec, crop_min=float(crop_min) if crop_min else None, crop_max=float(crop_max) if crop_max else None, smooth_window=smooth, smooth_poly=smooth_poly, baseline=baseline, derivative=derivative, normalization=norm)
    plot_type = st.selectbox("Spectral plot type", ["raw spectra overlay", "processed spectra overlay", "mean spectra by group", "mean ± SEM by group", "mean ± SD by group", "stacked spectra", "spectral heatmap", "PCA score plot", "PCA loading plot", "peak intensity vs metadata outcome", "feature importance vs wavelength/wavenumber"])
    group_col = st.selectbox("Group/color metadata column", ["", *processed.metadata.columns.tolist()]) or None

    fig = None
    if plot_type == "raw spectra overlay":
        fig, _ = sp.plot_overlay(spec, group_col=group_col, reverse_x=reverse_x, title="Raw spectra")
    elif plot_type == "processed spectra overlay":
        fig, _ = sp.plot_overlay(processed, group_col=group_col, reverse_x=reverse_x, title="Processed spectra")
    elif plot_type == "mean spectra by group":
        fig, _ = sp.plot_group_mean(processed, group_col=group_col or processed.metadata.columns[0], reverse_x=reverse_x)
    elif plot_type == "mean ± SEM by group":
        fig, _ = sp.plot_group_mean(processed, group_col=group_col or processed.metadata.columns[0], band="sem", reverse_x=reverse_x)
    elif plot_type == "mean ± SD by group":
        fig, _ = sp.plot_group_mean(processed, group_col=group_col or processed.metadata.columns[0], band="sd", reverse_x=reverse_x)
    elif plot_type == "stacked spectra":
        fig, _ = sp.plot_stacked(processed, reverse_x=reverse_x)
    elif plot_type == "spectral heatmap":
        fig, _ = sp.plot_heatmap(processed)
    else:
        scores, loadings, _ = sp.run_pca(processed)
        if plot_type == "PCA score plot":
            fig, ax = __import__('matplotlib.pyplot').pyplot.subplots(figsize=(6,4), dpi=300)
            if group_col and group_col in scores.columns:
                for g, d in scores.groupby(group_col):
                    ax.scatter(d['PC1'], d['PC2'], label=str(g))
                ax.legend()
            else:
                ax.scatter(scores['PC1'], scores['PC2'])
        elif plot_type == "PCA loading plot":
            fig, ax = __import__('matplotlib.pyplot').pyplot.subplots(figsize=(6,4), dpi=300)
            ax.plot(loadings.index.astype(float), loadings['PC1'])
        elif plot_type == "peak intensity vs metadata outcome":
            outcome = st.selectbox("Outcome metadata", processed.metadata.columns.tolist())
            peak = processed.X.max(axis=1)
            fig, ax = __import__('matplotlib.pyplot').pyplot.subplots(figsize=(6,4), dpi=300)
            ax.scatter(processed.metadata[outcome], peak)
        else:
            fig, ax = __import__('matplotlib.pyplot').pyplot.subplots(figsize=(6,4), dpi=300)
            ax.plot(processed.axis, loadings['PC1'].abs())

    st.pyplot(fig)
    png = fig_to_bytes(fig, "png")
    svg = fig_to_bytes(fig, "svg")
    pdf = fig_to_bytes(fig, "pdf")
    st.download_button("Download PNG", png, file_name="spectra_panel.png")
    st.download_button("Download SVG", svg, file_name="spectra_panel.svg")
    st.download_button("Download PDF", pdf, file_name="spectra_panel.pdf")
    if st.button("Add spectra panel"):
        st.session_state.panel_tray.append({"name": plot_type, "png": png, "svg": svg, "pdf": pdf})

    proc_df = processed.X.copy(); proc_df.columns = processed.axis
    if "sample_id" in processed.metadata.columns:
        proc_df.insert(0, "sample_id", processed.metadata["sample_id"])
    st.download_button("Export processed spectra CSV", proc_df.to_csv(index=False).encode(), file_name="processed_spectra.csv")

st.subheader("Multipanel builder")
rows = st.number_input("Rows", 1, 6, 2)
cols = st.number_input("Cols", 1, 6, 2)
if st.button("Build multipanel") and st.session_state.panel_tray:
    mfig = compose_multipanel([p["png"] for p in st.session_state.panel_tray], int(rows), int(cols))
    st.pyplot(mfig)
    msvg = fig_to_bytes(mfig, "svg")
    mpdf = fig_to_bytes(mfig, "pdf")
    st.download_button("Multipanel SVG", msvg, file_name="multipanel.svg")
    st.download_button("Multipanel PDF", mpdf, file_name="multipanel.pdf")
    figs = {f"panel_{i+1}.png": p["png"] for i,p in enumerate(st.session_state.panel_tray)}
    zip_b = build_export_zip(figs, st.session_state.stats_tables)
    st.download_button("ZIP", zip_b, file_name="export.zip")
