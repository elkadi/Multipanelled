from __future__ import annotations

import pandas as pd
import streamlit as st

from src.multipanneled_app.export import build_export_zip, fig_to_bytes
from src.multipanneled_app.io import detect_numeric_columns, load_csv_bytes, validate_columns
from src.multipanneled_app.multipanel import compose_multipanel
from src.multipanneled_app.plotting import PlotConfig, TEMPLATES, create_plot
from src.multipanneled_app import stats as stats_mod

st.set_page_config(page_title="Multipanelled", layout="wide")
st.title("Multipanelled: Manuscript Figure Builder")
st.markdown("Upload CSV files, build single panels, then compose publication-ready multi-panel figures.")

if "datasets" not in st.session_state:
    st.session_state.datasets = {}
if "panel_tray" not in st.session_state:
    st.session_state.panel_tray = []
if "stats_tables" not in st.session_state:
    st.session_state.stats_tables = {}

uploads = st.file_uploader("Upload one or more CSV files", type=["csv"], accept_multiple_files=True)
if uploads:
    for up in uploads:
        st.session_state.datasets[up.name] = load_csv_bytes(up.name, up.getvalue())

if not st.session_state.datasets:
    st.info("Upload at least one CSV file to begin.")
    st.stop()

selected_name = st.selectbox("Dataset", options=list(st.session_state.datasets.keys()))
df = st.session_state.datasets[selected_name]
st.subheader("Data preview")
st.dataframe(df.head(20), use_container_width=True)

cols = list(df.columns)
numeric_cols = detect_numeric_columns(df)
plot_type = st.selectbox("Plot type", ["scatter", "regression", "line", "bar", "box", "violin", "feature importance", "predicted-vs-measured"])
template = st.selectbox("Optional template", ["None", *TEMPLATES.keys()])
if template != "None":
    plot_type = TEMPLATES[template]

x = st.selectbox("X column", cols)
y = st.selectbox("Y column", cols)
hue = st.selectbox("Hue/group column (optional)", ["", *cols]) or None
sample_id = st.selectbox("Replicate/sample ID (optional)", ["", *cols]) or None

style = st.selectbox("Style preset", ["manuscript", "presentation", "grayscale", "colorblind-friendly"])
width = st.slider("Width", 3.0, 14.0, 6.0)
height = st.slider("Height", 2.0, 10.0, 4.0)
dpi = st.slider("DPI", 150, 600, 300)
font_scale = st.slider("Font scale", 0.6, 2.0, 1.0)
legend_pos = st.selectbox("Legend position", ["best", "upper right", "upper left", "lower right", "lower left"])
title = st.text_input("Title", "")
x_label = st.text_input("X label", "")
y_label = st.text_input("Y label", "")

needs_numeric = plot_type in {"scatter", "regression", "line", "bar", "violin", "feature importance", "predicted-vs-measured"}
validation = validate_columns(df, required=[x, y], numeric=[x, y] if needs_numeric else [])
for w in validation.warnings:
    st.warning(w)
for e in validation.errors:
    st.error(e)

if validation.ok:
    cfg = PlotConfig(plot_type=plot_type, x=x, y=y, hue=hue, title=title, x_label=x_label, y_label=y_label, legend_position=legend_pos, width=width, height=height, dpi=dpi, font_scale=font_scale, style_preset=style)
    fig, _ = create_plot(df, cfg)
    st.pyplot(fig, use_container_width=False)

    png = fig_to_bytes(fig, "png", dpi=dpi)
    svg = fig_to_bytes(fig, "svg", dpi=dpi)
    pdf = fig_to_bytes(fig, "pdf", dpi=dpi)
    st.download_button("Download PNG", png, file_name="panel.png", mime="image/png")
    st.download_button("Download SVG", svg, file_name="panel.svg", mime="image/svg+xml")
    st.download_button("Download PDF", pdf, file_name="panel.pdf", mime="application/pdf")

    if st.button("Add panel to tray"):
        st.session_state.panel_tray.append({"name": f"{selected_name}:{plot_type}", "png": png, "svg": svg, "pdf": pdf})

st.subheader("Optional statistics")
stat_kind = st.selectbox("Statistic", ["None", "Group summary", "ANOVA", "Tukey HSD", "Pearson", "Regression", "Repeated-measures corr"])
if stat_kind != "None":
    try:
        if stat_kind == "Group summary":
            table = stats_mod.group_summary(df, hue or x, y)
        elif stat_kind == "ANOVA":
            table = stats_mod.anova_one_way(df, hue or x, y)
        elif stat_kind == "Tukey HSD":
            table = stats_mod.tukey_hsd(df, hue or x, y)
        elif stat_kind == "Pearson":
            table = stats_mod.pearson_correlation(df, x, y)
        elif stat_kind == "Regression":
            table = stats_mod.regression_stats(df, x, y)
        else:
            if not sample_id:
                st.warning("Select replicate/sample ID column to run repeated-measures correlation.")
                table = pd.DataFrame()
            else:
                table = stats_mod.repeated_measures_corr(df, sample_id, x, y)
        if not table.empty:
            st.dataframe(table, use_container_width=True)
            st.session_state.stats_tables[f"stats_{stat_kind.lower().replace(' ', '_')}.csv"] = table
    except Exception as exc:
        st.error(f"Could not compute statistic: {exc}")

st.subheader("Panel tray & multipanel builder")
st.write(f"Panels in tray: {len(st.session_state.panel_tray)}")
for i, panel in enumerate(st.session_state.panel_tray):
    st.write(f"{i+1}. {panel['name']}")

nrows = st.number_input("Rows", min_value=1, max_value=6, value=2)
ncols = st.number_input("Columns", min_value=1, max_value=6, value=2)
if st.button("Build multipanel") and st.session_state.panel_tray:
    combined = compose_multipanel([p["png"] for p in st.session_state.panel_tray], int(nrows), int(ncols))
    st.pyplot(combined)
    m_png = fig_to_bytes(combined, "png", dpi=300)
    m_svg = fig_to_bytes(combined, "svg", dpi=300)
    m_pdf = fig_to_bytes(combined, "pdf", dpi=300)
    st.download_button("Download multipanel PNG", m_png, file_name="multipanel.png")
    st.download_button("Download multipanel SVG", m_svg, file_name="multipanel.svg")
    st.download_button("Download multipanel PDF", m_pdf, file_name="multipanel.pdf")

    all_figs = {f"panel_{i+1}.png": p["png"] for i, p in enumerate(st.session_state.panel_tray)}
    all_figs.update({"multipanel.png": m_png, "multipanel.svg": m_svg, "multipanel.pdf": m_pdf})
    zip_bytes = build_export_zip(all_figs, st.session_state.stats_tables)
    st.download_button("Download ZIP (figures + stats)", zip_bytes, file_name="multipanelled_export.zip", mime="application/zip")
