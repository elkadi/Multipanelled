from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA


@dataclass
class SpectraData:
    X: pd.DataFrame
    axis: pd.Index
    metadata: pd.DataFrame


def detect_spectral_axis_columns(df: pd.DataFrame, metadata_cols: list[str] | None = None) -> list[str]:
    metadata_cols = metadata_cols or []
    cols = []
    for c in df.columns:
        if c in metadata_cols:
            continue
        try:
            float(str(c))
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
        except ValueError:
            continue
    return cols


def from_wide(df: pd.DataFrame, metadata_cols: list[str]) -> SpectraData:
    axis_cols = detect_spectral_axis_columns(df, metadata_cols)
    if not axis_cols:
        raise ValueError("No spectral-axis columns detected in wide table.")
    X = df[axis_cols].apply(pd.to_numeric, errors="coerce")
    meta = df[[c for c in metadata_cols if c in df.columns]].copy()
    return SpectraData(X=X, axis=pd.Index([float(c) for c in axis_cols], name="axis"), metadata=meta)


def from_long(df: pd.DataFrame, sample_id: str, axis_col: str, intensity_col: str, metadata_cols: list[str] | None = None) -> SpectraData:
    metadata_cols = metadata_cols or []
    piv = df.pivot_table(index=sample_id, columns=axis_col, values=intensity_col, aggfunc="mean").sort_index(axis=1)
    meta_cols = [c for c in metadata_cols if c in df.columns and c not in {axis_col, intensity_col}]
    meta = df[[sample_id, *meta_cols]].drop_duplicates(subset=[sample_id]).set_index(sample_id).reindex(piv.index).reset_index()
    return SpectraData(X=piv.reset_index(drop=True), axis=pd.Index(piv.columns.astype(float), name="axis"), metadata=meta)


def from_files(file_map: dict[str, pd.DataFrame], axis_col: str, intensity_col: str, metadata: pd.DataFrame | None = None, sample_id_col: str = "sample_id") -> SpectraData:
    rows = []
    for fname, dfi in file_map.items():
        sid = Path(fname).stem
        tmp = dfi[[axis_col, intensity_col]].copy()
        tmp[sample_id_col] = sid
        rows.append(tmp)
    long_df = pd.concat(rows, ignore_index=True)
    meta_cols = [sample_id_col]
    if metadata is not None and sample_id_col in metadata.columns:
        long_df = long_df.merge(metadata, on=sample_id_col, how="left")
        meta_cols = list(metadata.columns)
    return from_long(long_df, sample_id_col, axis_col, intensity_col, meta_cols)


def preprocess_spectra(spec: SpectraData, crop_min=None, crop_max=None, smooth_window=0, smooth_poly=2, baseline=False, derivative=0, normalization="none") -> SpectraData:
    X = spec.X.copy().to_numpy(dtype=float)
    axis = spec.axis.to_numpy(dtype=float)

    if crop_min is not None or crop_max is not None:
        mask = np.ones_like(axis, dtype=bool)
        if crop_min is not None:
            mask &= axis >= float(crop_min)
        if crop_max is not None:
            mask &= axis <= float(crop_max)
        axis = axis[mask]
        X = X[:, mask]

    if smooth_window and smooth_window >= 3 and smooth_window % 2 == 1:
        X = savgol_filter(X, window_length=smooth_window, polyorder=min(smooth_poly, smooth_window - 1), axis=1)

    if baseline:
        X = X - X.min(axis=1, keepdims=True)

    if derivative in (1, 2):
        for _ in range(derivative):
            X = np.gradient(X, axis=1)

    if normalization == "min-max":
        mn = X.min(axis=1, keepdims=True)
        mx = X.max(axis=1, keepdims=True)
        X = (X - mn) / np.where((mx - mn) == 0, 1, (mx - mn))
    elif normalization == "vector norm":
        nm = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.where(nm == 0, 1, nm)
    elif normalization == "auc":
        auc = np.trapz(X, axis=1).reshape(-1, 1)
        X = X / np.where(auc == 0, 1, auc)
    elif normalization == "snv":
        X = (X - X.mean(axis=1, keepdims=True)) / np.where(X.std(axis=1, keepdims=True) == 0, 1, X.std(axis=1, keepdims=True))

    return SpectraData(X=pd.DataFrame(X), axis=pd.Index(axis, name="axis"), metadata=spec.metadata.copy())


def spectra_to_long(spec: SpectraData, sample_id_col: str = "sample_id") -> pd.DataFrame:
    sid = spec.metadata[sample_id_col] if sample_id_col in spec.metadata.columns else pd.Series(np.arange(len(spec.X)), name=sample_id_col)
    out = spec.X.copy()
    out.columns = spec.axis
    out[sample_id_col] = sid.values
    return out.melt(id_vars=[sample_id_col], var_name="axis", value_name="intensity")


def run_pca(spec: SpectraData, n_components: int = 2):
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(spec.X.to_numpy(dtype=float))
    score_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(n_components)])
    score_df = pd.concat([score_df, spec.metadata.reset_index(drop=True)], axis=1)
    loadings = pd.DataFrame(pca.components_.T, index=spec.axis, columns=[f"PC{i+1}" for i in range(n_components)])
    return score_df, loadings, pca


def plot_overlay(spec: SpectraData, group_col: str | None = None, reverse_x=False, title="Spectra overlay"):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    X = spec.X.to_numpy(dtype=float)
    for i in range(X.shape[0]):
        label = None
        if group_col and group_col in spec.metadata.columns:
            label = str(spec.metadata.iloc[i][group_col])
        ax.plot(spec.axis, X[i], alpha=0.6, label=label)
    if reverse_x:
        ax.invert_xaxis()
    ax.set_title(title)
    ax.set_xlabel(spec.axis.name or "axis")
    ax.set_ylabel("intensity")
    return fig, ax

def plot_group_mean(spec: SpectraData, group_col: str, band: str | None = None, reverse_x=False):
    long_df = spectra_to_long(spec)
    m = spec.metadata[[group_col]].reset_index(drop=True)
    long_df[group_col] = np.repeat(m[group_col].values, len(spec.axis))
    agg = long_df.groupby([group_col, 'axis'])['intensity'].agg(['mean', 'std', 'count']).reset_index()
    agg['sem'] = agg['std'] / np.sqrt(agg['count'])
    fig, ax = plt.subplots(figsize=(7,4), dpi=300)
    for grp, dfg in agg.groupby(group_col):
        ax.plot(dfg['axis'], dfg['mean'], label=str(grp))
        if band == 'sem':
            ax.fill_between(dfg['axis'], dfg['mean']-dfg['sem'], dfg['mean']+dfg['sem'], alpha=0.2)
        elif band == 'sd':
            ax.fill_between(dfg['axis'], dfg['mean']-dfg['std'], dfg['mean']+dfg['std'], alpha=0.2)
    if reverse_x:
        ax.invert_xaxis()
    ax.legend()
    return fig, ax


def plot_stacked(spec: SpectraData, reverse_x=False):
    fig, ax = plt.subplots(figsize=(7,4), dpi=300)
    X = spec.X.to_numpy(dtype=float)
    step = np.nanmax(np.abs(X)) * 0.1
    for i, row in enumerate(X):
        ax.plot(spec.axis, row + i * step)
    if reverse_x:
        ax.invert_xaxis()
    return fig, ax


def plot_heatmap(spec: SpectraData):
    fig, ax = plt.subplots(figsize=(7,4), dpi=300)
    sns.heatmap(spec.X, cmap='viridis', ax=ax)
    return fig, ax
