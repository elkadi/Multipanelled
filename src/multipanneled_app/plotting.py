from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

STYLE_PRESETS = {
    "manuscript": {"context": "paper", "style": "whitegrid", "palette": "deep"},
    "presentation": {"context": "talk", "style": "whitegrid", "palette": "tab10"},
    "grayscale": {"context": "paper", "style": "whitegrid", "palette": "Greys"},
    "colorblind-friendly": {"context": "paper", "style": "whitegrid", "palette": "colorblind"},
}

TEMPLATES = {
    "Fig 3 modulus/time": "line",
    "Fig 4 prediction/error": "predicted-vs-measured",
    "Fig 5 feature importance": "feature importance",
    "Fig 6 measured/predicted": "regression",
    "Graphical abstract": "scatter",
}


@dataclass
class PlotConfig:
    plot_type: str
    x: str
    y: str
    hue: str | None = None
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    legend_position: str = "best"
    width: float = 6.0
    height: float = 4.0
    dpi: int = 300
    font_scale: float = 1.0
    style_preset: str = "manuscript"


def apply_style(preset: str, font_scale: float = 1.0) -> None:
    cfg = STYLE_PRESETS.get(preset, STYLE_PRESETS["manuscript"])
    sns.set_theme(context=cfg["context"], style=cfg["style"], palette=cfg["palette"], font_scale=font_scale)
    plt.rcParams["svg.fonttype"] = "none"


def create_plot(df: pd.DataFrame, cfg: PlotConfig):
    apply_style(cfg.style_preset, cfg.font_scale)
    fig, ax = plt.subplots(figsize=(cfg.width, cfg.height), dpi=cfg.dpi)
    plot_df = df.dropna(subset=[c for c in [cfg.x, cfg.y, cfg.hue] if c])

    pt = cfg.plot_type.lower()
    if pt == "scatter":
        sns.scatterplot(data=plot_df, x=cfg.x, y=cfg.y, hue=cfg.hue, ax=ax)
    elif pt == "regression":
        sns.regplot(data=plot_df, x=cfg.x, y=cfg.y, ax=ax, scatter_kws={"alpha": 0.8})
    elif pt == "line":
        sns.lineplot(data=plot_df, x=cfg.x, y=cfg.y, hue=cfg.hue, marker="o", ax=ax)
    elif pt == "bar":
        sns.barplot(data=plot_df, x=cfg.x, y=cfg.y, hue=cfg.hue, ax=ax, errorbar="se")
    elif pt == "box":
        sns.boxplot(data=plot_df, x=cfg.x, y=cfg.y, hue=cfg.hue, ax=ax)
    elif pt == "violin":
        sns.violinplot(data=plot_df, x=cfg.x, y=cfg.y, hue=cfg.hue, ax=ax)
    elif pt == "feature importance":
        tmp = plot_df.sort_values(cfg.y, ascending=False)
        sns.barplot(data=tmp, x=cfg.y, y=cfg.x, ax=ax)
    elif pt == "predicted-vs-measured":
        sns.scatterplot(data=plot_df, x=cfg.x, y=cfg.y, hue=cfg.hue, ax=ax)
        mn = min(plot_df[cfg.x].min(), plot_df[cfg.y].min())
        mx = max(plot_df[cfg.x].max(), plot_df[cfg.y].max())
        ax.plot([mn, mx], [mn, mx], "--", color="black", linewidth=1)
    else:
        raise ValueError(f"Unsupported plot type: {cfg.plot_type}")

    ax.set_title(cfg.title)
    ax.set_xlabel(cfg.x_label or cfg.x)
    ax.set_ylabel(cfg.y_label or cfg.y)
    if ax.get_legend() is not None:
        ax.legend(loc=cfg.legend_position)
    fig.tight_layout()
    return fig, ax
