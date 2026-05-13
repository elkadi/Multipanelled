from __future__ import annotations

import io
import string

import matplotlib.pyplot as plt
from PIL import Image


def compose_multipanel(panels: list[bytes], nrows: int, ncols: int, labels: list[str] | None = None):
    if not labels:
        labels = list(string.ascii_uppercase[: len(panels)])
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows), dpi=300)
    axes = [axes] if not hasattr(axes, "flatten") else list(axes.flatten())

    for idx, ax in enumerate(axes):
        if idx < len(panels):
            img = Image.open(io.BytesIO(panels[idx])).convert("RGBA")
            ax.imshow(img)
            ax.text(0.02, 0.98, labels[idx], transform=ax.transAxes, va="top", ha="left", fontsize=12, weight="bold")
        ax.axis("off")

    fig.tight_layout()
    return fig
