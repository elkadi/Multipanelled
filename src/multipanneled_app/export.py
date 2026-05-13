from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd


def fig_to_bytes(fig, fmt: str = "png", dpi: int = 300) -> bytes:
    bio = io.BytesIO()
    fig.savefig(bio, format=fmt, dpi=dpi, bbox_inches="tight")
    return bio.getvalue()


def save_figure(fig, out_path: str | Path, fmt: str, dpi: int = 300) -> Path:
    out_path = Path(out_path)
    fig.savefig(out_path, format=fmt, dpi=dpi, bbox_inches="tight")
    return out_path


def build_export_zip(figures: dict[str, bytes], stats_tables: dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, b in figures.items():
            zf.writestr(name, b)
        for name, df in stats_tables.items():
            zf.writestr(name, df.to_csv(index=False).encode("utf-8"))
    return bio.getvalue()
