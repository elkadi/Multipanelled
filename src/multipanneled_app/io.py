from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Iterable

import pandas as pd


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]


def load_csv_bytes(file_name: str, content: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(content))


def detect_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def validate_columns(
    df: pd.DataFrame,
    *,
    required: Iterable[str],
    numeric: Iterable[str] = (),
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    required = [c for c in required if c]
    numeric = [c for c in numeric if c]

    for col in required:
        if col not in df.columns:
            errors.append(f"Missing required column: '{col}'")

    for col in numeric:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' must be numeric for this plot/stat operation")

    null_cols = [c for c in required if c in df.columns and df[c].isna().any()]
    if null_cols:
        warnings.append(f"Missing values detected in: {', '.join(null_cols)}. Rows with NaN may be dropped.")

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)
